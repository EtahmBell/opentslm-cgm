"""
Tiny repaired Stage-5 pairwise overfit diagnostic.

Purpose
-------
We previously established that the public Stage-5 ECG encoder
strongly contracts CGM information in Transformer blocks 5 and 6.

Encoder surgery showed that replacing BOTH Stage-5 T5/T6 with
their TSQA counterparts restores substantial CGM representation
variation.

This experiment asks the next question:

    Once the signal path is repaired, can Stage-5 MEMORIZE
    a deliberately tiny balanced CGM classification set?

Task
----
above_180

Tiny dataset:
    10 YES
    10 NO

Training:
    30 epochs
    same 20 examples every epoch
    fixed alternating YES/NO order

Objective:
    TRUE TWO-CLASS PAIRWISE CE

At the actual generation position after the prompt:

    logits[(a]
    logits[(b]

are extracted from the full vocabulary and CE is applied ONLY
between those two logits.

This is NOT the previous full-vocabulary class-token objective.

Conditions
----------
0. CONTROL
   Original Stage-5 T5/T6
   Stage-5 projector

1. MINIMAL REPAIR
   TSQA T5/T6 transplanted
   Stage-5 projector

2. STRONGER REPAIR
   TSQA T5/T6 transplanted
   TSQA projector transplanted

For ALL conditions:
    Stage-5 patch embedding + T1-T4 = FROZEN
    T5/T6                         = TRAINABLE
    projector                     = TRAINABLE
    Stage-5 LLM                   = FROZEN
    Stage-5 LoRA                  = FROZEN

No test participants are used.

This is a capacity/mechanism diagnostic, NOT a final model.
"""

import gc
import json
import math
import random
import sys

from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.cgm_multitask import (
    build_multitask_examples,
)

from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

SEED = 42

TSQA_REPO_ID = (
    "OpenTSLM/llama-3.2-1b-tsqa-sp"
)

STAGE5_REPO_ID = (
    "OpenTSLM/llama-3.2-1b-ecg-sp"
)

DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "stage5_repaired_tiny_overfit_pairwise.json"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

N_PER_CLASS = 10

EPOCHS = 30


# Keep these close to our earlier Stage-5 adaptation recipe.
LR_T5_T6 = 2e-4
LR_PROJECTOR = 1e-4

WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0


EVAL_EPOCHS = {
    0,
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    30,
}


# ============================================================
# Reproducibility
# ============================================================

random.seed(
    SEED
)

torch.manual_seed(
    SEED
)


# ============================================================
# Manifest
# ============================================================

with MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    manifest = json.load(f)


train_mean = (
    manifest[
        "normalization"
    ][
        "mean"
    ]
)

train_std = (
    manifest[
        "normalization"
    ][
        "std"
    ]
)


train_files = [
    DATA_DIR
    / item[
        "relative_path"
    ]

    for item
    in manifest[
        "train"
    ]
]


print(
    "Train participants:",
    len(train_files),
)

print(
    "TEST participants sealed:",
    len(
        manifest[
            "test"
        ]
    ),
)

print(
    f"Normalization mean: "
    f"{train_mean:.3f}"
)

print(
    f"Normalization SD: "
    f"{train_std:.3f}"
)


# ============================================================
# Device
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "\nDevice:",
    device,
)


# ============================================================
# Utility helpers
# ============================================================

def clone_state_dict(
    module,
):
    """
    Detached CPU copy of a module's parameters/buffers.
    """

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in module.state_dict().items()
    }


def count_parameters(
    parameters,
):

    return sum(
        p.numel()
        for p in parameters
    )


def grad_norm(
    parameters,
):

    total_sq = 0.0


    for p in parameters:

        if p.grad is None:

            continue


        value = float(
            p.grad.detach()
            .float()
            .norm()
            .item()
        )


        total_sq += (
            value
            * value
        )


    return math.sqrt(
        total_sq
    )


# ============================================================
# Load TSQA donor
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING TSQA DONOR"
)

print(
    "=" * 100
)


tsqa_model = (
    OpenTSLM.load_pretrained(
        TSQA_REPO_ID,
        device=device,
    )
)


EOS_TOKEN = (
    tsqa_model.get_eos_token()
)


tsqa_layers = (
    tsqa_model
    .encoder
    .encoder
    .layers
)


if len(
    tsqa_layers
) != 6:

    raise RuntimeError(
        "Expected exactly six TSQA "
        "Transformer layers."
    )


# Python index 4 = Transformer 5
# Python index 5 = Transformer 6

TSQA_T5_STATE = (
    clone_state_dict(
        tsqa_layers[
            4
        ]
    )
)

TSQA_T6_STATE = (
    clone_state_dict(
        tsqa_layers[
            5
        ]
    )
)

TSQA_PROJECTOR_STATE = (
    clone_state_dict(
        tsqa_model.projector
    )
)


print(
    "Saved TSQA donor states:"
)

print(
    "  Transformer 5"
)

print(
    "  Transformer 6"
)

print(
    "  Projector"
)


# ============================================================
# Build same tiny above_180 pool
# ============================================================

print(
    "\nBuilding above_180 examples..."
)


all_examples = (
    build_multitask_examples(
        files=train_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=[
            TASK_ID
        ],
    )
)


yes_examples = [
    x
    for x in all_examples
    if x[
        "_answer_clean"
    ] == "(a)"
]


no_examples = [
    x
    for x in all_examples
    if x[
        "_answer_clean"
    ] == "(b)"
]


print(
    "Available YES:",
    len(
        yes_examples
    ),
)

print(
    "Available NO:",
    len(
        no_examples
    ),
)


if (
    len(
        yes_examples
    )
    < N_PER_CLASS
):

    raise RuntimeError(
        "Not enough YES examples."
    )


if (
    len(
        no_examples
    )
    < N_PER_CLASS
):

    raise RuntimeError(
        "Not enough NO examples."
    )


# ============================================================
# SAME deterministic selection style as surgery diagnostic
# ============================================================

selection_rng = random.Random(
    SEED + 777
)


selected_yes = (
    selection_rng.sample(
        yes_examples,
        N_PER_CLASS,
    )
)


selected_no = (
    selection_rng.sample(
        no_examples,
        N_PER_CLASS,
    )
)


# Evaluation order:
# YES first then NO, just for readable reporting.

diagnostic_examples = (
    selected_yes
    + selected_no
)


# Training order:
# deterministically alternate classes.
#
# This avoids "the last ten updates were all NO" becoming
# another confound while keeping the order fixed every epoch.

training_examples = []


for yes_example, no_example in zip(
    selected_yes,
    selected_no,
):

    training_examples.append(
        yes_example
    )

    training_examples.append(
        no_example
    )


print(
    "\nTiny dataset:"
)

print(
    "YES:",
    len(
        selected_yes
    ),
)

print(
    "NO:",
    len(
        selected_no
    ),
)

print(
    "TOTAL:",
    len(
        diagnostic_examples
    ),
)


print(
    "\nFixed training order:"
)

print(
    "YES, NO, YES, NO, ... "
    "repeated identically each epoch"
)


# ============================================================
# Print selected examples
# ============================================================

print(
    "\nSelected examples:"
)


for i, example in enumerate(
    diagnostic_examples,
    start=1,
):

    print(
        f"{i:02d} | "
        f"{example['_answer_clean']} | "
        f"participant="
        f"{example.get('_participant_id')} | "
        f"window="
        f"{example.get('_window_start')} | "
        f"max="
        f"{example.get('_window_max')}"
    )


# ============================================================
# Release donor model
# ============================================================

del tsqa_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Decision-token IDs
# ============================================================

def get_decision_token_ids(
    model,
):

    # IMPORTANT:
    #
    # add_special_tokens=False because our pairwise objective
    # operates at the ACTUAL next-token generation position
    # after the prompt.
    #
    # We deliberately DO NOT append an artificial answer BOS.

    a_ids = (
        model.tokenizer(
            "(a)",
            add_special_tokens=False,
        ).input_ids
    )


    b_ids = (
        model.tokenizer(
            "(b)",
            add_special_tokens=False,
        ).input_ids
    )


    if (
        len(
            a_ids
        ) == 0
        or len(
            b_ids
        ) == 0
    ):

        raise RuntimeError(
            "Could not tokenize answers."
        )


    a_token_id = (
        int(
            a_ids[
                0
            ]
        )
    )


    b_token_id = (
        int(
            b_ids[
                0
            ]
        )
    )


    if (
        a_token_id
        == b_token_id
    ):

        raise RuntimeError(
            "First discriminating token "
            "for (a) and (b) is identical."
        )


    return (
        a_token_id,
        b_token_id,
        a_ids,
        b_ids,
    )


# ============================================================
# TRUE pairwise next-token logits
# ============================================================

def get_pair_logits(
    model,
    example,
    a_token_id,
    b_token_id,
):
    """
    Get only the two relevant class logits at the ACTUAL
    next-token generation position after the prompt.

    No answer tokens are appended.

    If prompt length = L, logits at position L-1 predict
    the next generated token.
    """

    (
        inputs_embeds,
        attention_mask,
    ) = (
        model.pad_and_apply_batch(
            [
                example
            ]
        )
    )


    outputs = (
        model.llm(
            inputs_embeds=
                inputs_embeds,

            attention_mask=
                attention_mask,

            return_dict=True,
        )
    )


    last_prompt_index = (
        int(
            attention_mask[
                0
            ]
            .sum()
            .item()
        )
        - 1
    )


    next_token_logits = (
        outputs.logits[
            0,
            last_prompt_index,
            :
        ]
    )


    pair_logits = torch.stack(
        [
            next_token_logits[
                a_token_id
            ],

            next_token_logits[
                b_token_id
            ],
        ]
    )


    return pair_logits


# ============================================================
# Evaluate pairwise decision on exact 20 examples
# ============================================================

def evaluate_pairwise(
    model,
    examples,
    a_token_id,
    b_token_id,
):

    model.encoder.eval()
    model.projector.eval()
    model.llm.eval()


    yes_total = 0
    yes_correct = 0

    no_total = 0
    no_correct = 0

    predicted_yes = 0
    predicted_no = 0

    correct_probabilities = []

    margins = []


    with torch.no_grad():

        for example in examples:

            pair_logits = (
                get_pair_logits(
                    model,
                    example,
                    a_token_id,
                    b_token_id,
                )
            )


            probabilities = (
                torch.softmax(
                    pair_logits.float(),
                    dim=0,
                )
            )


            p_a = float(
                probabilities[
                    0
                ].item()
            )


            p_b = float(
                probabilities[
                    1
                ].item()
            )


            prediction = (
                "(a)"
                if p_a >= p_b
                else "(b)"
            )


            gold = (
                example[
                    "_answer_clean"
                ]
            )


            margin = float(
                (
                    pair_logits[
                        0
                    ]
                    - pair_logits[
                        1
                    ]
                )
                .detach()
                .float()
                .item()
            )


            margins.append(
                margin
            )


            if prediction == "(a)":

                predicted_yes += 1

            else:

                predicted_no += 1


            if gold == "(a)":

                yes_total += 1

                correct_probabilities.append(
                    p_a
                )


                if prediction == gold:

                    yes_correct += 1


            else:

                no_total += 1

                correct_probabilities.append(
                    p_b
                )


                if prediction == gold:

                    no_correct += 1


    yes_accuracy = (
        yes_correct
        / yes_total
    )


    no_accuracy = (
        no_correct
        / no_total
    )


    balanced_accuracy = (
        0.5
        * (
            yes_accuracy
            + no_accuracy
        )
    )


    mean_p_correct = (
        sum(
            correct_probabilities
        )
        / len(
            correct_probabilities
        )
    )


    mean_margin = (
        sum(
            margins
        )
        / len(
            margins
        )
    )


    if len(
        margins
    ) > 1:

        margin_sd = float(
            torch.tensor(
                margins,
                dtype=torch.float32,
            )
            .std()
            .item()
        )

    else:

        margin_sd = 0.0


    return {
        "balanced_accuracy":
            balanced_accuracy,

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "predicted_yes":
            predicted_yes,

        "predicted_no":
            predicted_no,

        "mean_p_correct":
            mean_p_correct,

        "mean_margin_a_minus_b":
            mean_margin,

        "margin_sd":
            margin_sd,
    }


# ============================================================
# Representation check
#
# We want to ensure training doesn't immediately re-collapse
# the repaired encoder.
# ============================================================

def get_signal(
    example,
):

    signal = (
        example[
            "time_series"
        ][
            0
        ]
    )


    if not torch.is_tensor(
        signal
    ):

        signal = torch.tensor(
            signal,
            dtype=torch.float32,
        )


    signal = (
        signal
        .detach()
        .clone()
        .float()
    )


    if (
        signal.dim() == 2
        and signal.shape[-1] == 1
    ):

        signal = signal.squeeze(
            -1
        )


    if (
        signal.dim() == 2
        and signal.shape[0] == 1
    ):

        signal = signal.squeeze(
            0
        )


    return signal


def direct_representations(
    model,
    examples,
):

    encoder_vectors = []
    projector_vectors = []

    labels = []


    model.encoder.eval()
    model.projector.eval()


    with torch.no_grad():

        for example in examples:

            signal = (
                get_signal(
                    example
                )
            )


            x = (
                signal
                .to(
                    device
                )
                .unsqueeze(
                    0
                )
            )


            remainder = (
                x.shape[
                    1
                ]
                % model.patch_size
            )


            if remainder:

                pad_length = (
                    model.patch_size
                    - remainder
                )


                x = F.pad(
                    x,
                    (
                        0,
                        pad_length,
                    ),
                )


            encoded = (
                model.encoder(
                    x
                )
            )


            projected = (
                model.projector(
                    encoded
                )
            )


            encoder_vectors.append(
                encoded[
                    0
                ]
                .detach()
                .float()
                .cpu()
                .reshape(
                    -1
                )
            )


            projector_vectors.append(
                projected[
                    0
                ]
                .detach()
                .float()
                .cpu()
                .reshape(
                    -1
                )
            )


            labels.append(
                example[
                    "_answer_clean"
                ]
            )


    return (
        encoder_vectors,
        projector_vectors,
        labels,
    )


def representation_stats(
    vectors,
    labels,
):

    matrix = torch.stack(
        vectors
    ).float()


    feature_sd = float(
        matrix.std(
            dim=0
        )
        .mean()
        .item()
    )


    yes_indices = [
        i
        for i, label
        in enumerate(
            labels
        )
        if label == "(a)"
    ]


    no_indices = [
        i
        for i, label
        in enumerate(
            labels
        )
        if label == "(b)"
    ]


    yes_centroid = (
        matrix[
            yes_indices
        ]
        .mean(
            dim=0
        )
    )


    no_centroid = (
        matrix[
            no_indices
        ]
        .mean(
            dim=0
        )
    )


    centroid_cosine = float(
        F.cosine_similarity(
            yes_centroid.unsqueeze(
                0
            ),
            no_centroid.unsqueeze(
                0
            ),
        ).item()
    )


    centroid_l2 = float(
        torch.norm(
            yes_centroid
            - no_centroid
        ).item()
        / math.sqrt(
            matrix.shape[
                1
            ]
        )
    )


    return {
        "mean_feature_std":
            feature_sd,

        "yes_no_centroid_cosine":
            centroid_cosine,

        "yes_no_centroid_l2_per_sqrt_dim":
            centroid_l2,
    }


def evaluate_representations(
    model,
):

    (
        encoder_vectors,
        projector_vectors,
        labels,
    ) = (
        direct_representations(
            model,
            diagnostic_examples,
        )
    )


    return {
        "encoder":
            representation_stats(
                encoder_vectors,
                labels,
            ),

        "projector":
            representation_stats(
                projector_vectors,
                labels,
            ),
    }


# ============================================================
# Freeze / configure one condition
# ============================================================

def configure_condition(
    model,
    condition,
):

    layers = (
        model
        .encoder
        .encoder
        .layers
    )


    if len(
        layers
    ) != 6:

        raise RuntimeError(
            "Expected six Stage-5 "
            "Transformer layers."
        )


    # --------------------------------------------------------
    # Surgery
    # --------------------------------------------------------

    if condition in {
        "repair_stage5_projector",
        "repair_tsqa_projector",
    }:

        layers[
            4
        ].load_state_dict(
            TSQA_T5_STATE,
            strict=True,
        )


        layers[
            5
        ].load_state_dict(
            TSQA_T6_STATE,
            strict=True,
        )


    if (
        condition
        == "repair_tsqa_projector"
    ):

        model.projector.load_state_dict(
            TSQA_PROJECTOR_STATE,
            strict=True,
        )


    # --------------------------------------------------------
    # Freeze EVERYTHING first
    # --------------------------------------------------------

    for parameter in (
        model.encoder.parameters()
    ):

        parameter.requires_grad = False


    for parameter in (
        model.projector.parameters()
    ):

        parameter.requires_grad = False


    for parameter in (
        model.llm.parameters()
    ):

        parameter.requires_grad = False


    # --------------------------------------------------------
    # Unfreeze ONLY T5/T6
    # --------------------------------------------------------

    for parameter in (
        layers[
            4
        ].parameters()
    ):

        parameter.requires_grad = True


    for parameter in (
        layers[
            5
        ].parameters()
    ):

        parameter.requires_grad = True


    # --------------------------------------------------------
    # Projector trainable
    # --------------------------------------------------------

    for parameter in (
        model.projector.parameters()
    ):

        parameter.requires_grad = True


    # --------------------------------------------------------
    # Explicit audits
    # --------------------------------------------------------

    early_encoder_trainable = sum(
        p.numel()

        for component in [
            model.encoder.patch_embed,
            model.encoder.input_norm,
            *list(
                layers[
                    :4
                ]
            ),
        ]

        for p in component.parameters()

        if p.requires_grad
    )


    late_parameters = [
        p

        for layer in [
            layers[
                4
            ],
            layers[
                5
            ],
        ]

        for p in layer.parameters()

        if p.requires_grad
    ]


    projector_parameters = [
        p
        for p in model.projector.parameters()
        if p.requires_grad
    ]


    llm_trainable = sum(
        p.numel()
        for p in model.llm.parameters()
        if p.requires_grad
    )


    lora_trainable = sum(
        p.numel()

        for name, p
        in model.llm.named_parameters()

        if (
            p.requires_grad
            and "lora_" in name
        )
    )


    audit = {
        "early_encoder_trainable":
            early_encoder_trainable,

        "t5_t6_trainable":
            count_parameters(
                late_parameters
            ),

        "projector_trainable":
            count_parameters(
                projector_parameters
            ),

        "llm_trainable":
            llm_trainable,

        "lora_trainable":
            lora_trainable,
    }


    if (
        early_encoder_trainable
        != 0
    ):

        raise RuntimeError(
            "Early Stage-5 encoder "
            "unexpectedly trainable."
        )


    if (
        llm_trainable
        != 0
    ):

        raise RuntimeError(
            "LLM/LoRA unexpectedly trainable."
        )


    return (
        late_parameters,
        projector_parameters,
        audit,
    )


# ============================================================
# Training/evaluation modes
# ============================================================

def set_training_mode(
    model,
):

    # Keep frozen components deterministic.

    model.encoder.eval()
    model.llm.eval()


    # Only these are actively optimized.

    model.encoder.encoder.layers[
        4
    ].train()

    model.encoder.encoder.layers[
        5
    ].train()

    model.projector.train()


def set_eval_mode(
    model,
):

    model.encoder.eval()
    model.projector.eval()
    model.llm.eval()


# ============================================================
# Raw generation smoke
# ============================================================

def raw_generation_smoke(
    model,
    examples,
):

    set_eval_mode(
        model
    )


    outputs = []


    with torch.no_grad():

        for example in examples:

            generated = (
                model.generate(
                    [
                        example
                    ],
                    max_new_tokens=6,
                    do_sample=False,
                )[
                    0
                ]
            )


            outputs.append(
                {
                    "gold":
                        example[
                            "_answer_clean"
                        ],

                    "participant_id":
                        example.get(
                            "_participant_id"
                        ),

                    "window_start":
                        example.get(
                            "_window_start"
                        ),

                    "raw_generation":
                        generated,
                }
            )


    return outputs


# ============================================================
# Run one condition
# ============================================================

def run_condition(
    condition,
    display_name,
):

    print(
        "\n\n"
        + "#" * 110
    )

    print(
        display_name
    )

    print(
        "#" * 110
    )


    # Reset random state so trainable dropout etc. is matched
    # across conditions.

    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )


    # --------------------------------------------------------
    # Fresh public Stage-5 checkpoint every condition
    # --------------------------------------------------------

    model = (
        OpenTSLM.load_pretrained(
            STAGE5_REPO_ID,
            device=device,
            enable_lora=True,
        )
    )


    (
        a_token_id,
        b_token_id,
        a_ids,
        b_ids,
    ) = (
        get_decision_token_ids(
            model
        )
    )


    print(
        "\nAnswer tokenization "
        "(NO special tokens):"
    )

    print(
        "(a):",
        a_ids,
        "first decision token:",
        a_token_id,
        model.tokenizer.convert_ids_to_tokens(
            [
                a_token_id
            ]
        ),
    )

    print(
        "(b):",
        b_ids,
        "first decision token:",
        b_token_id,
        model.tokenizer.convert_ids_to_tokens(
            [
                b_token_id
            ]
        ),
    )


    (
        late_parameters,
        projector_parameters,
        audit,
    ) = (
        configure_condition(
            model,
            condition,
        )
    )


    print(
        "\n--- TRAINABLE PARAMETER AUDIT ---"
    )

    print(
        "Early encoder:",
        f"{audit['early_encoder_trainable']:,}"
    )

    print(
        "T5 + T6:",
        f"{audit['t5_t6_trainable']:,}"
    )

    print(
        "Projector:",
        f"{audit['projector_trainable']:,}"
    )

    print(
        "LLM total:",
        f"{audit['llm_trainable']:,}"
    )

    print(
        "LoRA:",
        f"{audit['lora_trainable']:,}"
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    late_parameters,

                "lr":
                    LR_T5_T6,
            },

            {
                "params":
                    projector_parameters,

                "lr":
                    LR_PROJECTOR,
            },
        ],

        weight_decay=
            WEIGHT_DECAY,
    )


    trainable_parameters = (
        late_parameters
        + projector_parameters
    )


    # --------------------------------------------------------
    # Initial representation diagnostic
    # --------------------------------------------------------

    initial_representation = (
        evaluate_representations(
            model
        )
    )


    print(
        "\n--- REPRESENTATION BEFORE TRAINING ---"
    )

    print(
        "Encoder feature SD:",
        f"{initial_representation['encoder']['mean_feature_std']:.8f}"
    )

    print(
        "Encoder Y/N cosine:",
        f"{initial_representation['encoder']['yes_no_centroid_cosine']:.8f}"
    )

    print(
        "Projector feature SD:",
        f"{initial_representation['projector']['mean_feature_std']:.8f}"
    )

    print(
        "Projector Y/N cosine:",
        f"{initial_representation['projector']['yes_no_centroid_cosine']:.8f}"
    )


    # --------------------------------------------------------
    # Initial pairwise evaluation
    # --------------------------------------------------------

    history = []


    initial_metrics = (
        evaluate_pairwise(
            model,
            diagnostic_examples,
            a_token_id,
            b_token_id,
        )
    )


    history.append(
        {
            "epoch":
                0,

            "mean_training_loss":
                None,

            **initial_metrics,
        }
    )


    print(
        "\nEpoch  0 | "
        f"BA="
        f"{100 * initial_metrics['balanced_accuracy']:6.1f}% | "
        f"YES="
        f"{100 * initial_metrics['yes_accuracy']:6.1f}% | "
        f"NO="
        f"{100 * initial_metrics['no_accuracy']:6.1f}% | "
        f"P(correct)="
        f"{initial_metrics['mean_p_correct']:.4f} | "
        f"pred A/B="
        f"{initial_metrics['predicted_yes']}/"
        f"{initial_metrics['predicted_no']} | "
        f"margin SD="
        f"{initial_metrics['margin_sd']:.4f}"
    )


    first_step_gradients = None


    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        set_training_mode(
            model
        )


        epoch_losses = []


        for step_index, example in enumerate(
            training_examples,
            start=1,
        ):

            optimizer.zero_grad(
                set_to_none=True
            )


            pair_logits = (
                get_pair_logits(
                    model,
                    example,
                    a_token_id,
                    b_token_id,
                )
            )


            gold = (
                example[
                    "_answer_clean"
                ]
            )


            target_index = (
                0
                if gold == "(a)"
                else 1
            )


            target = torch.tensor(
                [
                    target_index
                ],
                dtype=torch.long,
                device=device,
            )


            # ------------------------------------------------
            # TRUE TWO-WAY OBJECTIVE
            #
            # The rest of the vocabulary is completely absent
            # from this CE denominator.
            # ------------------------------------------------

            loss = (
                F.cross_entropy(
                    pair_logits
                    .float()
                    .unsqueeze(
                        0
                    ),

                    target,
                )
            )


            loss.backward()


            if (
                first_step_gradients
                is None
            ):

                first_step_gradients = {
                    "t5_t6_grad_norm":
                        grad_norm(
                            late_parameters
                        ),

                    "projector_grad_norm":
                        grad_norm(
                            projector_parameters
                        ),

                    "loss":
                        float(
                            loss
                            .detach()
                            .cpu()
                            .item()
                        ),

                    "gold":
                        gold,
                }


                print(
                    "\nFirst update:"
                )

                print(
                    "  Gold:",
                    gold
                )

                print(
                    "  Pairwise loss:",
                    f"{first_step_gradients['loss']:.6f}"
                )

                print(
                    "  T5/T6 grad norm:",
                    f"{first_step_gradients['t5_t6_grad_norm']:.8e}"
                )

                print(
                    "  Projector grad norm:",
                    f"{first_step_gradients['projector_grad_norm']:.8e}"
                )


            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                GRAD_CLIP_NORM,
            )


            optimizer.step()


            epoch_losses.append(
                float(
                    loss
                    .detach()
                    .cpu()
                    .item()
                )
            )


        mean_epoch_loss = (
            sum(
                epoch_losses
            )
            / len(
                epoch_losses
            )
        )


        # ----------------------------------------------------
        # Evaluate chosen checkpoints
        # ----------------------------------------------------

        if epoch in EVAL_EPOCHS:

            metrics = (
                evaluate_pairwise(
                    model,
                    diagnostic_examples,
                    a_token_id,
                    b_token_id,
                )
            )


            history.append(
                {
                    "epoch":
                        epoch,

                    "mean_training_loss":
                        mean_epoch_loss,

                    **metrics,
                }
            )


            print(
                f"Epoch {epoch:2d} | "
                f"loss="
                f"{mean_epoch_loss:8.5f} | "
                f"BA="
                f"{100 * metrics['balanced_accuracy']:6.1f}% | "
                f"YES="
                f"{100 * metrics['yes_accuracy']:6.1f}% | "
                f"NO="
                f"{100 * metrics['no_accuracy']:6.1f}% | "
                f"P(correct)="
                f"{metrics['mean_p_correct']:.4f} | "
                f"pred A/B="
                f"{metrics['predicted_yes']}/"
                f"{metrics['predicted_no']} | "
                f"margin SD="
                f"{metrics['margin_sd']:.4f}"
            )


    # ========================================================
    # Final diagnostics
    # ========================================================

    final_metrics = (
        evaluate_pairwise(
            model,
            diagnostic_examples,
            a_token_id,
            b_token_id,
        )
    )


    final_representation = (
        evaluate_representations(
            model
        )
    )


    print(
        "\n--- REPRESENTATION AFTER TRAINING ---"
    )

    print(
        "Encoder feature SD:",
        f"{final_representation['encoder']['mean_feature_std']:.8f}"
    )

    print(
        "Encoder Y/N cosine:",
        f"{final_representation['encoder']['yes_no_centroid_cosine']:.8f}"
    )

    print(
        "Projector feature SD:",
        f"{final_representation['projector']['mean_feature_std']:.8f}"
    )

    print(
        "Projector Y/N cosine:",
        f"{final_representation['projector']['yes_no_centroid_cosine']:.8f}"
    )


    # --------------------------------------------------------
    # Raw generation is SECONDARY.
    #
    # We only trained the FIRST discriminating decision token.
    # Do not use malformed continuation as primary failure.
    # --------------------------------------------------------

    generation_examples = [
        selected_yes[
            0
        ],
        selected_no[
            0
        ],
    ]


    raw_generations = (
        raw_generation_smoke(
            model,
            generation_examples,
        )
    )


    print(
        "\n--- RAW GENERATION SMOKE ---"
    )


    for item in raw_generations:

        print(
            f"Gold={item['gold']} | "
            f"participant="
            f"{item['participant_id']} | "
            f"window="
            f"{item['window_start']} | "
            f"raw={item['raw_generation']!r}"
        )


    result = {
        "condition":
            condition,

        "display_name":
            display_name,

        "parameter_audit":
            audit,

        "decision_tokens":
            {
                "a_full_ids":
                    a_ids,

                "b_full_ids":
                    b_ids,

                "a_decision_token_id":
                    a_token_id,

                "b_decision_token_id":
                    b_token_id,
            },

        "first_step_gradients":
            first_step_gradients,

        "initial_representation":
            initial_representation,

        "final_representation":
            final_representation,

        "history":
            history,

        "final_metrics":
            final_metrics,

        "raw_generation_smoke":
            raw_generations,
    }


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del optimizer
    del model

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# Conditions
# ============================================================

conditions = [
    (
        "original_pairwise",
        "0 — ORIGINAL STAGE-5 + TRUE PAIRWISE CONTROL",
    ),

    (
        "repair_stage5_projector",
        "1 — REPAIRED T5/T6 + STAGE-5 PROJECTOR",
    ),

    (
        "repair_tsqa_projector",
        "2 — REPAIRED T5/T6 + TSQA PROJECTOR",
    ),
]


all_results = {}


for (
    condition,
    display_name,
) in conditions:

    all_results[
        condition
    ] = (
        run_condition(
            condition,
            display_name,
        )
    )


# ============================================================
# Final comparison
# ============================================================

print(
    "\n\n"
    + "=" * 130
)

print(
    "TINY REPAIRED STAGE-5 PAIRWISE OVERFIT SUMMARY"
)

print(
    "=" * 130
)


print(
    f"{'Condition':30s} | "
    f"{'Initial BA':>10s} | "
    f"{'Final BA':>9s} | "
    f"{'YES acc':>8s} | "
    f"{'NO acc':>8s} | "
    f"{'P(correct)':>10s} | "
    f"{'Enc SD before':>13s} | "
    f"{'Enc SD after':>12s}"
)


print(
    "-" * 130
)


for (
    condition,
    _
) in conditions:

    result = (
        all_results[
            condition
        ]
    )


    initial_metrics = (
        result[
            "history"
        ][
            0
        ]
    )


    final_metrics = (
        result[
            "final_metrics"
        ]
    )


    enc_before = (
        result[
            "initial_representation"
        ][
            "encoder"
        ][
            "mean_feature_std"
        ]
    )


    enc_after = (
        result[
            "final_representation"
        ][
            "encoder"
        ][
            "mean_feature_std"
        ]
    )


    print(
        f"{condition:30s} | "
        f"{100 * initial_metrics['balanced_accuracy']:9.1f}% | "
        f"{100 * final_metrics['balanced_accuracy']:8.1f}% | "
        f"{100 * final_metrics['yes_accuracy']:7.1f}% | "
        f"{100 * final_metrics['no_accuracy']:7.1f}% | "
        f"{final_metrics['mean_p_correct']:10.4f} | "
        f"{enc_before:13.8f} | "
        f"{enc_after:12.8f}"
    )


# ============================================================
# Learning curves
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "BALANCED-ACCURACY LEARNING CURVES"
)

print(
    "=" * 100
)


for (
    condition,
    _
) in conditions:

    history = (
        all_results[
            condition
        ][
            "history"
        ]
    )


    curve = " | ".join(
        (
            f"E{item['epoch']}:"
            f"{100 * item['balanced_accuracy']:.1f}%"
        )

        for item in history
    )


    print(
        f"{condition:30s} | "
        f"{curve}"
    )


# ============================================================
# Interpretation helper
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "MECHANISTIC COMPARISON"
)

print(
    "=" * 100
)


control_ba = (
    all_results[
        "original_pairwise"
    ][
        "final_metrics"
    ][
        "balanced_accuracy"
    ]
)


minimal_ba = (
    all_results[
        "repair_stage5_projector"
    ][
        "final_metrics"
    ][
        "balanced_accuracy"
    ]
)


strong_ba = (
    all_results[
        "repair_tsqa_projector"
    ][
        "final_metrics"
    ][
        "balanced_accuracy"
    ]
)


print(
    "Original Stage-5 pairwise:",
    f"{100 * control_ba:.1f}%"
)

print(
    "Repaired T5/T6 + Stage5 projector:",
    f"{100 * minimal_ba:.1f}%"
)

print(
    "Repaired T5/T6 + TSQA projector:",
    f"{100 * strong_ba:.1f}%"
)


if (
    control_ba < 0.75
    and minimal_ba >= 0.90
):

    print(
        "\nPattern: late-encoder repair strongly "
        "restored tiny-set learnability."
    )


elif (
    control_ba >= 0.90
    and minimal_ba >= 0.90
):

    print(
        "\nPattern: the true pairwise objective "
        "was sufficient even from the collapsed "
        "Stage-5 initialization."
    )


elif (
    minimal_ba < 0.90
    and strong_ba >= 0.90
):

    print(
        "\nPattern: T5/T6 repair alone was not "
        "sufficient; restoring the TSQA projector "
        "was also important."
    )


elif (
    max(
        control_ba,
        minimal_ba,
        strong_ba,
    )
    < 0.75
):

    print(
        "\nPattern: none of the current conditions "
        "could reliably memorize the tiny set."
    )


else:

    print(
        "\nPattern is intermediate; inspect the "
        "learning curves and representation changes."
    )


# ============================================================
# Save results
# ============================================================

serializable = {
    "experiment":
        "stage5_repaired_tiny_overfit_pairwise",

    "seed":
        SEED,

    "task":
        TASK_ID,

    "epochs":
        EPOCHS,

    "n_yes":
        N_PER_CLASS,

    "n_no":
        N_PER_CLASS,

    "normalization":
        {
            "type":
                "fixed_training_cohort",

            "mean":
                train_mean,

            "std":
                train_std,
        },

    "optimizer":
        {
            "lr_t5_t6":
                LR_T5_T6,

            "lr_projector":
                LR_PROJECTOR,

            "weight_decay":
                WEIGHT_DECAY,

            "grad_clip_norm":
                GRAD_CLIP_NORM,
        },

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "selected_examples":
        [
            {
                "participant_id":
                    example.get(
                        "_participant_id"
                    ),

                "window_start":
                    example.get(
                        "_window_start"
                    ),

                "gold":
                    example[
                        "_answer_clean"
                    ],

                "window_min":
                    example.get(
                        "_window_min"
                    ),

                "window_max":
                    example.get(
                        "_window_max"
                    ),
            }

            for example
            in diagnostic_examples
        ],

    "results":
        all_results,
}


RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


with RESULTS_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        serializable,
        f,
        indent=2,
    )


print(
    "\nSaved diagnostic:"
)

print(
    RESULTS_PATH
)


print(
    "\nNo test participants were inspected."
)

print(
    "TEST participants remain sealed."
)

print(
    "No trained checkpoint was saved."
)