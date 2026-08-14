"""
Tiny OpenTSLM-style CGM factual grounding experiment.

PURPOSE
-------
Test whether the ORIGINAL public Stage-5 checkpoint can learn
a factual natural-language mapping from CGM -> language when
CGM is presented using an OpenTSLM-style interface.

This is NOT yet the full six-task alignment experiment.

Task:
    above_180

Dataset:
    10 YES
    10 NO
    exact same deterministic training examples used in our
    recent surgery / tiny-overfit diagnostics.

NEW INPUT FORMAT
----------------
Instead of fixed cohort normalization being the actual signal
representation, each individual CGM window is z-normalized
using ITS OWN mean and standard deviation.

The original physical scale is supplied as text:

    - CGM
    - mg/dL
    - 128 measurements
    - nominal 5-minute cadence
    - original mean
    - original SD

The target contains information NOT supplied in the text:

    Yes. Maximum glucose: 200 mg/dL.

or

    No. Maximum glucose: 178 mg/dL.

Therefore the model must use the waveform to recover the
maximum, while using mean/SD to recover absolute scale.

TRAINING
--------
Original public Stage-5 checkpoint.

Frozen:
    patch embed
    position embedding
    input norm
    T1-T4
    LLM
    LoRA

Trainable:
    Stage-5 T5
    Stage-5 T6
    Stage-5 projector

Loss:
    natural-language sequence CE
        +
    auxiliary TRUE Yes-vs-No pairwise CE

The pairwise term prevents the binary factual distinction
from being diluted by the full vocabulary.

Everything is kept in eval() mode during this tiny capacity
test so dropout does not add another source of noise.
Gradients still flow normally.

TEST PARTICIPANTS REMAIN SEALED.
"""

import gc
import json
import math
import random
import re
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
    / "tiny_cgm_native_grounding.json"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

N_PER_CLASS = 10

EPOCHS = 20

BATCH_SIZE = 2


LR_T5_T6 = 2e-4
LR_PROJECTOR = 1e-4

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0


# Pairwise factual auxiliary weight.
PAIRWISE_WEIGHT = 1.0


EVAL_EPOCHS = {
    0,
    1,
    2,
    5,
    10,
    15,
    20,
}


# Greedy generation is relatively expensive on CPU.
GENERATION_EPOCHS = {
    0,
    5,
    10,
    20,
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

    manifest = json.load(
        f
    )


train_mean = float(
    manifest[
        "normalization"
    ][
        "mean"
    ]
)

train_std = float(
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
    len(
        train_files
    ),
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
    "Existing fixed normalization:"
)

print(
    f"  mean={train_mean:.3f}"
)

print(
    f"  SD={train_std:.3f}"
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
# Load original Stage-5
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING ORIGINAL PUBLIC STAGE-5"
)

print(
    "=" * 100
)


model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


EOS_TOKEN = (
    model.get_eos_token()
)


# ============================================================
# Build OLD-format examples first
#
# We use the existing tested CGM loader/split code and then
# reconstruct the raw window exactly from the existing fixed
# normalization.
# ============================================================

print(
    "\nBuilding existing above_180 examples..."
)


base_examples = (
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


yes_pool = [
    example
    for example
    in base_examples
    if (
        example[
            "_answer_clean"
        ]
        == "(a)"
    )
]


no_pool = [
    example
    for example
    in base_examples
    if (
        example[
            "_answer_clean"
        ]
        == "(b)"
    )
]


print(
    "Available YES:",
    len(
        yes_pool
    ),
)

print(
    "Available NO:",
    len(
        no_pool
    ),
)


# ============================================================
# Same deterministic selection as recent diagnostics
# ============================================================

selection_rng = random.Random(
    SEED + 777
)


selected_yes = (
    selection_rng.sample(
        yes_pool,
        N_PER_CLASS,
    )
)


selected_no = (
    selection_rng.sample(
        no_pool,
        N_PER_CLASS,
    )
)


selected_base_examples = (
    selected_yes
    + selected_no
)


# ============================================================
# Convert fixed-normalized signal -> raw CGM
# ============================================================

def recover_raw_window(
    example,
):

    fixed_normalized = (
        example[
            "time_series"
        ][
            0
        ]
    )


    if not torch.is_tensor(
        fixed_normalized
    ):

        fixed_normalized = torch.tensor(
            fixed_normalized,
            dtype=torch.float32,
        )


    fixed_normalized = (
        fixed_normalized
        .detach()
        .clone()
        .float()
    )


    raw = (
        fixed_normalized
        * train_std
        + train_mean
    )


    expected_max = float(
        example[
            "_window_max"
        ]
    )


    reconstructed_max = float(
        raw.max().item()
    )


    if (
        abs(
            expected_max
            - reconstructed_max
        )
        > 1e-3
    ):

        raise RuntimeError(
            "Raw reconstruction mismatch: "
            f"metadata max={expected_max}, "
            f"reconstructed={reconstructed_max}"
        )


    return raw


# ============================================================
# OpenTSLM-style native CGM example
# ============================================================

def make_native_example(
    base_example,
):

    raw = (
        recover_raw_window(
            base_example
        )
    )


    window_mean = float(
        raw.mean().item()
    )


    # Population SD for normalization.
    window_std = float(
        raw.std(
            unbiased=False
        ).item()
    )


    if window_std < 1e-6:

        safe_std = 1.0

    else:

        safe_std = (
            window_std
        )


    normalized = (
        raw
        - window_mean
    ) / safe_std


    maximum = float(
        raw.max().item()
    )


    maximum_int = int(
        round(
            maximum
        )
    )


    label_yes = (
        maximum
        > 180.0
    )


    expected_label = (
        base_example[
            "_answer_clean"
        ]
        == "(a)"
    )


    if (
        label_yes
        != expected_label
    ):

        raise RuntimeError(
            "Grounding label does not match "
            "existing benchmark label."
        )


    if label_yes:

        answer_text = (
            f" Yes. Maximum glucose: "
            f"{maximum_int} mg/dL."
        )

    else:

        answer_text = (
            f" No. Maximum glucose: "
            f"{maximum_int} mg/dL."
        )


    # Important:
    #
    # Mean and SD restore physical scale.
    #
    # Maximum / TIR / excursions are NOT put in the
    # input text because those are facts we want the
    # waveform to teach the model.

    time_series_description = (
        "Continuous glucose monitor (CGM) signal "
        "measured in mg/dL. "
        f"This segment contains {len(raw)} measurements "
        "with a nominal sampling interval of 5 minutes. "
        f"Before normalization, the segment mean was "
        f"{window_mean:.2f} mg/dL and the standard deviation "
        f"was {window_std:.2f} mg/dL. "
        "The numeric signal supplied to the model has been "
        "z-normalized using this segment's own mean and "
        "standard deviation."
    )


    example = {
        "pre_prompt":
            (
                "Analyze the continuous glucose monitor "
                "signal and answer the factual question.\n"
            ),

        "time_series":
            [
                normalized
            ],

        "time_series_text":
            [
                time_series_description
            ],

        "post_prompt":
            (
                "\nQuestion: Did glucose exceed "
                "180 mg/dL during this segment?\n"
                "Answer Yes or No and report the "
                "maximum glucose value.\n"
                "Answer:"
            ),

        "answer":
            (
                answer_text
                + EOS_TOKEN
            ),

        # --------------------------------------------
        # Metadata not exposed to model
        # --------------------------------------------

        "_label_yes":
            label_yes,

        "_answer_text":
            answer_text.strip(),

        "_maximum":
            maximum,

        "_maximum_int":
            maximum_int,

        "_window_mean":
            window_mean,

        "_window_std":
            window_std,

        "_participant_id":
            base_example.get(
                "_participant_id"
            ),

        "_window_start":
            base_example.get(
                "_window_start"
            ),

        "_raw_window":
            raw,
    }


    return example


native_examples = [
    make_native_example(
        example
    )

    for example
    in selected_base_examples
]


native_yes = [
    example
    for example
    in native_examples
    if example[
        "_label_yes"
    ]
]


native_no = [
    example
    for example
    in native_examples
    if not example[
        "_label_yes"
    ]
]


print(
    "\nNative grounding dataset:"
)

print(
    "YES:",
    len(
        native_yes
    ),
)

print(
    "NO:",
    len(
        native_no
    ),
)


# ============================================================
# Deterministic balanced training order
# ============================================================

training_examples = []


for yes_example, no_example in zip(
    native_yes,
    native_no,
):

    training_examples.append(
        yes_example
    )

    training_examples.append(
        no_example
    )


print(
    "\nSelected examples:"
)


for index, example in enumerate(
    native_examples,
    start=1,
):

    print(
        f"{index:02d} | "
        f"{'YES' if example['_label_yes'] else 'NO ':3s} | "
        f"participant="
        f"{example['_participant_id']} | "
        f"window="
        f"{example['_window_start']} | "
        f"mean="
        f"{example['_window_mean']:.1f} | "
        f"SD="
        f"{example['_window_std']:.1f} | "
        f"max="
        f"{example['_maximum_int']}"
    )


# ============================================================
# Token IDs for auxiliary Yes-vs-No objective
# ============================================================

yes_ids = (
    model.tokenizer(
        " Yes",
        add_special_tokens=False,
    ).input_ids
)


no_ids = (
    model.tokenizer(
        " No",
        add_special_tokens=False,
    ).input_ids
)


if (
    len(
        yes_ids
    ) == 0
    or len(
        no_ids
    ) == 0
):

    raise RuntimeError(
        "Could not tokenize Yes/No."
    )


YES_TOKEN_ID = int(
    yes_ids[
        0
    ]
)

NO_TOKEN_ID = int(
    no_ids[
        0
    ]
)


if (
    YES_TOKEN_ID
    == NO_TOKEN_ID
):

    raise RuntimeError(
        "Yes and No decision tokens "
        "are identical."
    )


print(
    "\nDecision tokens:"
)

print(
    " Yes:",
    yes_ids,
    "first token=",
    YES_TOKEN_ID,
    model.tokenizer.convert_ids_to_tokens(
        [
            YES_TOKEN_ID
        ]
    ),
)

print(
    " No:",
    no_ids,
    "first token=",
    NO_TOKEN_ID,
    model.tokenizer.convert_ids_to_tokens(
        [
            NO_TOKEN_ID
        ]
    ),
)


# ============================================================
# Freeze model
# ============================================================

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
        "Expected six encoder Transformer layers."
    )


# Unfreeze Stage-5 T5/T6.

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


# Projector trainable.

for parameter in (
    model.projector.parameters()
):

    parameter.requires_grad = True


late_parameters = [
    parameter

    for layer in [
        layers[
            4
        ],
        layers[
            5
        ],
    ]

    for parameter
    in layer.parameters()

    if parameter.requires_grad
]


projector_parameters = [
    parameter

    for parameter
    in model.projector.parameters()

    if parameter.requires_grad
]


trainable_parameters = (
    late_parameters
    + projector_parameters
)


# ============================================================
# Parameter audit
# ============================================================

early_trainable = sum(
    parameter.numel()

    for name, parameter
    in model.encoder.named_parameters()

    if (
        parameter.requires_grad
        and not (
            name.startswith(
                "encoder.layers.4"
            )
            or name.startswith(
                "encoder.layers.5"
            )
        )
    )
)


llm_trainable = sum(
    parameter.numel()

    for parameter
    in model.llm.parameters()

    if parameter.requires_grad
)


t5_t6_trainable = sum(
    parameter.numel()
    for parameter
    in late_parameters
)


projector_trainable = sum(
    parameter.numel()
    for parameter
    in projector_parameters
)


print(
    "\n"
    + "=" * 100
)

print(
    "TRAINABLE PARAMETER AUDIT"
)

print(
    "=" * 100
)

print(
    "Early encoder trainable:",
    f"{early_trainable:,}"
)

print(
    "T5/T6 trainable:",
    f"{t5_t6_trainable:,}"
)

print(
    "Projector trainable:",
    f"{projector_trainable:,}"
)

print(
    "LLM / LoRA trainable:",
    f"{llm_trainable:,}"
)


if (
    early_trainable
    != 0
):

    raise RuntimeError(
        "Unexpected early encoder parameters trainable."
    )


if (
    llm_trainable
    != 0
):

    raise RuntimeError(
        "LLM or LoRA unexpectedly trainable."
    )


# ============================================================
# Keep EVERYTHING in eval mode
#
# eval() disables dropout but DOES NOT disable gradients.
# This is deliberate for a tiny memorization/capacity test.
# ============================================================

model.encoder.eval()
model.projector.eval()
model.llm.eval()


# ============================================================
# Optimizer
# ============================================================

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


# ============================================================
# Mini-batches
# ============================================================

training_batches = [
    training_examples[
        i:
        i + BATCH_SIZE
    ]

    for i
    in range(
        0,
        len(
            training_examples
        ),
        BATCH_SIZE,
    )
]


print(
    "\nTraining batches:",
    len(
        training_batches
    ),
)

print(
    "Batch size:",
    BATCH_SIZE,
)


# ============================================================
# Auxiliary TRUE Yes-vs-No loss
# ============================================================

def pairwise_loss_for_batch(
    batch,
):
    """
    Evaluate the actual next-token logits after the prompt.

    Only Yes and No compete in the auxiliary CE.
    """

    (
        inputs_embeds,
        attention_mask,
    ) = (
        model.pad_and_apply_batch(
            batch
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


    pair_logits_list = []

    targets = []


    for index, example in enumerate(
        batch
    ):

        last_prompt_position = (
            int(
                attention_mask[
                    index
                ]
                .sum()
                .item()
            )
            - 1
        )


        full_logits = (
            outputs.logits[
                index,
                last_prompt_position,
                :
            ]
        )


        pair_logits = torch.stack(
            [
                full_logits[
                    YES_TOKEN_ID
                ],

                full_logits[
                    NO_TOKEN_ID
                ],
            ]
        )


        pair_logits_list.append(
            pair_logits
        )


        targets.append(
            0
            if example[
                "_label_yes"
            ]
            else 1
        )


    pair_logits_tensor = torch.stack(
        pair_logits_list,
        dim=0,
    ).float()


    target_tensor = torch.tensor(
        targets,
        dtype=torch.long,
        device=device,
    )


    loss = F.cross_entropy(
        pair_logits_tensor,
        target_tensor,
    )


    return loss


# ============================================================
# Pairwise evaluation
# ============================================================

def evaluate_pairwise():
    """
    Evaluate Yes-vs-No at the actual next generation token.
    """

    correct_yes = 0
    correct_no = 0

    total_yes = 0
    total_no = 0

    predicted_yes = 0
    predicted_no = 0

    correct_probabilities = []

    margins = []


    with torch.no_grad():

        for example in native_examples:

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


            last_position = (
                int(
                    attention_mask[
                        0
                    ]
                    .sum()
                    .item()
                )
                - 1
            )


            full_logits = (
                outputs.logits[
                    0,
                    last_position,
                    :
                ]
            )


            pair_logits = torch.stack(
                [
                    full_logits[
                        YES_TOKEN_ID
                    ],

                    full_logits[
                        NO_TOKEN_ID
                    ],
                ]
            ).float()


            probabilities = torch.softmax(
                pair_logits,
                dim=0,
            )


            p_yes = float(
                probabilities[
                    0
                ].item()
            )


            p_no = float(
                probabilities[
                    1
                ].item()
            )


            prediction_yes = (
                p_yes
                >= p_no
            )


            gold_yes = (
                example[
                    "_label_yes"
                ]
            )


            margins.append(
                float(
                    (
                        pair_logits[
                            0
                        ]
                        - pair_logits[
                            1
                        ]
                    ).item()
                )
            )


            if prediction_yes:

                predicted_yes += 1

            else:

                predicted_no += 1


            if gold_yes:

                total_yes += 1

                correct_probabilities.append(
                    p_yes
                )


                if prediction_yes:

                    correct_yes += 1


            else:

                total_no += 1

                correct_probabilities.append(
                    p_no
                )


                if not prediction_yes:

                    correct_no += 1


    yes_accuracy = (
        correct_yes
        / total_yes
    )


    no_accuracy = (
        correct_no
        / total_no
    )


    balanced_accuracy = (
        0.5
        * (
            yes_accuracy
            + no_accuracy
        )
    )


    margin_tensor = torch.tensor(
        margins,
        dtype=torch.float32,
    )


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
            sum(
                correct_probabilities
            )
            / len(
                correct_probabilities
            ),

        "margin_sd":
            float(
                margin_tensor.std().item()
            ),
    }


# ============================================================
# Greedy natural-language evaluation
# ============================================================

def parse_generation(
    text,
):

    stripped = (
        text
        .strip()
    )


    lower = (
        stripped
        .lower()
    )


    if re.match(
        r"^yes\b",
        lower,
    ):

        decision = True

    elif re.match(
        r"^no\b",
        lower,
    ):

        decision = False

    else:

        decision = None


    # Look for a glucose value followed by mg/dL.
    match = re.search(
        r"(-?\d+(?:\.\d+)?)\s*mg\s*/?\s*d[lL]",
        stripped,
        flags=re.IGNORECASE,
    )


    if match:

        predicted_max = float(
            match.group(
                1
            )
        )

    else:

        predicted_max = None


    return (
        decision,
        predicted_max,
    )


def evaluate_generation():
    """
    Greedy generation on all 20 training examples.

    Metrics:
        - Yes/No validity
        - Yes/No accuracy
        - maximum-value extraction rate
        - max within +/-1 mg/dL
        - both facts correct
    """

    valid_decisions = 0
    correct_decisions = 0

    numeric_outputs = 0
    correct_max = 0

    fully_correct = 0

    samples = []


    with torch.no_grad():

        for example in native_examples:

            generated = (
                model.generate(
                    [
                        example
                    ],
                    max_new_tokens=24,
                    do_sample=False,
                )[
                    0
                ]
            )


            (
                predicted_decision,
                predicted_max,
            ) = (
                parse_generation(
                    generated
                )
            )


            gold_decision = (
                example[
                    "_label_yes"
                ]
            )


            gold_max = float(
                example[
                    "_maximum_int"
                ]
            )


            decision_correct = False
            max_correct = False


            if (
                predicted_decision
                is not None
            ):

                valid_decisions += 1


                if (
                    predicted_decision
                    == gold_decision
                ):

                    correct_decisions += 1
                    decision_correct = True


            if (
                predicted_max
                is not None
            ):

                numeric_outputs += 1


                if (
                    abs(
                        predicted_max
                        - gold_max
                    )
                    <= 1.0
                ):

                    correct_max += 1
                    max_correct = True


            if (
                decision_correct
                and max_correct
            ):

                fully_correct += 1


            samples.append(
                {
                    "gold_yes":
                        gold_decision,

                    "gold_max":
                        gold_max,

                    "predicted_yes":
                        predicted_decision,

                    "predicted_max":
                        predicted_max,

                    "participant_id":
                        example[
                            "_participant_id"
                        ],

                    "window_start":
                        example[
                            "_window_start"
                        ],

                    "raw_generation":
                        generated,
                }
            )


    n = len(
        native_examples
    )


    return {
        "valid_decision_rate":
            valid_decisions
            / n,

        "decision_accuracy":
            correct_decisions
            / n,

        "numeric_output_rate":
            numeric_outputs
            / n,

        "max_accuracy_within_1":
            correct_max
            / n,

        "fully_correct_rate":
            fully_correct
            / n,

        "samples":
            samples,
    }


# ============================================================
# Representation diagnostic
# ============================================================

def representation_stats():

    encoder_vectors = []
    projector_vectors = []

    labels = []


    with torch.no_grad():

        for example in native_examples:

            signal = (
                example[
                    "time_series"
                ][
                    0
                ]
                .detach()
                .float()
                .to(
                    device
                )
                .unsqueeze(
                    0
                )
            )


            encoded = (
                model.encoder(
                    signal
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
                    "_label_yes"
                ]
            )


    def summarize(
        vectors,
    ):

        matrix = torch.stack(
            vectors
        )


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
            if label
        ]


        no_indices = [
            i
            for i, label
            in enumerate(
                labels
            )
            if not label
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


        cosine = float(
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
                cosine,

            "yes_no_centroid_l2_per_sqrt_dim":
                centroid_l2,
        }


    return {
        "encoder":
            summarize(
                encoder_vectors
            ),

        "projector":
            summarize(
                projector_vectors
            ),
    }


# ============================================================
# Initial state
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "NATIVE CGM GROUNDING — BEFORE TRAINING"
)

print(
    "=" * 100
)


initial_pairwise = (
    evaluate_pairwise()
)


initial_representation = (
    representation_stats()
)


print(
    "Pairwise BA:",
    f"{100 * initial_pairwise['balanced_accuracy']:.1f}%"
)

print(
    "YES accuracy:",
    f"{100 * initial_pairwise['yes_accuracy']:.1f}%"
)

print(
    "NO accuracy:",
    f"{100 * initial_pairwise['no_accuracy']:.1f}%"
)

print(
    "Mean P(correct):",
    f"{initial_pairwise['mean_p_correct']:.4f}"
)


print(
    "\nEncoder feature SD:",
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


initial_generation = (
    evaluate_generation()
)


print(
    "\nGreedy generation:"
)

print(
    "Valid Yes/No:",
    f"{100 * initial_generation['valid_decision_rate']:.1f}%"
)

print(
    "Decision accuracy:",
    f"{100 * initial_generation['decision_accuracy']:.1f}%"
)

print(
    "Numeric output:",
    f"{100 * initial_generation['numeric_output_rate']:.1f}%"
)

print(
    "Correct maximum:",
    f"{100 * initial_generation['max_accuracy_within_1']:.1f}%"
)

print(
    "Fully correct:",
    f"{100 * initial_generation['fully_correct_rate']:.1f}%"
)


# ============================================================
# Training
# ============================================================

history = []


history.append(
    {
        "epoch":
            0,

        "sequence_loss":
            None,

        "pairwise_aux_loss":
            None,

        "total_loss":
            None,

        "pairwise_metrics":
            initial_pairwise,

        "generation_metrics":
            initial_generation,
    }
)


print(
    "\n"
    + "=" * 100
)

print(
    "TRAINING NATIVE CGM FACTUAL GROUNDING"
)

print(
    "=" * 100
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    sequence_losses = []
    pairwise_losses = []
    total_losses = []


    for batch in (
        training_batches
    ):

        optimizer.zero_grad(
            set_to_none=True
        )


        # --------------------------------------------
        # Natural-language grounding objective
        # --------------------------------------------

        sequence_loss = (
            model.compute_loss(
                batch
            )
        )


        # --------------------------------------------
        # Explicit factual Yes-vs-No auxiliary
        # --------------------------------------------

        pairwise_loss = (
            pairwise_loss_for_batch(
                batch
            )
        )


        total_loss = (
            sequence_loss
            + (
                PAIRWISE_WEIGHT
                * pairwise_loss
            )
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            trainable_parameters,
            GRAD_CLIP_NORM,
        )


        optimizer.step()


        sequence_losses.append(
            float(
                sequence_loss
                .detach()
                .float()
                .cpu()
                .item()
            )
        )


        pairwise_losses.append(
            float(
                pairwise_loss
                .detach()
                .float()
                .cpu()
                .item()
            )
        )


        total_losses.append(
            float(
                total_loss
                .detach()
                .float()
                .cpu()
                .item()
            )
        )


    mean_sequence_loss = (
        sum(
            sequence_losses
        )
        / len(
            sequence_losses
        )
    )


    mean_pairwise_loss = (
        sum(
            pairwise_losses
        )
        / len(
            pairwise_losses
        )
    )


    mean_total_loss = (
        sum(
            total_losses
        )
        / len(
            total_losses
        )
    )


    if (
        epoch
        in EVAL_EPOCHS
    ):

        pairwise_metrics = (
            evaluate_pairwise()
        )


        generation_metrics = None


        if (
            epoch
            in GENERATION_EPOCHS
        ):

            generation_metrics = (
                evaluate_generation()
            )


        history.append(
            {
                "epoch":
                    epoch,

                "sequence_loss":
                    mean_sequence_loss,

                "pairwise_aux_loss":
                    mean_pairwise_loss,

                "total_loss":
                    mean_total_loss,

                "pairwise_metrics":
                    pairwise_metrics,

                "generation_metrics":
                    generation_metrics,
            }
        )


        line = (
            f"Epoch {epoch:2d} | "
            f"seq={mean_sequence_loss:7.4f} | "
            f"pair={mean_pairwise_loss:7.4f} | "
            f"total={mean_total_loss:7.4f} | "
            f"BA="
            f"{100 * pairwise_metrics['balanced_accuracy']:5.1f}% | "
            f"YES="
            f"{100 * pairwise_metrics['yes_accuracy']:5.1f}% | "
            f"NO="
            f"{100 * pairwise_metrics['no_accuracy']:5.1f}% | "
            f"P(correct)="
            f"{pairwise_metrics['mean_p_correct']:.4f}"
        )


        if (
            generation_metrics
            is not None
        ):

            line += (
                " | GEN decision="
                f"{100 * generation_metrics['decision_accuracy']:.1f}%"
                " | max="
                f"{100 * generation_metrics['max_accuracy_within_1']:.1f}%"
                " | full="
                f"{100 * generation_metrics['fully_correct_rate']:.1f}%"
            )


        print(
            line
        )


# ============================================================
# Final representation
# ============================================================

final_representation = (
    representation_stats()
)


final_pairwise = (
    evaluate_pairwise()
)


final_generation = (
    evaluate_generation()
)


print(
    "\n"
    + "=" * 100
)

print(
    "FINAL NATIVE CGM GROUNDING RESULT"
)

print(
    "=" * 100
)


print(
    "Pairwise balanced accuracy:",
    f"{100 * final_pairwise['balanced_accuracy']:.1f}%"
)

print(
    "YES accuracy:",
    f"{100 * final_pairwise['yes_accuracy']:.1f}%"
)

print(
    "NO accuracy:",
    f"{100 * final_pairwise['no_accuracy']:.1f}%"
)

print(
    "Mean P(correct):",
    f"{final_pairwise['mean_p_correct']:.4f}"
)


print(
    "\nNatural-language generation:"
)

print(
    "Valid Yes/No:",
    f"{100 * final_generation['valid_decision_rate']:.1f}%"
)

print(
    "Decision accuracy:",
    f"{100 * final_generation['decision_accuracy']:.1f}%"
)

print(
    "Numeric output rate:",
    f"{100 * final_generation['numeric_output_rate']:.1f}%"
)

print(
    "Maximum within ±1 mg/dL:",
    f"{100 * final_generation['max_accuracy_within_1']:.1f}%"
)

print(
    "Fully correct:",
    f"{100 * final_generation['fully_correct_rate']:.1f}%"
)


print(
    "\nRepresentation:"
)

print(
    "Encoder SD before:",
    f"{initial_representation['encoder']['mean_feature_std']:.8f}"
)

print(
    "Encoder SD after:",
    f"{final_representation['encoder']['mean_feature_std']:.8f}"
)

print(
    "Encoder Y/N cosine before:",
    f"{initial_representation['encoder']['yes_no_centroid_cosine']:.8f}"
)

print(
    "Encoder Y/N cosine after:",
    f"{final_representation['encoder']['yes_no_centroid_cosine']:.8f}"
)

print(
    "Projector SD before:",
    f"{initial_representation['projector']['mean_feature_std']:.8f}"
)

print(
    "Projector SD after:",
    f"{final_representation['projector']['mean_feature_std']:.8f}"
)


# ============================================================
# Show representative generations
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "REPRESENTATIVE FINAL GENERATIONS"
)

print(
    "=" * 100
)


for item in (
    final_generation[
        "samples"
    ][
        :5
    ]
):

    print(
        f"\nGold: "
        f"{'YES' if item['gold_yes'] else 'NO'} "
        f"| max={item['gold_max']:.0f}"
    )

    print(
        "Generated:",
        repr(
            item[
                "raw_generation"
            ]
        ),
    )


print(
    "\n--- NO EXAMPLES ---"
)


no_samples = [
    item

    for item
    in final_generation[
        "samples"
    ]

    if not item[
        "gold_yes"
    ]
]


for item in (
    no_samples[
        :5
    ]
):

    print(
        f"\nGold: NO "
        f"| max={item['gold_max']:.0f}"
    )

    print(
        "Generated:",
        repr(
            item[
                "raw_generation"
            ]
        ),
    )


# ============================================================
# Save results
# ============================================================

serializable_examples = [
    {
        "participant_id":
            example[
                "_participant_id"
            ],

        "window_start":
            example[
                "_window_start"
            ],

        "label_yes":
            example[
                "_label_yes"
            ],

        "window_mean":
            example[
                "_window_mean"
            ],

        "window_std":
            example[
                "_window_std"
            ],

        "maximum":
            example[
                "_maximum"
            ],

        "answer_text":
            example[
                "_answer_text"
            ],
    }

    for example
    in native_examples
]


results = {
    "experiment":
        "tiny_cgm_native_grounding",

    "stage5_repo":
        STAGE5_REPO_ID,

    "seed":
        SEED,

    "task":
        TASK_ID,

    "epochs":
        EPOCHS,

    "batch_size":
        BATCH_SIZE,

    "n_yes":
        N_PER_CLASS,

    "n_no":
        N_PER_CLASS,

    "input_format":
        {
            "normalization":
                "per_window_zscore",

            "normalization_std":
                "population_sd_unbiased_false",

            "text_context":
                [
                    "sensor_type_cgm",
                    "units_mg_dl",
                    "measurement_count",
                    "nominal_5_minute_sampling",
                    "original_window_mean",
                    "original_window_std",
                ],

            "target":
                (
                    "Yes/No plus maximum glucose "
                    "in mg/dL"
                ),
        },

    "training":
        {
            "frozen":
                [
                    "stage5_patch_embedding",
                    "stage5_position_embedding",
                    "stage5_input_norm",
                    "stage5_transformers_1_4",
                    "stage5_llm",
                    "stage5_lora",
                ],

            "trainable":
                [
                    "stage5_transformer_5",
                    "stage5_transformer_6",
                    "stage5_projector",
                ],

            "lr_t5_t6":
                LR_T5_T6,

            "lr_projector":
                LR_PROJECTOR,

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip":
                GRAD_CLIP_NORM,

            "pairwise_aux_weight":
                PAIRWISE_WEIGHT,

            "dropout":
                "disabled_via_eval_mode",
        },

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "examples":
        serializable_examples,

    "initial_pairwise":
        initial_pairwise,

    "initial_generation":
        initial_generation,

    "initial_representation":
        initial_representation,

    "history":
        history,

    "final_pairwise":
        final_pairwise,

    "final_generation":
        final_generation,

    "final_representation":
        final_representation,
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
        results,
        f,
        indent=2,
    )


print(
    "\nSaved:"
)

print(
    RESULTS_PATH
)


print(
    "\nNo checkpoint was saved."
)

print(
    "No validation/test participants were used."
)

print(
    "TEST participants remain sealed."
)


# ============================================================
# Cleanup
# ============================================================

del optimizer
del model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()