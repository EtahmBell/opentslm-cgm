"""
Tiny Stage-5 CGM overfit diagnostic.

Purpose
-------
Our Stage-5 class-token experiment stayed near chance even on
training examples.

Before changing datasets or alignment formats, ask the simplest
possible question:

    Can Stage 5 memorize 20 CGM examples?

Task:
    above_180

Training set:
    10 YES examples
    10 NO examples

Two conditions:

A. FROZEN LoRA
    base Llama      frozen
    Stage-5 LoRA    frozen
    encoder          trainable
    projector        trainable

B. TRAINABLE LoRA
    base Llama      frozen
    Stage-5 LoRA    trainable
    encoder          trainable
    projector        trainable

Both conditions:
    - start from the SAME original Stage-5 checkpoint
    - use the SAME 20 examples
    - use the SAME epoch order
    - use class-token-only loss
    - run 30 epochs

Interpretation
--------------
Both reach ~100%:
    architecture can learn; one-pass 1200-example schedule is likely
    the issue.

Frozen learns, trainable fails:
    LoRA adaptation is interfering.

Frozen fails, trainable learns:
    language-side adaptation is required.

Neither learns:
    investigate Stage-5 signal pathway / optimization more deeply.

TEST participants remain sealed.
"""

import json
import random
import sys

from pathlib import Path

import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


# ============================================================
# Project paths
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
    / "stage5_tiny_overfit_above180_results.json"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

N_PER_CLASS = 10

EPOCHS = 30
EVAL_EVERY = 5


ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4
LORA_LR = 1e-4

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0


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
# We need the EOS token before constructing examples.
#
# Load one temporary Stage-5 model just to obtain tokenizer/EOS
# and determine the answer token IDs.
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "PREPARING TOKENIZER AND DATA"
)

print(
    "=" * 78
)


template_model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


EOS_TOKEN = (
    template_model.get_eos_token()
)


# ============================================================
# Build all training examples for above_180
# ============================================================

raw_examples = (
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
    for x in raw_examples

    if x[
        "_answer_clean"
    ] == "(a)"
]


no_examples = [

    x
    for x in raw_examples

    if x[
        "_answer_clean"
    ] == "(b)"
]


print(
    "\nAvailable YES:",
    len(yes_examples),
)

print(
    "Available NO:",
    len(no_examples),
)


if len(yes_examples) < N_PER_CLASS:

    raise RuntimeError(
        "Not enough YES examples."
    )


if len(no_examples) < N_PER_CLASS:

    raise RuntimeError(
        "Not enough NO examples."
    )


# ============================================================
# Pick EXACT same 20 examples for both conditions
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


tiny_train_examples = (
    selected_yes
    + selected_no
)


print(
    "\nTiny training set:"
)

print(
    "YES:",
    len(selected_yes),
)

print(
    "NO:",
    len(selected_no),
)

print(
    "TOTAL:",
    len(tiny_train_examples),
)


# ============================================================
# Determine exact answer-token structure
# ============================================================

ANSWER_A = next(
    x[
        "answer"
    ]
    for x in tiny_train_examples
    if x[
        "_answer_clean"
    ] == "(a)"
)


ANSWER_B = next(
    x[
        "answer"
    ]
    for x in tiny_train_examples
    if x[
        "_answer_clean"
    ] == "(b)"
)


def tokenize_answer(
    model,
    answer,
):

    return (
        model.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
        )
        .input_ids[
            0
        ]
        .to(
            device
        )
    )


answer_a_ids = (
    tokenize_answer(
        template_model,
        ANSWER_A,
    )
)


answer_b_ids = (
    tokenize_answer(
        template_model,
        ANSWER_B,
    )
)


common_prefix_len = 0


while (
    common_prefix_len
    < min(
        len(answer_a_ids),
        len(answer_b_ids),
    )
    and answer_a_ids[
        common_prefix_len
    ].item()
    == answer_b_ids[
        common_prefix_len
    ].item()
):

    common_prefix_len += 1


if common_prefix_len >= min(
    len(answer_a_ids),
    len(answer_b_ids),
):

    raise RuntimeError(
        "No discriminating answer token found."
    )


COMMON_PREFIX_IDS = (
    answer_a_ids[
        :common_prefix_len
    ]
    .detach()
    .clone()
)


A_TOKEN_ID = int(
    answer_a_ids[
        common_prefix_len
    ].item()
)


B_TOKEN_ID = int(
    answer_b_ids[
        common_prefix_len
    ].item()
)


print(
    "\nAnswer A IDs:",
    answer_a_ids.tolist(),
)

print(
    "Answer B IDs:",
    answer_b_ids.tolist(),
)

print(
    "Common prefix IDs:",
    COMMON_PREFIX_IDS.tolist(),
)

print(
    "A decision token:",
    A_TOKEN_ID,
    template_model.tokenizer.convert_ids_to_tokens(
        [
            A_TOKEN_ID
        ]
    ),
)

print(
    "B decision token:",
    B_TOKEN_ID,
    template_model.tokenizer.convert_ids_to_tokens(
        [
            B_TOKEN_ID
        ]
    ),
)


# We no longer need the temporary model.
del template_model


# ============================================================
# Fixed epoch order
#
# Both conditions receive the exact same example order
# within every epoch.
# ============================================================

epoch_orders = []


for epoch_index in range(
    EPOCHS
):

    indices = list(
        range(
            len(
                tiny_train_examples
            )
        )
    )


    epoch_rng = random.Random(
        SEED
        + 100000
        + epoch_index
    )


    epoch_rng.shuffle(
        indices
    )


    epoch_orders.append(
        indices
    )


# ============================================================
# Model helpers
# ============================================================

def get_class_decision_logits(
    model,
    example,
):
    """
    Get full-vocabulary logits at the exact point where
    the model must choose "(a" versus "(b".
    """

    (
        prompt_embeds,
        prompt_mask,
    ) = (
        model.pad_and_apply_batch(
            [
                example
            ]
        )
    )


    # Remove trailing padding, if any.
    valid_prompt_length = int(
        prompt_mask[
            0
        ]
        .sum()
        .item()
    )


    prompt_embeds = (
        prompt_embeds[
            :,
            :valid_prompt_length,
            :
        ]
    )


    prompt_mask = (
        prompt_mask[
            :,
            :valid_prompt_length
        ]
    )


    prefix_ids = (
        COMMON_PREFIX_IDS.to(
            prompt_embeds.device
        )
    )


    if prefix_ids.numel() > 0:

        prefix_embeds = (
            model.llm
            .get_input_embeddings()(
                prefix_ids.unsqueeze(
                    0
                )
            )
        )


        prefix_embeds = (
            prefix_embeds.to(
                dtype=
                    prompt_embeds.dtype,
                device=
                    prompt_embeds.device,
            )
        )


        prefix_mask = (
            torch.ones(
                (
                    1,
                    prefix_ids.numel(),
                ),
                dtype=
                    prompt_mask.dtype,
                device=
                    prompt_mask.device,
            )
        )


        inputs_embeds = (
            torch.cat(
                [
                    prompt_embeds,
                    prefix_embeds,
                ],
                dim=1,
            )
        )


        attention_mask = (
            torch.cat(
                [
                    prompt_mask,
                    prefix_mask,
                ],
                dim=1,
            )
        )


    else:

        inputs_embeds = (
            prompt_embeds
        )

        attention_mask = (
            prompt_mask
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


    return (
        outputs.logits[
            0,
            -1,
            :
        ]
        .float()
    )


# ============================================================
# Class-token loss
# ============================================================

def compute_class_token_loss(
    model,
    example,
):

    logits = (
        get_class_decision_logits(
            model,
            example,
        )
    )


    gold = (
        example[
            "_answer_clean"
        ]
    )


    if gold == "(a)":

        target_id = (
            A_TOKEN_ID
        )

    elif gold == "(b)":

        target_id = (
            B_TOKEN_ID
        )

    else:

        raise RuntimeError(
            f"Unexpected gold label: "
            f"{gold}"
        )


    target = torch.tensor(
        [
            target_id
        ],
        dtype=torch.long,
        device=
            logits.device,
    )


    return (
        F.cross_entropy(
            logits.unsqueeze(
                0
            ),
            target,
        )
    )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    examples,
):

    model.eval()


    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    predicted_yes = 0
    predicted_no = 0

    probabilities = []


    with torch.no_grad():

        for example in examples:

            logits = (
                get_class_decision_logits(
                    model,
                    example,
                )
            )


            pair_logits = (
                torch.stack(
                    [
                        logits[
                            A_TOKEN_ID
                        ],

                        logits[
                            B_TOKEN_ID
                        ],
                    ]
                )
            )


            probs = (
                torch.softmax(
                    pair_logits,
                    dim=0,
                )
            )


            p_a = float(
                probs[
                    0
                ].cpu()
            )


            p_b = float(
                probs[
                    1
                ].cpu()
            )


            pred = (
                "(a)"
                if p_a > p_b
                else "(b)"
            )


            gold = (
                example[
                    "_answer_clean"
                ]
            )


            if pred == "(a)":

                predicted_yes += 1

            else:

                predicted_no += 1


            if gold == "(a)":

                yes_total += 1

                if pred == "(a)":

                    yes_correct += 1

                p_correct = p_a

            else:

                no_total += 1

                if pred == "(b)":

                    no_correct += 1

                p_correct = p_b


            probabilities.append(
                {
                    "gold":
                        gold,

                    "prediction":
                        pred,

                    "p_a":
                        p_a,

                    "p_b":
                        p_b,

                    "p_correct":
                        p_correct,

                    "participant_id":
                        example.get(
                            "_participant_id"
                        ),

                    "window_start":
                        example.get(
                            "_window_start"
                        ),
                }
            )


    yes_accuracy = (
        yes_correct
        / yes_total
    )


    no_accuracy = (
        no_correct
        / no_total
    )


    balanced_accuracy = (
        yes_accuracy
        + no_accuracy
    ) / 2.0


    mean_p_correct = (
        sum(
            x[
                "p_correct"
            ]

            for x in probabilities
        )
        / len(
            probabilities
        )
    )


    return {
        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "predicted_yes":
            predicted_yes,

        "predicted_no":
            predicted_no,

        "mean_probability_correct":
            mean_p_correct,

        "predictions":
            probabilities,
    }


# ============================================================
# Train one condition
# ============================================================

def run_condition(
    condition_name,
    train_lora,
):

    print(
        "\n\n"
        + "#" * 90
    )

    print(
        condition_name
    )

    print(
        "#" * 90
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # Fresh original Stage-5 model for EACH condition.
    # --------------------------------------------------------

    torch.manual_seed(
        SEED
    )

    model = (
        OpenTSLM.load_pretrained(
            STAGE5_REPO_ID,
            device=device,
            enable_lora=True,
        )
    )


    # --------------------------------------------------------
    # Freeze complete LLM first.
    # --------------------------------------------------------

    for parameter in (
        model.llm.parameters()
    ):

        parameter.requires_grad = False


    # --------------------------------------------------------
    # Optionally re-enable LoRA.
    # --------------------------------------------------------

    if train_lora:

        for name, parameter in (
            model.llm.named_parameters()
        ):

            if "lora_" in name:

                parameter.requires_grad = True


    # --------------------------------------------------------
    # Encoder/projector always train.
    # --------------------------------------------------------

    for parameter in (
        model.encoder.parameters()
    ):

        parameter.requires_grad = True


    for parameter in (
        model.projector.parameters()
    ):

        parameter.requires_grad = True


    encoder_parameters = list(
        model.encoder.parameters()
    )


    projector_parameters = list(
        model.projector.parameters()
    )


    lora_parameters = list(
        model.get_lora_parameters()
    )


    trainable_lora_parameters = [

        p
        for p in lora_parameters

        if p.requires_grad
    ]


    base_llm_trainable = sum(

        p.numel()

        for name, p
        in model.llm.named_parameters()

        if (
            p.requires_grad
            and "lora_" not in name
        )
    )


    encoder_count = sum(

        p.numel()

        for p in encoder_parameters

        if p.requires_grad
    )


    projector_count = sum(

        p.numel()

        for p in projector_parameters

        if p.requires_grad
    )


    lora_count = sum(

        p.numel()

        for p in trainable_lora_parameters
    )


    print(
        "\nTrainable encoder:",
        f"{encoder_count:,}"
    )

    print(
        "Trainable projector:",
        f"{projector_count:,}"
    )

    print(
        "Trainable LoRA:",
        f"{lora_count:,}"
    )

    print(
        "Trainable non-LoRA LLM:",
        f"{base_llm_trainable:,}"
    )


    if base_llm_trainable != 0:

        raise RuntimeError(
            "Base Llama unexpectedly trainable."
        )


    if train_lora:

        if lora_count == 0:

            raise RuntimeError(
                "LoRA should be trainable "
                "but zero parameters found."
            )

    else:

        if lora_count != 0:

            raise RuntimeError(
                "LoRA should be frozen "
                "but trainable parameters found."
            )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    parameter_groups = [
        {
            "params":
                encoder_parameters,

            "lr":
                ENCODER_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },

        {
            "params":
                projector_parameters,

            "lr":
                PROJECTOR_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },
    ]


    if train_lora:

        parameter_groups.append(
            {
                "params":
                    trainable_lora_parameters,

                "lr":
                    LORA_LR,

                "weight_decay":
                    WEIGHT_DECAY,
            }
        )


    optimizer = AdamW(
        parameter_groups
    )


    trainable_parameters = [

        p
        for group in parameter_groups
        for p in group[
            "params"
        ]
        if p.requires_grad
    ]


    # --------------------------------------------------------
    # Initial performance
    # --------------------------------------------------------

    initial = (
        evaluate(
            model,
            tiny_train_examples,
        )
    )


    print(
        "\nBefore training:"
    )

    print(
        f"YES accuracy:      "
        f"{initial['yes_accuracy'] * 100:.1f}%"
    )

    print(
        f"NO accuracy:       "
        f"{initial['no_accuracy'] * 100:.1f}%"
    )

    print(
        f"Balanced accuracy: "
        f"{initial['balanced_accuracy'] * 100:.1f}%"
    )

    print(
        f"Mean P(correct):   "
        f"{initial['mean_probability_correct']:.4f}"
    )


    history = [
        {
            "epoch":
                0,

            "mean_loss":
                None,

            "evaluation":
                initial,
        }
    ]


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.encoder.train()
        model.projector.train()

        # Keep LLM in train mode only when LoRA itself trains.
        #
        # Otherwise eval mode keeps the frozen language side
        # deterministic while encoder/projector adapt.
        if train_lora:

            model.llm.train()

        else:

            model.llm.eval()


        epoch_losses = []


        indices = (
            epoch_orders[
                epoch - 1
            ]
        )


        for index in indices:

            example = (
                tiny_train_examples[
                    index
                ]
            )


            optimizer.zero_grad(
                set_to_none=True
            )


            loss = (
                compute_class_token_loss(
                    model,
                    example,
                )
            )


            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    f"Non-finite loss at "
                    f"epoch {epoch}"
                )


            loss.backward()


            clip_grad_norm_(
                trainable_parameters,
                GRAD_CLIP_NORM,
            )


            optimizer.step()


            epoch_losses.append(
                float(
                    loss
                    .detach()
                    .cpu()
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
        # Evaluate every 5 epochs + first + final
        # ----------------------------------------------------

        if (
            epoch == 1
            or epoch % EVAL_EVERY == 0
            or epoch == EPOCHS
        ):

            result = (
                evaluate(
                    model,
                    tiny_train_examples,
                )
            )


            history.append(
                {
                    "epoch":
                        epoch,

                    "mean_loss":
                        mean_epoch_loss,

                    "evaluation":
                        result,
                }
            )


            print(
                f"\nEpoch "
                f"{epoch:2d}/{EPOCHS}"
            )

            print(
                f"  mean loss:        "
                f"{mean_epoch_loss:.4f}"
            )

            print(
                f"  YES accuracy:     "
                f"{result['yes_accuracy'] * 100:5.1f}%"
            )

            print(
                f"  NO accuracy:      "
                f"{result['no_accuracy'] * 100:5.1f}%"
            )

            print(
                f"  balanced acc:     "
                f"{result['balanced_accuracy'] * 100:5.1f}%"
            )

            print(
                f"  predicted YES:    "
                f"{result['predicted_yes']}"
            )

            print(
                f"  predicted NO:     "
                f"{result['predicted_no']}"
            )

            print(
                f"  mean P(correct):  "
                f"{result['mean_probability_correct']:.4f}"
            )


    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    final_result = (
        evaluate(
            model,
            tiny_train_examples,
        )
    )


    return {
        "condition":
            condition_name,

        "train_lora":
            train_lora,

        "trainable_parameters":
            {
                "encoder":
                    encoder_count,

                "projector":
                    projector_count,

                "lora":
                    lora_count,

                "base_llm":
                    base_llm_trainable,
            },

        "history":
            history,

        "final":
            final_result,
    }


# ============================================================
# Run both conditions
# ============================================================

frozen_result = (
    run_condition(
        condition_name=
            "CONDITION A — STAGE-5 LoRA FROZEN",

        train_lora=False,
    )
)


trainable_result = (
    run_condition(
        condition_name=
            "CONDITION B — STAGE-5 LoRA TRAINABLE",

        train_lora=True,
    )
)


# ============================================================
# Final comparison
# ============================================================

print(
    "\n\n"
    + "=" * 90
)

print(
    "TINY-OVERFIT FINAL COMPARISON"
)

print(
    "=" * 90
)


print(
    f"{'Condition':30s} "
    f"{'YES':>8s} "
    f"{'NO':>8s} "
    f"{'BA':>8s} "
    f"{'P(correct)':>12s}"
)

print(
    "-" * 72
)


for result in [
    frozen_result,
    trainable_result,
]:

    final = (
        result[
            "final"
        ]
    )


    label = (
        "LoRA FROZEN"
        if not result[
            "train_lora"
        ]
        else "LoRA TRAINABLE"
    )


    print(
        f"{label:30s} "
        f"{final['yes_accuracy'] * 100:7.1f}% "
        f"{final['no_accuracy'] * 100:7.1f}% "
        f"{final['balanced_accuracy'] * 100:7.1f}% "
        f"{final['mean_probability_correct']:11.4f}"
    )


# ============================================================
# Automatic interpretation hint
# ============================================================

frozen_ba = (
    frozen_result[
        "final"
    ][
        "balanced_accuracy"
    ]
)


trainable_ba = (
    trainable_result[
        "final"
    ][
        "balanced_accuracy"
    ]
)


print(
    "\n"
    + "=" * 90
)

print(
    "DIAGNOSTIC INTERPRETATION"
)

print(
    "=" * 90
)


if (
    frozen_ba >= 0.95
    and trainable_ba >= 0.95
):

    print(
        "Both conditions memorized the 20 examples."
    )

    print(
        "→ Stage-5 can learn CGM class distinctions."
    )

    print(
        "→ Focus next on the one-pass / multitask training regime."
    )


elif (
    frozen_ba >= 0.95
    and trainable_ba < 0.80
):

    print(
        "Frozen LoRA memorized, trainable LoRA did not."
    )

    print(
        "→ Strong evidence that LoRA adaptation is interfering."
    )


elif (
    trainable_ba >= 0.95
    and frozen_ba < 0.80
):

    print(
        "Trainable LoRA memorized, frozen LoRA did not."
    )

    print(
        "→ Language-side adaptation appears necessary."
    )


elif (
    frozen_ba < 0.80
    and trainable_ba < 0.80
):

    print(
        "Neither condition cleanly memorized the 20 examples."
    )

    print(
        "→ Investigate Stage-5 signal/gradient pathway next."
    )


else:

    print(
        "Mixed / partial-learning result."
    )

    print(
        "→ Inspect learning curves before changing anything."
    )


# ============================================================
# Save results
# ============================================================

results = {
    "experiment":
        "stage5_tiny_overfit_above180",

    "seed":
        SEED,

    "task":
        TASK_ID,

    "n_yes":
        N_PER_CLASS,

    "n_no":
        N_PER_CLASS,

    "epochs":
        EPOCHS,

    "learning_rates":
        {
            "encoder":
                ENCODER_LR,

            "projector":
                PROJECTOR_LR,

            "lora":
                LORA_LR,
        },

    "normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "answer_tokens":
        {
            "a":
                A_TOKEN_ID,

            "b":
                B_TOKEN_ID,

            "common_prefix":
                COMMON_PREFIX_IDS
                .cpu()
                .tolist(),
        },

    "selected_examples":
        [
            {
                "participant_id":
                    x.get(
                        "_participant_id"
                    ),

                "window_start":
                    x.get(
                        "_window_start"
                    ),

                "gold":
                    x[
                        "_answer_clean"
                    ],
            }

            for x in tiny_train_examples
        ],

    "frozen_lora":
        frozen_result,

    "trainable_lora":
        trainable_result,
}


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
    "\nSaved results:"
)

print(
    RESULTS_PATH
)


print(
    "\nTEST participants remain sealed."
)