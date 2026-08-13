"""
Stage-5 OpenTSLM full CGM adaptation.

Question
--------
Our first Stage-5 CGM experiment froze the Stage-5 LoRA adapters and
trained only the time-series encoder + projector. That adaptation failed.

This experiment instead continues the Stage-5 SP curriculum more naturally:

    base Llama weights    FROZEN
    Stage-5 LoRA          TRAINABLE
    time-series encoder   TRAINABLE
    projector             TRAINABLE

We use the SAME:
    - 23 train participants
    - 5 validation participants
    - 6 sealed test participants
    - normalization
    - six CGM tasks
    - 100 YES + 100 NO / task
    - 1200 training examples
    - seed-42 training order
    - encoder/projector learning rates

Initial diagnostic:
    seed = 42 only

If this works, repeat seeds 43/44 afterward.

TEST participants remain sealed.
"""

import json
import random
import re
import sys

from pathlib import Path

import torch

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
    balance_multitask_examples,
    sample_across_participants,
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

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_alignment_stage5_full_adaptation_seed42.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_stage5_full_adaptation_seed42_results.json"
)


ACTIVE_TASKS = [
    "above_160",
    "above_180",
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
    "tir_70_180_above_90",
]


WINDOW_SIZE = 128
STRIDE = 128

TRAIN_EXAMPLES_PER_CLASS = 100

ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4

# OpenTSLM curriculum uses projector LR for LoRA.
LORA_LR = PROJECTOR_LR

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0

EVAL_BATCH_SIZE = 4

MAX_NEW_TOKENS = 6

PRINT_EVERY = 50


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
# Directories
# ============================================================

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
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


def manifest_files(
    split_name,
):

    return [

        DATA_DIR
        / item[
            "relative_path"
        ]

        for item
        in manifest[
            split_name
        ]
    ]


train_files = (
    manifest_files(
        "train"
    )
)

val_files = (
    manifest_files(
        "validation"
    )
)


print(
    "Train participants:",
    len(train_files),
)

print(
    "Validation participants:",
    len(val_files),
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
# Load Stage-5 checkpoint WITH LoRA
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "LOADING STAGE-5 ECG CHECKPOINT WITH LoRA"
)

print(
    "=" * 78
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
# Configure full adaptation
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "CONFIGURING TRAINABLE PARAMETERS"
)

print(
    "=" * 78
)


# ------------------------------------------------------------
# First freeze every LLM parameter.
# ------------------------------------------------------------

for parameter in (
    model.llm.parameters()
):

    parameter.requires_grad = False


# ------------------------------------------------------------
# Then selectively re-enable ONLY LoRA parameters.
# Base Llama stays frozen.
# ------------------------------------------------------------

for name, parameter in (
    model.llm.named_parameters()
):

    if "lora_" in name:

        parameter.requires_grad = True


# ------------------------------------------------------------
# Encoder and projector remain trainable.
# ------------------------------------------------------------

for parameter in (
    model.encoder.parameters()
):

    parameter.requires_grad = True


for parameter in (
    model.projector.parameters()
):

    parameter.requires_grad = True


# ============================================================
# Gather parameter groups
# ============================================================

encoder_parameters = list(
    model.encoder.parameters()
)

projector_parameters = list(
    model.projector.parameters()
)

lora_parameters = (
    model.get_lora_parameters()
)


if len(
    lora_parameters
) == 0:

    raise RuntimeError(
        "LoRA is enabled but no trainable "
        "LoRA parameters were found."
    )


trainable_parameters = (
    encoder_parameters
    + projector_parameters
    + lora_parameters
)


# ============================================================
# Parameter audit
# ============================================================

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
    for p in lora_parameters
    if p.requires_grad
)

base_llm_trainable_count = sum(
    p.numel()
    for name, p
    in model.llm.named_parameters()

    if (
        p.requires_grad
        and "lora_" not in name
    )
)


print(
    "Trainable encoder parameters:",
    f"{encoder_count:,}"
)

print(
    "Trainable projector parameters:",
    f"{projector_count:,}"
)

print(
    "Trainable LoRA parameters:",
    f"{lora_count:,}"
)

print(
    "Trainable non-LoRA LLM parameters:",
    f"{base_llm_trainable_count:,}"
)


if base_llm_trainable_count != 0:

    raise RuntimeError(
        "Base LLM unexpectedly contains "
        "trainable non-LoRA parameters."
    )


print(
    "\n✓ Base Llama frozen."
)

print(
    "✓ Stage-5 LoRA trainable."
)

print(
    "✓ Encoder trainable."
)

print(
    "✓ Projector trainable."
)


# ============================================================
# Build training examples
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "BUILDING CGM TRAINING DATA"
)

print(
    "=" * 78
)


raw_train_examples = (
    build_multitask_examples(
        files=train_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=ACTIVE_TASKS,
    )
)


train_examples = (
    balance_multitask_examples(
        examples=raw_train_examples,
        task_ids=ACTIVE_TASKS,
        per_class=TRAIN_EXAMPLES_PER_CLASS,
        seed=SEED,
    )
)


if len(
    train_examples
) != 1200:

    raise RuntimeError(
        f"Expected 1200 training examples, "
        f"found {len(train_examples)}"
    )


print(
    "\nFixed training examples:",
    len(train_examples),
)

print(
    "Training order: original seed-42 mixed order"
)


# ============================================================
# Build validation examples
# ============================================================

print(
    "\nBuilding validation examples..."
)


raw_val_examples = (
    build_multitask_examples(
        files=val_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=ACTIVE_TASKS,
    )
)


# ============================================================
# Same balanced validation selection as previous experiments
# ============================================================

def build_balanced_validation(
    examples,
    task_ids,
    seed,
):

    output = {}


    for task_index, task_id in enumerate(
        task_ids
    ):

        task_examples = [

            x
            for x in examples

            if x[
                "_task_id"
            ] == task_id
        ]


        yes_examples = [

            x
            for x in task_examples

            if x[
                "_answer_clean"
            ] == "(a)"
        ]


        no_examples = [

            x
            for x in task_examples

            if x[
                "_answer_clean"
            ] == "(b)"
        ]


        n_per_class = min(
            len(
                yes_examples
            ),
            len(
                no_examples
            ),
        )


        if n_per_class == 0:

            raise RuntimeError(
                f"{task_id} has only "
                "one validation class."
            )


        task_seed = (
            seed
            + 10000
            + task_index * 1000
        )


        selected_yes = (
            sample_across_participants(
                yes_examples,
                n=n_per_class,
                seed=task_seed,
            )
        )


        selected_no = (
            sample_across_participants(
                no_examples,
                n=n_per_class,
                seed=task_seed + 1,
            )
        )


        balanced = (
            selected_yes
            + selected_no
        )


        rng = random.Random(
            task_seed + 2
        )

        rng.shuffle(
            balanced
        )


        output[
            task_id
        ] = balanced


        print(
            f"{task_id:28s} | "
            f"YES={len(selected_yes):3d} | "
            f"NO={len(selected_no):3d} | "
            f"total={len(balanced):3d}"
        )


    return output


val_by_task = (
    build_balanced_validation(
        raw_val_examples,
        ACTIVE_TASKS,
        SEED,
    )
)


# ============================================================
# Prediction helpers
# ============================================================

def clean_prediction(
    prediction,
):

    text = (
        str(
            prediction
        )
        .strip()
        .lower()
    )


    match = re.search(
        r"\([ab]\)",
        text,
    )


    if match:

        return match.group(0)


    return None


def chunks(
    items,
    batch_size,
):

    for start in range(
        0,
        len(items),
        batch_size,
    ):

        yield items[
            start:
            start + batch_size
        ]


# ============================================================
# Evaluation
# ============================================================

def evaluate_examples(
    examples,
):

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    pred_yes = 0
    pred_no = 0

    invalid = 0


    model.eval()


    with torch.no_grad():

        for batch in chunks(
            examples,
            EVAL_BATCH_SIZE,
        ):

            predictions = (
                model.generate(
                    batch,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            )


            for example, raw_pred in zip(
                batch,
                predictions,
            ):

                gold = (
                    example[
                        "_answer_clean"
                    ]
                )

                pred = (
                    clean_prediction(
                        raw_pred
                    )
                )


                if pred == "(a)":

                    pred_yes += 1

                elif pred == "(b)":

                    pred_no += 1

                else:

                    invalid += 1


                if gold == "(a)":

                    yes_total += 1

                    if pred == gold:

                        yes_correct += 1

                else:

                    no_total += 1

                    if pred == gold:

                        no_correct += 1


    yes_accuracy = (
        yes_correct
        / yes_total
        if yes_total
        else 0.0
    )

    no_accuracy = (
        no_correct
        / no_total
        if no_total
        else 0.0
    )

    balanced_accuracy = (
        yes_accuracy
        + no_accuracy
    ) / 2.0


    valid_output_rate = (
        (
            len(examples)
            - invalid
        )
        / len(examples)
    )


    return {
        "n":
            len(examples),

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "predicted_yes":
            pred_yes,

        "predicted_no":
            pred_no,

        "invalid":
            invalid,

        "valid_output_rate":
            valid_output_rate,
    }


def evaluate_all(
    label,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        label
    )

    print(
        "=" * 78
    )


    results = {}


    for task_id in ACTIVE_TASKS:

        result = (
            evaluate_examples(
                val_by_task[
                    task_id
                ]
            )
        )


        results[
            task_id
        ] = result


        print(
            f"\n{task_id}"
        )

        print(
            f"  n:                 "
            f"{result['n']}"
        )

        print(
            f"  YES accuracy:       "
            f"{result['yes_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  NO accuracy:        "
            f"{result['no_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  balanced accuracy:  "
            f"{result['balanced_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  valid output rate:  "
            f"{result['valid_output_rate'] * 100:5.1f}%"
        )

        print(
            f"  predicted YES:      "
            f"{result['predicted_yes']}"
        )

        print(
            f"  predicted NO:       "
            f"{result['predicted_no']}"
        )

        print(
            f"  invalid outputs:    "
            f"{result['invalid']}"
        )


    macro = (
        sum(
            result[
                "balanced_accuracy"
            ]

            for result
            in results.values()
        )
        / len(results)
    )


    macro_valid = (
        sum(
            result[
                "valid_output_rate"
            ]

            for result
            in results.values()
        )
        / len(results)
    )


    print(
        "\nMACRO BALANCED ACCURACY:",
        f"{macro * 100:.1f}%"
    )

    print(
        "MACRO VALID OUTPUT RATE:",
        f"{macro_valid * 100:.1f}%"
    )


    return {
        "tasks":
            results,

        "macro_balanced_accuracy":
            macro,

        "macro_valid_output_rate":
            macro_valid,
    }


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    [
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

        {
            "params":
                lora_parameters,

            "lr":
                LORA_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },
    ]
)


print(
    "\nLearning rates:"
)

print(
    f"Encoder:   {ENCODER_LR:.2e}"
)

print(
    f"Projector: {PROJECTOR_LR:.2e}"
)

print(
    f"LoRA:      {LORA_LR:.2e}"
)


# ============================================================
# Smoke-test BEFORE training
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STAGE-5 FULL-ADAPTATION SMOKE TEST"
)

print(
    "=" * 78
)


smoke_example = (
    train_examples[
        0
    ]
)


with torch.no_grad():

    smoke_prediction = (
        model.generate(
            [
                smoke_example
            ],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    )[0]


print(
    "Task:",
    smoke_example[
        "_task_id"
    ],
)

print(
    "Gold:",
    smoke_example[
        "_answer_clean"
    ],
)

print(
    "Raw output:",
    smoke_prediction,
)


# ============================================================
# Training
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STAGE-5 FULL CGM ADAPTATION — SEED 42"
)

print(
    "=" * 78
)


model.encoder.train()
model.projector.train()

# PeftModel/LoRA needs training mode for adapters.
model.llm.train()


losses = []


for step, example in enumerate(
    train_examples,
    start=1,
):

    optimizer.zero_grad(
        set_to_none=True
    )


    loss = (
        model.compute_loss(
            [
                example
            ]
        )
    )


    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            f"Non-finite loss at "
            f"step {step}: "
            f"{loss.item()}"
        )


    loss.backward()


    clip_grad_norm_(
        trainable_parameters,
        GRAD_CLIP_NORM,
    )


    optimizer.step()


    loss_value = float(
        loss.detach().cpu()
    )


    losses.append(
        loss_value
    )


    if (
        step == 1
        or step % PRINT_EVERY == 0
        or step == len(train_examples)
    ):

        recent = (
            losses[
                -PRINT_EVERY:
            ]
        )


        recent_mean = (
            sum(
                recent
            )
            / len(
                recent
            )
        )


        print(
            f"Step "
            f"{step:4d}/"
            f"{len(train_examples)} | "
            f"Loss "
            f"{loss_value:.4f} | "
            f"Recent mean "
            f"{recent_mean:.4f}"
        )


mean_loss = (
    sum(
        losses
    )
    / len(
        losses
    )
)


print(
    "\nMean training loss:",
    f"{mean_loss:.4f}"
)


# ============================================================
# Validation
# ============================================================

model.eval()


validation_results = (
    evaluate_all(
        "STAGE-5 FULL ADAPTATION — VALIDATION"
    )
)


# ============================================================
# Previous frozen-LoRA / TSQA seed-42 references
# ============================================================

FROZEN_STAGE5_SEED42 = {
    "above_160":
        0.000,

    "above_180":
        0.000,

    "sustained_above_180_15m":
        0.000,

    "rise_50_30m":
        0.000,

    "fall_50_30m":
        0.500,

    "tir_70_180_above_90":
        0.083,
}


TSQA_SEED42 = {
    "above_160":
        0.806,

    "above_180":
        0.837,

    "sustained_above_180_15m":
        0.892,

    "rise_50_30m":
        0.707,

    "fall_50_30m":
        0.694,

    "tir_70_180_above_90":
        0.500,
}


frozen_macro = (
    sum(
        FROZEN_STAGE5_SEED42.values()
    )
    / len(
        FROZEN_STAGE5_SEED42
    )
)


tsqa_macro = (
    sum(
        TSQA_SEED42.values()
    )
    / len(
        TSQA_SEED42
    )
)


# ============================================================
# Comparison
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "SEED-42 ADAPTATION COMPARISON"
)

print(
    "=" * 90
)


print(
    f"{'Task':28s} "
    f"{'TSQA':>9s} "
    f"{'S5 frozen':>11s} "
    f"{'S5 full':>9s}"
)

print(
    "-" * 64
)


for task_id in ACTIVE_TASKS:

    tsqa = (
        TSQA_SEED42[
            task_id
        ]
    )

    frozen = (
        FROZEN_STAGE5_SEED42[
            task_id
        ]
    )

    full = (
        validation_results[
            "tasks"
        ][
            task_id
        ][
            "balanced_accuracy"
        ]
    )


    print(
        f"{task_id:28s} "
        f"{tsqa * 100:8.1f}% "
        f"{frozen * 100:10.1f}% "
        f"{full * 100:8.1f}%"
    )


full_macro = (
    validation_results[
        "macro_balanced_accuracy"
    ]
)


print(
    "-" * 64
)

print(
    f"{'MACRO':28s} "
    f"{tsqa_macro * 100:8.1f}% "
    f"{frozen_macro * 100:10.1f}% "
    f"{full_macro * 100:8.1f}%"
)


# ============================================================
# Save checkpoint
#
# LoRA parameters are currently trainable, so OpenTSLM's
# store_to_file() will save them with encoder/projector.
# ============================================================

model.store_to_file(
    str(
        CHECKPOINT_PATH
    )
)


# ============================================================
# Save results
# ============================================================

results = {

    "experiment":
        "stage5_full_adaptation_seed42",

    "seed":
        SEED,

    "repo_id":
        STAGE5_REPO_ID,

    "train_participants":
        len(
            train_files
        ),

    "validation_participants":
        len(
            val_files
        ),

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "training_examples":
        len(
            train_examples
        ),

    "trainable_parameters":
        {
            "encoder":
                encoder_count,

            "projector":
                projector_count,

            "lora":
                lora_count,

            "base_llm_non_lora":
                base_llm_trainable_count,
        },

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

    "mean_training_loss":
        mean_loss,

    "loss_history":
        losses,

    "validation":
        validation_results,

    "references":
        {
            "tsqa_seed42":
                TSQA_SEED42,

            "stage5_frozen_lora_seed42":
                FROZEN_STAGE5_SEED42,
        },
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
    "\nSaved checkpoint:"
)

print(
    CHECKPOINT_PATH
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