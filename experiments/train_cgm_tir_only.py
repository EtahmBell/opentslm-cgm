"""
TIR-only CGM alignment experiment.

Purpose
-------
Determine whether the failure to learn TIR in the six-task experiment
was caused by:

1. TIR being difficult/unlearnable for this OpenTSLM representation, OR
2. interference from simultaneous training on the other five CGM tasks.

IMPORTANT
---------
This script reconstructs the SAME TIR training examples and SAME
validation examples used in the multitask experiment.

Starting checkpoint:
    OpenTSLM/llama-3.2-1b-tsqa-sp

Training:
    23 participants
    100 YES + 100 NO TIR examples

Validation:
    5 unseen participants
    24 YES + 24 NO TIR examples

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
    sample_across_participants,
)


from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

SEED = 42

REPO_ID = (
    "OpenTSLM/llama-3.2-1b-tsqa-sp"
)

TIR_TASK = (
    "tir_70_180_above_90"
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
    / "cgm_alignment_tir_only_tsqa.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_tir_only_results.json"
)


WINDOW_SIZE = 128
STRIDE = 128

TRAIN_PER_CLASS = 100

NUM_EPOCHS = 1

ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0

EVAL_BATCH_SIZE = 4

MAX_NEW_TOKENS = 6

PRINT_EVERY = 25


# ============================================================
# Important seed reconstruction
# ============================================================

# In the original multitask experiment TIR was the sixth task:
#
# task_index = 5
#
# balance_multitask_examples used:
#
# seed + task_index * 1000
#
# Therefore the exact TIR training selection used:
#
# YES seed = 5042
# NO  seed = 5043

MULTITASK_TIR_INDEX = 5

TRAIN_TIR_SEED = (
    SEED
    + MULTITASK_TIR_INDEX * 1000
)


# Validation used:
#
# seed + 10000 + task_index * 1000
#
# Therefore:
#
# YES = 15042
# NO  = 15043
# shuffle = 15044

VAL_TIR_SEED = (
    SEED
    + 10000
    + MULTITASK_TIR_INDEX * 1000
)


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
# Load manifest
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
    len(manifest["test"]),
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
# Load original TSQA checkpoint
# ============================================================

print(
    "\nLoading original TSQA OpenTSLM..."
)

model = (
    OpenTSLM.load_pretrained(
        REPO_ID,
        device=device,
    )
)

model.eval()


EOS_TOKEN = (
    model.get_eos_token()
)


# ============================================================
# Build TIR training pool
# ============================================================

print(
    "\nBuilding TIR training pool..."
)


raw_train = (
    build_multitask_examples(
        files=train_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=[
            TIR_TASK
        ],
    )
)


train_yes = [
    x
    for x in raw_train
    if x["_answer_clean"] == "(a)"
]

train_no = [
    x
    for x in raw_train
    if x["_answer_clean"] == "(b)"
]


# ============================================================
# Reconstruct EXACT TIR training selection
# from multitask run
# ============================================================

selected_train_yes = (
    sample_across_participants(
        train_yes,
        n=TRAIN_PER_CLASS,
        seed=TRAIN_TIR_SEED,
    )
)

selected_train_no = (
    sample_across_participants(
        train_no,
        n=TRAIN_PER_CLASS,
        seed=TRAIN_TIR_SEED + 1,
    )
)


train_examples = (
    selected_train_yes
    + selected_train_no
)


# Same final global shuffle used by the multitask
# balance function.
train_rng = random.Random(
    SEED
)

train_rng.shuffle(
    train_examples
)


print(
    "TIR training YES:",
    len(selected_train_yes),
)

print(
    "TIR training NO:",
    len(selected_train_no),
)

print(
    "TIR training total:",
    len(train_examples),
)


# ============================================================
# Build TIR validation pool
# ============================================================

print(
    "\nBuilding TIR validation pool..."
)


raw_val = (
    build_multitask_examples(
        files=val_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=[
            TIR_TASK
        ],
    )
)


val_yes = [
    x
    for x in raw_val
    if x["_answer_clean"] == "(a)"
]

val_no = [
    x
    for x in raw_val
    if x["_answer_clean"] == "(b)"
]


n_val_per_class = min(
    len(val_yes),
    len(val_no),
)


selected_val_yes = (
    sample_across_participants(
        val_yes,
        n=n_val_per_class,
        seed=VAL_TIR_SEED,
    )
)

selected_val_no = (
    sample_across_participants(
        val_no,
        n=n_val_per_class,
        seed=VAL_TIR_SEED + 1,
    )
)


val_examples = (
    selected_val_yes
    + selected_val_no
)


val_rng = random.Random(
    VAL_TIR_SEED + 2
)

val_rng.shuffle(
    val_examples
)


print(
    "TIR validation YES:",
    len(selected_val_yes),
)

print(
    "TIR validation NO:",
    len(selected_val_no),
)

print(
    "TIR validation total:",
    len(val_examples),
)


# ============================================================
# Prediction helpers
# ============================================================

def clean_prediction(
    prediction,
):

    text = (
        str(prediction)
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

def evaluate(
    examples,
    label,
):

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    invalid = 0

    pred_yes = 0
    pred_no = 0


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


    print(
        "\n"
        + "=" * 72
    )

    print(
        label
    )

    print(
        "=" * 72
    )

    print(
        "Examples:",
        len(examples),
    )

    print(
        "YES accuracy:",
        f"{yes_accuracy * 100:.1f}%"
    )

    print(
        "NO accuracy:",
        f"{no_accuracy * 100:.1f}%"
    )

    print(
        "Balanced accuracy:",
        f"{balanced_accuracy * 100:.1f}%"
    )

    print(
        "Predicted YES:",
        pred_yes,
    )

    print(
        "Predicted NO:",
        pred_no,
    )

    print(
        "Invalid outputs:",
        invalid,
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

        "pred_yes":
            pred_yes,

        "pred_no":
            pred_no,

        "invalid":
            invalid,
    }


# ============================================================
# Baseline validation
# ============================================================

baseline_val = (
    evaluate(
        val_examples,
        "ORIGINAL TSQA — TIR VALIDATION",
    )
)


# ============================================================
# Train encoder + projector
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "TIR-ONLY TRAINING"
)

print(
    "=" * 72
)


# Freeze LLM.
for parameter in (
    model.llm.parameters()
):

    parameter.requires_grad = False


# Train time-series components.
for parameter in (
    model.encoder.parameters()
):

    parameter.requires_grad = True


for parameter in (
    model.projector.parameters()
):

    parameter.requires_grad = True


model.encoder.train()
model.projector.train()

model.llm.eval()


encoder_parameters = list(
    model.encoder.parameters()
)

projector_parameters = list(
    model.projector.parameters()
)


trainable_parameters = (
    encoder_parameters
    + projector_parameters
)


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
    ]
)


loss_history = []


# ============================================================
# Training
# ============================================================

for epoch in range(
    NUM_EPOCHS
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/"
        f"{NUM_EPOCHS}"
    )


    epoch_examples = list(
        train_examples
    )


    epoch_rng = random.Random(
        SEED + epoch
    )

    epoch_rng.shuffle(
        epoch_examples
    )


    epoch_losses = []


    for step, example in enumerate(
        epoch_examples,
        start=1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )


        loss = (
            model.compute_loss(
                [example]
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


        loss_history.append(
            loss_value
        )

        epoch_losses.append(
            loss_value
        )


        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == len(epoch_examples)
        ):

            recent = (
                epoch_losses[
                    -PRINT_EVERY:
                ]
            )

            recent_mean = (
                sum(recent)
                / len(recent)
            )


            print(
                f"Step "
                f"{step:3d}/"
                f"{len(epoch_examples)} | "
                f"Loss "
                f"{loss_value:.4f} | "
                f"Recent mean "
                f"{recent_mean:.4f}"
            )


    print(
        "\nEpoch mean loss:",
        f"{sum(epoch_losses) / len(epoch_losses):.4f}"
    )


# ============================================================
# Post-training evaluation
# ============================================================

model.eval()


train_result = (
    evaluate(
        train_examples,
        "TIR-ONLY MODEL — TRAIN",
    )
)


val_result = (
    evaluate(
        val_examples,
        "TIR-ONLY MODEL — VALIDATION",
    )
)


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "TIR-ONLY SUMMARY"
)

print(
    "=" * 72
)


print(
    "Baseline validation:",
    f"{baseline_val['balanced_accuracy'] * 100:.1f}%"
)

print(
    "After training — TRAIN:",
    f"{train_result['balanced_accuracy'] * 100:.1f}%"
)

print(
    "After training — VALIDATION:",
    f"{val_result['balanced_accuracy'] * 100:.1f}%"
)


print(
    "\nPrevious MULTITASK TIR:"
)

print(
    "Train:      50.0%"
)

print(
    "Validation: 50.0%"
)


# ============================================================
# Save checkpoint
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
        "cgm_tir_only_tsqa",

    "seed":
        SEED,

    "repo_id":
        REPO_ID,

    "train_participants":
        len(train_files),

    "validation_participants":
        len(val_files),

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "train_examples":
        len(train_examples),

    "validation_examples":
        len(val_examples),

    "normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "baseline_validation":
        baseline_val,

    "trained_train":
        train_result,

    "trained_validation":
        val_result,

    "loss_history":
        loss_history,
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