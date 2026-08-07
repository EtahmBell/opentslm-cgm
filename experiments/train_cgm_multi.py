"""
First multi-participant OpenTSLM -> CGM alignment experiment.

Goal:
    Determine whether OpenTSLM can learn a CGM-specific rule
    that GENERALIZES across participants.

Task:
    Did glucose exceed 160 mg/dL?

Key differences from the participant-1001 smoke test:
    - multiple participants
    - split by participant
    - fixed training-set normalization
    - non-overlapping windows
    - balanced training labels
    - validation participants were never seen during training
"""

import json
import random
import sys
from pathlib import Path

import torch
from torch.optim import AdamW

from opentslm import OpenTSLM


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


from src.cgm_multidata import (
    build_examples,
    balance_binary_examples,
)


DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

CHECKPOINT_DIR.mkdir(
    exist_ok=True
)

OUTPUT_CHECKPOINT = (
    CHECKPOINT_DIR
    / "cgm_alignment_multi.pt"
)


# ============================================================
# Training configuration
# ============================================================

REPO_ID = (
    "OpenTSLM/llama-3.2-1b-tsqa-sp"
)

SEED = 42

WINDOW_SIZE = 128
STRIDE = 128

THRESHOLD = 160

ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4

WEIGHT_DECAY = 1e-2

# Start with ONE epoch.
#
# Increase later after we inspect validation behavior.
EPOCHS = 1

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

    manifest = json.load(
        f
    )


train_mean = (
    manifest[
        "normalization"
    ]["mean"]
)

train_std = (
    manifest[
        "normalization"
    ]["std"]
)


def manifest_files(split_name):

    return [
        DATA_DIR
        / item["filename"]

        for item
        in manifest[split_name]
    ]


train_files = manifest_files(
    "train"
)

val_files = manifest_files(
    "validation"
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
    "Normalization mean:",
    round(
        train_mean,
        3,
    ),
)

print(
    "Normalization SD:",
    round(
        train_std,
        3,
    ),
)


# ============================================================
# Load OpenTSLM
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


model = OpenTSLM.load_pretrained(
    REPO_ID,
    device=device,
)


EOS = model.get_eos_token()


# ============================================================
# Build examples
# ============================================================

print(
    "\nBuilding training examples..."
)


train_examples_all = (
    build_examples(
        train_files,

        normalization_mean=
            train_mean,

        normalization_std=
            train_std,

        eos_token=
            EOS,

        window_size=
            WINDOW_SIZE,

        stride=
            STRIDE,

        threshold=
            THRESHOLD,
    )
)


train_examples = (
    balance_binary_examples(
        train_examples_all,
        seed=SEED,
    )
)


print(
    "Raw training examples:",
    len(train_examples_all),
)

print(
    "Balanced training examples:",
    len(train_examples),
)


print(
    "Training YES:",
    sum(
        x["_answer_clean"] == "(a)"
        for x in train_examples
    ),
)


print(
    "Training NO:",
    sum(
        x["_answer_clean"] == "(b)"
        for x in train_examples
    ),
)


print(
    "\nBuilding validation examples..."
)


val_examples = (
    build_examples(
        val_files,

        normalization_mean=
            train_mean,

        normalization_std=
            train_std,

        eos_token=
            EOS,

        window_size=
            WINDOW_SIZE,

        stride=
            STRIDE,

        threshold=
            THRESHOLD,
    )
)


print(
    "Validation examples:",
    len(val_examples),
)


# ============================================================
# OpenTSLM-facing sample
# ============================================================

def model_sample(example):

    return {
        "pre_prompt":
            example["pre_prompt"],

        "time_series_text":
            example["time_series_text"],

        "time_series":
            example["time_series"],

        "post_prompt":
            example["post_prompt"],

        "answer":
            example["answer"],
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    examples,
    label,
    max_examples=None,
):

    if max_examples is not None:
        examples = examples[
            :max_examples
        ]

    correct = 0

    yes_correct = 0
    yes_total = 0

    no_correct = 0
    no_total = 0


    print(
        "\n"
        + "=" * 70
    )

    print(
        label
    )

    print(
        "=" * 70
    )


    for i, example in enumerate(
        examples
    ):

        prediction = (
            model.generate(
                [
                    model_sample(
                        example
                    )
                ],
                max_new_tokens=5,
            )[0]
            .strip()
        )

        gold = (
            example[
                "_answer_clean"
            ]
        )

        is_correct = (
            prediction.startswith(
                gold
            )
        )


        if gold == "(a)":

            yes_total += 1

            if is_correct:
                yes_correct += 1

        else:

            no_total += 1

            if is_correct:
                no_correct += 1


        if is_correct:
            correct += 1


        print(
            f"{i + 1:03d} | "
            f"participant="
            f"{example['_participant_id']} | "
            f"max="
            f"{example['_max_glucose']:6.1f} | "
            f"gold={gold} | "
            f"pred={repr(prediction)} | "
            f"{'✓' if is_correct else '✗'}"
        )


    total = len(
        examples
    )


    accuracy = (
        correct / total
        if total
        else 0
    )


    yes_accuracy = (
        yes_correct / yes_total
        if yes_total
        else 0
    )


    no_accuracy = (
        no_correct / no_total
        if no_total
        else 0
    )


    balanced_accuracy = (
        (
            yes_accuracy
            + no_accuracy
        )
        / 2
    )


    print(
        f"\nOverall accuracy: "
        f"{accuracy * 100:.1f}%"
    )


    print(
        f"YES accuracy: "
        f"{yes_accuracy * 100:.1f}%"
    )


    print(
        f"NO accuracy: "
        f"{no_accuracy * 100:.1f}%"
    )


    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy * 100:.1f}%"
    )


    return {
        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,
    }


# ============================================================
# Baseline validation
# ============================================================

model.encoder.eval()
model.projector.eval()
model.llm.eval()


# Generation on CPU is slow.
#
# Use only first 30 validation examples for the initial
# baseline sanity check.
baseline_metrics = evaluate(
    val_examples,
    "ORIGINAL OPENTSLM — VALIDATION",
    max_examples=30,
)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    [
        {
            "params":
                model.encoder.parameters(),

            "lr":
                ENCODER_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },

        {
            "params":
                model.projector.parameters(),

            "lr":
                PROJECTOR_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },
    ]
)


model.llm.eval()

model.encoder.train()
model.projector.train()


# ============================================================
# Training
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "TRAINING"
)

print(
    "=" * 70
)


rng = random.Random(
    SEED
)


step = 0


for epoch in range(
    EPOCHS
):

    rng.shuffle(
        train_examples
    )

    epoch_loss = 0.0


    for example in train_examples:

        step += 1


        optimizer.zero_grad()


        loss = model.compute_loss(
            [
                model_sample(
                    example
                )
            ]
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            list(
                model.encoder.parameters()
            )
            +
            list(
                model.projector.parameters()
            ),
            max_norm=1.0,
        )


        optimizer.step()


        epoch_loss += (
            loss.item()
        )


        if (
            step % 25 == 0
            or step == 1
        ):

            print(
                f"Epoch "
                f"{epoch + 1}/{EPOCHS} | "
                f"Step {step} | "
                f"Loss "
                f"{loss.item():.4f}"
            )


    mean_epoch_loss = (
        epoch_loss
        / len(train_examples)
    )


    print(
        f"\nEpoch "
        f"{epoch + 1} "
        f"mean loss: "
        f"{mean_epoch_loss:.4f}"
    )


# ============================================================
# Validation after alignment training
# ============================================================

model.encoder.eval()
model.projector.eval()
model.llm.eval()


aligned_metrics = evaluate(
    val_examples,
    "CGM-ALIGNED OPENTSLM — VALIDATION",
)


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 70
)

print(
    "SUMMARY"
)

print(
    "=" * 70
)


print(
    "Baseline balanced accuracy:",
    f"{baseline_metrics['balanced_accuracy'] * 100:.1f}%"
)


print(
    "Aligned balanced accuracy:",
    f"{aligned_metrics['balanced_accuracy'] * 100:.1f}%"
)


# ============================================================
# Save checkpoint
# ============================================================

model.store_to_file(
    str(
        OUTPUT_CHECKPOINT
    )
)


print(
    "\nSaved checkpoint:"
)

print(
    OUTPUT_CHECKPOINT
)