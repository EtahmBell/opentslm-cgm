"""
Evaluate the final multitask CGM checkpoint on the exact
training examples used in the original experiment.

Main question:
    Did the model actually learn the TIR task during training?

We also report training accuracy for all six tasks for context.

IMPORTANT:
The six test participants are never loaded.
"""

import json
import random
import re
import sys

from collections import defaultdict
from pathlib import Path

import torch


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
)


from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

SEED = 42

REPO_ID = (
    "OpenTSLM/llama-3.2-1b-tsqa-sp"
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
    / "cgm_alignment_multitask_tsqa.pt"
)


ACTIVE_TASKS = [
    "above_160",
    "above_180",
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
    "tir_70_180_above_90",
]


TIR_TASK = (
    "tir_70_180_above_90"
)

WINDOW_SIZE = 128
STRIDE = 128

TRAIN_EXAMPLES_PER_CLASS = 100

EVAL_BATCH_SIZE = 4
MAX_NEW_TOKENS = 6


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
    "Validation participants NOT used:",
    len(
        manifest[
            "validation"
        ]
    ),
)

print(
    "TEST participants remain sealed:",
    len(
        manifest[
            "test"
        ]
    ),
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
# Load original model + our trained checkpoint
# ============================================================

print(
    "\nLoading original TSQA model..."
)

model = (
    OpenTSLM.load_pretrained(
        REPO_ID,
        device=device,
    )
)


print(
    "\nLoading trained CGM checkpoint..."
)

model.load_from_file(
    str(
        CHECKPOINT_PATH
    )
)

model.eval()


EOS_TOKEN = (
    model.get_eos_token()
)


# ============================================================
# Reconstruct exact training pool
# ============================================================

print(
    "\nReconstructing training examples..."
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


# IMPORTANT:
# Reconstruct ALL six tasks together.
#
# This preserves the exact task_index-based seeds that were
# used in train_cgm_multitask.py.
train_examples = (
    balance_multitask_examples(
        examples=raw_train_examples,
        task_ids=ACTIVE_TASKS,
        per_class=TRAIN_EXAMPLES_PER_CLASS,
        seed=SEED,
    )
)


print(
    "\nExact balanced training set:",
    len(train_examples),
)


# ============================================================
# Prediction parsing
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


# ============================================================
# Batch helper
# ============================================================

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
# Evaluate examples
# ============================================================

def evaluate_examples(
    examples,
):

    correct = 0

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    invalid = 0

    rows = []


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

                gold = example[
                    "_answer_clean"
                ]

                pred = (
                    clean_prediction(
                        raw_pred
                    )
                )


                if gold == "(a)":

                    yes_total += 1

                    if pred == gold:
                        yes_correct += 1


                else:

                    no_total += 1

                    if pred == gold:
                        no_correct += 1


                if pred == gold:
                    correct += 1


                if pred is None:
                    invalid += 1


                rows.append(
                    {
                        "task_id":
                            example[
                                "_task_id"
                            ],

                        "participant_id":
                            example[
                                "_participant_id"
                            ],

                        "window_start":
                            example[
                                "_window_start"
                            ],

                        "metric_value":
                            float(
                                example[
                                    "_metric_value"
                                ]
                            ),

                        "gold":
                            gold,

                        "pred":
                            pred,

                        "correct":
                            pred == gold,
                    }
                )


    total = len(
        examples
    )


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


    overall_accuracy = (
        correct
        / total
        if total
        else 0.0
    )


    balanced_accuracy = (
        yes_accuracy
        + no_accuracy
    ) / 2


    return {
        "n":
            total,

        "accuracy":
            overall_accuracy,

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "invalid":
            invalid,

        "rows":
            rows,
    }


# ============================================================
# Evaluate each task
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "TRAINING-SET FIT"
)

print(
    "=" * 78
)


task_results = {}


for task_id in ACTIVE_TASKS:

    task_examples = [

        example

        for example
        in train_examples

        if example[
            "_task_id"
        ] == task_id
    ]


    result = (
        evaluate_examples(
            task_examples
        )
    )


    task_results[
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
        f"  accuracy:           "
        f"{result['accuracy'] * 100:5.1f}%"
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
        f"  invalid outputs:    "
        f"{result['invalid']}"
    )


# ============================================================
# Detailed TIR analysis
# ============================================================

tir_result = (
    task_results[
        TIR_TASK
    ]
)

tir_rows = (
    tir_result[
        "rows"
    ]
)


print(
    "\n"
    + "=" * 78
)

print(
    "TIR DIAGNOSTIC"
)

print(
    "=" * 78
)


pred_yes = sum(

    row["pred"]
    == "(a)"

    for row
    in tir_rows
)


pred_no = sum(

    row["pred"]
    == "(b)"

    for row
    in tir_rows
)


print(
    "Training TIR examples:",
    len(tir_rows),
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
    "Training TIR balanced accuracy:",
    f"{tir_result['balanced_accuracy'] * 100:.1f}%"
)


# ============================================================
# TIR performance by actual proportion
# ============================================================

# metric_value for this task is a fraction:
# e.g. 0.93 = 93% of readings in range.

TIR_BINS = [
    (
        "<=75%",
        lambda x:
            x <= 0.75,
    ),

    (
        "75-90%",
        lambda x:
            0.75 < x <= 0.90,
    ),

    (
        "90-95%",
        lambda x:
            0.90 < x <= 0.95,
    ),

    (
        ">95%",
        lambda x:
            x > 0.95,
    ),
]


print(
    "\nTIR accuracy by actual proportion:"
)


for label, condition in TIR_BINS:

    subset = [

        row

        for row
        in tir_rows

        if condition(
            row[
                "metric_value"
            ]
        )
    ]


    if not subset:
        continue


    n_correct = sum(

        row[
            "correct"
        ]

        for row
        in subset
    )


    accuracy = (
        n_correct
        / len(subset)
    )


    print(
        f"  {label:8s} | "
        f"n={len(subset):3d} | "
        f"accuracy={accuracy * 100:5.1f}%"
    )


# ============================================================
# Incorrect TIR examples
# ============================================================

incorrect_tir = [

    row

    for row
    in tir_rows

    if not row[
        "correct"
    ]
]


incorrect_tir = sorted(
    incorrect_tir,
    key=lambda x:
        abs(
            x[
                "metric_value"
            ]
            - 0.90
        ),
)


print(
    "\nClosest incorrect TIR examples "
    "to the 90% boundary:"
)


for row in incorrect_tir[:15]:

    print(
        f"  participant="
        f"{row['participant_id']:18s} | "
        f"TIR="
        f"{row['metric_value'] * 100:6.2f}% | "
        f"gold={row['gold']} | "
        f"pred={row['pred']}"
    )


print(
    "\nTEST participants remain sealed."
)