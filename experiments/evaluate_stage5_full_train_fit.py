"""
Evaluate Stage-5 full-adaptation train fit.

Purpose
-------
The Stage-5 full-adaptation model reached low training loss (~0.30)
but produced an all-YES classifier on unseen validation participants.

This script asks:

    Did the model actually learn the TRAINING examples?

We evaluate the saved checkpoint using model.generate() on:

    1. Exact same 1200 balanced training examples
    2. Same balanced validation examples

We report per task:
    - YES accuracy
    - NO accuracy
    - balanced accuracy
    - predicted YES / NO
    - invalid outputs

We also print representative raw generations for:
    - TRAIN YES
    - TRAIN NO
    - VALIDATION YES
    - VALIDATION NO

TEST participants remain sealed.
"""

import json
import random
import re
import sys

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

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_alignment_stage5_full_adaptation_seed42.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_stage5_full_train_fit_results.json"
)

DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
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

EVAL_BATCH_SIZE = 4

MAX_NEW_TOKENS = 6

RAW_EXAMPLES_PER_CLASS = 2


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
# Load original Stage-5 model WITH LoRA
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "LOADING ORIGINAL STAGE-5 MODEL"
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


# ============================================================
# Load our trained CGM checkpoint
# ============================================================

print(
    "\nLoading Stage-5 full-adaptation CGM checkpoint:"
)

print(
    CHECKPOINT_PATH
)


model.load_from_file(
    str(
        CHECKPOINT_PATH
    )
)


model.eval()


print(
    "\n✓ Full-adaptation checkpoint loaded."
)


EOS_TOKEN = (
    model.get_eos_token()
)


# ============================================================
# Build EXACT SAME training examples
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "RECONSTRUCTING EXACT TRAINING SET"
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
    "\nExact balanced training set:",
    len(train_examples),
)


# ============================================================
# Group training examples by task
# ============================================================

train_by_task = {}


for task_id in ACTIVE_TASKS:

    task_examples = [
        x
        for x in train_examples
        if x[
            "_task_id"
        ] == task_id
    ]


    yes_count = sum(
        x[
            "_answer_clean"
        ] == "(a)"
        for x in task_examples
    )

    no_count = sum(
        x[
            "_answer_clean"
        ] == "(b)"
        for x in task_examples
    )


    train_by_task[
        task_id
    ] = task_examples


    print(
        f"{task_id:28s} | "
        f"YES={yes_count:3d} | "
        f"NO={no_count:3d} | "
        f"total={len(task_examples):3d}"
    )


# ============================================================
# Build SAME balanced validation set
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "RECONSTRUCTING VALIDATION SET"
)

print(
    "=" * 78
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
# Helpers
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
# Full generation evaluation
# ============================================================

def evaluate_examples(
    examples,
):

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    predicted_yes = 0
    predicted_no = 0

    invalid = 0


    raw_records = []


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

                    predicted_yes += 1

                elif pred == "(b)":

                    predicted_no += 1

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


                raw_records.append(
                    {
                        "task_id":
                            example[
                                "_task_id"
                            ],

                        "participant_id":
                            example.get(
                                "_participant_id"
                            ),

                        "window_start":
                            example.get(
                                "_window_start"
                            ),

                        "gold":
                            gold,

                        "parsed_prediction":
                            pred,

                        "raw_prediction":
                            str(
                                raw_pred
                            ),
                    }
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


    balanced_accuracy = (
        yes_accuracy
        + no_accuracy
    ) / 2.0


    accuracy = (
        yes_correct
        + no_correct
    ) / len(
        examples
    )


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

        "accuracy":
            accuracy,

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

        "invalid":
            invalid,

        "valid_output_rate":
            valid_output_rate,

        "raw_records":
            raw_records,
    }


# ============================================================
# Print one task
# ============================================================

def print_result(
    task_id,
    result,
):

    print(
        f"\n{task_id}"
    )

    print(
        f"n:                 "
        f"{result['n']}"
    )

    print(
        f"accuracy:           "
        f"{result['accuracy'] * 100:5.1f}%"
    )

    print(
        f"YES accuracy:       "
        f"{result['yes_accuracy'] * 100:5.1f}%"
    )

    print(
        f"NO accuracy:        "
        f"{result['no_accuracy'] * 100:5.1f}%"
    )

    print(
        f"balanced accuracy:  "
        f"{result['balanced_accuracy'] * 100:5.1f}%"
    )

    print(
        f"predicted YES:      "
        f"{result['predicted_yes']}"
    )

    print(
        f"predicted NO:       "
        f"{result['predicted_no']}"
    )

    print(
        f"invalid outputs:    "
        f"{result['invalid']}"
    )

    print(
        f"valid output rate:  "
        f"{result['valid_output_rate'] * 100:5.1f}%"
    )


# ============================================================
# Evaluate training set
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "EXACT TRAINING-SET GENERATION PERFORMANCE"
)

print(
    "=" * 78
)


train_results = {}


for task_id in ACTIVE_TASKS:

    result = (
        evaluate_examples(
            train_by_task[
                task_id
            ]
        )
    )


    train_results[
        task_id
    ] = result


    print_result(
        task_id,
        result,
    )


train_macro = (
    sum(
        x[
            "balanced_accuracy"
        ]
        for x in train_results.values()
    )
    / len(
        train_results
    )
)


print(
    "\nTRAIN MACRO BALANCED ACCURACY:",
    f"{train_macro * 100:.1f}%"
)


# ============================================================
# Evaluate validation
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "VALIDATION GENERATION PERFORMANCE"
)

print(
    "=" * 78
)


val_results = {}


for task_id in ACTIVE_TASKS:

    result = (
        evaluate_examples(
            val_by_task[
                task_id
            ]
        )
    )


    val_results[
        task_id
    ] = result


    print_result(
        task_id,
        result,
    )


val_macro = (
    sum(
        x[
            "balanced_accuracy"
        ]
        for x in val_results.values()
    )
    / len(
        val_results
    )
)


print(
    "\nVALIDATION MACRO BALANCED ACCURACY:",
    f"{val_macro * 100:.1f}%"
)


# ============================================================
# Train-vs-validation comparison
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "TRAIN vs VALIDATION"
)

print(
    "=" * 78
)


print(
    f"{'Task':28s} "
    f"{'Train BA':>10s} "
    f"{'Val BA':>10s} "
    f"{'Gap':>10s}"
)

print(
    "-" * 62
)


for task_id in ACTIVE_TASKS:

    train_ba = (
        train_results[
            task_id
        ][
            "balanced_accuracy"
        ]
    )

    val_ba = (
        val_results[
            task_id
        ][
            "balanced_accuracy"
        ]
    )


    print(
        f"{task_id:28s} "
        f"{train_ba * 100:9.1f}% "
        f"{val_ba * 100:9.1f}% "
        f"{(train_ba - val_ba) * 100:+9.1f}"
    )


print(
    "-" * 62
)

print(
    f"{'MACRO':28s} "
    f"{train_macro * 100:9.1f}% "
    f"{val_macro * 100:9.1f}% "
    f"{(train_macro - val_macro) * 100:+9.1f}"
)


# ============================================================
# Representative RAW outputs
# ============================================================

def show_raw_examples(
    split_name,
    task_id,
    result,
):

    records = (
        result[
            "raw_records"
        ]
    )


    yes_records = [
        x
        for x in records
        if x[
            "gold"
        ] == "(a)"
    ]


    no_records = [
        x
        for x in records
        if x[
            "gold"
        ] == "(b)"
    ]


    print(
        "\n"
        + "-" * 78
    )

    print(
        f"{split_name} — {task_id}"
    )

    print(
        "-" * 78
    )


    print(
        "\nGold YES examples:"
    )


    for record in (
        yes_records[
            :RAW_EXAMPLES_PER_CLASS
        ]
    ):

        print(
            f"  participant="
            f"{record['participant_id']} | "
            f"window="
            f"{record['window_start']} | "
            f"gold="
            f"{record['gold']} | "
            f"parsed="
            f"{record['parsed_prediction']} | "
            f"raw="
            f"{record['raw_prediction']!r}"
        )


    print(
        "\nGold NO examples:"
    )


    for record in (
        no_records[
            :RAW_EXAMPLES_PER_CLASS
        ]
    ):

        print(
            f"  participant="
            f"{record['participant_id']} | "
            f"window="
            f"{record['window_start']} | "
            f"gold="
            f"{record['gold']} | "
            f"parsed="
            f"{record['parsed_prediction']} | "
            f"raw="
            f"{record['raw_prediction']!r}"
        )


print(
    "\n"
    + "=" * 78
)

print(
    "REPRESENTATIVE RAW GENERATIONS"
)

print(
    "=" * 78
)


for task_id in ACTIVE_TASKS:

    show_raw_examples(
        "TRAIN",
        task_id,
        train_results[
            task_id
        ],
    )

    show_raw_examples(
        "VALIDATION",
        task_id,
        val_results[
            task_id
        ],
    )


# ============================================================
# Strip raw records before compact JSON summary
# ============================================================

def compact_result(
    result,
):

    return {
        key: value
        for key, value in result.items()
        if key != "raw_records"
    }


compact_train = {
    task_id:
        compact_result(
            result
        )

    for task_id, result
    in train_results.items()
}


compact_val = {
    task_id:
        compact_result(
            result
        )

    for task_id, result
    in val_results.items()
}


# ============================================================
# Save results
# ============================================================

results = {
    "experiment":
        "stage5_full_train_fit",

    "seed":
        SEED,

    "repo_id":
        STAGE5_REPO_ID,

    "checkpoint":
        str(
            CHECKPOINT_PATH
        ),

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

    "train_macro_balanced_accuracy":
        train_macro,

    "validation_macro_balanced_accuracy":
        val_macro,

    "train":
        compact_train,

    "validation":
        compact_val,

    "raw_training_predictions": {
        task_id:
            train_results[
                task_id
            ][
                "raw_records"
            ]

        for task_id
        in ACTIVE_TASKS
    },

    "raw_validation_predictions": {
        task_id:
            val_results[
                task_id
            ][
                "raw_records"
            ]

        for task_id
        in ACTIVE_TASKS
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
    "\nSaved results:"
)

print(
    RESULTS_PATH
)


print(
    "\nTEST participants remain sealed."
)