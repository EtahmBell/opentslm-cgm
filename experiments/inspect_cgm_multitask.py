"""
Inspect class balance for every CGM alignment task.

TRAIN and VALIDATION only.
TEST remains sealed.
"""

import json
import sys

from collections import defaultdict
from pathlib import Path


# ============================================================
# Project imports
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.cgm_multitask import (
    TASK_IDS,
    build_multitask_examples,
)


# ============================================================
# Paths
# ============================================================

DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
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


# ============================================================
# Build examples
# ============================================================

print(
    "Building TRAIN examples..."
)

train_examples = (
    build_multitask_examples(
        files=train_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token="",
        window_size=128,
        stride=128,
    )
)


print(
    "Building VALIDATION examples..."
)

val_examples = (
    build_multitask_examples(
        files=val_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token="",
        window_size=128,
        stride=128,
    )
)


# ============================================================
# Summary
# ============================================================

def summarize(
    examples,
    split_name,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        split_name
    )

    print(
        "=" * 78
    )


    for task_id in TASK_IDS:

        task_examples = [

            example

            for example
            in examples

            if example[
                "_task_id"
            ] == task_id
        ]


        yes_examples = [

            x
            for x
            in task_examples

            if x[
                "_answer_clean"
            ] == "(a)"
        ]


        no_examples = [

            x
            for x
            in task_examples

            if x[
                "_answer_clean"
            ] == "(b)"
        ]


        total = len(
            task_examples
        )

        yes_n = len(
            yes_examples
        )

        no_n = len(
            no_examples
        )


        # ------------------------------------
        # Participant diversity
        # ------------------------------------

        participants_yes = set(
            x[
                "_participant_id"
            ]
            for x
            in yes_examples
        )

        participants_no = set(
            x[
                "_participant_id"
            ]
            for x
            in no_examples
        )


        usable_balanced = (
            2
            * min(
                yes_n,
                no_n,
            )
        )


        print(
            f"\n{task_id}"
        )

        print(
            "-" * len(task_id)
        )

        print(
            f"Total windows:      "
            f"{total}"
        )

        print(
            f"YES:                "
            f"{yes_n:4d} "
            f"({yes_n / total * 100:5.1f}%)"
        )

        print(
            f"NO:                 "
            f"{no_n:4d} "
            f"({no_n / total * 100:5.1f}%)"
        )

        print(
            f"Participants w/ YES:"
            f" {len(participants_yes)}"
        )

        print(
            f"Participants w/ NO: "
            f"{len(participants_no)}"
        )

        print(
            f"Balanced examples:  "
            f"{usable_balanced}"
        )


# ============================================================
# Output
# ============================================================

print(
    "\nNormalization:"
)

print(
    f"Mean = "
    f"{train_mean:.3f}"
)

print(
    f"SD   = "
    f"{train_std:.3f}"
)


summarize(
    train_examples,
    "TRAIN",
)

summarize(
    val_examples,
    "VALIDATION",
)


print(
    "\n"
    + "=" * 78
)

print(
    "TEST"
)

print(
    "=" * 78
)

print(
    f"{len(manifest['test'])} "
    "participants remain sealed."
)