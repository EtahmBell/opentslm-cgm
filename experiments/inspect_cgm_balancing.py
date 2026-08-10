"""
Verify the participant-aware multitask training sampler.
"""

import json
import sys

from collections import Counter, defaultdict
from pathlib import Path


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


# ============================================================
# Configuration
# ============================================================

ACTIVE_TASKS = [
    "above_160",
    "above_180",
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
    "tir_70_180_above_90",
]

PER_CLASS = 100

SEED = 42


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
# Manifest
# ============================================================

with MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    manifest = json.load(f)


train_mean = (
    manifest["normalization"]["mean"]
)

train_std = (
    manifest["normalization"]["std"]
)


train_files = [

    DATA_DIR
    / item["relative_path"]

    for item
    in manifest["train"]
]


# ============================================================
# Build raw pool
# ============================================================

examples = (
    build_multitask_examples(
        files=train_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token="",
        window_size=128,
        stride=128,
        task_ids=ACTIVE_TASKS,
    )
)


print(
    "Raw multitask examples:",
    len(examples),
)


# ============================================================
# Balance
# ============================================================

balanced = (
    balance_multitask_examples(
        examples=examples,
        task_ids=ACTIVE_TASKS,
        per_class=PER_CLASS,
        seed=SEED,
    )
)


print(
    "\nBalanced training examples:",
    len(balanced),
)


# ============================================================
# Verify counts
# ============================================================

print(
    "\n"
    + "=" * 72
)

print(
    "BALANCED DATASET"
)

print(
    "=" * 72
)


for task_id in ACTIVE_TASKS:

    task_examples = [

        x
        for x in balanced

        if x["_task_id"]
        == task_id
    ]


    yes = sum(

        x["_answer_clean"]
        == "(a)"

        for x
        in task_examples
    )


    no = sum(

        x["_answer_clean"]
        == "(b)"

        for x
        in task_examples
    )


    participants = set(

        x["_participant_id"]

        for x
        in task_examples
    )


    yes_participants = set(

        x["_participant_id"]

        for x
        in task_examples

        if x["_answer_clean"]
        == "(a)"
    )


    no_participants = set(

        x["_participant_id"]

        for x
        in task_examples

        if x["_answer_clean"]
        == "(b)"
    )


    print(
        f"\n{task_id}"
    )

    print(
        f"  total:            "
        f"{len(task_examples)}"
    )

    print(
        f"  YES:              "
        f"{yes}"
    )

    print(
        f"  NO:               "
        f"{no}"
    )

    print(
        f"  participants:     "
        f"{len(participants)}"
    )

    print(
        f"  YES participants: "
        f"{len(yes_participants)}"
    )

    print(
        f"  NO participants:  "
        f"{len(no_participants)}"
    )


print(
    "\nTEST participants remain untouched."
)