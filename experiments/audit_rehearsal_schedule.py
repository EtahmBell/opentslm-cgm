"""
Audit the exact TIR rehearsal curriculum without training.

Checks:
1. class balance of TIR warm-up
2. class balance of TIR rehearsal
3. class/task balance through Stage 2
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cgm_multitask import (
    build_multitask_examples,
    balance_multitask_examples,
)

SEED = 42

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

TIR_TASK = "tir_70_180_above_90"


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
    DATA_DIR / item["relative_path"]
    for item in manifest["train"]
]


# We don't need a real EOS token for schedule auditing.
raw_examples = build_multitask_examples(
    files=train_files,
    normalization_mean=train_mean,
    normalization_std=train_std,
    eos_token="",
    window_size=128,
    stride=128,
    task_ids=ACTIVE_TASKS,
)


train_examples = balance_multitask_examples(
    examples=raw_examples,
    task_ids=ACTIVE_TASKS,
    per_class=100,
    seed=SEED,
)


tir_examples = [
    x for x in train_examples
    if x["_task_id"] == TIR_TASK
]

other_examples = [
    x for x in train_examples
    if x["_task_id"] != TIR_TASK
]


tir_warmup = tir_examples[:100]
tir_rehearsal = tir_examples[100:]


def print_class_counts(label, examples):

    counts = Counter(
        x["_answer_clean"]
        for x in examples
    )

    print(f"\n{label}")
    print("-" * len(label))

    print("Total:", len(examples))
    print("YES:", counts["(a)"])
    print("NO: ", counts["(b)"])


print_class_counts(
    "TIR WARM-UP",
    tir_warmup,
)

print_class_counts(
    "TIR REHEARSAL",
    tir_rehearsal,
)


# Rebuild same Stage-2 schedule.
others_per_rehearsal = (
    len(other_examples)
    // len(tir_rehearsal)
)

stage2 = []

other_index = 0

for tir_example in tir_rehearsal:

    next_index = (
        other_index
        + others_per_rehearsal
    )

    stage2.extend(
        other_examples[
            other_index:
            next_index
        ]
    )

    stage2.append(
        tir_example
    )

    other_index = next_index


print_class_counts(
    "FULL STAGE 2",
    stage2,
)


print(
    "\nStage-2 task counts:"
)

task_counts = Counter(
    x["_task_id"]
    for x in stage2
)

for task_id in ACTIVE_TASKS:

    print(
        f"{task_id:28s}",
        task_counts[task_id],
    )


print(
    "\nStage-2 chunks:"
)

for start in range(
    0,
    len(stage2),
    100,
):

    chunk = stage2[
        start:
        start + 100
    ]

    class_counts = Counter(
        x["_answer_clean"]
        for x in chunk
    )

    task_counts = Counter(
        x["_task_id"]
        for x in chunk
    )

    print(
        f"\nSteps "
        f"{start + 1:4d}-"
        f"{start + len(chunk):4d}"
    )

    print(
        "  YES:",
        class_counts["(a)"],
        "| NO:",
        class_counts["(b)"],
    )

    print(
        "  TIR:",
        task_counts[TIR_TASK],
    )