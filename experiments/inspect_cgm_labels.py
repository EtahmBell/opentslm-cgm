"""
Inspect CGM alignment label distributions before training.

We inspect TRAIN and VALIDATION only.

The TEST participants remain untouched until the final experiment.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


# ============================================================
# Project imports
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from src.cgm_multidata import build_examples


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
# Configuration
# ============================================================

WINDOW_SIZE = 128
STRIDE = 128
THRESHOLD = 160


# ============================================================
# Load manifest
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


def manifest_files(split_name):
    return [
        DATA_DIR / item["relative_path"]
        for item in manifest[split_name]
    ]


train_files = manifest_files(
    "train"
)

val_files = manifest_files(
    "validation"
)


# ============================================================
# Build examples
# ============================================================

# We do not need a real EOS token because this script
# never sends examples into OpenTSLM.

train_examples = build_examples(
    train_files,
    normalization_mean=train_mean,
    normalization_std=train_std,
    eos_token="",
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    threshold=THRESHOLD,
)

val_examples = build_examples(
    val_files,
    normalization_mean=train_mean,
    normalization_std=train_std,
    eos_token="",
    window_size=WINDOW_SIZE,
    stride=STRIDE,
    threshold=THRESHOLD,
)


# ============================================================
# Summary helper
# ============================================================

def summarize(examples, label):

    yes = [
        x for x in examples
        if x["_answer_clean"] == "(a)"
    ]

    no = [
        x for x in examples
        if x["_answer_clean"] == "(b)"
    ]

    total = len(examples)

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    print(
        "Total windows:",
        total,
    )

    print(
        "YES (>160):",
        len(yes),
        f"({len(yes) / total * 100:.1f}%)"
        if total else "",
    )

    print(
        "NO (<=160):",
        len(no),
        f"({len(no) / total * 100:.1f}%)"
        if total else "",
    )

    # --------------------------------------------------------
    # Participant-level distribution
    # --------------------------------------------------------

    by_participant = defaultdict(
        lambda: {
            "yes": 0,
            "no": 0,
        }
    )

    for example in examples:

        pid = example[
            "_participant_id"
        ]

        if (
            example["_answer_clean"]
            == "(a)"
        ):
            by_participant[pid][
                "yes"
            ] += 1

        else:
            by_participant[pid][
                "no"
            ] += 1


    print(
        "\nParticipant distributions:"
    )

    for pid in sorted(
        by_participant
    ):

        counts = (
            by_participant[pid]
        )

        participant_total = (
            counts["yes"]
            + counts["no"]
        )

        print(
            f"{pid:20s} | "
            f"windows={participant_total:3d} | "
            f"YES={counts['yes']:3d} | "
            f"NO={counts['no']:3d}"
        )


# ============================================================
# Results
# ============================================================

print(
    "Fixed normalization:"
)

print(
    f"Mean = {train_mean:.3f} mg/dL"
)

print(
    f"SD   = {train_std:.3f} mg/dL"
)


summarize(
    train_examples,
    "TRAIN",
)

summarize(
    val_examples,
    "VALIDATION",
)


print("\n" + "=" * 60)
print("TEST SET")
print("=" * 60)

print(
    f"{len(manifest['test'])} participants "
    "reserved and not inspected."
)