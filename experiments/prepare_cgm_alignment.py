"""
Prepare the multi-participant CGM experiment.

This script:
1. discovers local AI-READI CGM files
2. randomly selects participants
3. splits BY PARTICIPANT
4. calculates normalization from TRAIN participants only
5. saves the experiment manifest locally

No raw CGM data is copied.
"""

import json
import random
import sys
from pathlib import Path


# ============================================================
# Make project src/ importable
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.cgm_multidata import (
    discover_cgm_files,
    split_participant_files,
    compute_training_normalization,
    load_cgm_file,
)


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

ARTIFACT_DIR = (
    ROOT / "artifacts"
)

ARTIFACT_DIR.mkdir(
    exist_ok=True
)

OUTPUT_PATH = (
    ARTIFACT_DIR
    / "cgm_alignment_manifest.json"
)

SEED = 42

# Start small.
#
# Change to 40 once you have >=40 downloaded.
MAX_PARTICIPANTS = 40


# ============================================================
# Discover CGM files
# ============================================================

files = discover_cgm_files(
    DATA_DIR
)

print(
    "CGM participant files found:",
    len(files),
)

if len(files) < 10:
    print(
        "\nWARNING:"
        "\nYou currently have fewer than 10 participants."
        "\nDownload more CGM JSON files before real training."
    )


# ============================================================
# Randomly select participant subset
# ============================================================

rng = random.Random(
    SEED
)

rng.shuffle(
    files
)

if MAX_PARTICIPANTS is not None:
    files = files[
        :MAX_PARTICIPANTS
    ]


print(
    "Participants selected:",
    len(files),
)


# ============================================================
# Participant split
# ============================================================

(
    train_files,
    val_files,
    test_files,
) = split_participant_files(
    files,
    train_fraction=0.70,
    val_fraction=0.15,
    seed=SEED,
)


print("\nParticipant split")

print(
    "Train:",
    len(train_files),
)

print(
    "Validation:",
    len(val_files),
)

print(
    "Test:",
    len(test_files),
)


# ============================================================
# Train-only normalization
# ============================================================

train_mean, train_std = (
    compute_training_normalization(
        train_files
    )
)


print(
    "\nTRAIN normalization"
)

print(
    "Mean:",
    round(
        train_mean,
        4,
    ),
)

print(
    "SD:",
    round(
        train_std,
        4,
    ),
)


# ============================================================
# Get participant IDs
# ============================================================

def describe_files(paths):

    output = []

    for path in paths:

        participant_id, glucose = (
            load_cgm_file(
                path
            )
        )

        output.append(
            {
                "participant_id":
                    participant_id,

                "filename":
                    path.name,

                "n_readings":
                    len(glucose),
            }
        )

    return output


manifest = {
    "seed":
        SEED,

    "window_size":
        128,

    "stride":
        128,

    "threshold_mg_dl":
        160,

    "normalization": {
        "mean":
            train_mean,

        "std":
            train_std,

        "computed_from":
            "training participants only",
    },

    "train":
        describe_files(
            train_files
        ),

    "validation":
        describe_files(
            val_files
        ),

    "test":
        describe_files(
            test_files
        ),
}


# ============================================================
# Save local experiment manifest
# ============================================================

with OUTPUT_PATH.open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


print(
    "\nSaved manifest:"
)

print(
    OUTPUT_PATH
)