"""
Audit CGM timestamp spacing across TRAIN and VALIDATION participants.

Purpose
-------
Our temporal tasks currently assume Dexcom readings occur every ~5 minutes.

This script checks whether that assumption is actually true across the
participants we are using.

It does NOT inspect CGM labels or the held-out TEST set.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

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

EXPECTED_INTERVAL_MINUTES = 5.0

# Same threshold we previously used for participant 1001:
# anything >7.5 minutes is treated as a meaningful gap.
GAP_THRESHOLD_MINUTES = 7.5

# Small timing deviations around five minutes are fine.
NORMAL_MIN_MINUTES = 4.5
NORMAL_MAX_MINUTES = 5.5


# ============================================================
# Timestamp extraction
# ============================================================

TIMESTAMP_KEYS = [
    "system_time",
    "systemTime",
    "display_time",
    "displayTime",
    "timestamp",
    "date_time",
    "datetime",
    "time",
]


def extract_cgm_records(file_path):
    """
    Load raw CGM records from an AI-READI Dexcom JSON file.
    """

    with Path(file_path).open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)

    if (
        "body" not in payload
        or "cgm" not in payload["body"]
    ):
        raise ValueError(
            f"No body['cgm'] found in {file_path}"
        )

    return payload["body"]["cgm"]


def find_timestamp_key(records):
    """
    Find the timestamp field used in the CGM records.
    """

    if not records:
        raise ValueError(
            "CGM record list is empty."
        )

    first = records[0]

    for key in TIMESTAMP_KEYS:

        if key in first:
            return key

    raise ValueError(
        "Could not identify timestamp field.\n"
        f"Available fields: {list(first.keys())}"
    )


def parse_timestamps(records):
    """
    Extract timestamps from the AI-READI CGM schema.

    Structure:
        effective_time_frame
            -> time_interval
                -> start_date_time

    CGM entries represent individual readings, so the
    start and end timestamps should be identical.
    """

    raw_values = []

    mismatched_start_end = 0


    for row in records:

        try:

            interval = (
                row[
                    "effective_time_frame"
                ][
                    "time_interval"
                ]
            )

            start_time = (
                interval[
                    "start_date_time"
                ]
            )

            end_time = (
                interval.get(
                    "end_date_time"
                )
            )


        except (
            KeyError,
            TypeError,
        ) as exc:

            raise ValueError(
                "Could not extract timestamp from "
                "effective_time_frame -> "
                "time_interval -> start_date_time.\n"
                f"Problem record: {row}"
            ) from exc


        raw_values.append(
            start_time
        )


        if (
            end_time is not None
            and end_time != start_time
        ):

            mismatched_start_end += 1


    timestamps = pd.to_datetime(
        raw_values,
        utc=True,
        errors="coerce",
    )


    invalid = int(
        timestamps.isna().sum()
    )


    if invalid:

        raise ValueError(
            f"{invalid} timestamps could not be parsed."
        )


    return (
        "effective_time_frame.time_interval.start_date_time",
        timestamps,
        mismatched_start_end,
    )

# ============================================================
# Manifest
# ============================================================

with MANIFEST_PATH.open(
    "r",
    encoding="utf-8",
) as f:

    manifest = json.load(f)


def manifest_files(split_name):

    return [

        (
            item["participant_id"],
            DATA_DIR / item["relative_path"],
        )

        for item
        in manifest[split_name]
    ]


# IMPORTANT:
# Do not inspect the six test participants yet.
participants = (
    manifest_files("train")
    + manifest_files("validation")
)


# ============================================================
# Audit
# ============================================================

print(
    "Participants audited:",
    len(participants),
)

print(
    "Splits: TRAIN + VALIDATION only"
)

print(
    f"Expected interval: "
    f"{EXPECTED_INTERVAL_MINUTES:.1f} min"
)

print(
    f"Meaningful gap threshold: "
    f">{GAP_THRESHOLD_MINUTES:.1f} min"
)


all_deltas = []

participant_results = []


for participant_id, file_path in participants:

    records = extract_cgm_records(
        file_path
    )

    (
        timestamp_key,
        timestamps,
        mismatched_start_end,
    ) = parse_timestamps(
        records
    )


    timestamp_series = pd.Series(
        timestamps
    )

    deltas = (
        timestamp_series
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )


    all_deltas.extend(
        deltas.tolist()
    )


    meaningful_gaps = (
        deltas
        > GAP_THRESHOLD_MINUTES
    )

    abnormal_intervals = (
        (deltas < NORMAL_MIN_MINUTES)
        | (deltas > NORMAL_MAX_MINUTES)
    )

    duplicates = (
        deltas == 0
    )

    backwards = (
        deltas < 0
    )


    result = {
        "participant_id":
            participant_id,

        "n_readings":
            len(records),

        "timestamp_key":
            timestamp_key,

        "median_delta":
            float(deltas.median()),

        "max_delta":
            float(deltas.max()),

        "meaningful_gaps":
            int(meaningful_gaps.sum()),

        "abnormal_intervals":
            int(abnormal_intervals.sum()),

        "duplicates":
            int(duplicates.sum()),

        "backwards":
            int(backwards.sum()),

        "mismatched_start_end":
            mismatched_start_end,
    }

    participant_results.append(
        result
    )


    print(
        f"{participant_id:18s} | "
        f"n={len(records):4d} | "
        f"median={deltas.median():5.2f}m | "
        f"max={deltas.max():7.2f}m | "
        f"gaps>{GAP_THRESHOLD_MINUTES:g}m="
        f"{meaningful_gaps.sum():3d} | "
        f"start/end mismatch="
        f"{mismatched_start_end}"
    )


# ============================================================
# Overall summary
# ============================================================

all_deltas = pd.Series(
    all_deltas,
    dtype=float,
)


print(
    "\n"
    + "=" * 72
)

print(
    "OVERALL TIMESTAMP AUDIT"
)

print(
    "=" * 72
)


print(
    "Total intervals:",
    len(all_deltas),
)

print(
    f"Median interval: "
    f"{all_deltas.median():.3f} min"
)

print(
    f"Mean interval:   "
    f"{all_deltas.mean():.3f} min"
)

print(
    f"95th percentile: "
    f"{all_deltas.quantile(0.95):.3f} min"
)

print(
    f"99th percentile: "
    f"{all_deltas.quantile(0.99):.3f} min"
)

print(
    f"Maximum interval:"
    f" {all_deltas.max():.3f} min"
)


normal_count = int(
    (
        (all_deltas >= NORMAL_MIN_MINUTES)
        & (all_deltas <= NORMAL_MAX_MINUTES)
    ).sum()
)

meaningful_gap_count = int(
    (
        all_deltas
        > GAP_THRESHOLD_MINUTES
    ).sum()
)


print(
    "\nNormal 4.5–5.5 min intervals:",
    normal_count,
    (
        f"({normal_count / len(all_deltas) * 100:.2f}%)"
    ),
)

print(
    f"Intervals >"
    f"{GAP_THRESHOLD_MINUTES:g} min:",
    meaningful_gap_count,
    (
        f"({meaningful_gap_count / len(all_deltas) * 100:.3f}%)"
    ),
)


participants_with_gaps = [

    x
    for x
    in participant_results

    if x["meaningful_gaps"] > 0
]

total_duplicates = sum(
    x["duplicates"]
    for x in participant_results
)

total_backwards = sum(
    x["backwards"]
    for x in participant_results
)

total_start_end_mismatches = sum(
    x["mismatched_start_end"]
    for x in participant_results
)


print(
    "\nDuplicate timestamps:",
    total_duplicates,
)

print(
    "Backward timestamp jumps:",
    total_backwards,
)

print(
    "Start/end mismatches:",
    total_start_end_mismatches,
)


print(
    "\nParticipants with meaningful gaps:",
    len(participants_with_gaps),
    "/",
    len(participant_results),
)


if participants_with_gaps:

    print(
        "\nParticipants requiring attention:"
    )

    for result in participants_with_gaps:

        print(
            f"  {result['participant_id']} | "
            f"gaps={result['meaningful_gaps']} | "
            f"max={result['max_delta']:.1f} min"
        )


print(
    "\nTEST participants were not inspected."
)