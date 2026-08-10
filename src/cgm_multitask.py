"""
Multi-task CGM alignment examples.

All labels are calculated deterministically from the RAW glucose values.

The model receives the normalized signal.
The label generator always uses the original mg/dL values.

Current tasks
-------------
1. glucose > 140 mg/dL
2. glucose > 160 mg/dL
3. glucose > 180 mg/dL
4. glucose < 70 mg/dL
5. sustained glucose > 180 mg/dL for >= 15 min
6. rise > 50 mg/dL within 30 min
7. fall > 50 mg/dL within 30 min
8. > 90% time in range 70-180 mg/dL
"""

from pathlib import Path

import torch

from src.cgm_multidata import load_cgm_file

import json
import pandas as pd


# ============================================================
# Configuration
# ============================================================

DEFAULT_WINDOW_SIZE = 128
DEFAULT_STRIDE = 128

# Dexcom G6 readings are nominally every 5 minutes.
READING_INTERVAL_MINUTES = 5

# 30 minutes / 5 minutes = 6 index steps.
STEPS_30_MINUTES = 6

# We use the same convention as before:
# 3 consecutive 5-minute readings = a >=15-minute event.
READINGS_15_MINUTES = 3


# ============================================================
# Task definitions
# ============================================================

TASK_IDS = [
    "above_140",
    "above_160",
    "above_180",
    "below_70",
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
    "tir_70_180_above_90",
]


QUESTIONS = {

    "above_140":
        "Did glucose exceed 140 mg/dL during this period?",

    "above_160":
        "Did glucose exceed 160 mg/dL during this period?",

    "above_180":
        "Did glucose exceed 180 mg/dL during this period?",

    "below_70":
        "Did glucose fall below 70 mg/dL during this period?",

    "sustained_above_180_15m":
        (
            "Was glucose above 180 mg/dL for at least "
            "15 continuous minutes?"
        ),

    "rise_50_30m":
        (
            "Did glucose rise by more than 50 mg/dL "
            "within any 30-minute period?"
        ),

    "fall_50_30m":
        (
            "Did glucose fall by more than 50 mg/dL "
            "within any 30-minute period?"
        ),

    "tir_70_180_above_90":
        (
            "Was more than 90% of the glucose data "
            "between 70 and 180 mg/dL?"
        ),
}


# ============================================================
# Utility functions
# ============================================================

def longest_true_run(mask):
    """
    Return the longest consecutive run of True values.
    """

    longest = 0
    current = 0

    for value in mask:

        if bool(value):
            current += 1
            longest = max(
                longest,
                current,
            )

        else:
            current = 0

    return longest


def max_30m_rise(window):
    """
    Largest increase over exactly 6 readings = 30 minutes.
    """

    if len(window) <= STEPS_30_MINUTES:
        return 0.0

    changes = (
        window[STEPS_30_MINUTES:]
        - window[:-STEPS_30_MINUTES]
    )

    return float(
        torch.max(changes)
    )


def max_30m_fall(window):
    """
    Largest decrease over exactly 6 readings = 30 minutes.

    Returned as a positive magnitude.
    """

    if len(window) <= STEPS_30_MINUTES:
        return 0.0

    changes = (
        window[:-STEPS_30_MINUTES]
        - window[STEPS_30_MINUTES:]
    )

    return float(
        torch.max(changes)
    )

TEMPORAL_TASKS = {
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
}


# ============================================================
# Timestamp utilities
# ============================================================

GAP_THRESHOLD_MINUTES = 7.5


def load_cgm_timestamps(file_path):
    """
    Load timestamps in the same record order as the CGM values.

    Returns a pandas DatetimeIndex.
    """

    with Path(file_path).open(
        "r",
        encoding="utf-8",
    ) as f:
        payload = json.load(f)


    records = payload[
        "body"
    ][
        "cgm"
    ]


    raw_timestamps = [

        row[
            "effective_time_frame"
        ][
            "time_interval"
        ][
            "start_date_time"
        ]

        for row in records
    ]


    timestamps = pd.to_datetime(
        raw_timestamps,
        utc=True,
        errors="raise",
    )


    return timestamps


def window_has_gap(
    timestamps,
    start,
    window_size,
    gap_threshold_minutes=GAP_THRESHOLD_MINUTES,
):
    """
    Return True if a window contains any consecutive
    timestamp interval greater than the allowed threshold.
    """

    window_timestamps = (
        timestamps[
            start:
            start + window_size
        ]
    )


    if len(window_timestamps) < 2:
        return False


    timestamp_series = pd.Series(
        window_timestamps
    )


    deltas = (
        timestamp_series
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60.0
    )


    return bool(
        (
            deltas
            > gap_threshold_minutes
        ).any()
    )

# ============================================================
# Ground-truth task calculations
# ============================================================

def evaluate_task(
    raw_window,
    task_id,
):
    """
    Calculate the deterministic label for one task.

    Returns
    -------
    label : bool
    metric_name : str
    metric_value : float
    """

    raw_window = raw_window.float()

    minimum = float(
        torch.min(raw_window)
    )

    maximum = float(
        torch.max(raw_window)
    )


    # --------------------------------------------------------
    # Absolute thresholds
    # --------------------------------------------------------

    if task_id == "above_140":

        return (
            maximum > 140.0,
            "max_glucose",
            maximum,
        )


    if task_id == "above_160":

        return (
            maximum > 160.0,
            "max_glucose",
            maximum,
        )


    if task_id == "above_180":

        return (
            maximum > 180.0,
            "max_glucose",
            maximum,
        )


    if task_id == "below_70":

        return (
            minimum < 70.0,
            "min_glucose",
            minimum,
        )


    # --------------------------------------------------------
    # Sustained hyperglycemia
    # --------------------------------------------------------

    if (
        task_id
        == "sustained_above_180_15m"
    ):

        mask = (
            raw_window > 180.0
        )

        longest_run = (
            longest_true_run(mask)
        )

        return (
            longest_run
            >= READINGS_15_MINUTES,

            "longest_run_above_180_readings",

            float(longest_run),
        )


    # --------------------------------------------------------
    # 30-minute change
    # --------------------------------------------------------

    if task_id == "rise_50_30m":

        rise = max_30m_rise(
            raw_window
        )

        return (
            rise > 50.0,
            "max_30m_rise",
            rise,
        )


    if task_id == "fall_50_30m":

        fall = max_30m_fall(
            raw_window
        )

        return (
            fall > 50.0,
            "max_30m_fall",
            fall,
        )


    # --------------------------------------------------------
    # Time in range
    # --------------------------------------------------------

    if (
        task_id
        == "tir_70_180_above_90"
    ):

        in_range = (
            (raw_window >= 70.0)
            & (raw_window <= 180.0)
        )

        tir = float(
            in_range.float().mean()
        )

        return (
            tir > 0.90,
            "tir_70_180",
            tir,
        )


    raise ValueError(
        f"Unknown task: {task_id}"
    )


# ============================================================
# OpenTSLM example construction
# ============================================================

def make_multitask_example(
    raw_window,
    participant_id,
    window_start,
    task_id,
    normalization_mean,
    normalization_std,
    eos_token="",
):
    """
    Create one OpenTSLM-compatible binary CGM example.
    """

    label, metric_name, metric_value = (
        evaluate_task(
            raw_window,
            task_id,
        )
    )


    # IMPORTANT:
    # Use fixed TRAINING-SET normalization.
    #
    # Do NOT normalize each window independently.
    normalized_window = (
        raw_window.float()
        - normalization_mean
    ) / (
        normalization_std
        + 1e-8
    )


    answer_clean = (
        "(a)"
        if label
        else "(b)"
    )


    question = QUESTIONS[
        task_id
    ]


    example = {

        # ------------------------------------
        # OpenTSLM fields
        # ------------------------------------

        "pre_prompt":
            (
                "Analyze the continuous glucose monitor "
                "time series and answer the question.\n"
            ),

        "time_series": [
            normalized_window
        ],

        "time_series_text": [
            (
                "Continuous glucose monitor measurements "
                "sampled approximately every 5 minutes."
            )
        ],

        "post_prompt":
            (
                f"\nQuestion: {question}\n"
                "(a) Yes\n"
                "(b) No\n"
                "Answer:"
            ),

        "answer":
            answer_clean
            + eos_token,


        # ------------------------------------
        # Metadata — NOT given to model
        # ------------------------------------

        "_answer_clean":
            answer_clean,

        "_label":
            bool(label),

        "_task_id":
            task_id,

        "_question":
            question,

        "_participant_id":
            participant_id,

        "_window_start":
            int(window_start),

        "_window_end":
            int(
                window_start
                + len(raw_window)
                - 1
            ),

        "_window_min":
            float(
                torch.min(raw_window)
            ),

        "_window_max":
            float(
                torch.max(raw_window)
            ),

        "_metric_name":
            metric_name,

        "_metric_value":
            metric_value,
    }

    return example


# ============================================================
# Build all examples
# ============================================================

def build_multitask_examples(
    files,
    normalization_mean,
    normalization_std,
    eos_token="",
    window_size=DEFAULT_WINDOW_SIZE,
    stride=DEFAULT_STRIDE,
    task_ids=None,
):
    """
    Generate all task/window examples for a collection
    of participant files.
    """

    if task_ids is None:
        task_ids = TASK_IDS

    examples = []

    for file_path in files:

        participant_id, glucose = (
            load_cgm_file(
                Path(file_path)
            )
        )

        timestamps = (
            load_cgm_timestamps(
                Path(file_path)
            )
        )

        if len(timestamps) != len(glucose):

            raise ValueError(
                f"Timestamp/glucose length mismatch "
                f"for {participant_id}: "
                f"{len(timestamps)} timestamps vs "
                f"{len(glucose)} glucose values"
            )

        # ------------------------------------
        # Non-overlapping windows by default
        # ------------------------------------

        for start in range(
            0,
            len(glucose)
            - window_size
            + 1,
            stride,
        ):

            raw_window = (
                glucose[
                    start:
                    start + window_size
                ]
            )

            has_gap = window_has_gap(
                timestamps=timestamps,
                start=start,
                window_size=window_size,
            )

            for task_id in task_ids:

                # Temporal labels assume regular ~5-minute sampling.
                # Do not create temporal examples for windows
                # containing meaningful missing-data gaps.
                if (
                    task_id in TEMPORAL_TASKS
                    and has_gap
                ):
                    continue

                example = (
                    make_multitask_example(
                        raw_window=raw_window,
                        participant_id=participant_id,
                        window_start=start,
                        task_id=task_id,
                        normalization_mean=normalization_mean,
                        normalization_std=normalization_std,
                        eos_token=eos_token,
                    )
                )

                example[
                    "_contains_timestamp_gap"
                ] = has_gap

                examples.append(
                    example
                )

    return examples

# ============================================================
# Participant-aware class balancing
# ============================================================

import random
from collections import defaultdict


def sample_across_participants(
    examples,
    n,
    seed=42,
):
    """
    Select up to n examples while spreading examples across
    participants as evenly as possible.

    This prevents one participant from dominating a class.
    """

    rng = random.Random(
        seed
    )


    by_participant = defaultdict(
        list
    )


    for example in examples:

        participant_id = example[
            "_participant_id"
        ]

        by_participant[
            participant_id
        ].append(
            example
        )


    # Shuffle windows within each participant.
    for participant_examples in (
        by_participant.values()
    ):

        rng.shuffle(
            participant_examples
        )


    participant_ids = list(
        by_participant.keys()
    )

    rng.shuffle(
        participant_ids
    )


    selected = []


    # Round-robin:
    # one example per participant per pass.
    while (
        len(selected) < n
        and participant_ids
    ):

        remaining_participants = []


        for participant_id in participant_ids:

            participant_examples = (
                by_participant[
                    participant_id
                ]
            )


            if participant_examples:

                selected.append(
                    participant_examples.pop()
                )


                if len(selected) >= n:
                    break


            if participant_examples:

                remaining_participants.append(
                    participant_id
                )


        participant_ids = (
            remaining_participants
        )


    return selected


def balance_multitask_examples(
    examples,
    task_ids,
    per_class=100,
    seed=42,
):
    """
    Create a balanced multi-task training set.

    For every task:
        per_class YES examples
        per_class NO examples

    Sampling is participant-aware.
    """

    rng = random.Random(
        seed
    )

    balanced = []


    for task_index, task_id in enumerate(
        task_ids
    ):

        task_examples = [

            example
            for example in examples

            if example["_task_id"]
            == task_id
        ]


        yes_examples = [

            example
            for example in task_examples

            if example[
                "_answer_clean"
            ] == "(a)"
        ]


        no_examples = [

            example
            for example in task_examples

            if example[
                "_answer_clean"
            ] == "(b)"
        ]


        available_per_class = min(
            len(yes_examples),
            len(no_examples),
        )


        n_to_sample = min(
            per_class,
            available_per_class,
        )


        task_seed = (
            seed
            + task_index * 1000
        )


        selected_yes = (
            sample_across_participants(
                yes_examples,
                n=n_to_sample,
                seed=task_seed,
            )
        )


        selected_no = (
            sample_across_participants(
                no_examples,
                n=n_to_sample,
                seed=task_seed + 1,
            )
        )


        print(
            f"{task_id:28s} | "
            f"YES={len(selected_yes):3d} | "
            f"NO={len(selected_no):3d}"
        )


        balanced.extend(
            selected_yes
        )

        balanced.extend(
            selected_no
        )


    # Mix tasks/classes together before training.
    rng.shuffle(
        balanced
    )


    return balanced