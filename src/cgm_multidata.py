"""
Utilities for multi-participant AI-READI CGM alignment experiments.

Important design choices:
- Split by PARTICIPANT, never by window.
- Compute normalization statistics using TRAIN participants only.
- Apply the same normalization to train / validation / test.
- Use non-overlapping windows initially.
"""

import json
import math
import random
from pathlib import Path

import torch


# ============================================================
# CGM loading
# ============================================================

def load_cgm_file(file_path):
    """
    Load one AI-READI Dexcom JSON file.

    Returns
    -------
    participant_id : str
    glucose : torch.Tensor
        1D tensor of glucose values in mg/dL.
    """

    file_path = Path(file_path)

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    if "body" not in data:
        raise ValueError(
            f"{file_path} has no body field."
        )

    if "cgm" not in data["body"]:
        raise ValueError(
            f"{file_path} has no CGM field."
        )

    participant_id = data.get(
        "header",
        {},
    ).get(
        "patient_id",
        file_path.stem,
    )

    values = []

    for record in data["body"]["cgm"]:

        blood_glucose = record.get(
            "blood_glucose",
            {},
        )

        value = blood_glucose.get(
            "value"
        )

        if value is not None:
            values.append(float(value))

    if not values:
        raise ValueError(
            f"No CGM readings in {file_path}"
        )

    glucose = torch.tensor(
        values,
        dtype=torch.float32,
    )

    return participant_id, glucose


# ============================================================
# Find participant files
# ============================================================

def discover_cgm_files(data_dir):
    """
    Find CGM JSON files.

    We validate each JSON instead of assuming every JSON
    in the directory contains CGM.
    """

    data_dir = Path(data_dir)

    valid_files = []

    for path in sorted(
        data_dir.rglob("*_DEX.json")
    ):
        try:
            _, glucose = load_cgm_file(path)

            if len(glucose) >= 128:
                valid_files.append(path)

        except Exception:
            # Ignore non-CGM JSON files.
            continue

    return valid_files


# ============================================================
# Participant-level splitting
# ============================================================

def split_participant_files(
    files,
    train_fraction=0.70,
    val_fraction=0.15,
    seed=42,
):
    """
    Deterministically split PARTICIPANTS.

    No participant can appear in more than one split.
    """

    files = list(files)

    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)

    n_train = int(
        n * train_fraction
    )

    n_val = int(
        n * val_fraction
    )

    train_files = files[
        :n_train
    ]

    val_files = files[
        n_train:n_train + n_val
    ]

    test_files = files[
        n_train + n_val:
    ]

    return (
        train_files,
        val_files,
        test_files,
    )


# ============================================================
# Fixed training-set normalization
# ============================================================

def compute_training_normalization(
    train_files,
):
    """
    Compute ONE mean and standard deviation from all CGM
    readings belonging to TRAIN participants.

    Validation/test participants are never used here.
    """

    all_values = []

    for path in train_files:

        _, glucose = load_cgm_file(
            path
        )

        all_values.append(
            glucose
        )

    combined = torch.cat(
        all_values
    )

    mean = combined.mean().item()

    # Population SD.
    std = combined.std(
        unbiased=False
    ).item()

    return mean, std


# ============================================================
# Window creation
# ============================================================

def make_windows(
    glucose,
    window_size=128,
    stride=128,
):
    """
    Cut a participant's CGM into windows.

    Initially use stride == window_size so windows do not
    overlap.
    """

    windows = []

    for start in range(
        0,
        len(glucose) - window_size + 1,
        stride,
    ):

        window = glucose[
            start:start + window_size
        ]

        windows.append(
            (
                start,
                window,
            )
        )

    return windows


# ============================================================
# Build one OpenTSLM CGM example
# ============================================================

def make_threshold_example(
    participant_id,
    start,
    raw_window,
    normalization_mean,
    normalization_std,
    eos_token,
    threshold=160,
):
    """
    Create:

        raw CGM window
            +
        "Did glucose exceed 160 mg/dL?"
            ->
        (a) Yes / (b) No

    Unlike the smoke test, normalization is FIXED across
    all participants.
    """

    normalized = (
        raw_window
        - normalization_mean
    ) / (
        normalization_std
        + 1e-8
    )

    max_glucose = (
        raw_window.max().item()
    )

    answer_yes = (
        max_glucose > threshold
    )

    answer = (
        "(a)"
        if answer_yes
        else "(b)"
    )

    question = f"""
The following time series contains continuous glucose monitor
measurements sampled approximately every 5 minutes.

Did the glucose concentration ever exceed {threshold} mg/dL
during this interval?

Choose one answer:
(a) Yes
(b) No

Only answer (a) or (b).
""".strip()

    # Notice:
    # We no longer give the model the window mean/std.
    #
    # Since fixed normalization preserves absolute scale,
    # we want the time-series representation itself to carry
    # the glucose-level information.
    description = (
        "This is a continuous glucose monitor time series "
        "sampled approximately every 5 minutes."
    )

    return {
        "pre_prompt":
            question,

        "time_series_text":
            [description],

        "time_series":
            [normalized.clone()],

        "post_prompt":
            "Predict the CGM Threshold Answer:",

        "answer":
            answer + eos_token,

        # Metadata never sent to OpenTSLM.
        "_participant_id":
            participant_id,

        "_start":
            start,

        "_max_glucose":
            max_glucose,

        "_answer_clean":
            answer,

        "_threshold":
            threshold,
    }


# ============================================================
# Build examples across participants
# ============================================================

def build_examples(
    files,
    normalization_mean,
    normalization_std,
    eos_token,
    window_size=128,
    stride=128,
    threshold=160,
):
    examples = []

    for file_path in files:

        participant_id, glucose = (
            load_cgm_file(
                file_path
            )
        )

        windows = make_windows(
            glucose,
            window_size=window_size,
            stride=stride,
        )

        for start, window in windows:

            example = (
                make_threshold_example(
                    participant_id=
                        participant_id,

                    start=
                        start,

                    raw_window=
                        window,

                    normalization_mean=
                        normalization_mean,

                    normalization_std=
                        normalization_std,

                    eos_token=
                        eos_token,

                    threshold=
                        threshold,
                )
            )

            examples.append(
                example
            )

    return examples


# ============================================================
# Balance training labels
# ============================================================

def balance_binary_examples(
    examples,
    seed=42,
):
    """
    Equal numbers of YES and NO examples.

    This prevents the model from learning:
        "always answer (a)"
    or
        "always answer (b)"
    """

    positive = [
        x for x in examples
        if x["_answer_clean"] == "(a)"
    ]

    negative = [
        x for x in examples
        if x["_answer_clean"] == "(b)"
    ]

    rng = random.Random(seed)

    rng.shuffle(positive)
    rng.shuffle(negative)

    n = min(
        len(positive),
        len(negative),
    )

    balanced = (
        positive[:n]
        + negative[:n]
    )

    rng.shuffle(
        balanced
    )

    return balanced