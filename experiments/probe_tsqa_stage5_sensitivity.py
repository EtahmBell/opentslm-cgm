"""
Synthetic/scaling sensitivity probe for TSQA vs Stage-5.

Purpose
-------
Previous diagnostic showed:

Original TSQA:
    meaningful representation variation across CGM windows

Original Stage-5 ECG:
    encoder/projector representations were nearly identical
    across CGM windows.

Question
--------
Is Stage-5:

A) globally insensitive to input changes?

or

B) capable of responding to signals, but relatively insensitive
   to the amplitude/frequency regime occupied by CGM?

No training is performed.

We compare ORIGINAL:
    1. TSQA
    2. Stage-5 ECG

Signals:
    - one real normalized CGM window
    - scaled versions of that CGM
    - constant signals
    - ramps
    - sinusoids at several frequencies/amplitudes
    - noise
    - spikes
    - alternating signals

For each input we extract:

    CGM/synthetic signal
        ↓
    encoder representation
        ↓
    projector representation

We report distance from the real CGM x1 baseline and
pairwise representation variability.

TEST participants remain sealed.
"""

import gc
import json
import math
import random
import sys

from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# Paths
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
)

from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

SEED = 42

TSQA_REPO_ID = (
    "OpenTSLM/llama-3.2-1b-tsqa-sp"
)

STAGE5_REPO_ID = (
    "OpenTSLM/llama-3.2-1b-ecg-sp"
)

DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "tsqa_stage5_synthetic_sensitivity.json"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128


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
# Load TSQA temporarily to get EOS
# ============================================================

print(
    "\nLoading TSQA to construct CGM examples..."
)


temp_model = (
    OpenTSLM.load_pretrained(
        TSQA_REPO_ID,
        device=device,
    )
)


EOS_TOKEN = (
    temp_model.get_eos_token()
)


# ============================================================
# Build CGM windows
# ============================================================

raw_examples = (
    build_multitask_examples(
        files=train_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=[
            TASK_ID
        ],
    )
)


yes_examples = [
    x
    for x in raw_examples
    if x[
        "_answer_clean"
    ] == "(a)"
]


no_examples = [
    x
    for x in raw_examples
    if x[
        "_answer_clean"
    ] == "(b)"
]


if (
    len(yes_examples) == 0
    or len(no_examples) == 0
):

    raise RuntimeError(
        "Could not find both YES and NO CGM windows."
    )


# ============================================================
# Choose representative real CGM windows
#
# Use same deterministic selection style as earlier diagnostics.
# ============================================================

rng = random.Random(
    SEED + 777
)


real_yes_example = (
    rng.choice(
        yes_examples
    )
)

real_no_example = (
    rng.choice(
        no_examples
    )
)


def get_signal(
    example,
):

    time_series = (
        example[
            "time_series"
        ]
    )


    if len(time_series) != 1:

        raise RuntimeError(
            "Expected exactly one time-series "
            "segment per example."
        )


    signal = (
        time_series[
            0
        ]
    )


    if not torch.is_tensor(
        signal
    ):

        signal = torch.tensor(
            signal,
            dtype=torch.float32,
        )


    signal = (
        signal
        .detach()
        .clone()
        .float()
    )


    # [T, 1] -> [T]
    if (
        signal.dim() == 2
        and signal.shape[-1] == 1
    ):

        signal = signal.squeeze(
            -1
        )


    # [1, T] -> [T]
    if (
        signal.dim() == 2
        and signal.shape[0] == 1
    ):

        signal = signal.squeeze(
            0
        )


    if signal.dim() != 1:

        raise RuntimeError(
            "Expected one-dimensional signal, "
            f"got {tuple(signal.shape)}"
        )


    return signal


real_yes = (
    get_signal(
        real_yes_example
    )
)


real_no = (
    get_signal(
        real_no_example
    )
)


if len(real_yes) != WINDOW_SIZE:

    raise RuntimeError(
        f"Expected YES length {WINDOW_SIZE}, "
        f"got {len(real_yes)}"
    )


if len(real_no) != WINDOW_SIZE:

    raise RuntimeError(
        f"Expected NO length {WINDOW_SIZE}, "
        f"got {len(real_no)}"
    )


print(
    "\nSelected real YES example:"
)

print(
    "Participant:",
    real_yes_example.get(
        "_participant_id"
    ),
)

print(
    "Window:",
    real_yes_example.get(
        "_window_start"
    ),
)


print(
    "\nSelected real NO example:"
)

print(
    "Participant:",
    real_no_example.get(
        "_participant_id"
    ),
)

print(
    "Window:",
    real_no_example.get(
        "_window_start"
    ),
)


# ============================================================
# Remove temporary model
# ============================================================

del temp_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Input statistics
# ============================================================

def signal_stats(
    signal,
):

    return {
        "min":
            float(
                signal.min().item()
            ),

        "max":
            float(
                signal.max().item()
            ),

        "mean":
            float(
                signal.mean().item()
            ),

        "std":
            float(
                signal.std().item()
            ),

        "range":
            float(
                (
                    signal.max()
                    - signal.min()
                ).item()
            ),
    }


print(
    "\n"
    + "=" * 90
)

print(
    "REAL CGM INPUT STATISTICS"
)

print(
    "=" * 90
)


print(
    "\nNormalized YES:",
    signal_stats(
        real_yes
    ),
)

print(
    "Normalized NO:",
    signal_stats(
        real_no
    ),
)


# ============================================================
# Construct synthetic signals
#
# Everything is in MODEL INPUT SPACE.
#
# Since CGM is already normalized:
#
#     z = (glucose - mean) / SD
#
# values around +/-1 represent roughly +/-1 training SD.
# ============================================================

t = torch.linspace(
    0.0,
    1.0,
    WINDOW_SIZE,
)


generator = torch.Generator()

generator.manual_seed(
    SEED
)


signals = {}


# ------------------------------------------------------------
# Real CGM baselines
# ------------------------------------------------------------

signals[
    "real_cgm_yes_x1"
] = (
    real_yes.clone()
)

signals[
    "real_cgm_no_x1"
] = (
    real_no.clone()
)


# ------------------------------------------------------------
# Scale the same REAL YES window around normalized zero.
#
# In normalized coordinates, zero = training-set mean glucose.
# ------------------------------------------------------------

for scale in [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]:

    signals[
        f"real_yes_x{scale:g}"
    ] = (
        real_yes
        * scale
    )


# ------------------------------------------------------------
# Scale real NO as well.
# ------------------------------------------------------------

for scale in [
    0.1,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
]:

    signals[
        f"real_no_x{scale:g}"
    ] = (
        real_no
        * scale
    )


# ------------------------------------------------------------
# Constant signals
# ------------------------------------------------------------

signals[
    "constant_zero"
] = torch.zeros(
    WINDOW_SIZE
)


signals[
    "constant_plus1"
] = torch.ones(
    WINDOW_SIZE
)


signals[
    "constant_minus1"
] = (
    -torch.ones(
        WINDOW_SIZE
    )
)


signals[
    "constant_plus5"
] = (
    torch.ones(
        WINDOW_SIZE
    )
    * 5.0
)


signals[
    "constant_minus5"
] = (
    -torch.ones(
        WINDOW_SIZE
    )
    * 5.0
)


# ------------------------------------------------------------
# Linear ramps
# ------------------------------------------------------------

signals[
    "ramp_minus1_to_plus1"
] = torch.linspace(
    -1.0,
    1.0,
    WINDOW_SIZE,
)


signals[
    "ramp_minus5_to_plus5"
] = torch.linspace(
    -5.0,
    5.0,
    WINDOW_SIZE,
)


# ------------------------------------------------------------
# Sinusoids
# ------------------------------------------------------------

signals[
    "sine_1cycle_amp1"
] = torch.sin(
    2.0
    * math.pi
    * t
)


signals[
    "sine_1cycle_amp5"
] = (
    5.0
    * torch.sin(
        2.0
        * math.pi
        * t
    )
)


signals[
    "sine_4cycle_amp1"
] = torch.sin(
    2.0
    * math.pi
    * 4.0
    * t
)


signals[
    "sine_16cycle_amp1"
] = torch.sin(
    2.0
    * math.pi
    * 16.0
    * t
)


signals[
    "sine_32cycle_amp1"
] = torch.sin(
    2.0
    * math.pi
    * 32.0
    * t
)


signals[
    "sine_16cycle_amp5"
] = (
    5.0
    * torch.sin(
        2.0
        * math.pi
        * 16.0
        * t
    )
)


# ------------------------------------------------------------
# Noise
# ------------------------------------------------------------

signals[
    "noise_std0.1"
] = (
    0.1
    * torch.randn(
        WINDOW_SIZE,
        generator=generator,
    )
)


signals[
    "noise_std1"
] = torch.randn(
    WINDOW_SIZE,
    generator=generator,
)


signals[
    "noise_std5"
] = (
    5.0
    * torch.randn(
        WINDOW_SIZE,
        generator=generator,
    )
)


# ------------------------------------------------------------
# Single spikes
# ------------------------------------------------------------

spike_1 = torch.zeros(
    WINDOW_SIZE
)

spike_1[
    WINDOW_SIZE // 2
] = 1.0


signals[
    "single_spike_amp1"
] = spike_1


spike_5 = torch.zeros(
    WINDOW_SIZE
)

spike_5[
    WINDOW_SIZE // 2
] = 5.0


signals[
    "single_spike_amp5"
] = spike_5


spike_10 = torch.zeros(
    WINDOW_SIZE
)

spike_10[
    WINDOW_SIZE // 2
] = 10.0


signals[
    "single_spike_amp10"
] = spike_10


# ------------------------------------------------------------
# Alternating high-frequency pattern
# ------------------------------------------------------------

alternating = torch.ones(
    WINDOW_SIZE
)

alternating[
    1::2
] = -1.0


signals[
    "alternating_plusminus1"
] = alternating


signals[
    "alternating_plusminus5"
] = (
    alternating
    * 5.0
)


# ============================================================
# Print synthetic input stats
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "SYNTHETIC SIGNAL INPUT STATISTICS"
)

print(
    "=" * 90
)


for name, signal in signals.items():

    stats = signal_stats(
        signal
    )


    print(
        f"{name:28s} | "
        f"min={stats['min']:8.3f} | "
        f"max={stats['max']:8.3f} | "
        f"mean={stats['mean']:8.3f} | "
        f"std={stats['std']:8.3f}"
    )


# ============================================================
# Encoder + projector forward helper
# ============================================================

def encode_signal(
    model,
    signal,
):

    x = (
        signal
        .detach()
        .clone()
        .float()
        .to(
            device
        )
        .unsqueeze(
            0
        )
    )


    # Pad exactly as needed for patching.
    remainder = (
        x.shape[
            1
        ]
        % model.patch_size
    )


    if remainder:

        pad_length = (
            model.patch_size
            - remainder
        )


        x = F.pad(
            x,
            (
                0,
                pad_length,
            ),
        )


    model.encoder.eval()
    model.projector.eval()


    with torch.no_grad():

        encoded = (
            model.encoder(
                x
            )
        )


        projected = (
            model.projector(
                encoded
            )
        )


    encoded_flat = (
        encoded[
            0
        ]
        .float()
        .cpu()
        .reshape(
            -1
        )
    )


    projected_flat = (
        projected[
            0
        ]
        .float()
        .cpu()
        .reshape(
            -1
        )
    )


    return (
        encoded_flat,
        projected_flat,
    )


# ============================================================
# Representation distance
# ============================================================

def compare_vectors(
    reference,
    other,
):

    cosine = float(
        F.cosine_similarity(
            reference.unsqueeze(
                0
            ),
            other.unsqueeze(
                0
            ),
        ).item()
    )


    l2 = float(
        torch.norm(
            reference
            - other
        ).item()
    )


    normalized_l2 = (
        l2
        / math.sqrt(
            reference.numel()
        )
    )


    relative_l2 = (
        l2
        / (
            float(
                torch.norm(
                    reference
                ).item()
            )
            + 1e-12
        )
    )


    return {
        "cosine":
            cosine,

        "l2":
            l2,

        "l2_per_sqrt_dim":
            normalized_l2,

        "relative_l2":
            relative_l2,
    }


# ============================================================
# Representation variation across all signals
# ============================================================

def representation_variation(
    vectors,
):

    matrix = torch.stack(
        list(
            vectors.values()
        )
    ).float()


    mean_feature_std = float(
        matrix.std(
            dim=0
        )
        .mean()
        .item()
    )


    centered = (
        matrix
        - matrix.mean(
            dim=0,
            keepdim=True,
        )
    )


    rms_distance_from_mean = float(
        torch.sqrt(
            (
                centered
                * centered
            )
            .mean()
        ).item()
    )


    # Pairwise cosine statistics
    normalized = F.normalize(
        matrix,
        dim=1,
    )


    cosine_matrix = (
        normalized
        @ normalized.T
    )


    pairwise_cosines = []


    for i in range(
        matrix.shape[
            0
        ]
    ):

        for j in range(
            i + 1,
            matrix.shape[
                0
            ]
        ):

            pairwise_cosines.append(
                float(
                    cosine_matrix[
                        i,
                        j
                    ].item()
                )
            )


    return {
        "mean_feature_std":
            mean_feature_std,

        "rms_distance_from_mean":
            rms_distance_from_mean,

        "mean_pairwise_cosine":
            sum(
                pairwise_cosines
            )
            / len(
                pairwise_cosines
            ),

        "min_pairwise_cosine":
            min(
                pairwise_cosines
            ),

        "max_pairwise_cosine":
            max(
                pairwise_cosines
            ),
    }


# ============================================================
# Run one model
# ============================================================

def run_probe(
    repo_id,
    label,
    enable_lora,
):

    print(
        "\n\n"
        + "#" * 100
    )

    print(
        label
    )

    print(
        "#" * 100
    )


    model = (
        OpenTSLM.load_pretrained(
            repo_id,
            device=device,
            enable_lora=enable_lora,
        )
    )


    encoder_vectors = {}

    projector_vectors = {}


    for name, signal in signals.items():

        (
            encoder_vector,
            projector_vector,
        ) = (
            encode_signal(
                model,
                signal,
            )
        )


        encoder_vectors[
            name
        ] = encoder_vector


        projector_vectors[
            name
        ] = projector_vector


    # --------------------------------------------------------
    # Baselines
    # --------------------------------------------------------

    encoder_reference = (
        encoder_vectors[
            "real_yes_x1"
        ]
    )


    projector_reference = (
        projector_vectors[
            "real_yes_x1"
        ]
    )


    comparisons = {}


    print(
        "\n"
        + "=" * 100
    )

    print(
        "DISTANCE FROM REAL CGM YES ×1 BASELINE"
    )

    print(
        "=" * 100
    )


    print(
        f"{'Signal':28s} | "
        f"{'Enc cosine':>10s} | "
        f"{'Enc L2/√d':>11s} | "
        f"{'Proj cosine':>11s} | "
        f"{'Proj L2/√d':>12s}"
    )


    print(
        "-" * 100
    )


    for name in signals.keys():

        enc_comparison = (
            compare_vectors(
                encoder_reference,
                encoder_vectors[
                    name
                ],
            )
        )


        proj_comparison = (
            compare_vectors(
                projector_reference,
                projector_vectors[
                    name
                ],
            )
        )


        comparisons[
            name
        ] = {
            "encoder":
                enc_comparison,

            "projector":
                proj_comparison,
        }


        print(
            f"{name:28s} | "
            f"{enc_comparison['cosine']:10.6f} | "
            f"{enc_comparison['l2_per_sqrt_dim']:11.6f} | "
            f"{proj_comparison['cosine']:11.6f} | "
            f"{proj_comparison['l2_per_sqrt_dim']:12.6f}"
        )


    # --------------------------------------------------------
    # Overall variation
    # --------------------------------------------------------

    encoder_variation = (
        representation_variation(
            encoder_vectors
        )
    )


    projector_variation = (
        representation_variation(
            projector_vectors
        )
    )


    print(
        "\n--- OVERALL ENCODER VARIATION ---"
    )

    print(
        "Mean feature std:",
        f"{encoder_variation['mean_feature_std']:.8f}"
    )

    print(
        "RMS distance from mean:",
        f"{encoder_variation['rms_distance_from_mean']:.8f}"
    )

    print(
        "Mean pairwise cosine:",
        f"{encoder_variation['mean_pairwise_cosine']:.8f}"
    )

    print(
        "Minimum pairwise cosine:",
        f"{encoder_variation['min_pairwise_cosine']:.8f}"
    )


    print(
        "\n--- OVERALL PROJECTOR VARIATION ---"
    )

    print(
        "Mean feature std:",
        f"{projector_variation['mean_feature_std']:.8f}"
    )

    print(
        "RMS distance from mean:",
        f"{projector_variation['rms_distance_from_mean']:.8f}"
    )

    print(
        "Mean pairwise cosine:",
        f"{projector_variation['mean_pairwise_cosine']:.8f}"
    )

    print(
        "Minimum pairwise cosine:",
        f"{projector_variation['min_pairwise_cosine']:.8f}"
    )


    # --------------------------------------------------------
    # Useful targeted scaling summary
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "REAL-CGM SCALING SUMMARY"
    )

    print(
        "=" * 100
    )


    for name in [
        "real_yes_x0.1",
        "real_yes_x0.5",
        "real_yes_x1",
        "real_yes_x2",
        "real_yes_x5",
        "real_yes_x10",
    ]:

        result = (
            comparisons[
                name
            ]
        )


        print(
            f"{name:20s} | "
            f"enc cosine="
            f"{result['encoder']['cosine']:.8f} | "
            f"enc L2/√d="
            f"{result['encoder']['l2_per_sqrt_dim']:.8f} | "
            f"proj cosine="
            f"{result['projector']['cosine']:.8f}"
        )


    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    result = {
        "label":
            label,

        "comparisons_to_real_yes_x1":
            comparisons,

        "encoder_variation":
            encoder_variation,

        "projector_variation":
            projector_variation,
    }


    del model

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# TSQA
# ============================================================

tsqa_result = (
    run_probe(
        repo_id=
            TSQA_REPO_ID,

        label=
            "ORIGINAL TSQA",

        enable_lora=False,
    )
)


# ============================================================
# Stage 5
# ============================================================

stage5_result = (
    run_probe(
        repo_id=
            STAGE5_REPO_ID,

        label=
            "ORIGINAL STAGE-5 ECG",

        enable_lora=True,
    )
)


# ============================================================
# Final comparison
# ============================================================

print(
    "\n\n"
    + "=" * 100
)

print(
    "TSQA vs STAGE-5 SENSITIVITY SUMMARY"
)

print(
    "=" * 100
)


print(
    f"{'Model':24s} | "
    f"{'Encoder feature SD':>18s} | "
    f"{'Enc min cosine':>14s} | "
    f"{'Projector SD':>14s} | "
    f"{'Proj min cosine':>15s}"
)


print(
    "-" * 100
)


for label, result in [
    (
        "TSQA",
        tsqa_result,
    ),
    (
        "Stage-5 ECG",
        stage5_result,
    ),
]:

    print(
        f"{label:24s} | "
        f"{result['encoder_variation']['mean_feature_std']:18.8f} | "
        f"{result['encoder_variation']['min_pairwise_cosine']:14.8f} | "
        f"{result['projector_variation']['mean_feature_std']:14.8f} | "
        f"{result['projector_variation']['min_pairwise_cosine']:15.8f}"
    )


# ============================================================
# Targeted Stage-5 interpretation values
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "STAGE-5 KEY RESPONSES"
)

print(
    "=" * 100
)


for name in [
    "real_yes_x0.1",
    "real_yes_x0.5",
    "real_yes_x2",
    "real_yes_x5",
    "real_yes_x10",
    "constant_zero",
    "sine_1cycle_amp1",
    "sine_16cycle_amp1",
    "sine_16cycle_amp5",
    "noise_std1",
    "noise_std5",
    "single_spike_amp10",
    "alternating_plusminus5",
]:

    result = (
        stage5_result[
            "comparisons_to_real_yes_x1"
        ][
            name
        ]
    )


    print(
        f"{name:28s} | "
        f"enc cosine="
        f"{result['encoder']['cosine']:.8f} | "
        f"enc L2/√d="
        f"{result['encoder']['l2_per_sqrt_dim']:.8f} | "
        f"proj cosine="
        f"{result['projector']['cosine']:.8f} | "
        f"proj L2/√d="
        f"{result['projector']['l2_per_sqrt_dim']:.8f}"
    )


# ============================================================
# Save
# ============================================================

results = {
    "experiment":
        "tsqa_stage5_synthetic_scaling_sensitivity",

    "seed":
        SEED,

    "normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "real_examples":
        {
            "yes":
                {
                    "participant_id":
                        real_yes_example.get(
                            "_participant_id"
                        ),

                    "window_start":
                        real_yes_example.get(
                            "_window_start"
                        ),

                    "input_stats":
                        signal_stats(
                            real_yes
                        ),
                },

            "no":
                {
                    "participant_id":
                        real_no_example.get(
                            "_participant_id"
                        ),

                    "window_start":
                        real_no_example.get(
                            "_window_start"
                        ),

                    "input_stats":
                        signal_stats(
                            real_no
                        ),
                },
        },

    "signal_input_stats":
        {
            name:
                signal_stats(
                    signal
                )

            for name, signal
            in signals.items()
        },

    "tsqa":
        tsqa_result,

    "stage5":
        stage5_result,
}


RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


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
    "\nSaved diagnostic:"
)

print(
    RESULTS_PATH
)


print(
    "\nNo training was performed."
)

print(
    "TEST participants remain sealed."
)