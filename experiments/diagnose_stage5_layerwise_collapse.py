"""
Layer-wise TSQA vs Stage-5 encoder collapse diagnostic.

Purpose
-------
Previous experiments showed:

TSQA:
    Different CGM and synthetic signals produce substantially
    different encoder representations.

Stage-5 ECG:
    Even enormous changes to the input produce nearly identical
    final encoder/projector representations.

This script asks:

    WHERE inside the Stage-5 encoder does that information disappear?

We manually reproduce the official TransformerCNNEncoder forward path:

    raw signal
        ↓
    Conv1D patch embedding
        ↓
    + positional embedding
        ↓
    input LayerNorm
        ↓
    Transformer layer 1
        ↓
    ...
    Transformer layer 6
        ↓
    encoder final
        ↓
    projector

We compare ORIGINAL:
    1. TSQA
    2. Stage-5 ECG

Using three deliberately different signals:
    - a real normalized CGM window
    - all zeros
    - alternating +5 / -5

At every stage we measure:
    - activation norm / RMS
    - variation across the three inputs
    - cosine(real, zero)
    - cosine(real, alternating)
    - L2 distance / sqrt(dim)
    - local input-gradient sensitivity

We also compare:
    - Conv1D patch weight magnitude
    - positional embedding magnitude
    - patch activation magnitude vs positional embedding magnitude

No optimization is performed.

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
    / "stage5_layerwise_collapse.json"
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
# Load temporary TSQA model to construct CGM examples
# ============================================================

print(
    "\nLoading TSQA temporarily "
    "to construct CGM examples..."
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
# Build train CGM windows
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


if len(
    yes_examples
) == 0:

    raise RuntimeError(
        "No above_180 YES examples found."
    )


# ============================================================
# Same deterministic selection style as earlier probes
# ============================================================

selection_rng = random.Random(
    SEED + 777
)


real_example = (
    selection_rng.choice(
        yes_examples
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


    if len(
        time_series
    ) != 1:

        raise RuntimeError(
            "Expected exactly one "
            "time-series segment."
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


    if (
        signal.dim() == 2
        and signal.shape[
            -1
        ] == 1
    ):

        signal = signal.squeeze(
            -1
        )


    if (
        signal.dim() == 2
        and signal.shape[
            0
        ] == 1
    ):

        signal = signal.squeeze(
            0
        )


    if signal.dim() != 1:

        raise RuntimeError(
            "Expected 1-D signal, got "
            f"{tuple(signal.shape)}"
        )


    return signal


real_signal = (
    get_signal(
        real_example
    )
)


if len(
    real_signal
) != WINDOW_SIZE:

    raise RuntimeError(
        f"Expected window length "
        f"{WINDOW_SIZE}, got "
        f"{len(real_signal)}"
    )


print(
    "\nSelected real CGM:"
)

print(
    "Participant:",
    real_example.get(
        "_participant_id"
    ),
)

print(
    "Window:",
    real_example.get(
        "_window_start"
    ),
)

print(
    "Gold:",
    real_example[
        "_answer_clean"
    ],
)


# ============================================================
# Release temporary model
# ============================================================

del temp_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Three deliberately different signals
# ============================================================

zero_signal = torch.zeros(
    WINDOW_SIZE,
    dtype=torch.float32,
)


alternating_signal = torch.ones(
    WINDOW_SIZE,
    dtype=torch.float32,
)


alternating_signal[
    1::2
] = -1.0


alternating_signal *= 5.0


signals = {
    "real_cgm":
        real_signal,

    "zeros":
        zero_signal,

    "alternating_plusminus5":
        alternating_signal,
}


# ============================================================
# Print input statistics
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
    }


print(
    "\n"
    + "=" * 100
)

print(
    "INPUT SIGNALS"
)

print(
    "=" * 100
)


for name, signal in signals.items():

    stats = (
        signal_stats(
            signal
        )
    )


    print(
        f"{name:28s} | "
        f"min={stats['min']:8.4f} | "
        f"max={stats['max']:8.4f} | "
        f"mean={stats['mean']:8.4f} | "
        f"std={stats['std']:8.4f}"
    )


# ============================================================
# Architecture check
# ============================================================

def check_encoder_architecture(
    model,
):

    encoder = (
        model.encoder
    )


    required = [
        "patch_embed",
        "pos_embed",
        "input_norm",
        "input_dropout",
        "encoder",
        "patch_size",
    ]


    missing = [
        name
        for name in required
        if not hasattr(
            encoder,
            name,
        )
    ]


    if missing:

        raise RuntimeError(
            "Unexpected encoder architecture. "
            "Missing attributes: "
            f"{missing}"
        )


    if not hasattr(
        encoder.encoder,
        "layers",
    ):

        raise RuntimeError(
            "Expected TransformerEncoder "
            "with .layers attribute."
        )


# ============================================================
# Pad signal exactly to patch size
# ============================================================

def prepare_signal(
    model,
    signal,
    requires_grad=False,
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


    remainder = (
        x.shape[
            1
        ]
        % model.encoder.patch_size
    )


    if remainder:

        pad_length = (
            model.encoder.patch_size
            - remainder
        )


        x = F.pad(
            x,
            (
                0,
                pad_length,
            ),
        )


    if requires_grad:

        x.requires_grad_(
            True
        )


    return x


# ============================================================
# Manual layer-by-layer forward
# ============================================================

def layerwise_forward(
    model,
    signal,
    requires_grad=False,
):

    check_encoder_architecture(
        model
    )


    encoder = (
        model.encoder
    )


    x_input = (
        prepare_signal(
            model,
            signal,
            requires_grad=
                requires_grad,
        )
    )


    stages = {}


    # --------------------------------------------------------
    # Conv1D patch embedding
    #
    # [B, L]
    #   -> [B, 1, L]
    #   -> [B, D, N]
    #   -> [B, N, D]
    # --------------------------------------------------------

    x = (
        x_input.unsqueeze(
            1
        )
    )


    x = (
        encoder.patch_embed(
            x
        )
    )


    x = (
        x.transpose(
            1,
            2,
        )
    )


    stages[
        "01_patch_embed"
    ] = x


    # --------------------------------------------------------
    # Position embedding
    # --------------------------------------------------------

    n_patches = (
        x.shape[
            1
        ]
    )


    pos = (
        encoder.pos_embed[
            :,
            :n_patches,
            :
        ]
    )


    x = (
        x
        + pos
    )


    stages[
        "02_plus_position"
    ] = x


    # --------------------------------------------------------
    # Input LayerNorm
    # --------------------------------------------------------

    x = (
        encoder.input_norm(
            x
        )
    )


    stages[
        "03_input_norm"
    ] = x


    # --------------------------------------------------------
    # Input dropout
    # --------------------------------------------------------

    x = (
        encoder.input_dropout(
            x
        )
    )


    stages[
        "04_input_dropout"
    ] = x


    # --------------------------------------------------------
    # Individual TransformerEncoder layers
    # --------------------------------------------------------

    for index, layer in enumerate(
        encoder.encoder.layers,
        start=1,
    ):

        x = (
            layer(
                x
            )
        )


        stages[
            f"{4 + index:02d}_transformer_{index}"
        ] = x


    # --------------------------------------------------------
    # TransformerEncoder may optionally have final norm.
    # --------------------------------------------------------

    final_norm = getattr(
        encoder.encoder,
        "norm",
        None,
    )


    if final_norm is not None:

        x = (
            final_norm(
                x
            )
        )


        stages[
            "11_transformer_final_norm"
        ] = x


    # --------------------------------------------------------
    # Final encoder output
    # --------------------------------------------------------

    stages[
        "12_encoder_final"
    ] = x


    # --------------------------------------------------------
    # Project into LLM hidden space
    # --------------------------------------------------------

    projected = (
        model.projector(
            x
        )
    )


    stages[
        "13_projector"
    ] = projected


    return (
        x_input,
        stages,
        pos,
    )


# ============================================================
# Validate manual encoder forward
#
# Important sanity check:
# our manual layer-by-layer path should reproduce
# model.encoder(signal).
# ============================================================

def validate_manual_forward(
    model,
    signal,
):

    model.encoder.eval()
    model.projector.eval()


    x_input = (
        prepare_signal(
            model,
            signal,
            requires_grad=False,
        )
    )


    with torch.no_grad():

        direct = (
            model.encoder(
                x_input
            )
        )


        (
            _,
            stages,
            _,
        ) = (
            layerwise_forward(
                model,
                signal,
                requires_grad=False,
            )
        )


        manual = (
            stages[
                "12_encoder_final"
            ]
        )


    max_abs_difference = float(
        torch.max(
            torch.abs(
                direct
                - manual
            )
        )
        .cpu()
        .item()
    )


    return (
        max_abs_difference
    )


# ============================================================
# Tensor helpers
# ============================================================

def flatten_cpu(
    tensor,
):

    return (
        tensor
        .detach()
        .float()
        .cpu()
        .reshape(
            -1
        )
    )


def tensor_rms(
    tensor,
):

    tensor = (
        tensor
        .detach()
        .float()
    )


    return float(
        torch.sqrt(
            torch.mean(
                tensor
                * tensor
            )
        )
        .cpu()
        .item()
    )


def tensor_norm(
    tensor,
):

    return float(
        torch.norm(
            tensor
            .detach()
            .float()
        )
        .cpu()
        .item()
    )


# ============================================================
# Compare vectors
# ============================================================

def compare_vectors(
    a,
    b,
):

    a = (
        a.float()
        .reshape(
            -1
        )
    )


    b = (
        b.float()
        .reshape(
            -1
        )
    )


    cosine = float(
        F.cosine_similarity(
            a.unsqueeze(
                0
            ),
            b.unsqueeze(
                0
            ),
        )
        .item()
    )


    l2 = float(
        torch.norm(
            a
            - b
        )
        .item()
    )


    l2_per_sqrt_dim = (
        l2
        / math.sqrt(
            a.numel()
        )
    )


    denominator = (
        float(
            torch.norm(
                a
            )
            .item()
        )
        + 1e-12
    )


    relative_l2 = (
        l2
        / denominator
    )


    return {
        "cosine":
            cosine,

        "l2":
            l2,

        "l2_per_sqrt_dim":
            l2_per_sqrt_dim,

        "relative_l2":
            relative_l2,
    }


# ============================================================
# Variation across the three signal inputs
# ============================================================

def three_signal_variation(
    vectors,
):

    matrix = torch.stack(
        [
            vectors[
                "real_cgm"
            ],

            vectors[
                "zeros"
            ],

            vectors[
                "alternating_plusminus5"
            ],
        ]
    ).float()


    feature_std = float(
        matrix.std(
            dim=0
        )
        .mean()
        .item()
    )


    rms_feature_std = float(
        torch.sqrt(
            torch.mean(
                matrix.std(
                    dim=0
                )
                ** 2
            )
        )
        .item()
    )


    return {
        "mean_feature_std":
            feature_std,

        "rms_feature_std":
            rms_feature_std,
    }


# ============================================================
# Local input-gradient sensitivity
#
# We use a deterministic random projection of each stage's
# activation, then measure || d scalar / d input ||.
#
# This is a cheap scalar proxy for local Jacobian sensitivity.
# ============================================================

def gradient_sensitivity(
    model,
    signal,
):

    model.encoder.eval()
    model.projector.eval()


    (
        x_input,
        stages,
        _,
    ) = (
        layerwise_forward(
            model,
            signal,
            requires_grad=True,
        )
    )


    gradient_results = {}


    stage_items = list(
        stages.items()
    )


    for stage_index, (
        stage_name,
        activation,
    ) in enumerate(
        stage_items
    ):

        # ----------------------------------------------------
        # Deterministic random projection.
        #
        # Different seed per layer, same seed/model-independent
        # construction for fair comparison.
        # ----------------------------------------------------

        generator = (
            torch.Generator(
                device="cpu"
            )
        )


        generator.manual_seed(
            SEED
            + 900000
            + stage_index
        )


        probe = torch.randn(
            activation.numel(),
            generator=generator,
            dtype=torch.float32,
        )


        probe = (
            probe
            .to(
                activation.device
            )
            .reshape_as(
                activation
            )
        )


        probe = (
            probe
            / math.sqrt(
                activation.numel()
            )
        )


        scalar = (
            activation.float()
            * probe
        ).sum()


        gradient = (
            torch.autograd.grad(
                scalar,
                x_input,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )[
                0
            ]
        )


        if gradient is None:

            grad_norm = 0.0
            grad_rms = 0.0

        else:

            gradient = (
                gradient
                .detach()
                .float()
            )


            grad_norm = float(
                torch.norm(
                    gradient
                )
                .cpu()
                .item()
            )


            grad_rms = float(
                torch.sqrt(
                    torch.mean(
                        gradient
                        * gradient
                    )
                )
                .cpu()
                .item()
            )


        gradient_results[
            stage_name
        ] = {
            "input_gradient_norm":
                grad_norm,

            "input_gradient_rms":
                grad_rms,
        }


    return (
        gradient_results
    )


# ============================================================
# Run one model
# ============================================================

def run_model(
    repo_id,
    label,
    enable_lora,
):

    print(
        "\n\n"
        + "#" * 110
    )

    print(
        label
    )

    print(
        "#" * 110
    )


    model = (
        OpenTSLM.load_pretrained(
            repo_id,
            device=device,
            enable_lora=
                enable_lora,
        )
    )


    model.encoder.eval()
    model.projector.eval()
    model.llm.eval()


    # --------------------------------------------------------
    # Verify manual path
    # --------------------------------------------------------

    manual_difference = (
        validate_manual_forward(
            model,
            real_signal,
        )
    )


    print(
        "\nManual-vs-direct encoder "
        "max abs difference:",
        f"{manual_difference:.10e}"
    )


    if manual_difference > 1e-4:

        raise RuntimeError(
            "Manual encoder path does not "
            "match model.encoder(). "
            f"Difference={manual_difference}"
        )


    print(
        "✓ Manual layer-by-layer forward "
        "matches model.encoder()."
    )


    # --------------------------------------------------------
    # Weight / scale diagnostics
    # --------------------------------------------------------

    patch_weights = (
        model.encoder
        .patch_embed
        .weight
        .detach()
        .float()
    )


    patch_weight_norm = float(
        torch.norm(
            patch_weights
        )
        .cpu()
        .item()
    )


    patch_weight_rms = float(
        torch.sqrt(
            torch.mean(
                patch_weights
                * patch_weights
            )
        )
        .cpu()
        .item()
    )


    # Need position tensor shape for current window.
    with torch.no_grad():

        (
            _,
            real_stages_temp,
            pos,
        ) = (
            layerwise_forward(
                model,
                real_signal,
                requires_grad=False,
            )
        )


    position_rms = (
        tensor_rms(
            pos
        )
    )


    real_patch_rms = (
        tensor_rms(
            real_stages_temp[
                "01_patch_embed"
            ]
        )
    )


    alternating_patch_rms = None


    with torch.no_grad():

        (
            _,
            alt_stages_temp,
            _,
        ) = (
            layerwise_forward(
                model,
                alternating_signal,
                requires_grad=False,
            )
        )


        alternating_patch_rms = (
            tensor_rms(
                alt_stages_temp[
                    "01_patch_embed"
                ]
            )
        )


    patch_to_position_ratio = (
        real_patch_rms
        / (
            position_rms
            + 1e-12
        )
    )


    print(
        "\n--- INPUT / POSITION SCALE ---"
    )

    print(
        "Patch Conv weight norm:",
        f"{patch_weight_norm:.8f}"
    )

    print(
        "Patch Conv weight RMS:",
        f"{patch_weight_rms:.8f}"
    )

    print(
        "Position embedding RMS:",
        f"{position_rms:.8f}"
    )

    print(
        "Real-CGM patch-output RMS:",
        f"{real_patch_rms:.8f}"
    )

    print(
        "Alternating ±5 patch-output RMS:",
        f"{alternating_patch_rms:.8f}"
    )

    print(
        "Real patch / position RMS ratio:",
        f"{patch_to_position_ratio:.8e}"
    )


    # --------------------------------------------------------
    # Extract every stage for all 3 signals
    # --------------------------------------------------------

    signal_stages = {}


    with torch.no_grad():

        for signal_name, signal in signals.items():

            (
                _,
                stages,
                _,
            ) = (
                layerwise_forward(
                    model,
                    signal,
                    requires_grad=False,
                )
            )


            signal_stages[
                signal_name
            ] = {
                stage_name:
                    flatten_cpu(
                        value
                    )

                for stage_name, value
                in stages.items()
            }


    stage_names = list(
        signal_stages[
            "real_cgm"
        ].keys()
    )


    # --------------------------------------------------------
    # Compute stage-wise statistics
    # --------------------------------------------------------

    stage_results = {}


    print(
        "\n"
        + "=" * 145
    )

    print(
        "LAYER-BY-LAYER REPRESENTATION SENSITIVITY"
    )

    print(
        "=" * 145
    )


    print(
        f"{'Stage':29s} | "
        f"{'Real RMS':>10s} | "
        f"{'3-input SD':>11s} | "
        f"{'cos R/0':>10s} | "
        f"{'L2 R/0':>10s} | "
        f"{'cos R/Alt':>10s} | "
        f"{'L2 R/Alt':>11s}"
    )


    print(
        "-" * 145
    )


    for stage_name in stage_names:

        vectors = {
            signal_name:
                signal_stages[
                    signal_name
                ][
                    stage_name
                ]

            for signal_name
            in signals.keys()
        }


        real_vector = (
            vectors[
                "real_cgm"
            ]
        )


        zero_vector = (
            vectors[
                "zeros"
            ]
        )


        alt_vector = (
            vectors[
                "alternating_plusminus5"
            ]
        )


        real_zero = (
            compare_vectors(
                real_vector,
                zero_vector,
            )
        )


        real_alt = (
            compare_vectors(
                real_vector,
                alt_vector,
            )
        )


        variation = (
            three_signal_variation(
                vectors
            )
        )


        real_rms = float(
            torch.sqrt(
                torch.mean(
                    real_vector
                    * real_vector
                )
            ).item()
        )


        stage_results[
            stage_name
        ] = {
            "real_rms":
                real_rms,

            "variation":
                variation,

            "real_vs_zero":
                real_zero,

            "real_vs_alternating":
                real_alt,
        }


        print(
            f"{stage_name:29s} | "
            f"{real_rms:10.6f} | "
            f"{variation['mean_feature_std']:11.8f} | "
            f"{real_zero['cosine']:10.7f} | "
            f"{real_zero['l2_per_sqrt_dim']:10.7f} | "
            f"{real_alt['cosine']:10.7f} | "
            f"{real_alt['l2_per_sqrt_dim']:11.7f}"
        )


    # --------------------------------------------------------
    # Gradient sensitivity
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 100
    )

    print(
        "LOCAL INPUT-GRADIENT SENSITIVITY — REAL CGM"
    )

    print(
        "=" * 100
    )


    gradient_results = (
        gradient_sensitivity(
            model,
            real_signal,
        )
    )


    print(
        f"{'Stage':29s} | "
        f"{'Input grad norm':>18s} | "
        f"{'Input grad RMS':>18s}"
    )


    print(
        "-" * 75
    )


    for stage_name in stage_names:

        result = (
            gradient_results[
                stage_name
            ]
        )


        print(
            f"{stage_name:29s} | "
            f"{result['input_gradient_norm']:18.8e} | "
            f"{result['input_gradient_rms']:18.8e}"
        )


    model_result = {
        "label":
            label,

        "manual_forward_max_abs_difference":
            manual_difference,

        "scale_diagnostics":
            {
                "patch_weight_norm":
                    patch_weight_norm,

                "patch_weight_rms":
                    patch_weight_rms,

                "position_embedding_rms":
                    position_rms,

                "real_patch_output_rms":
                    real_patch_rms,

                "alternating_patch_output_rms":
                    alternating_patch_rms,

                "real_patch_to_position_rms_ratio":
                    patch_to_position_ratio,
            },

        "stages":
            stage_results,

        "gradient_sensitivity":
            gradient_results,
    }


    del model

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return (
        model_result
    )


# ============================================================
# TSQA
# ============================================================

tsqa_result = (
    run_model(
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
    run_model(
        repo_id=
            STAGE5_REPO_ID,

        label=
            "ORIGINAL STAGE-5 ECG",

        enable_lora=True,
    )
)


# ============================================================
# Compact comparison
# ============================================================

print(
    "\n\n"
    + "=" * 145
)

print(
    "TSQA vs STAGE-5 — WHERE DOES VARIATION DISAPPEAR?"
)

print(
    "=" * 145
)


print(
    f"{'Stage':29s} | "
    f"{'TSQA SD':>11s} | "
    f"{'S5 SD':>11s} | "
    f"{'TSQA cos R/0':>13s} | "
    f"{'S5 cos R/0':>12s} | "
    f"{'TSQA cos R/A':>13s} | "
    f"{'S5 cos R/A':>12s}"
)


print(
    "-" * 145
)


common_stages = [
    stage_name
    for stage_name
    in tsqa_result[
        "stages"
    ].keys()
    if stage_name
    in stage5_result[
        "stages"
    ]
]


for stage_name in common_stages:

    tsqa = (
        tsqa_result[
            "stages"
        ][
            stage_name
        ]
    )


    stage5 = (
        stage5_result[
            "stages"
        ][
            stage_name
        ]
    )


    print(
        f"{stage_name:29s} | "
        f"{tsqa['variation']['mean_feature_std']:11.8f} | "
        f"{stage5['variation']['mean_feature_std']:11.8f} | "
        f"{tsqa['real_vs_zero']['cosine']:13.8f} | "
        f"{stage5['real_vs_zero']['cosine']:12.8f} | "
        f"{tsqa['real_vs_alternating']['cosine']:13.8f} | "
        f"{stage5['real_vs_alternating']['cosine']:12.8f}"
    )


# ============================================================
# Scale comparison
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "PATCH-EMBEDDING / POSITION SCALE COMPARISON"
)

print(
    "=" * 100
)


print(
    f"{'Metric':40s} | "
    f"{'TSQA':>18s} | "
    f"{'Stage-5':>18s}"
)


print(
    "-" * 85
)


scale_keys = [
    (
        "Patch Conv weight RMS",
        "patch_weight_rms",
    ),

    (
        "Position embedding RMS",
        "position_embedding_rms",
    ),

    (
        "Real CGM patch-output RMS",
        "real_patch_output_rms",
    ),

    (
        "Alternating ±5 patch-output RMS",
        "alternating_patch_output_rms",
    ),

    (
        "Real patch / position RMS ratio",
        "real_patch_to_position_rms_ratio",
    ),
]


for label, key in scale_keys:

    print(
        f"{label:40s} | "
        f"{tsqa_result['scale_diagnostics'][key]:18.8e} | "
        f"{stage5_result['scale_diagnostics'][key]:18.8e}"
    )


# ============================================================
# Gradient comparison
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "INPUT-GRADIENT SENSITIVITY COMPARISON"
)

print(
    "=" * 120
)


print(
    f"{'Stage':29s} | "
    f"{'TSQA grad RMS':>18s} | "
    f"{'Stage-5 grad RMS':>18s} | "
    f"{'S5 / TSQA':>14s}"
)


print(
    "-" * 90
)


for stage_name in common_stages:

    tsqa_grad = (
        tsqa_result[
            "gradient_sensitivity"
        ][
            stage_name
        ][
            "input_gradient_rms"
        ]
    )


    stage5_grad = (
        stage5_result[
            "gradient_sensitivity"
        ][
            stage_name
        ][
            "input_gradient_rms"
        ]
    )


    ratio = (
        stage5_grad
        / (
            tsqa_grad
            + 1e-30
        )
    )


    print(
        f"{stage_name:29s} | "
        f"{tsqa_grad:18.8e} | "
        f"{stage5_grad:18.8e} | "
        f"{ratio:14.6e}"
    )


# ============================================================
# Automatic localization hint
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "AUTOMATIC LOCALIZATION HINT"
)

print(
    "=" * 100
)


patch_stage = (
    "01_patch_embed"
)


tsqa_patch_sd = (
    tsqa_result[
        "stages"
    ][
        patch_stage
    ][
        "variation"
    ][
        "mean_feature_std"
    ]
)


stage5_patch_sd = (
    stage5_result[
        "stages"
    ][
        patch_stage
    ][
        "variation"
    ][
        "mean_feature_std"
    ]
)


patch_ratio = (
    stage5_patch_sd
    / (
        tsqa_patch_sd
        + 1e-30
    )
)


print(
    "TSQA patch variation:",
    f"{tsqa_patch_sd:.8e}"
)

print(
    "Stage-5 patch variation:",
    f"{stage5_patch_sd:.8e}"
)

print(
    "Stage-5 / TSQA patch variation:",
    f"{patch_ratio:.8e}"
)


if patch_ratio < 1e-3:

    print(
        "\nStage-5 variation is already "
        "dramatically suppressed at the "
        "Conv1D patch embedding."
    )

    print(
        "→ The first input projection is the "
        "leading place to investigate."
    )


else:

    # Find first stage where S5 becomes dramatically
    # less variable relative to TSQA.
    collapse_stage = None


    for stage_name in common_stages:

        tsqa_sd = (
            tsqa_result[
                "stages"
            ][
                stage_name
            ][
                "variation"
            ][
                "mean_feature_std"
            ]
        )


        stage5_sd = (
            stage5_result[
                "stages"
            ][
                stage_name
            ][
                "variation"
            ][
                "mean_feature_std"
            ]
        )


        ratio = (
            stage5_sd
            / (
                tsqa_sd
                + 1e-30
            )
        )


        if ratio < 1e-3:

            collapse_stage = (
                stage_name
            )

            break


    if collapse_stage is None:

        print(
            "\nNo stage crossed the "
            "1e-3 variation-ratio threshold."
        )

        print(
            "→ Inspect the gradual layer-by-layer "
            "trend manually."
        )


    else:

        print(
            "\nFirst dramatic suppression detected at:"
        )

        print(
            collapse_stage
        )


# ============================================================
# Save
# ============================================================

results = {
    "experiment":
        "stage5_layerwise_collapse",

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

    "real_example":
        {
            "participant_id":
                real_example.get(
                    "_participant_id"
                ),

            "window_start":
                real_example.get(
                    "_window_start"
                ),

            "gold":
                real_example[
                    "_answer_clean"
                ],

            "signal_stats":
                signal_stats(
                    real_signal
                ),
        },

    "input_signals":
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