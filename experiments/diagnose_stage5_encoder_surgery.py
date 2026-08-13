"""
Stage-5 encoder surgery diagnostic.

Question
--------
The Stage-5 ECG encoder preserves signal variation through
Transformer blocks 1-4, then sharply collapses it in block 5
and almost completely removes it in block 6.

Can we restore CGM sensitivity by surgically changing only
those late encoder blocks?

NO TRAINING is performed.

Conditions
----------
1. Original Stage-5
2. TSQA Transformer 5 transplanted into Stage-5
3. TSQA Transformer 6 transplanted into Stage-5
4. TSQA Transformer 5 + 6 transplanted into Stage-5
5. Stage-5 with Transformer 5 + 6 skipped
6. Full TSQA encoder + Stage-5 projector
7. Full TSQA encoder + TSQA projector

We evaluate each condition on:

A. Stress-test inputs
   - real CGM
   - all zeros
   - alternating +/-5

B. Twenty real above_180 windows
   - 10 YES
   - 10 NO

Metrics
-------
- encoder feature variation
- projector feature variation
- real-vs-zero / real-vs-alternating cosine
- YES/NO centroid cosine
- same-class vs opposite-class cosine
- normalized L2 distances

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
    / "stage5_encoder_surgery.json"
)

TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

N_PER_CLASS = 10


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)


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
    f"Normalization mean: {train_mean:.3f}"
)

print(
    f"Normalization SD: {train_std:.3f}"
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
# Helpers
# ============================================================

def clone_state_dict(
    module,
):
    """
    Make a detached CPU clone so subsequent mutations to the
    live model cannot modify our saved original weights.
    """

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in module.state_dict().items()
    }


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
        signal.detach()
        .clone()
        .float()
    )


    if (
        signal.dim() == 2
        and signal.shape[-1] == 1
    ):

        signal = signal.squeeze(-1)


    if (
        signal.dim() == 2
        and signal.shape[0] == 1
    ):

        signal = signal.squeeze(0)


    if signal.dim() != 1:

        raise RuntimeError(
            "Expected one-dimensional signal; "
            f"got {tuple(signal.shape)}"
        )


    return signal


def prepare_signal(
    model,
    signal,
):

    x = (
        signal.detach()
        .clone()
        .float()
        .to(device)
        .unsqueeze(0)
    )


    patch_size = (
        model.encoder.patch_size
    )


    remainder = (
        x.shape[1]
        % patch_size
    )


    if remainder:

        pad_length = (
            patch_size
            - remainder
        )


        x = F.pad(
            x,
            (
                0,
                pad_length,
            ),
        )


    return x


def flatten_cpu(
    tensor,
):

    return (
        tensor.detach()
        .float()
        .cpu()
        .reshape(-1)
    )


# ============================================================
# Manual encoder forward
#
# Allows us to skip T5/T6 while keeping everything else
# exactly the same.
# ============================================================

def manual_encoder_forward(
    model,
    signal,
    layers_to_run=None,
):

    encoder = (
        model.encoder
    )


    if layers_to_run is None:

        layers_to_run = list(
            range(
                len(
                    encoder.encoder.layers
                )
            )
        )


    x = (
        prepare_signal(
            model,
            signal,
        )
    )


    # [B, L] -> [B, 1, L]
    x = x.unsqueeze(1)


    # Conv patch embedding
    x = (
        encoder.patch_embed(
            x
        )
    )


    # [B, D, patches]
    # ->
    # [B, patches, D]
    x = x.transpose(
        1,
        2,
    )


    n_patches = (
        x.shape[1]
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


    x = (
        encoder.input_norm(
            x
        )
    )


    x = (
        encoder.input_dropout(
            x
        )
    )


    for layer_index in layers_to_run:

        x = (
            encoder.encoder.layers[
                layer_index
            ](
                x
            )
        )


    final_norm = getattr(
        encoder.encoder,
        "norm",
        None,
    )


    if final_norm is not None:

        x = final_norm(x)


    return x


# ============================================================
# Sanity check:
# manual six-layer forward must reproduce model.encoder()
# ============================================================

def validate_manual_forward(
    model,
    signal,
):

    x = (
        prepare_signal(
            model,
            signal,
        )
    )


    with torch.no_grad():

        direct = (
            model.encoder(
                x
            )
        )


        manual = (
            manual_encoder_forward(
                model,
                signal,
                layers_to_run=
                    list(
                        range(
                            len(
                                model.encoder
                                .encoder
                                .layers
                            )
                        )
                    ),
            )
        )


    return float(
        torch.max(
            torch.abs(
                direct
                - manual
            )
        )
        .cpu()
        .item()
    )


# ============================================================
# One signal through encoder + projector
# ============================================================

def encode_signal(
    model,
    signal,
    layers_to_run=None,
):

    model.encoder.eval()
    model.projector.eval()


    with torch.no_grad():

        encoded = (
            manual_encoder_forward(
                model,
                signal,
                layers_to_run=
                    layers_to_run,
            )
        )


        projected = (
            model.projector(
                encoded
            )
        )


    return (
        flatten_cpu(
            encoded
        ),

        flatten_cpu(
            projected
        ),
    )


# ============================================================
# Vector comparisons
# ============================================================

def compare_vectors(
    a,
    b,
):

    a = a.float()
    b = b.float()


    cosine = float(
        F.cosine_similarity(
            a.unsqueeze(0),
            b.unsqueeze(0),
        ).item()
    )


    l2 = float(
        torch.norm(
            a - b
        ).item()
    )


    l2_per_sqrt_dim = (
        l2
        / math.sqrt(
            a.numel()
        )
    )


    relative_l2 = (
        l2
        / (
            float(
                torch.norm(
                    a
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
            l2_per_sqrt_dim,

        "relative_l2":
            relative_l2,
    }


# ============================================================
# Variation across arbitrary vectors
# ============================================================

def vector_variation(
    vectors,
):

    matrix = torch.stack(
        vectors
    ).float()


    mean_feature_std = float(
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
        matrix.shape[0]
    ):

        for j in range(
            i + 1,
            matrix.shape[0],
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

        "rms_feature_std":
            rms_feature_std,

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
# Class-separation metrics on 10 YES + 10 NO
# ============================================================

def class_representation_stats(
    vectors,
    labels,
):

    matrix = torch.stack(
        vectors
    ).float()


    normalized = F.normalize(
        matrix,
        dim=1,
    )


    cosine_matrix = (
        normalized
        @ normalized.T
    )


    same_cosines = []
    opposite_cosines = []

    same_l2 = []
    opposite_l2 = []


    for i in range(
        len(labels)
    ):

        for j in range(
            i + 1,
            len(labels),
        ):

            cosine = float(
                cosine_matrix[
                    i,
                    j
                ].item()
            )


            distance = float(
                torch.norm(
                    matrix[i]
                    - matrix[j]
                ).item()
                / math.sqrt(
                    matrix.shape[1]
                )
            )


            if labels[i] == labels[j]:

                same_cosines.append(
                    cosine
                )

                same_l2.append(
                    distance
                )

            else:

                opposite_cosines.append(
                    cosine
                )

                opposite_l2.append(
                    distance
                )


    yes_indices = [
        i
        for i, label
        in enumerate(labels)
        if label == "(a)"
    ]


    no_indices = [
        i
        for i, label
        in enumerate(labels)
        if label == "(b)"
    ]


    yes_centroid = (
        matrix[
            yes_indices
        ]
        .mean(
            dim=0
        )
    )


    no_centroid = (
        matrix[
            no_indices
        ]
        .mean(
            dim=0
        )
    )


    centroid_cosine = float(
        F.cosine_similarity(
            yes_centroid.unsqueeze(0),
            no_centroid.unsqueeze(0),
        ).item()
    )


    centroid_l2 = float(
        torch.norm(
            yes_centroid
            - no_centroid
        ).item()
        / math.sqrt(
            matrix.shape[1]
        )
    )


    return {
        "dimension":
            int(
                matrix.shape[1]
            ),

        "mean_feature_std":
            float(
                matrix.std(
                    dim=0
                )
                .mean()
                .item()
            ),

        "same_class_cosine":
            sum(
                same_cosines
            )
            / len(
                same_cosines
            ),

        "opposite_class_cosine":
            sum(
                opposite_cosines
            )
            / len(
                opposite_cosines
            ),

        "yes_no_centroid_cosine":
            centroid_cosine,

        "same_class_l2_per_sqrt_dim":
            sum(
                same_l2
            )
            / len(
                same_l2
            ),

        "opposite_class_l2_per_sqrt_dim":
            sum(
                opposite_l2
            )
            / len(
                opposite_l2
            ),

        "yes_no_centroid_l2_per_sqrt_dim":
            centroid_l2,
    }


# ============================================================
# Load TSQA first
#
# We use it for:
# 1. EOS token / data construction
# 2. donor weights
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING TSQA DONOR MODEL"
)

print(
    "=" * 100
)


tsqa_model = (
    OpenTSLM.load_pretrained(
        TSQA_REPO_ID,
        device=device,
    )
)


EOS_TOKEN = (
    tsqa_model.get_eos_token()
)


# ============================================================
# Verify layer count
# ============================================================

tsqa_layers = (
    tsqa_model
    .encoder
    .encoder
    .layers
)


print(
    "\nTSQA transformer layers:",
    len(tsqa_layers),
)


if len(tsqa_layers) != 6:

    raise RuntimeError(
        "Expected six TSQA transformer layers."
    )


# ============================================================
# Save TSQA donor weights
#
# Python indices:
# layer 5 -> index 4
# layer 6 -> index 5
# ============================================================

tsqa_t5_state = (
    clone_state_dict(
        tsqa_layers[4]
    )
)


tsqa_t6_state = (
    clone_state_dict(
        tsqa_layers[5]
    )
)


tsqa_encoder_state = (
    clone_state_dict(
        tsqa_model.encoder
    )
)


tsqa_projector_state = (
    clone_state_dict(
        tsqa_model.projector
    )
)


# ============================================================
# Build CGM examples
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


print(
    "\nAvailable YES:",
    len(yes_examples),
)

print(
    "Available NO:",
    len(no_examples),
)


# ============================================================
# Exact deterministic 20-example selection
# ============================================================

selection_rng = random.Random(
    SEED + 777
)


selected_yes = (
    selection_rng.sample(
        yes_examples,
        N_PER_CLASS,
    )
)


selected_no = (
    selection_rng.sample(
        no_examples,
        N_PER_CLASS,
    )
)


diagnostic_examples = (
    selected_yes
    + selected_no
)


diagnostic_labels = [
    x[
        "_answer_clean"
    ]
    for x in diagnostic_examples
]


print(
    "\nReal-CGM diagnostic set:"
)

print(
    "YES:",
    len(selected_yes),
)

print(
    "NO:",
    len(selected_no),
)

print(
    "TOTAL:",
    len(diagnostic_examples),
)


# ============================================================
# Stress-test signals
# ============================================================

real_signal = (
    get_signal(
        selected_yes[0]
    )
)


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


stress_signals = {
    "real_cgm":
        real_signal,

    "zeros":
        zero_signal,

    "alternating_plusminus5":
        alternating_signal,
}


# ============================================================
# Release TSQA donor model
# ============================================================

del tsqa_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Load Stage 5 once
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING STAGE-5 BASE MODEL"
)

print(
    "=" * 100
)


stage5_model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


stage5_layers = (
    stage5_model
    .encoder
    .encoder
    .layers
)


print(
    "\nStage-5 transformer layers:",
    len(stage5_layers),
)


if len(stage5_layers) != 6:

    raise RuntimeError(
        "Expected six Stage-5 transformer layers."
    )


# ============================================================
# Architecture compatibility checks
# ============================================================

print(
    "\nChecking TSQA -> Stage-5 donor compatibility..."
)


try:

    stage5_layers[
        4
    ].load_state_dict(
        tsqa_t5_state,
        strict=True,
    )


    stage5_layers[
        5
    ].load_state_dict(
        tsqa_t6_state,
        strict=True,
    )


except Exception as exc:

    raise RuntimeError(
        "TSQA T5/T6 weights are not directly "
        "compatible with Stage-5 architecture."
    ) from exc


print(
    "✓ TSQA T5/T6 states are structurally compatible."
)


# ============================================================
# Save pristine Stage-5 state AFTER compatibility check
#
# Important: reload original model first because we just
# temporarily placed TSQA weights in T5/T6.
# ============================================================

del stage5_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


stage5_model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


stage5_encoder_original = (
    clone_state_dict(
        stage5_model.encoder
    )
)


stage5_projector_original = (
    clone_state_dict(
        stage5_model.projector
    )
)


# ============================================================
# Manual forward sanity check
# ============================================================

manual_difference = (
    validate_manual_forward(
        stage5_model,
        real_signal,
    )
)


print(
    "\nManual-vs-direct Stage-5 encoder "
    "max abs difference:",
    f"{manual_difference:.10e}"
)


if manual_difference > 1e-4:

    raise RuntimeError(
        "Manual encoder forward does not "
        "reproduce Stage-5 encoder."
    )


print(
    "✓ Manual surgery path matches original encoder."
)


# ============================================================
# Restore pristine Stage-5
# ============================================================

def restore_stage5():

    stage5_model.encoder.load_state_dict(
        stage5_encoder_original,
        strict=True,
    )


    stage5_model.projector.load_state_dict(
        stage5_projector_original,
        strict=True,
    )


    stage5_model.encoder.eval()
    stage5_model.projector.eval()


# ============================================================
# Apply condition
# ============================================================

def configure_condition(
    condition_name,
):

    restore_stage5()


    layers_to_run = [
        0,
        1,
        2,
        3,
        4,
        5,
    ]


    if condition_name == "stage5_original":

        pass


    elif condition_name == "tsqa_t5_only":

        stage5_model.encoder.encoder.layers[
            4
        ].load_state_dict(
            tsqa_t5_state,
            strict=True,
        )


    elif condition_name == "tsqa_t6_only":

        stage5_model.encoder.encoder.layers[
            5
        ].load_state_dict(
            tsqa_t6_state,
            strict=True,
        )


    elif condition_name == "tsqa_t5_t6":

        stage5_model.encoder.encoder.layers[
            4
        ].load_state_dict(
            tsqa_t5_state,
            strict=True,
        )


        stage5_model.encoder.encoder.layers[
            5
        ].load_state_dict(
            tsqa_t6_state,
            strict=True,
        )


    elif condition_name == "skip_t5_t6":

        # Run only transformer layers 1-4.
        layers_to_run = [
            0,
            1,
            2,
            3,
        ]


    elif condition_name == "full_tsqa_encoder":

        stage5_model.encoder.load_state_dict(
            tsqa_encoder_state,
            strict=True,
        )


    elif condition_name == "full_tsqa_encoder_projector":

        stage5_model.encoder.load_state_dict(
            tsqa_encoder_state,
            strict=True,
        )


        stage5_model.projector.load_state_dict(
            tsqa_projector_state,
            strict=True,
        )


    else:

        raise ValueError(
            f"Unknown condition: {condition_name}"
        )


    stage5_model.encoder.eval()
    stage5_model.projector.eval()


    return layers_to_run


# ============================================================
# Evaluate one surgery condition
# ============================================================

def evaluate_condition(
    condition_name,
    display_name,
):

    print(
        "\n\n"
        + "#" * 110
    )

    print(
        display_name
    )

    print(
        "#" * 110
    )


    layers_to_run = (
        configure_condition(
            condition_name
        )
    )


    # --------------------------------------------------------
    # Stress signals
    # --------------------------------------------------------

    stress_encoder_vectors = {}
    stress_projector_vectors = {}


    for signal_name, signal in stress_signals.items():

        (
            encoder_vector,
            projector_vector,
        ) = (
            encode_signal(
                stage5_model,
                signal,
                layers_to_run=
                    layers_to_run,
            )
        )


        stress_encoder_vectors[
            signal_name
        ] = encoder_vector


        stress_projector_vectors[
            signal_name
        ] = projector_vector


    stress_encoder_variation = (
        vector_variation(
            list(
                stress_encoder_vectors.values()
            )
        )
    )


    stress_projector_variation = (
        vector_variation(
            list(
                stress_projector_vectors.values()
            )
        )
    )


    enc_real_zero = (
        compare_vectors(
            stress_encoder_vectors[
                "real_cgm"
            ],
            stress_encoder_vectors[
                "zeros"
            ],
        )
    )


    enc_real_alt = (
        compare_vectors(
            stress_encoder_vectors[
                "real_cgm"
            ],
            stress_encoder_vectors[
                "alternating_plusminus5"
            ],
        )
    )


    proj_real_zero = (
        compare_vectors(
            stress_projector_vectors[
                "real_cgm"
            ],
            stress_projector_vectors[
                "zeros"
            ],
        )
    )


    proj_real_alt = (
        compare_vectors(
            stress_projector_vectors[
                "real_cgm"
            ],
            stress_projector_vectors[
                "alternating_plusminus5"
            ],
        )
    )


    print(
        "\n--- STRESS TEST ---"
    )

    print(
        "Encoder feature SD:",
        f"{stress_encoder_variation['mean_feature_std']:.8f}"
    )

    print(
        "Encoder real/zero cosine:",
        f"{enc_real_zero['cosine']:.8f}"
    )

    print(
        "Encoder real/alternating cosine:",
        f"{enc_real_alt['cosine']:.8f}"
    )

    print(
        "Encoder real/alternating L2/sqrt(dim):",
        f"{enc_real_alt['l2_per_sqrt_dim']:.8f}"
    )


    print(
        "\nProjector feature SD:",
        f"{stress_projector_variation['mean_feature_std']:.8f}"
    )

    print(
        "Projector real/zero cosine:",
        f"{proj_real_zero['cosine']:.8f}"
    )

    print(
        "Projector real/alternating cosine:",
        f"{proj_real_alt['cosine']:.8f}"
    )

    print(
        "Projector real/alternating L2/sqrt(dim):",
        f"{proj_real_alt['l2_per_sqrt_dim']:.8f}"
    )


    # --------------------------------------------------------
    # Twenty real CGM examples
    # --------------------------------------------------------

    real_encoder_vectors = []
    real_projector_vectors = []


    for example in diagnostic_examples:

        signal = (
            get_signal(
                example
            )
        )


        (
            encoder_vector,
            projector_vector,
        ) = (
            encode_signal(
                stage5_model,
                signal,
                layers_to_run=
                    layers_to_run,
            )
        )


        real_encoder_vectors.append(
            encoder_vector
        )


        real_projector_vectors.append(
            projector_vector
        )


    encoder_class_stats = (
        class_representation_stats(
            real_encoder_vectors,
            diagnostic_labels,
        )
    )


    projector_class_stats = (
        class_representation_stats(
            real_projector_vectors,
            diagnostic_labels,
        )
    )


    print(
        "\n--- 20 REAL CGM WINDOWS: ENCODER ---"
    )

    print(
        "Mean feature SD:",
        f"{encoder_class_stats['mean_feature_std']:.8f}"
    )

    print(
        "Same-class cosine:",
        f"{encoder_class_stats['same_class_cosine']:.8f}"
    )

    print(
        "Opposite-class cosine:",
        f"{encoder_class_stats['opposite_class_cosine']:.8f}"
    )

    print(
        "YES/NO centroid cosine:",
        f"{encoder_class_stats['yes_no_centroid_cosine']:.8f}"
    )

    print(
        "YES/NO centroid L2/sqrt(dim):",
        f"{encoder_class_stats['yes_no_centroid_l2_per_sqrt_dim']:.8f}"
    )


    print(
        "\n--- 20 REAL CGM WINDOWS: PROJECTOR ---"
    )

    print(
        "Mean feature SD:",
        f"{projector_class_stats['mean_feature_std']:.8f}"
    )

    print(
        "Same-class cosine:",
        f"{projector_class_stats['same_class_cosine']:.8f}"
    )

    print(
        "Opposite-class cosine:",
        f"{projector_class_stats['opposite_class_cosine']:.8f}"
    )

    print(
        "YES/NO centroid cosine:",
        f"{projector_class_stats['yes_no_centroid_cosine']:.8f}"
    )

    print(
        "YES/NO centroid L2/sqrt(dim):",
        f"{projector_class_stats['yes_no_centroid_l2_per_sqrt_dim']:.8f}"
    )


    return {
        "condition":
            condition_name,

        "display_name":
            display_name,

        "layers_to_run":
            layers_to_run,

        "stress_test":
            {
                "encoder_variation":
                    stress_encoder_variation,

                "projector_variation":
                    stress_projector_variation,

                "encoder_real_vs_zero":
                    enc_real_zero,

                "encoder_real_vs_alternating":
                    enc_real_alt,

                "projector_real_vs_zero":
                    proj_real_zero,

                "projector_real_vs_alternating":
                    proj_real_alt,
            },

        "real_cgm_20":
            {
                "encoder":
                    encoder_class_stats,

                "projector":
                    projector_class_stats,
            },
    }


# ============================================================
# Conditions
# ============================================================

conditions = [
    (
        "stage5_original",
        "1 — ORIGINAL STAGE-5",
    ),

    (
        "tsqa_t5_only",
        "2 — STAGE-5 + TSQA TRANSFORMER 5",
    ),

    (
        "tsqa_t6_only",
        "3 — STAGE-5 + TSQA TRANSFORMER 6",
    ),

    (
        "tsqa_t5_t6",
        "4 — STAGE-5 + TSQA TRANSFORMERS 5 AND 6",
    ),

    (
        "skip_t5_t6",
        "5 — STAGE-5 SKIPPING TRANSFORMERS 5 AND 6",
    ),

    (
        "full_tsqa_encoder",
        "6 — FULL TSQA ENCODER + STAGE-5 PROJECTOR",
    ),

    (
        "full_tsqa_encoder_projector",
        "7 — FULL TSQA ENCODER + TSQA PROJECTOR",
    ),
]


results = {}


for (
    condition_name,
    display_name,
) in conditions:

    results[
        condition_name
    ] = (
        evaluate_condition(
            condition_name,
            display_name,
        )
    )


# ============================================================
# Compact summary
# ============================================================

print(
    "\n\n"
    + "=" * 145
)

print(
    "STAGE-5 ENCODER SURGERY SUMMARY"
)

print(
    "=" * 145
)


print(
    f"{'Condition':32s} | "
    f"{'Stress Enc SD':>13s} | "
    f"{'Stress Proj SD':>14s} | "
    f"{'Real20 Enc SD':>13s} | "
    f"{'Real20 Proj SD':>14s} | "
    f"{'Enc Y/N cos':>11s} | "
    f"{'Proj Y/N cos':>12s}"
)


print(
    "-" * 145
)


for condition_name, _ in conditions:

    result = (
        results[
            condition_name
        ]
    )


    print(
        f"{condition_name:32s} | "
        f"{result['stress_test']['encoder_variation']['mean_feature_std']:13.8f} | "
        f"{result['stress_test']['projector_variation']['mean_feature_std']:14.8f} | "
        f"{result['real_cgm_20']['encoder']['mean_feature_std']:13.8f} | "
        f"{result['real_cgm_20']['projector']['mean_feature_std']:14.8f} | "
        f"{result['real_cgm_20']['encoder']['yes_no_centroid_cosine']:11.8f} | "
        f"{result['real_cgm_20']['projector']['yes_no_centroid_cosine']:12.8f}"
    )


# ============================================================
# Relative recovery vs original Stage-5
# ============================================================

original = (
    results[
        "stage5_original"
    ]
)


original_enc_sd = (
    original[
        "real_cgm_20"
    ][
        "encoder"
    ][
        "mean_feature_std"
    ]
)


original_proj_sd = (
    original[
        "real_cgm_20"
    ][
        "projector"
    ][
        "mean_feature_std"
    ]
)


print(
    "\n"
    + "=" * 120
)

print(
    "REAL-CGM VARIATION RECOVERY RELATIVE TO ORIGINAL STAGE-5"
)

print(
    "=" * 120
)


print(
    f"{'Condition':32s} | "
    f"{'Encoder x original':>20s} | "
    f"{'Projector x original':>22s}"
)


print(
    "-" * 85
)


for condition_name, _ in conditions:

    result = (
        results[
            condition_name
        ]
    )


    enc_sd = (
        result[
            "real_cgm_20"
        ][
            "encoder"
        ][
            "mean_feature_std"
        ]
    )


    proj_sd = (
        result[
            "real_cgm_20"
        ][
            "projector"
        ][
            "mean_feature_std"
        ]
    )


    enc_ratio = (
        enc_sd
        / (
            original_enc_sd
            + 1e-30
        )
    )


    proj_ratio = (
        proj_sd
        / (
            original_proj_sd
            + 1e-30
        )
    )


    print(
        f"{condition_name:32s} | "
        f"{enc_ratio:20.2f} | "
        f"{proj_ratio:22.2f}"
    )


# ============================================================
# Targeted interpretation
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "TARGETED T5/T6 COMPARISON"
)

print(
    "=" * 100
)


for condition_name in [
    "stage5_original",
    "tsqa_t5_only",
    "tsqa_t6_only",
    "tsqa_t5_t6",
    "skip_t5_t6",
]:

    result = (
        results[
            condition_name
        ]
    )


    print(
        f"{condition_name:24s} | "
        f"Enc SD="
        f"{result['real_cgm_20']['encoder']['mean_feature_std']:.8f} | "
        f"Proj SD="
        f"{result['real_cgm_20']['projector']['mean_feature_std']:.8f} | "
        f"Enc Y/N cosine="
        f"{result['real_cgm_20']['encoder']['yes_no_centroid_cosine']:.8f}"
    )


# ============================================================
# Save
# ============================================================

serializable = {
    "experiment":
        "stage5_encoder_surgery",

    "seed":
        SEED,

    "task":
        TASK_ID,

    "n_yes":
        N_PER_CLASS,

    "n_no":
        N_PER_CLASS,

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

    "manual_forward_max_abs_difference":
        manual_difference,

    "selected_examples":
        [
            {
                "participant_id":
                    example.get(
                        "_participant_id"
                    ),

                "window_start":
                    example.get(
                        "_window_start"
                    ),

                "gold":
                    example[
                        "_answer_clean"
                    ],
            }

            for example
            in diagnostic_examples
        ],

    "results":
        results,
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
        serializable,
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
    "No checkpoint was modified or saved."
)

print(
    "TEST participants remain sealed."
)


# ============================================================
# Cleanup
# ============================================================

del stage5_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()