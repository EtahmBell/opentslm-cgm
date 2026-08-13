"""
TSQA vs Stage-5 CGM signal-path diagnostic.

Purpose
-------
Stage-5 could not memorize even 20 above_180 CGM examples using
our previous full-vocabulary class-token objective.

Before changing the dataset or training more, compare where the
signal path differs between:

    1. Original TSQA checkpoint
    2. CGM-aligned TSQA checkpoint (positive control, if available)
    3. Original Stage-5 ECG checkpoint

We use the EXACT SAME 20 above_180 examples from the tiny-overfit run:
    10 YES
    10 NO

NO optimization steps are performed.

For each model we measure:

A. Encoder representation sensitivity
   Do different CGM windows produce different encoder embeddings?

B. Projector representation sensitivity
   Do different CGM windows remain distinct after projection into
   the LLM hidden space?

C. LLM class-margin sensitivity
   With identical above_180 question text, how much does

       logit("(a") - logit("(b")

   change across different CGM windows?

D. Gradient pathway
   Using a TRUE two-way loss between "(a" and "(b)", do gradients
   reach the encoder and projector for a known YES and known NO
   example?

Interpretation
--------------
Stage-5 representations nearly identical:
    -> encoder/projector representation collapse or specialization.

Representations vary, but class margin is constant:
    -> LLM/soft-prompt interface is insensitive to CGM differences.

Healthy representations + healthy gradients:
    -> signal pathway probably works; optimization/loss is more likely
       the issue.

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
# Project paths
# ============================================================

ROOT = Path(
    __file__
).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.cgm_multitask import build_multitask_examples
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

TSQA_ALIGNED_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "cgm_alignment_multitask_tsqa.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "tsqa_vs_stage5_signal_path.json"
)

DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

N_PER_CLASS = 10


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
# Utility
# ============================================================

def clear_model(
    model,
):

    del model

    gc.collect()

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


# ============================================================
# Load TSQA first so we can obtain EOS and build the exact data
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "LOADING ORIGINAL TSQA"
)

print(
    "=" * 90
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
# Reconstruct exact same above_180 pool
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
# EXACT same selection as tiny-overfit script
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


print(
    "\nDiagnostic set:"
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
# Answer-token information
# ============================================================

ANSWER_A = next(
    x[
        "answer"
    ]
    for x in diagnostic_examples
    if x[
        "_answer_clean"
    ] == "(a)"
)


ANSWER_B = next(
    x[
        "answer"
    ]
    for x in diagnostic_examples
    if x[
        "_answer_clean"
    ] == "(b)"
)


def get_answer_token_info(
    model,
):

    a_ids = (
        model.tokenizer(
            ANSWER_A,
            return_tensors="pt",
            truncation=True,
        )
        .input_ids[
            0
        ]
        .to(
            device
        )
    )


    b_ids = (
        model.tokenizer(
            ANSWER_B,
            return_tensors="pt",
            truncation=True,
        )
        .input_ids[
            0
        ]
        .to(
            device
        )
    )


    prefix_len = 0


    while (
        prefix_len
        < min(
            len(a_ids),
            len(b_ids),
        )
        and a_ids[
            prefix_len
        ].item()
        == b_ids[
            prefix_len
        ].item()
    ):

        prefix_len += 1


    if prefix_len >= min(
        len(a_ids),
        len(b_ids),
    ):

        raise RuntimeError(
            "Could not find discriminating "
            "answer token."
        )


    prefix_ids = (
        a_ids[
            :prefix_len
        ]
    )


    a_token_id = int(
        a_ids[
            prefix_len
        ].item()
    )


    b_token_id = int(
        b_ids[
            prefix_len
        ].item()
    )


    return {
        "a_ids":
            a_ids,

        "b_ids":
            b_ids,

        "prefix_ids":
            prefix_ids,

        "a_token_id":
            a_token_id,

        "b_token_id":
            b_token_id,
    }


# ============================================================
# Extract raw time series from an example
# ============================================================

def get_single_timeseries(
    example,
):

    time_series = (
        example[
            "time_series"
        ]
    )


    # build_multitask_examples currently gives one
    # time-series segment per CGM example.
    if len(time_series) != 1:

        raise RuntimeError(
            "Expected one time-series segment, "
            f"found {len(time_series)}"
        )


    ts = (
        time_series[
            0
        ]
    )


    if not torch.is_tensor(
        ts
    ):

        ts = torch.tensor(
            ts,
            dtype=torch.float32,
        )


    ts = (
        ts.detach()
        .clone()
        .float()
    )


    # Handle [T, 1].
    if (
        ts.dim() == 2
        and ts.shape[
            -1
        ] == 1
    ):

        ts = (
            ts.squeeze(
                -1
            )
        )


    # Handle any accidental singleton batch dimension.
    if (
        ts.dim() == 2
        and ts.shape[
            0
        ] == 1
    ):

        ts = (
            ts.squeeze(
                0
            )
        )


    if ts.dim() != 1:

        raise RuntimeError(
            "Expected a 1D CGM signal after "
            f"squeezing, got shape {tuple(ts.shape)}"
        )


    return ts


# ============================================================
# Direct encoder/projector representation
# ============================================================

def extract_signal_representations(
    model,
    example,
):

    ts = (
        get_single_timeseries(
            example
        )
        .to(
            device
        )
    )


    x = (
        ts.unsqueeze(
            0
        )
    )


    # Match OpenTSLM pad_and_apply_batch behavior:
    # pad length to a multiple of patch_size.
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


        x = (
            F.pad(
                x,
                (
                    0,
                    pad_length,
                ),
            )
        )


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


    return (
        encoded[
            0
        ]
        .float()
        .cpu()
        .reshape(
            -1
        ),

        projected[
            0
        ]
        .float()
        .cpu()
        .reshape(
            -1
        ),
    )


# ============================================================
# Representation statistics
# ============================================================

def representation_statistics(
    vectors,
    labels,
):

    matrix = torch.stack(
        vectors
    ).float()


    normalized = (
        F.normalize(
            matrix,
            dim=1,
        )
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
        len(vectors)
    ):

        for j in range(
            i + 1,
            len(vectors),
        ):

            cosine = float(
                cosine_matrix[
                    i,
                    j
                ].item()
            )


            # Normalize L2 by sqrt(feature dimension)
            # so encoder/projector scales are easier to compare.
            l2 = float(
                torch.norm(
                    matrix[
                        i
                    ]
                    - matrix[
                        j
                    ]
                ).item()
                / math.sqrt(
                    matrix.shape[
                        1
                    ]
                )
            )


            if labels[
                i
            ] == labels[
                j
            ]:

                same_cosines.append(
                    cosine
                )

                same_l2.append(
                    l2
                )

            else:

                opposite_cosines.append(
                    cosine
                )

                opposite_l2.append(
                    l2
                )


    yes_matrix = (
        matrix[
            [
                i
                for i, label
                in enumerate(
                    labels
                )
                if label == "(a)"
            ]
        ]
    )


    no_matrix = (
        matrix[
            [
                i
                for i, label
                in enumerate(
                    labels
                )
                if label == "(b)"
            ]
        ]
    )


    yes_centroid = (
        yes_matrix.mean(
            dim=0
        )
    )


    no_centroid = (
        no_matrix.mean(
            dim=0
        )
    )


    centroid_cosine = float(
        F.cosine_similarity(
            yes_centroid.unsqueeze(
                0
            ),
            no_centroid.unsqueeze(
                0
            ),
        ).item()
    )


    centroid_l2 = float(
        torch.norm(
            yes_centroid
            - no_centroid
        ).item()
        / math.sqrt(
            matrix.shape[
                1
            ]
        )
    )


    # Average featurewise standard deviation across windows.
    mean_feature_std = float(
        matrix.std(
            dim=0
        )
        .mean()
        .item()
    )


    return {
        "dimension":
            int(
                matrix.shape[
                    1
                ]
            ),

        "mean_same_class_cosine":
            sum(
                same_cosines
            )
            / len(
                same_cosines
            ),

        "mean_opposite_class_cosine":
            sum(
                opposite_cosines
            )
            / len(
                opposite_cosines
            ),

        "mean_same_class_l2_per_sqrt_dim":
            sum(
                same_l2
            )
            / len(
                same_l2
            ),

        "mean_opposite_class_l2_per_sqrt_dim":
            sum(
                opposite_l2
            )
            / len(
                opposite_l2
            ),

        "yes_no_centroid_cosine":
            centroid_cosine,

        "yes_no_centroid_l2_per_sqrt_dim":
            centroid_l2,

        "mean_feature_std_across_windows":
            mean_feature_std,
    }


# ============================================================
# Class-decision logits
# ============================================================

def get_class_logits(
    model,
    example,
    token_info,
):

    (
        prompt_embeds,
        prompt_mask,
    ) = (
        model.pad_and_apply_batch(
            [
                example
            ]
        )
    )


    valid_prompt_length = int(
        prompt_mask[
            0
        ]
        .sum()
        .item()
    )


    prompt_embeds = (
        prompt_embeds[
            :,
            :valid_prompt_length,
            :
        ]
    )


    prompt_mask = (
        prompt_mask[
            :,
            :valid_prompt_length
        ]
    )


    prefix_ids = (
        token_info[
            "prefix_ids"
        ]
        .to(
            prompt_embeds.device
        )
    )


    if prefix_ids.numel() > 0:

        prefix_embeds = (
            model.llm
            .get_input_embeddings()(
                prefix_ids.unsqueeze(
                    0
                )
            )
        )


        prefix_embeds = (
            prefix_embeds.to(
                dtype=
                    prompt_embeds.dtype,
                device=
                    prompt_embeds.device,
            )
        )


        prefix_mask = (
            torch.ones(
                (
                    1,
                    prefix_ids.numel(),
                ),
                dtype=
                    prompt_mask.dtype,
                device=
                    prompt_mask.device,
            )
        )


        inputs_embeds = (
            torch.cat(
                [
                    prompt_embeds,
                    prefix_embeds,
                ],
                dim=1,
            )
        )


        attention_mask = (
            torch.cat(
                [
                    prompt_mask,
                    prefix_mask,
                ],
                dim=1,
            )
        )


    else:

        inputs_embeds = (
            prompt_embeds
        )

        attention_mask = (
            prompt_mask
        )


    outputs = (
        model.llm(
            inputs_embeds=
                inputs_embeds,

            attention_mask=
                attention_mask,

            return_dict=True,
        )
    )


    logits = (
        outputs.logits[
            0,
            -1,
            :
        ]
        .float()
    )


    a_logit = (
        logits[
            token_info[
                "a_token_id"
            ]
        ]
    )


    b_logit = (
        logits[
            token_info[
                "b_token_id"
            ]
        ]
    )


    return (
        a_logit,
        b_logit,
    )


# ============================================================
# Class-margin sensitivity across the 20 windows
# ============================================================

def class_margin_diagnostic(
    model,
    examples,
    token_info,
):

    model.eval()


    records = []


    with torch.no_grad():

        for example in examples:

            (
                a_logit,
                b_logit,
            ) = (
                get_class_logits(
                    model,
                    example,
                    token_info,
                )
            )


            margin = float(
                (
                    a_logit
                    - b_logit
                )
                .cpu()
                .item()
            )


            p_a = float(
                torch.sigmoid(
                    a_logit
                    - b_logit
                )
                .cpu()
                .item()
            )


            pred = (
                "(a)"
                if margin > 0
                else "(b)"
            )


            records.append(
                {
                    "gold":
                        example[
                            "_answer_clean"
                        ],

                    "prediction":
                        pred,

                    "margin_a_minus_b":
                        margin,

                    "p_a_given_a_or_b":
                        p_a,

                    "participant_id":
                        example.get(
                            "_participant_id"
                        ),

                    "window_start":
                        example.get(
                            "_window_start"
                        ),
                }
            )


    margins = torch.tensor(
        [
            x[
                "margin_a_minus_b"
            ]
            for x in records
        ],
        dtype=torch.float32,
    )


    yes_margins = torch.tensor(
        [
            x[
                "margin_a_minus_b"
            ]
            for x in records
            if x[
                "gold"
            ] == "(a)"
        ],
        dtype=torch.float32,
    )


    no_margins = torch.tensor(
        [
            x[
                "margin_a_minus_b"
            ]
            for x in records
            if x[
                "gold"
            ] == "(b)"
        ],
        dtype=torch.float32,
    )


    yes_correct = sum(
        x[
            "prediction"
        ] == "(a)"
        for x in records
        if x[
            "gold"
        ] == "(a)"
    )


    no_correct = sum(
        x[
            "prediction"
        ] == "(b)"
        for x in records
        if x[
            "gold"
        ] == "(b)"
    )


    yes_accuracy = (
        yes_correct
        / N_PER_CLASS
    )


    no_accuracy = (
        no_correct
        / N_PER_CLASS
    )


    return {
        "balanced_accuracy":
            (
                yes_accuracy
                + no_accuracy
            )
            / 2.0,

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "mean_margin":
            float(
                margins.mean().item()
            ),

        "margin_std":
            float(
                margins.std().item()
            ),

        "margin_min":
            float(
                margins.min().item()
            ),

        "margin_max":
            float(
                margins.max().item()
            ),

        "mean_yes_margin":
            float(
                yes_margins.mean().item()
            ),

        "mean_no_margin":
            float(
                no_margins.mean().item()
            ),

        "records":
            records,
    }


# ============================================================
# Gradient norm helper
# ============================================================

def parameter_gradient_norm(
    parameters,
):

    squared_sum = 0.0


    for parameter in parameters:

        if parameter.grad is None:

            continue


        gradient = (
            parameter.grad
            .detach()
            .float()
        )


        squared_sum += float(
            torch.sum(
                gradient
                * gradient
            ).item()
        )


    return math.sqrt(
        squared_sum
    )


# ============================================================
# TRUE pairwise A-vs-B gradient diagnostic
#
# Important:
# This is NOT the previous full-vocabulary class-token loss.
#
# We take only:
#
#     logit("(a")
#     logit("(b")
#
# and perform a 2-class cross entropy.
# ============================================================

def gradient_diagnostic(
    model,
    example,
    token_info,
):

    # Freeze the entire language model, including Stage-5 LoRA.
    # Gradients can still propagate THROUGH it to inputs.
    for parameter in (
        model.llm.parameters()
    ):

        parameter.requires_grad = False


    for parameter in (
        model.encoder.parameters()
    ):

        parameter.requires_grad = True


    for parameter in (
        model.projector.parameters()
    ):

        parameter.requires_grad = True


    model.encoder.eval()
    model.projector.eval()
    model.llm.eval()


    model.encoder.zero_grad(
        set_to_none=True
    )

    model.projector.zero_grad(
        set_to_none=True
    )

    model.llm.zero_grad(
        set_to_none=True
    )


    (
        a_logit,
        b_logit,
    ) = (
        get_class_logits(
            model,
            example,
            token_info,
        )
    )


    pair_logits = (
        torch.stack(
            [
                a_logit,
                b_logit,
            ]
        )
        .unsqueeze(
            0
        )
    )


    gold = (
        example[
            "_answer_clean"
        ]
    )


    target_index = (
        0
        if gold == "(a)"
        else 1
    )


    target = torch.tensor(
        [
            target_index
        ],
        dtype=torch.long,
        device=
            pair_logits.device,
    )


    pairwise_loss = (
        F.cross_entropy(
            pair_logits,
            target,
        )
    )


    pairwise_loss.backward()


    encoder_grad_norm = (
        parameter_gradient_norm(
            model.encoder.parameters()
        )
    )


    projector_grad_norm = (
        parameter_gradient_norm(
            model.projector.parameters()
        )
    )


    pair_probs = (
        torch.softmax(
            pair_logits.detach(),
            dim=-1,
        )[
            0
        ]
    )


    result = {
        "gold":
            gold,

        "pairwise_loss":
            float(
                pairwise_loss
                .detach()
                .cpu()
                .item()
            ),

        "p_a":
            float(
                pair_probs[
                    0
                ]
                .cpu()
                .item()
            ),

        "p_b":
            float(
                pair_probs[
                    1
                ]
                .cpu()
                .item()
            ),

        "encoder_gradient_norm":
            encoder_grad_norm,

        "projector_gradient_norm":
            projector_grad_norm,

        "participant_id":
            example.get(
                "_participant_id"
            ),

        "window_start":
            example.get(
                "_window_start"
            ),
    }


    # Clear gradients before next example.
    model.encoder.zero_grad(
        set_to_none=True
    )

    model.projector.zero_grad(
        set_to_none=True
    )

    model.llm.zero_grad(
        set_to_none=True
    )


    return result


# ============================================================
# Full diagnostic for one model
# ============================================================

def run_model_diagnostic(
    model,
    label,
):

    print(
        "\n\n"
        + "#" * 90
    )

    print(
        label
    )

    print(
        "#" * 90
    )


    model.eval()


    token_info = (
        get_answer_token_info(
            model
        )
    )


    print(
        "\nA token:",
        token_info[
            "a_token_id"
        ],
        model.tokenizer.convert_ids_to_tokens(
            [
                token_info[
                    "a_token_id"
                ]
            ]
        ),
    )


    print(
        "B token:",
        token_info[
            "b_token_id"
        ],
        model.tokenizer.convert_ids_to_tokens(
            [
                token_info[
                    "b_token_id"
                ]
            ]
        ),
    )


    # --------------------------------------------------------
    # Representation extraction
    # --------------------------------------------------------

    encoder_vectors = []
    projector_vectors = []

    labels = []


    for example in diagnostic_examples:

        (
            encoder_vector,
            projector_vector,
        ) = (
            extract_signal_representations(
                model,
                example,
            )
        )


        encoder_vectors.append(
            encoder_vector
        )

        projector_vectors.append(
            projector_vector
        )

        labels.append(
            example[
                "_answer_clean"
            ]
        )


    encoder_stats = (
        representation_statistics(
            encoder_vectors,
            labels,
        )
    )


    projector_stats = (
        representation_statistics(
            projector_vectors,
            labels,
        )
    )


    print(
        "\n--- ENCODER REPRESENTATIONS ---"
    )

    print(
        "Dimension:",
        encoder_stats[
            "dimension"
        ],
    )

    print(
        "Same-class cosine:",
        f"{encoder_stats['mean_same_class_cosine']:.6f}"
    )

    print(
        "Opposite-class cosine:",
        f"{encoder_stats['mean_opposite_class_cosine']:.6f}"
    )

    print(
        "YES/NO centroid cosine:",
        f"{encoder_stats['yes_no_centroid_cosine']:.6f}"
    )

    print(
        "Same-class L2/sqrt(dim):",
        f"{encoder_stats['mean_same_class_l2_per_sqrt_dim']:.6f}"
    )

    print(
        "Opposite-class L2/sqrt(dim):",
        f"{encoder_stats['mean_opposite_class_l2_per_sqrt_dim']:.6f}"
    )

    print(
        "YES/NO centroid L2/sqrt(dim):",
        f"{encoder_stats['yes_no_centroid_l2_per_sqrt_dim']:.6f}"
    )

    print(
        "Mean feature std across windows:",
        f"{encoder_stats['mean_feature_std_across_windows']:.6f}"
    )


    print(
        "\n--- PROJECTOR REPRESENTATIONS ---"
    )

    print(
        "Dimension:",
        projector_stats[
            "dimension"
        ],
    )

    print(
        "Same-class cosine:",
        f"{projector_stats['mean_same_class_cosine']:.6f}"
    )

    print(
        "Opposite-class cosine:",
        f"{projector_stats['mean_opposite_class_cosine']:.6f}"
    )

    print(
        "YES/NO centroid cosine:",
        f"{projector_stats['yes_no_centroid_cosine']:.6f}"
    )

    print(
        "Same-class L2/sqrt(dim):",
        f"{projector_stats['mean_same_class_l2_per_sqrt_dim']:.6f}"
    )

    print(
        "Opposite-class L2/sqrt(dim):",
        f"{projector_stats['mean_opposite_class_l2_per_sqrt_dim']:.6f}"
    )

    print(
        "YES/NO centroid L2/sqrt(dim):",
        f"{projector_stats['yes_no_centroid_l2_per_sqrt_dim']:.6f}"
    )

    print(
        "Mean feature std across windows:",
        f"{projector_stats['mean_feature_std_across_windows']:.6f}"
    )


    # --------------------------------------------------------
    # Class margin
    # --------------------------------------------------------

    margin_result = (
        class_margin_diagnostic(
            model,
            diagnostic_examples,
            token_info,
        )
    )


    print(
        "\n--- LLM A-vs-B MARGIN ---"
    )

    print(
        "Balanced accuracy:",
        f"{margin_result['balanced_accuracy'] * 100:.1f}%"
    )

    print(
        "YES accuracy:",
        f"{margin_result['yes_accuracy'] * 100:.1f}%"
    )

    print(
        "NO accuracy:",
        f"{margin_result['no_accuracy'] * 100:.1f}%"
    )

    print(
        "Mean margin A-B:",
        f"{margin_result['mean_margin']:.6f}"
    )

    print(
        "Margin SD across windows:",
        f"{margin_result['margin_std']:.6f}"
    )

    print(
        "Margin range:",
        f"{margin_result['margin_min']:.6f}",
        "to",
        f"{margin_result['margin_max']:.6f}"
    )

    print(
        "Mean YES margin:",
        f"{margin_result['mean_yes_margin']:.6f}"
    )

    print(
        "Mean NO margin:",
        f"{margin_result['mean_no_margin']:.6f}"
    )


    # --------------------------------------------------------
    # Gradient path on one YES and one NO
    # --------------------------------------------------------

    yes_example = (
        selected_yes[
            0
        ]
    )

    no_example = (
        selected_no[
            0
        ]
    )


    yes_gradient = (
        gradient_diagnostic(
            model,
            yes_example,
            token_info,
        )
    )


    no_gradient = (
        gradient_diagnostic(
            model,
            no_example,
            token_info,
        )
    )


    print(
        "\n--- TRUE PAIRWISE A-vs-B GRADIENT PATH ---"
    )


    print(
        "\nGold YES example:"
    )

    print(
        "  Pairwise loss:",
        f"{yes_gradient['pairwise_loss']:.6f}"
    )

    print(
        "  P(a):",
        f"{yes_gradient['p_a']:.6f}"
    )

    print(
        "  P(b):",
        f"{yes_gradient['p_b']:.6f}"
    )

    print(
        "  Encoder grad norm:",
        f"{yes_gradient['encoder_gradient_norm']:.8e}"
    )

    print(
        "  Projector grad norm:",
        f"{yes_gradient['projector_gradient_norm']:.8e}"
    )


    print(
        "\nGold NO example:"
    )

    print(
        "  Pairwise loss:",
        f"{no_gradient['pairwise_loss']:.6f}"
    )

    print(
        "  P(a):",
        f"{no_gradient['p_a']:.6f}"
    )

    print(
        "  P(b):",
        f"{no_gradient['p_b']:.6f}"
    )

    print(
        "  Encoder grad norm:",
        f"{no_gradient['encoder_gradient_norm']:.8e}"
    )

    print(
        "  Projector grad norm:",
        f"{no_gradient['projector_gradient_norm']:.8e}"
    )


    return {
        "encoder":
            encoder_stats,

        "projector":
            projector_stats,

        "class_margin":
            margin_result,

        "gradient_yes":
            yes_gradient,

        "gradient_no":
            no_gradient,
    }


# ============================================================
# 1. Original TSQA
# ============================================================

results = {}


results[
    "tsqa_original"
] = (
    run_model_diagnostic(
        tsqa_model,
        "MODEL 1 — ORIGINAL TSQA",
    )
)


# ============================================================
# 2. CGM-aligned TSQA positive control
# ============================================================

if TSQA_ALIGNED_CHECKPOINT.exists():

    print(
        "\n"
        + "=" * 90
    )

    print(
        "LOADING SUCCESSFUL CGM-ALIGNED TSQA CHECKPOINT"
    )

    print(
        "=" * 90
    )

    print(
        TSQA_ALIGNED_CHECKPOINT
    )


    tsqa_model.load_from_file(
        str(
            TSQA_ALIGNED_CHECKPOINT
        )
    )


    results[
        "tsqa_cgm_aligned"
    ] = (
        run_model_diagnostic(
            tsqa_model,
            "MODEL 2 — CGM-ALIGNED TSQA",
        )
    )


else:

    print(
        "\nWARNING: aligned TSQA checkpoint not found:"
    )

    print(
        TSQA_ALIGNED_CHECKPOINT
    )

    print(
        "Skipping aligned TSQA positive control."
    )


# ============================================================
# Remove TSQA model before loading Stage 5
# ============================================================

clear_model(
    tsqa_model
)


# ============================================================
# 3. Original Stage-5
# ============================================================

print(
    "\n"
    + "=" * 90
)

print(
    "LOADING ORIGINAL STAGE-5 ECG MODEL"
)

print(
    "=" * 90
)


stage5_model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


results[
    "stage5_original"
] = (
    run_model_diagnostic(
        stage5_model,
        "MODEL 3 — ORIGINAL STAGE-5 ECG",
    )
)


clear_model(
    stage5_model
)


# ============================================================
# Summary table
# ============================================================

print(
    "\n\n"
    + "=" * 110
)

print(
    "TSQA vs STAGE-5 SIGNAL-PATH SUMMARY"
)

print(
    "=" * 110
)


print(
    f"{'Model':28s} "
    f"{'Enc cos Y/N':>12s} "
    f"{'Proj cos Y/N':>13s} "
    f"{'Margin SD':>11s} "
    f"{'Margin BA':>11s} "
    f"{'Enc grad Y':>13s} "
    f"{'Enc grad N':>13s}"
)

print(
    "-" * 110
)


for key, result in results.items():

    print(
        f"{key:28s} "
        f"{result['encoder']['yes_no_centroid_cosine']:12.6f} "
        f"{result['projector']['yes_no_centroid_cosine']:13.6f} "
        f"{result['class_margin']['margin_std']:11.6f} "
        f"{result['class_margin']['balanced_accuracy'] * 100:10.1f}% "
        f"{result['gradient_yes']['encoder_gradient_norm']:13.4e} "
        f"{result['gradient_no']['encoder_gradient_norm']:13.4e}"
    )


print(
    "\n"
    + "=" * 110
)

print(
    "PROJECTOR GRADIENT SUMMARY"
)

print(
    "=" * 110
)


for key, result in results.items():

    print(
        f"{key:28s} | "
        f"YES grad="
        f"{result['gradient_yes']['projector_gradient_norm']:.4e} | "
        f"NO grad="
        f"{result['gradient_no']['projector_gradient_norm']:.4e}"
    )


# ============================================================
# Save
# ============================================================

serializable_results = {
    "experiment":
        "tsqa_vs_stage5_signal_path",

    "task":
        TASK_ID,

    "seed":
        SEED,

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

    "selected_examples":
        [
            {
                "participant_id":
                    x.get(
                        "_participant_id"
                    ),

                "window_start":
                    x.get(
                        "_window_start"
                    ),

                "gold":
                    x[
                        "_answer_clean"
                    ],
            }

            for x in diagnostic_examples
        ],

    "models":
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
        serializable_results,
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
    "\nNo optimization steps were performed."
)

print(
    "TEST participants remain sealed."
)