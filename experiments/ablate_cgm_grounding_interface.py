"""
CGM grounding interface ablation.

GOAL
----
Determine WHY our first OpenTSLM-style CGM grounding smoke test
failed.

We vary three things:

1. SIGNAL NORMALIZATION
   A) fixed training-cohort normalization
   B) per-window z-normalization

2. TARGET
   A) binary pairwise (a) vs (b)
   B) natural-language "Yes." vs "No." plus pairwise auxiliary

3. LANGUAGE ADAPTATION
   A) Stage-5 LoRA frozen
   B) Stage-5 LoRA trainable

We deliberately DO NOT ask for exact maximum glucose yet.

DATA
----
Same deterministic:
    10 YES
    10 NO
above_180 examples.

TEST PARTICIPANTS REMAIN SEALED.

MODEL
-----
Every condition starts from a fresh copy of:

    OpenTSLM/llama-3.2-1b-ecg-sp

Always frozen:
    patch embedding
    input norm
    T1-T4
    base LLM weights

Always trainable:
    Stage-5 T5
    Stage-5 T6
    Stage-5 projector

LoRA:
    frozen or trainable depending on condition.

CONDITIONS
----------
A_fixed_pairwise
    fixed normalization
    binary (a)/(b)
    pairwise loss only
    LoRA frozen

B_window_pairwise
    per-window normalization
    same binary prompt
    pairwise loss only
    LoRA frozen

C_fixed_language_frozen
    fixed normalization
    natural Yes./No.
    sequence + pairwise
    LoRA frozen

D_window_language_frozen
    per-window normalization
    natural Yes./No.
    sequence + pairwise
    LoRA frozen

E_fixed_language_lora
    fixed normalization
    natural Yes./No.
    sequence + pairwise
    LoRA trainable

F_window_language_lora
    per-window normalization
    natural Yes./No.
    sequence + pairwise
    LoRA trainable

No checkpoint is saved.
"""

import gc
import json
import math
import random
import re
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

from src.cgm_multitask import build_multitask_examples
from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

SEED = 42

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
    / "cgm_grounding_interface_ablation.json"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

N_PER_CLASS = 10

# Keep 30 because the original Stage-5 pairwise control
# did not escape chance until around epoch 20.
EPOCHS = 30


LR_T5_T6 = 2e-4
LR_PROJECTOR = 1e-4

# LoRA gets a much smaller LR.
LR_LORA = 2e-5

WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0

PAIRWISE_WEIGHT = 1.0


EVAL_EPOCHS = {
    0,
    1,
    2,
    5,
    10,
    15,
    20,
    25,
    30,
}


# Generation is only useful for natural-language conditions.
GENERATION_EPOCHS = {
    0,
    5,
    10,
    20,
    30,
}


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


train_mean = float(
    manifest[
        "normalization"
    ][
        "mean"
    ]
)

train_std = float(
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
    f"Fixed normalization mean: "
    f"{train_mean:.3f}"
)

print(
    f"Fixed normalization SD: "
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
# Temporary model just to get EOS token
# ============================================================

print(
    "\nLoading temporary Stage-5 model "
    "for data construction..."
)


temporary_model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


EOS_TOKEN = (
    temporary_model.get_eos_token()
)


# ============================================================
# Build existing fixed-normalized examples
# ============================================================

base_examples = (
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


yes_pool = [
    example

    for example
    in base_examples

    if (
        example[
            "_answer_clean"
        ]
        == "(a)"
    )
]


no_pool = [
    example

    for example
    in base_examples

    if (
        example[
            "_answer_clean"
        ]
        == "(b)"
    )
]


print(
    "\nAvailable YES:",
    len(yes_pool),
)

print(
    "Available NO:",
    len(no_pool),
)


# ============================================================
# Exact same deterministic selection
# ============================================================

selection_rng = random.Random(
    SEED + 777
)


selected_yes = (
    selection_rng.sample(
        yes_pool,
        N_PER_CLASS,
    )
)


selected_no = (
    selection_rng.sample(
        no_pool,
        N_PER_CLASS,
    )
)


selected_base = (
    selected_yes
    + selected_no
)


# Alternating fixed order.

ordered_base = []


for yes_example, no_example in zip(
    selected_yes,
    selected_no,
):

    ordered_base.append(
        yes_example
    )

    ordered_base.append(
        no_example
    )


# ============================================================
# Recover physical CGM values
# ============================================================

def recover_raw_window(
    example,
):

    signal = (
        example[
            "time_series"
        ][
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


    raw = (
        signal
        * train_std
        + train_mean
    )


    reconstructed_max = float(
        raw.max().item()
    )


    expected_max = float(
        example[
            "_window_max"
        ]
    )


    if (
        abs(
            reconstructed_max
            - expected_max
        )
        > 1e-3
    ):

        raise RuntimeError(
            "Raw-window reconstruction mismatch: "
            f"{reconstructed_max} "
            f"vs {expected_max}"
        )


    return raw


# ============================================================
# Precompute raw records
# ============================================================

raw_records = []


for base_example in ordered_base:

    raw = (
        recover_raw_window(
            base_example
        )
    )


    window_mean = float(
        raw.mean().item()
    )


    window_std = float(
        raw.std(
            unbiased=False
        ).item()
    )


    maximum = float(
        raw.max().item()
    )


    is_yes = (
        base_example[
            "_answer_clean"
        ]
        == "(a)"
    )


    if (
        is_yes
        != (
            maximum
            > 180.0
        )
    ):

        raise RuntimeError(
            "Label/max mismatch."
        )


    raw_records.append(
        {
            "raw":
                raw,

            "mean":
                window_mean,

            "std":
                window_std,

            "max":
                maximum,

            "is_yes":
                is_yes,

            "participant_id":
                base_example.get(
                    "_participant_id"
                ),

            "window_start":
                base_example.get(
                    "_window_start"
                ),
        }
    )


print(
    "\nSelected data:"
)


for index, record in enumerate(
    raw_records,
    start=1,
):

    print(
        f"{index:02d} | "
        f"{'YES' if record['is_yes'] else 'NO ':3s} | "
        f"participant="
        f"{record['participant_id']} | "
        f"window="
        f"{record['window_start']} | "
        f"mean="
        f"{record['mean']:.1f} | "
        f"SD="
        f"{record['std']:.1f} | "
        f"max="
        f"{record['max']:.0f}"
    )


# We no longer need temporary model.

del temporary_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Condition definitions
# ============================================================

CONDITIONS = [
    {
        "name":
            "A_fixed_pairwise",

        "display":
            "A — FIXED NORMALIZATION + PAIRWISE",

        "normalization":
            "fixed",

        "target_mode":
            "binary",

        "sequence_loss":
            False,

        "train_lora":
            False,
    },

    {
        "name":
            "B_window_pairwise",

        "display":
            "B — PER-WINDOW NORMALIZATION + PAIRWISE",

        "normalization":
            "window",

        "target_mode":
            "binary",

        "sequence_loss":
            False,

        "train_lora":
            False,
    },

    {
        "name":
            "C_fixed_language_frozen",

        "display":
            "C — FIXED + LANGUAGE, LoRA FROZEN",

        "normalization":
            "fixed",

        "target_mode":
            "language",

        "sequence_loss":
            True,

        "train_lora":
            False,
    },

    {
        "name":
            "D_window_language_frozen",

        "display":
            "D — PER-WINDOW + LANGUAGE, LoRA FROZEN",

        "normalization":
            "window",

        "target_mode":
            "language",

        "sequence_loss":
            True,

        "train_lora":
            False,
    },

    {
        "name":
            "E_fixed_language_lora",

        "display":
            "E — FIXED + LANGUAGE, LoRA TRAINABLE",

        "normalization":
            "fixed",

        "target_mode":
            "language",

        "sequence_loss":
            True,

        "train_lora":
            True,
    },

    {
        "name":
            "F_window_language_lora",

        "display":
            "F — PER-WINDOW + LANGUAGE, LoRA TRAINABLE",

        "normalization":
            "window",

        "target_mode":
            "language",

        "sequence_loss":
            True,

        "train_lora":
            True,
    },
]


# ============================================================
# Construct condition-specific example
# ============================================================

def make_example(
    record,
    condition,
):

    raw = (
        record[
            "raw"
        ]
    )


    if (
        condition[
            "normalization"
        ]
        == "fixed"
    ):

        signal = (
            raw
            - train_mean
        ) / train_std


        normalization_text = (
            "The numeric signal was normalized using "
            f"the fixed training-cohort mean "
            f"{train_mean:.2f} mg/dL and standard deviation "
            f"{train_std:.2f} mg/dL."
        )


    elif (
        condition[
            "normalization"
        ]
        == "window"
    ):

        safe_std = (
            record[
                "std"
            ]
            if (
                record[
                    "std"
                ]
                >= 1e-6
            )
            else 1.0
        )


        signal = (
            raw
            - record[
                "mean"
            ]
        ) / safe_std


        normalization_text = (
            "The numeric signal was z-normalized using "
            "this segment's own original mean and "
            "standard deviation."
        )


    else:

        raise ValueError(
            condition[
                "normalization"
            ]
        )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Both normalization conditions get the SAME physical
    # metadata: mean, SD, units, cadence, count.
    #
    # This makes the normalization comparison cleaner.
    # --------------------------------------------------------

    time_series_text = (
        "Continuous glucose monitor (CGM) signal "
        "measured in mg/dL. "
        f"This segment contains {len(raw)} measurements "
        "with a nominal sampling interval of 5 minutes. "
        f"The original segment mean was "
        f"{record['mean']:.2f} mg/dL and the original "
        f"standard deviation was "
        f"{record['std']:.2f} mg/dL. "
        + normalization_text
    )


    if (
        condition[
            "target_mode"
        ]
        == "binary"
    ):

        post_prompt = (
            "\nQuestion: Did glucose exceed "
            "180 mg/dL during this segment?\n"
            "(a) Yes\n"
            "(b) No\n"
            "Answer:"
        )


        answer_clean = (
            "(a)"
            if record[
                "is_yes"
            ]
            else "(b)"
        )


        answer = (
            answer_clean
            + EOS_TOKEN
        )


    elif (
        condition[
            "target_mode"
        ]
        == "language"
    ):

        post_prompt = (
            "\nQuestion: Did glucose exceed "
            "180 mg/dL during this segment?\n"
            "Answer Yes or No.\n"
            "Answer:"
        )


        answer_clean = (
            "Yes."
            if record[
                "is_yes"
            ]
            else "No."
        )


        # Leading space is intentional for tokenizer context.
        answer = (
            " "
            + answer_clean
            + EOS_TOKEN
        )


    else:

        raise ValueError(
            condition[
                "target_mode"
            ]
        )


    return {
        "pre_prompt":
            (
                "Analyze the continuous glucose monitor "
                "signal and answer the factual question.\n"
            ),

        "time_series":
            [
                signal
                .detach()
                .clone()
                .float()
            ],

        "time_series_text":
            [
                time_series_text
            ],

        "post_prompt":
            post_prompt,

        "answer":
            answer,

        "_is_yes":
            record[
                "is_yes"
            ],

        "_answer_clean":
            answer_clean,

        "_participant_id":
            record[
                "participant_id"
            ],

        "_window_start":
            record[
                "window_start"
            ],

        "_mean":
            record[
                "mean"
            ],

        "_std":
            record[
                "std"
            ],

        "_max":
            record[
                "max"
            ],
    }


# ============================================================
# Token helpers
# ============================================================

def get_decision_tokens(
    model,
    condition,
):

    if (
        condition[
            "target_mode"
        ]
        == "binary"
    ):

        yes_string = "(a)"
        no_string = "(b)"


    else:

        yes_string = " Yes"
        no_string = " No"


    yes_ids = (
        model.tokenizer(
            yes_string,
            add_special_tokens=False,
        ).input_ids
    )


    no_ids = (
        model.tokenizer(
            no_string,
            add_special_tokens=False,
        ).input_ids
    )


    if (
        len(
            yes_ids
        )
        == 0
        or len(
            no_ids
        )
        == 0
    ):

        raise RuntimeError(
            "Unable to tokenize decisions."
        )


    yes_token = int(
        yes_ids[
            0
        ]
    )


    no_token = int(
        no_ids[
            0
        ]
    )


    if (
        yes_token
        == no_token
    ):

        raise RuntimeError(
            "YES and NO first decision tokens match."
        )


    return (
        yes_token,
        no_token,
        yes_ids,
        no_ids,
    )


# ============================================================
# Pairwise logits
# ============================================================

def pair_logits_for_example(
    model,
    example,
    yes_token,
    no_token,
):

    (
        inputs_embeds,
        attention_mask,
    ) = (
        model.pad_and_apply_batch(
            [
                example
            ]
        )
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


    last_position = (
        int(
            attention_mask[
                0
            ]
            .sum()
            .item()
        )
        - 1
    )


    logits = (
        outputs.logits[
            0,
            last_position,
            :
        ]
    )


    return torch.stack(
        [
            logits[
                yes_token
            ],

            logits[
                no_token
            ],
        ]
    )


# ============================================================
# Pairwise loss
# ============================================================

def pairwise_loss(
    model,
    example,
    yes_token,
    no_token,
):

    pair_logits = (
        pair_logits_for_example(
            model,
            example,
            yes_token,
            no_token,
        )
    )


    target = torch.tensor(
        [
            0
            if example[
                "_is_yes"
            ]
            else 1
        ],
        dtype=torch.long,
        device=device,
    )


    return F.cross_entropy(
        pair_logits
        .float()
        .unsqueeze(
            0
        ),

        target,
    )


# ============================================================
# Pairwise evaluation
# ============================================================

def evaluate_pairwise(
    model,
    examples,
    yes_token,
    no_token,
):

    yes_correct = 0
    no_correct = 0

    yes_total = 0
    no_total = 0

    predicted_yes = 0
    predicted_no = 0

    p_correct = []

    margins = []


    with torch.no_grad():

        for example in examples:

            pair_logits = (
                pair_logits_for_example(
                    model,
                    example,
                    yes_token,
                    no_token,
                )
            ).float()


            probabilities = (
                torch.softmax(
                    pair_logits,
                    dim=0,
                )
            )


            p_yes = float(
                probabilities[
                    0
                ].item()
            )


            p_no = float(
                probabilities[
                    1
                ].item()
            )


            prediction_yes = (
                p_yes
                >= p_no
            )


            gold_yes = (
                example[
                    "_is_yes"
                ]
            )


            margins.append(
                float(
                    (
                        pair_logits[
                            0
                        ]
                        - pair_logits[
                            1
                        ]
                    ).item()
                )
            )


            if prediction_yes:

                predicted_yes += 1

            else:

                predicted_no += 1


            if gold_yes:

                yes_total += 1

                p_correct.append(
                    p_yes
                )


                if prediction_yes:

                    yes_correct += 1


            else:

                no_total += 1

                p_correct.append(
                    p_no
                )


                if not prediction_yes:

                    no_correct += 1


    yes_accuracy = (
        yes_correct
        / yes_total
    )


    no_accuracy = (
        no_correct
        / no_total
    )


    balanced_accuracy = (
        0.5
        * (
            yes_accuracy
            + no_accuracy
        )
    )


    margin_tensor = torch.tensor(
        margins,
        dtype=torch.float32,
    )


    return {
        "balanced_accuracy":
            balanced_accuracy,

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "predicted_yes":
            predicted_yes,

        "predicted_no":
            predicted_no,

        "mean_p_correct":
            sum(
                p_correct
            )
            / len(
                p_correct
            ),

        "margin_sd":
            float(
                margin_tensor.std().item()
            ),
    }


# ============================================================
# Natural-language generation evaluation
# ============================================================

def parse_yes_no(
    text,
):

    cleaned = (
        text
        .strip()
        .lower()
    )


    if re.match(
        r"^yes\b",
        cleaned,
    ):

        return True


    if re.match(
        r"^no\b",
        cleaned,
    ):

        return False


    return None


def evaluate_generation(
    model,
    examples,
):

    valid = 0
    correct = 0

    predicted_yes = 0
    predicted_no = 0

    samples = []


    with torch.no_grad():

        for example in examples:

            generated = (
                model.generate(
                    [
                        example
                    ],
                    max_new_tokens=12,
                    do_sample=False,
                )[
                    0
                ]
            )


            prediction = (
                parse_yes_no(
                    generated
                )
            )


            gold = (
                example[
                    "_is_yes"
                ]
            )


            if (
                prediction
                is not None
            ):

                valid += 1


                if prediction:

                    predicted_yes += 1

                else:

                    predicted_no += 1


                if prediction == gold:

                    correct += 1


            samples.append(
                {
                    "gold_yes":
                        gold,

                    "predicted_yes":
                        prediction,

                    "raw_generation":
                        generated,

                    "participant_id":
                        example[
                            "_participant_id"
                        ],

                    "window_start":
                        example[
                            "_window_start"
                        ],
                }
            )


    n = len(
        examples
    )


    return {
        "valid_rate":
            valid
            / n,

        "accuracy":
            correct
            / n,

        "predicted_yes":
            predicted_yes,

        "predicted_no":
            predicted_no,

        "samples":
            samples,
    }


# ============================================================
# Representation diagnostics
# ============================================================

def get_patch_size(
    model,
):

    if hasattr(
        model,
        "patch_size",
    ):

        return int(
            model.patch_size
        )


    if hasattr(
        model.encoder,
        "patch_size",
    ):

        return int(
            model.encoder.patch_size
        )


    raise RuntimeError(
        "Unable to find patch size."
    )


def representation_stats(
    model,
    examples,
):

    encoder_vectors = []
    projector_vectors = []

    labels = []

    patch_size = (
        get_patch_size(
            model
        )
    )


    with torch.no_grad():

        for example in examples:

            signal = (
                example[
                    "time_series"
                ][
                    0
                ]
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
                signal.shape[
                    1
                ]
                % patch_size
            )


            if remainder:

                signal = F.pad(
                    signal,
                    (
                        0,
                        patch_size
                        - remainder,
                    ),
                )


            encoded = (
                model.encoder(
                    signal
                )
            )


            projected = (
                model.projector(
                    encoded
                )
            )


            encoder_vectors.append(
                encoded[
                    0
                ]
                .detach()
                .float()
                .cpu()
                .reshape(
                    -1
                )
            )


            projector_vectors.append(
                projected[
                    0
                ]
                .detach()
                .float()
                .cpu()
                .reshape(
                    -1
                )
            )


            labels.append(
                example[
                    "_is_yes"
                ]
            )


    def summarize(
        vectors,
    ):

        matrix = torch.stack(
            vectors
        ).float()


        feature_sd = float(
            matrix.std(
                dim=0
            )
            .mean()
            .item()
        )


        yes_indices = [
            i

            for i, label
            in enumerate(
                labels
            )

            if label
        ]


        no_indices = [
            i

            for i, label
            in enumerate(
                labels
            )

            if not label
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


        cosine = float(
            F.cosine_similarity(
                yes_centroid.unsqueeze(
                    0
                ),

                no_centroid.unsqueeze(
                    0
                ),
            ).item()
        )


        l2 = float(
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


        return {
            "feature_sd":
                feature_sd,

            "yes_no_cosine":
                cosine,

            "yes_no_l2_per_sqrt_dim":
                l2,
        }


    return {
        "encoder":
            summarize(
                encoder_vectors
            ),

        "projector":
            summarize(
                projector_vectors
            ),
    }


# ============================================================
# Gradient norm helper
# ============================================================

def parameter_grad_norm(
    parameters,
):

    squared = 0.0


    for parameter in parameters:

        if parameter.grad is None:

            continue


        norm = float(
            parameter.grad
            .detach()
            .float()
            .norm()
            .item()
        )


        squared += (
            norm
            * norm
        )


    return math.sqrt(
        squared
    )


# ============================================================
# Configure trainable parameters
# ============================================================

def configure_training(
    model,
    condition,
):

    # Freeze everything.

    for parameter in (
        model.encoder.parameters()
    ):

        parameter.requires_grad = False


    for parameter in (
        model.projector.parameters()
    ):

        parameter.requires_grad = False


    for parameter in (
        model.llm.parameters()
    ):

        parameter.requires_grad = False


    layers = (
        model
        .encoder
        .encoder
        .layers
    )


    if len(
        layers
    ) != 6:

        raise RuntimeError(
            "Expected six encoder layers."
        )


    # T5/T6 trainable.

    for parameter in (
        layers[
            4
        ].parameters()
    ):

        parameter.requires_grad = True


    for parameter in (
        layers[
            5
        ].parameters()
    ):

        parameter.requires_grad = True


    # Projector trainable.

    for parameter in (
        model.projector.parameters()
    ):

        parameter.requires_grad = True


    # Optionally unfreeze Stage-5 LoRA only.

    if condition[
        "train_lora"
    ]:

        for name, parameter in (
            model.llm.named_parameters()
        ):

            if (
                "lora_" in name.lower()
            ):

                parameter.requires_grad = True


    late_parameters = [
        parameter

        for layer in [
            layers[
                4
            ],
            layers[
                5
            ],
        ]

        for parameter
        in layer.parameters()

        if parameter.requires_grad
    ]


    projector_parameters = [
        parameter

        for parameter
        in model.projector.parameters()

        if parameter.requires_grad
    ]


    lora_parameters = [
        parameter

        for name, parameter
        in model.llm.named_parameters()

        if (
            parameter.requires_grad
            and "lora_" in name.lower()
        )
    ]


    other_llm_trainable = [
        parameter

        for name, parameter
        in model.llm.named_parameters()

        if (
            parameter.requires_grad
            and "lora_" not in name.lower()
        )
    ]


    early_encoder_trainable = sum(
        parameter.numel()

        for name, parameter
        in model.encoder.named_parameters()

        if (
            parameter.requires_grad
            and not (
                name.startswith(
                    "encoder.layers.4"
                )
                or name.startswith(
                    "encoder.layers.5"
                )
            )
        )
    )


    if (
        early_encoder_trainable
        != 0
    ):

        raise RuntimeError(
            "Early encoder unexpectedly trainable."
        )


    if (
        len(
            other_llm_trainable
        )
        != 0
    ):

        raise RuntimeError(
            "Non-LoRA LLM weights became trainable."
        )


    if (
        condition[
            "train_lora"
        ]
        and len(
            lora_parameters
        )
        == 0
    ):

        raise RuntimeError(
            "LoRA condition requested, "
            "but no trainable LoRA parameters found."
        )


    if (
        not condition[
            "train_lora"
        ]
        and len(
            lora_parameters
        )
        != 0
    ):

        raise RuntimeError(
            "LoRA should be frozen."
        )


    audit = {
        "early_encoder_trainable":
            early_encoder_trainable,

        "t5_t6_trainable":
            sum(
                p.numel()
                for p
                in late_parameters
            ),

        "projector_trainable":
            sum(
                p.numel()
                for p
                in projector_parameters
            ),

        "lora_trainable":
            sum(
                p.numel()
                for p
                in lora_parameters
            ),
    }


    return (
        late_parameters,
        projector_parameters,
        lora_parameters,
        audit,
    )


# ============================================================
# Run one condition
# ============================================================

def run_condition(
    condition,
):

    print(
        "\n\n"
        + "#" * 110
    )

    print(
        condition[
            "display"
        ]
    )

    print(
        "#" * 110
    )


    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )


    # --------------------------------------------------------
    # Fresh Stage-5 checkpoint
    # --------------------------------------------------------

    model = (
        OpenTSLM.load_pretrained(
            STAGE5_REPO_ID,
            device=device,
            enable_lora=True,
        )
    )


    examples = [
        make_example(
            record,
            condition,
        )

        for record
        in raw_records
    ]


    (
        yes_token,
        no_token,
        yes_ids,
        no_ids,
    ) = (
        get_decision_tokens(
            model,
            condition,
        )
    )


    print(
        "\nDecision tokens:"
    )

    print(
        "YES:",
        yes_ids,
        "->",
        yes_token,
        model.tokenizer.convert_ids_to_tokens(
            [
                yes_token
            ]
        ),
    )

    print(
        "NO:",
        no_ids,
        "->",
        no_token,
        model.tokenizer.convert_ids_to_tokens(
            [
                no_token
            ]
        ),
    )


    (
        late_parameters,
        projector_parameters,
        lora_parameters,
        audit,
    ) = (
        configure_training(
            model,
            condition,
        )
    )


    print(
        "\nTrainable parameter audit:"
    )

    print(
        "Early encoder:",
        f"{audit['early_encoder_trainable']:,}"
    )

    print(
        "T5/T6:",
        f"{audit['t5_t6_trainable']:,}"
    )

    print(
        "Projector:",
        f"{audit['projector_trainable']:,}"
    )

    print(
        "LoRA:",
        f"{audit['lora_trainable']:,}"
    )


    # --------------------------------------------------------
    # Deterministic capacity test:
    # disable dropout but retain gradients.
    # --------------------------------------------------------

    model.encoder.eval()
    model.projector.eval()
    model.llm.eval()


    optimizer_groups = [
        {
            "params":
                late_parameters,

            "lr":
                LR_T5_T6,

            "weight_decay":
                WEIGHT_DECAY,
        },

        {
            "params":
                projector_parameters,

            "lr":
                LR_PROJECTOR,

            "weight_decay":
                WEIGHT_DECAY,
        },
    ]


    if len(
        lora_parameters
    ) > 0:

        optimizer_groups.append(
            {
                "params":
                    lora_parameters,

                "lr":
                    LR_LORA,

                "weight_decay":
                    0.0,
            }
        )


    optimizer = (
        torch.optim.AdamW(
            optimizer_groups
        )
    )


    trainable_parameters = (
        late_parameters
        + projector_parameters
        + lora_parameters
    )


    # --------------------------------------------------------
    # Before training
    # --------------------------------------------------------

    initial_pairwise = (
        evaluate_pairwise(
            model,
            examples,
            yes_token,
            no_token,
        )
    )


    initial_representation = (
        representation_stats(
            model,
            examples,
        )
    )


    initial_generation = None


    if (
        condition[
            "target_mode"
        ]
        == "language"
    ):

        initial_generation = (
            evaluate_generation(
                model,
                examples,
            )
        )


    print(
        "\nBEFORE TRAINING"
    )

    print(
        "Pairwise BA:",
        f"{100 * initial_pairwise['balanced_accuracy']:.1f}%"
    )

    print(
        "YES:",
        f"{100 * initial_pairwise['yes_accuracy']:.1f}%"
    )

    print(
        "NO:",
        f"{100 * initial_pairwise['no_accuracy']:.1f}%"
    )

    print(
        "P(correct):",
        f"{initial_pairwise['mean_p_correct']:.4f}"
    )

    print(
        "Encoder SD:",
        f"{initial_representation['encoder']['feature_sd']:.8f}"
    )

    print(
        "Encoder Y/N cosine:",
        f"{initial_representation['encoder']['yes_no_cosine']:.8f}"
    )

    print(
        "Projector SD:",
        f"{initial_representation['projector']['feature_sd']:.8f}"
    )


    if (
        initial_generation
        is not None
    ):

        print(
            "Generation accuracy:",
            f"{100 * initial_generation['accuracy']:.1f}%"
        )

        print(
            "Generation valid:",
            f"{100 * initial_generation['valid_rate']:.1f}%"
        )


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    history = [
        {
            "epoch":
                0,

            "pairwise_metrics":
                initial_pairwise,

            "generation_metrics":
                initial_generation,

            "representation":
                initial_representation,

            "mean_pairwise_loss":
                None,

            "mean_sequence_loss":
                None,

            "mean_total_loss":
                None,
        }
    ]


    first_gradient_audit = None


    # ========================================================
    # Training
    # ========================================================

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        pair_losses = []
        sequence_losses = []
        total_losses = []


        # Same alternating:
        # YES, NO, YES, NO...
        #
        # Same exact ordering every epoch.

        for example in examples:

            optimizer.zero_grad(
                set_to_none=True
            )


            p_loss = (
                pairwise_loss(
                    model,
                    example,
                    yes_token,
                    no_token,
                )
            )


            if (
                condition[
                    "sequence_loss"
                ]
            ):

                s_loss = (
                    model.compute_loss(
                        [
                            example
                        ]
                    )
                )


                total_loss = (
                    s_loss
                    + (
                        PAIRWISE_WEIGHT
                        * p_loss
                    )
                )


            else:

                s_loss = None

                total_loss = p_loss


            total_loss.backward()


            if (
                first_gradient_audit
                is None
            ):

                first_gradient_audit = {
                    "t5_t6":
                        parameter_grad_norm(
                            late_parameters
                        ),

                    "projector":
                        parameter_grad_norm(
                            projector_parameters
                        ),

                    "lora":
                        parameter_grad_norm(
                            lora_parameters
                        )
                        if len(
                            lora_parameters
                        ) > 0
                        else 0.0,
                }


                print(
                    "\nFirst gradient:"
                )

                print(
                    "T5/T6:",
                    f"{first_gradient_audit['t5_t6']:.8e}"
                )

                print(
                    "Projector:",
                    f"{first_gradient_audit['projector']:.8e}"
                )

                print(
                    "LoRA:",
                    f"{first_gradient_audit['lora']:.8e}"
                )


            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
                GRAD_CLIP_NORM,
            )


            optimizer.step()


            pair_losses.append(
                float(
                    p_loss
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
            )


            if (
                s_loss
                is not None
            ):

                sequence_losses.append(
                    float(
                        s_loss
                        .detach()
                        .float()
                        .cpu()
                        .item()
                    )
                )


            total_losses.append(
                float(
                    total_loss
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )
            )


        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        if epoch in EVAL_EPOCHS:

            pair_metrics = (
                evaluate_pairwise(
                    model,
                    examples,
                    yes_token,
                    no_token,
                )
            )


            generation_metrics = None


            if (
                condition[
                    "target_mode"
                ]
                == "language"
                and epoch
                in GENERATION_EPOCHS
            ):

                generation_metrics = (
                    evaluate_generation(
                        model,
                        examples,
                    )
                )


            representation = (
                representation_stats(
                    model,
                    examples,
                )
            )


            mean_pair_loss = (
                sum(
                    pair_losses
                )
                / len(
                    pair_losses
                )
            )


            mean_sequence_loss = (
                (
                    sum(
                        sequence_losses
                    )
                    / len(
                        sequence_losses
                    )
                )
                if len(
                    sequence_losses
                ) > 0
                else None
            )


            mean_total_loss = (
                sum(
                    total_losses
                )
                / len(
                    total_losses
                )
            )


            history.append(
                {
                    "epoch":
                        epoch,

                    "pairwise_metrics":
                        pair_metrics,

                    "generation_metrics":
                        generation_metrics,

                    "representation":
                        representation,

                    "mean_pairwise_loss":
                        mean_pair_loss,

                    "mean_sequence_loss":
                        mean_sequence_loss,

                    "mean_total_loss":
                        mean_total_loss,
                }
            )


            line = (
                f"Epoch {epoch:2d} | "
                f"pair="
                f"{mean_pair_loss:7.4f}"
            )


            if (
                mean_sequence_loss
                is not None
            ):

                line += (
                    f" | seq="
                    f"{mean_sequence_loss:7.4f}"
                )


            line += (
                f" | BA="
                f"{100 * pair_metrics['balanced_accuracy']:5.1f}%"
                f" | YES="
                f"{100 * pair_metrics['yes_accuracy']:5.1f}%"
                f" | NO="
                f"{100 * pair_metrics['no_accuracy']:5.1f}%"
                f" | P="
                f"{pair_metrics['mean_p_correct']:.4f}"
                f" | EncSD="
                f"{representation['encoder']['feature_sd']:.6f}"
            )


            if (
                generation_metrics
                is not None
            ):

                line += (
                    f" | GEN="
                    f"{100 * generation_metrics['accuracy']:.1f}%"
                    f" | valid="
                    f"{100 * generation_metrics['valid_rate']:.1f}%"
                )


            print(
                line
            )


    # ========================================================
    # Final
    # ========================================================

    final_pairwise = (
        evaluate_pairwise(
            model,
            examples,
            yes_token,
            no_token,
        )
    )


    final_representation = (
        representation_stats(
            model,
            examples,
        )
    )


    final_generation = None


    if (
        condition[
            "target_mode"
        ]
        == "language"
    ):

        final_generation = (
            evaluate_generation(
                model,
                examples,
            )
        )


    print(
        "\nFINAL"
    )

    print(
        "Pairwise BA:",
        f"{100 * final_pairwise['balanced_accuracy']:.1f}%"
    )

    print(
        "YES:",
        f"{100 * final_pairwise['yes_accuracy']:.1f}%"
    )

    print(
        "NO:",
        f"{100 * final_pairwise['no_accuracy']:.1f}%"
    )

    print(
        "Encoder SD:",
        f"{final_representation['encoder']['feature_sd']:.8f}"
    )

    print(
        "Encoder Y/N cosine:",
        f"{final_representation['encoder']['yes_no_cosine']:.8f}"
    )

    print(
        "Projector SD:",
        f"{final_representation['projector']['feature_sd']:.8f}"
    )


    if (
        final_generation
        is not None
    ):

        print(
            "Generation accuracy:",
            f"{100 * final_generation['accuracy']:.1f}%"
        )

        print(
            "Generation valid:",
            f"{100 * final_generation['valid_rate']:.1f}%"
        )


        print(
            "\nRepresentative generations:"
        )


        for item in (
            final_generation[
                "samples"
            ][
                :3
            ]
        ):

            print(
                f"  Gold="
                f"{'YES' if item['gold_yes'] else 'NO'}"
                f" | {item['raw_generation']!r}"
            )


        no_samples = [
            item

            for item
            in final_generation[
                "samples"
            ]

            if not item[
                "gold_yes"
            ]
        ]


        for item in (
            no_samples[
                :3
            ]
        ):

            print(
                f"  Gold=NO"
                f" | {item['raw_generation']!r}"
            )


    result = {
        "condition":
            condition,

        "parameter_audit":
            audit,

        "decision_tokens":
            {
                "yes_ids":
                    yes_ids,

                "no_ids":
                    no_ids,

                "yes_first":
                    yes_token,

                "no_first":
                    no_token,
            },

        "first_gradient_audit":
            first_gradient_audit,

        "initial_pairwise":
            initial_pairwise,

        "initial_representation":
            initial_representation,

        "initial_generation":
            initial_generation,

        "history":
            history,

        "final_pairwise":
            final_pairwise,

        "final_representation":
            final_representation,

        "final_generation":
            final_generation,
    }


    del optimizer
    del model

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# Run all six
# ============================================================

all_results = {}


for condition in CONDITIONS:

    all_results[
        condition[
            "name"
        ]
    ] = (
        run_condition(
            condition
        )
    )


# ============================================================
# Summary
# ============================================================

print(
    "\n\n"
    + "=" * 150
)

print(
    "CGM GROUNDING INTERFACE ABLATION — FINAL SUMMARY"
)

print(
    "=" * 150
)


print(
    f"{'Condition':30s} | "
    f"{'Final BA':>9s} | "
    f"{'YES':>6s} | "
    f"{'NO':>6s} | "
    f"{'GEN acc':>8s} | "
    f"{'GEN valid':>9s} | "
    f"{'Enc SD':>11s} | "
    f"{'Enc Y/N cos':>11s} | "
    f"{'Proj SD':>11s}"
)


print(
    "-" * 150
)


for condition in CONDITIONS:

    name = (
        condition[
            "name"
        ]
    )


    result = (
        all_results[
            name
        ]
    )


    pairwise = (
        result[
            "final_pairwise"
        ]
    )


    representation = (
        result[
            "final_representation"
        ]
    )


    generation = (
        result[
            "final_generation"
        ]
    )


    gen_acc = (
        (
            f"{100 * generation['accuracy']:.1f}%"
        )
        if generation
        is not None
        else "-"
    )


    gen_valid = (
        (
            f"{100 * generation['valid_rate']:.1f}%"
        )
        if generation
        is not None
        else "-"
    )


    print(
        f"{name:30s} | "
        f"{100 * pairwise['balanced_accuracy']:8.1f}% | "
        f"{100 * pairwise['yes_accuracy']:5.1f}% | "
        f"{100 * pairwise['no_accuracy']:5.1f}% | "
        f"{gen_acc:>8s} | "
        f"{gen_valid:>9s} | "
        f"{representation['encoder']['feature_sd']:11.8f} | "
        f"{representation['encoder']['yes_no_cosine']:11.8f} | "
        f"{representation['projector']['feature_sd']:11.8f}"
    )


# ============================================================
# Direct causal comparisons
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "DIRECT COMPARISONS"
)

print(
    "=" * 110
)


def ba(
    name,
):

    return (
        all_results[
            name
        ][
            "final_pairwise"
        ][
            "balanced_accuracy"
        ]
    )


def gen(
    name,
):

    value = (
        all_results[
            name
        ][
            "final_generation"
        ]
    )


    if value is None:

        return None


    return (
        value[
            "accuracy"
        ]
    )


print(
    "\n1. NORMALIZATION — binary pairwise:"
)

print(
    "Fixed:",
    f"{100 * ba('A_fixed_pairwise'):.1f}%"
)

print(
    "Per-window:",
    f"{100 * ba('B_window_pairwise'):.1f}%"
)

print(
    "Delta window - fixed:",
    f"{100 * (ba('B_window_pairwise') - ba('A_fixed_pairwise')):+.1f} pp"
)


print(
    "\n2. NORMALIZATION — language, LoRA frozen:"
)

print(
    "Fixed pairwise BA:",
    f"{100 * ba('C_fixed_language_frozen'):.1f}%"
)

print(
    "Window pairwise BA:",
    f"{100 * ba('D_window_language_frozen'):.1f}%"
)

print(
    "Fixed generation:",
    f"{100 * gen('C_fixed_language_frozen'):.1f}%"
)

print(
    "Window generation:",
    f"{100 * gen('D_window_language_frozen'):.1f}%"
)


print(
    "\n3. LoRA EFFECT — fixed normalization:"
)

print(
    "LoRA frozen generation:",
    f"{100 * gen('C_fixed_language_frozen'):.1f}%"
)

print(
    "LoRA trainable generation:",
    f"{100 * gen('E_fixed_language_lora'):.1f}%"
)

print(
    "Delta:",
    f"{100 * (gen('E_fixed_language_lora') - gen('C_fixed_language_frozen')):+.1f} pp"
)


print(
    "\n4. LoRA EFFECT — per-window normalization:"
)

print(
    "LoRA frozen generation:",
    f"{100 * gen('D_window_language_frozen'):.1f}%"
)

print(
    "LoRA trainable generation:",
    f"{100 * gen('F_window_language_lora'):.1f}%"
)

print(
    "Delta:",
    f"{100 * (gen('F_window_language_lora') - gen('D_window_language_frozen')):+.1f} pp"
)


print(
    "\n5. NORMALIZATION — language with LoRA trainable:"
)

print(
    "Fixed generation:",
    f"{100 * gen('E_fixed_language_lora'):.1f}%"
)

print(
    "Window generation:",
    f"{100 * gen('F_window_language_lora'):.1f}%"
)


# ============================================================
# Save results
# ============================================================

serializable_records = [
    {
        "participant_id":
            record[
                "participant_id"
            ],

        "window_start":
            record[
                "window_start"
            ],

        "is_yes":
            record[
                "is_yes"
            ],

        "mean":
            record[
                "mean"
            ],

        "std":
            record[
                "std"
            ],

        "max":
            record[
                "max"
            ],
    }

    for record
    in raw_records
]


output = {
    "experiment":
        "cgm_grounding_interface_ablation",

    "seed":
        SEED,

    "stage5_repo":
        STAGE5_REPO_ID,

    "task":
        TASK_ID,

    "epochs":
        EPOCHS,

    "n_yes":
        N_PER_CLASS,

    "n_no":
        N_PER_CLASS,

    "fixed_normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "hyperparameters":
        {
            "lr_t5_t6":
                LR_T5_T6,

            "lr_projector":
                LR_PROJECTOR,

            "lr_lora":
                LR_LORA,

            "weight_decay":
                WEIGHT_DECAY,

            "grad_clip":
                GRAD_CLIP_NORM,

            "pairwise_weight":
                PAIRWISE_WEIGHT,
        },

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "records":
        serializable_records,

    "conditions":
        CONDITIONS,

    "results":
        all_results,
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
        output,
        f,
        indent=2,
    )


print(
    "\nSaved:"
)

print(
    RESULTS_PATH
)


print(
    "\nNo checkpoint was saved."
)

print(
    "No validation/test participants were used."
)

print(
    "TEST participants remain sealed."
)