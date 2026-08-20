"""
CGM SIGNAL-NECESSITY / GROUNDING DIAGNOSTIC

Question
--------
Can Stage-5 learn CGM -> language grounding that genuinely
depends on the CGM waveform, rather than textual shortcuts?

Key design
----------
START:
    Original public Stage-5 ECG checkpoint.

INPUT:
    Fixed global normalization ONLY:

        (raw_mg_dl - 142.143) / 37.232

    Text includes ONLY invariant metadata:
        - sensor = CGM
        - units = mg/dL
        - 128 readings
        - ~5 minute cadence
        - SAME global normalization mean/SD for every example

    NO per-window:
        - mean
        - SD
        - max
        - TIR
        - threshold statistics
        - participant ID
        - window index

TARGET:
    " Yes."
    " No."

TRAINABLE:
    Stage-5 Transformer 5
    Stage-5 Transformer 6
    Stage-5 projector
    Stage-5 LoRA

FROZEN:
    patch embedding
    T1-T4
    base LLM weights

LOSS:
    sequence loss
    +
    true Yes-vs-No pairwise loss

PARTICIPANT SPLIT:
    training examples -> TRAIN participants only
    validation examples -> VALIDATION participants only
    TEST participants untouched

VALIDATION ABLATIONS:
    1. REAL
       correct validation waveform

    2. ZERO
       exact same text, but waveform replaced by zeros

    3. SHUFFLED-OPPOSITE
       exact same text, but waveform replaced by a validation
       waveform from the OPPOSITE label.

For shuffled inputs we report both:

    accuracy vs ORIGINAL label
    accuracy vs DONOR-SIGNAL label

Strong evidence of waveform use would look like:

    REAL:
        high accuracy

    ZERO:
        ~50% balanced accuracy

    SHUFFLED:
        low accuracy vs original labels
        high accuracy vs donor-signal labels

This is a diagnostic, not final model training.

TEST PARTICIPANTS REMAIN SEALED.
"""

import copy
import gc
import json
import math
import random
import re
import sys

from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F


# ============================================================
# Paths
# ============================================================

ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

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
    / "cgm_signal_necessity.json"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128


# ------------------------------------------------------------
# Small-but-nontrivial grounding smoke test.
#
# Increase later if this works.
# ------------------------------------------------------------

TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 20

BATCH_SIZE = 4

EPOCHS = 15


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

LR_T5_T6 = 2e-4

LR_PROJECTOR = 1e-4

LR_LORA = 2e-5


WEIGHT_DECAY = 1e-2


# IMPORTANT:
#
# Clip groups separately.
#
# Previous LoRA gradients were much larger than sensor-path
# gradients. Global clipping would allow LoRA to dominate the
# norm and effectively shrink T5/T6 updates too.

CLIP_T5_T6 = 1.0

CLIP_PROJECTOR = 1.0

CLIP_LORA = 1.0


SEQUENCE_WEIGHT = 1.0

PAIRWISE_WEIGHT = 1.0


EVAL_EPOCHS = {
    0,
    1,
    2,
    5,
    10,
    15,
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


validation_manifest = (
    manifest.get(
        "validation"
    )
)


if validation_manifest is None:

    validation_manifest = (
        manifest.get(
            "val"
        )
    )


if validation_manifest is None:

    raise RuntimeError(
        "Could not find validation split "
        "in manifest."
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


validation_files = [
    DATA_DIR
    / item[
        "relative_path"
    ]

    for item
    in validation_manifest
]


print(
    "Train participants:",
    len(train_files),
)

print(
    "Validation participants:",
    len(validation_files),
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
# Temporary model for EOS token
# ============================================================

print(
    "\nLoading Stage-5 tokenizer/model "
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
# Build train + validation pools
#
# IMPORTANT:
# test split is NEVER loaded here.
# ============================================================

print(
    "\nBuilding TRAIN above_180 pool..."
)


train_pool = (
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


print(
    "Building VALIDATION above_180 pool..."
)


validation_pool = (
    build_multitask_examples(
        files=validation_files,
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


# ============================================================
# Split by class
# ============================================================

def split_by_class(
    examples,
):

    yes = [
        example

        for example
        in examples

        if (
            example[
                "_answer_clean"
            ]
            == "(a)"
        )
    ]


    no = [
        example

        for example
        in examples

        if (
            example[
                "_answer_clean"
            ]
            == "(b)"
        )
    ]


    return (
        yes,
        no,
    )


(
    train_yes_pool,
    train_no_pool,
) = (
    split_by_class(
        train_pool
    )
)


(
    val_yes_pool,
    val_no_pool,
) = (
    split_by_class(
        validation_pool
    )
)


print(
    "\nTRAIN available:"
)

print(
    "  YES:",
    len(
        train_yes_pool
    ),
)

print(
    "  NO:",
    len(
        train_no_pool
    ),
)


print(
    "\nVALIDATION available:"
)

print(
    "  YES:",
    len(
        val_yes_pool
    ),
)

print(
    "  NO:",
    len(
        val_no_pool
    ),
)


if (
    len(
        train_yes_pool
    )
    < TRAIN_PER_CLASS
    or
    len(
        train_no_pool
    )
    < TRAIN_PER_CLASS
):

    raise RuntimeError(
        "Not enough balanced training examples."
    )


if (
    len(
        val_yes_pool
    )
    < VAL_PER_CLASS
    or
    len(
        val_no_pool
    )
    < VAL_PER_CLASS
):

    raise RuntimeError(
        "Not enough balanced validation examples. "
        f"Requested {VAL_PER_CLASS}/class."
    )


# ============================================================
# Participant-aware deterministic sampler
#
# Spreads selection across participants rather than letting
# one participant dominate the tiny diagnostic.
# ============================================================

def participant_round_robin_sample(
    examples,
    n,
    seed,
):

    rng = random.Random(
        seed
    )


    grouped = defaultdict(
        list
    )


    for example in examples:

        participant = (
            example.get(
                "_participant_id"
            )
        )


        grouped[
            participant
        ].append(
            example
        )


    participant_ids = list(
        grouped.keys()
    )


    rng.shuffle(
        participant_ids
    )


    for participant in participant_ids:

        rng.shuffle(
            grouped[
                participant
            ]
        )


    selected = []


    while (
        len(
            selected
        )
        < n
    ):

        made_progress = False


        for participant in participant_ids:

            if (
                grouped[
                    participant
                ]
            ):

                selected.append(
                    grouped[
                        participant
                    ].pop()
                )


                made_progress = True


                if (
                    len(
                        selected
                    )
                    == n
                ):

                    break


        if not made_progress:

            raise RuntimeError(
                "Could not sample requested "
                "number of examples."
            )


    return selected


# ============================================================
# Select examples
# ============================================================

selected_train_yes = (
    participant_round_robin_sample(
        train_yes_pool,
        TRAIN_PER_CLASS,
        SEED + 101,
    )
)


selected_train_no = (
    participant_round_robin_sample(
        train_no_pool,
        TRAIN_PER_CLASS,
        SEED + 102,
    )
)


selected_val_yes = (
    participant_round_robin_sample(
        val_yes_pool,
        VAL_PER_CLASS,
        SEED + 201,
    )
)


selected_val_no = (
    participant_round_robin_sample(
        val_no_pool,
        VAL_PER_CLASS,
        SEED + 202,
    )
)


# ============================================================
# Participant split audit
# ============================================================

train_participants = {
    example.get(
        "_participant_id"
    )

    for example
    in (
        selected_train_yes
        + selected_train_no
    )
}


val_participants = {
    example.get(
        "_participant_id"
    )

    for example
    in (
        selected_val_yes
        + selected_val_no
    )
}


participant_overlap = (
    train_participants
    & val_participants
)


if participant_overlap:

    raise RuntimeError(
        "TRAIN/VALIDATION participant leakage: "
        f"{sorted(participant_overlap)}"
    )


print(
    "\nParticipant split audit:"
)

print(
    "  selected train participants:",
    len(
        train_participants
    ),
)

print(
    "  selected validation participants:",
    len(
        val_participants
    ),
)

print(
    "  overlap:",
    len(
        participant_overlap
    ),
)


# ============================================================
# Common invariant prompt
#
# CRITICAL:
# There is ZERO per-example textual signal information.
# ============================================================

COMMON_PRE_PROMPT = (
    "Analyze the continuous glucose monitor "
    "signal and answer the factual question.\n"
)


COMMON_TS_TEXT = (
    "This is a continuous glucose monitor (CGM) "
    "signal measured in mg/dL. "
    f"It contains {WINDOW_SIZE} readings sampled "
    "approximately every 5 minutes. "
    "The numeric CGM signal was normalized using "
    "the same fixed training-set normalization for "
    "every segment: "
    f"mean {train_mean:.3f} mg/dL and "
    f"standard deviation {train_std:.3f} mg/dL."
)


COMMON_POST_PROMPT = (
    "\nQuestion: Did glucose exceed "
    "180 mg/dL during this segment?\n"
    "Answer Yes or No.\n"
    "Answer:"
)


# ============================================================
# Convert existing examples to grounding examples
# ============================================================

def make_grounding_example(
    base_example,
):

    signal = (
        base_example[
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


    is_yes = (
        base_example[
            "_answer_clean"
        ]
        == "(a)"
    )


    answer = (
        " Yes."
        if is_yes
        else " No."
    )


    return {
        "pre_prompt":
            COMMON_PRE_PROMPT,

        "time_series":
            [
                signal
            ],

        "time_series_text":
            [
                COMMON_TS_TEXT
            ],

        "post_prompt":
            COMMON_POST_PROMPT,

        "answer":
            (
                answer
                + EOS_TOKEN
            ),

        # --------------------------------------------
        # Metadata NEVER exposed to model
        # --------------------------------------------

        "_is_yes":
            is_yes,

        "_participant_id":
            base_example.get(
                "_participant_id"
            ),

        "_window_start":
            base_example.get(
                "_window_start"
            ),

        "_window_max":
            base_example.get(
                "_window_max"
            ),

        "_signal_source":
            "real",

        "_donor_is_yes":
            None,
    }


train_examples = [
    make_grounding_example(
        example
    )

    for example
    in (
        selected_train_yes
        + selected_train_no
    )
]


val_real_examples = [
    make_grounding_example(
        example
    )

    for example
    in (
        selected_val_yes
        + selected_val_no
    )
]


# ============================================================
# Verify textual input is literally invariant
# ============================================================

unique_text_inputs = {
    (
        example[
            "pre_prompt"
        ],
        example[
            "time_series_text"
        ][
            0
        ],
        example[
            "post_prompt"
        ],
    )

    for example
    in (
        train_examples
        + val_real_examples
    )
}


print(
    "\nUnique prompt/text combinations:",
    len(
        unique_text_inputs
    ),
)


if (
    len(
        unique_text_inputs
    )
    != 1
):

    raise RuntimeError(
        "Per-example textual variation detected."
    )


print(
    "✓ Text is identical across every example."
)


# ============================================================
# Balance validation order
# ============================================================

def alternate_classes(
    examples,
):

    yes = [
        example

        for example
        in examples

        if example[
            "_is_yes"
        ]
    ]


    no = [
        example

        for example
        in examples

        if not example[
            "_is_yes"
        ]
    ]


    if (
        len(
            yes
        )
        != len(
            no
        )
    ):

        raise RuntimeError(
            "Expected balanced examples."
        )


    output = []


    for y, n in zip(
        yes,
        no,
    ):

        output.append(
            y
        )

        output.append(
            n
        )


    return output


train_examples = (
    alternate_classes(
        train_examples
    )
)


val_real_examples = (
    alternate_classes(
        val_real_examples
    )
)


# ============================================================
# Build ZERO validation
# ============================================================

def clone_with_signal(
    example,
    signal,
    source,
    donor_is_yes=None,
):

    cloned = {
        key:
            value

        for key, value
        in example.items()
    }


    cloned[
        "time_series"
    ] = [
        signal
        .detach()
        .clone()
        .float()
    ]


    cloned[
        "_signal_source"
    ] = source


    cloned[
        "_donor_is_yes"
    ] = donor_is_yes


    return cloned


val_zero_examples = []


for example in val_real_examples:

    original_signal = (
        example[
            "time_series"
        ][
            0
        ]
    )


    val_zero_examples.append(
        clone_with_signal(
            example,
            torch.zeros_like(
                original_signal
            ),
            source="zero",
            donor_is_yes=None,
        )
    )


# ============================================================
# Build OPPOSITE-LABEL shuffled validation
#
# Every YES example receives a NO waveform.
# Every NO example receives a YES waveform.
#
# This is stronger than an arbitrary permutation.
# ============================================================

val_yes_examples = [
    example

    for example
    in val_real_examples

    if example[
        "_is_yes"
    ]
]


val_no_examples = [
    example

    for example
    in val_real_examples

    if not example[
        "_is_yes"
    ]
]


shuffle_rng = random.Random(
    SEED + 999
)


yes_donors = list(
    val_yes_examples
)

no_donors = list(
    val_no_examples
)


shuffle_rng.shuffle(
    yes_donors
)

shuffle_rng.shuffle(
    no_donors
)


# We need counters because validation examples are alternating.

yes_donor_index = 0
no_donor_index = 0

val_shuffled_examples = []


for original in val_real_examples:

    if original[
        "_is_yes"
    ]:

        donor = (
            no_donors[
                no_donor_index
            ]
        )

        no_donor_index += 1


    else:

        donor = (
            yes_donors[
                yes_donor_index
            ]
        )

        yes_donor_index += 1


    donor_signal = (
        donor[
            "time_series"
        ][
            0
        ]
    )


    val_shuffled_examples.append(
        clone_with_signal(
            original,
            donor_signal,
            source="opposite_label_shuffle",
            donor_is_yes=donor[
                "_is_yes"
            ],
        )
    )


# ============================================================
# Shuffled audit
# ============================================================

for example in val_shuffled_examples:

    if (
        example[
            "_donor_is_yes"
        ]
        == example[
            "_is_yes"
        ]
    ):

        raise RuntimeError(
            "Shuffled donor did not have "
            "opposite label."
        )


print(
    "✓ Every shuffled validation signal "
    "comes from the opposite label."
)


# ============================================================
# Print selected split
# ============================================================

print(
    "\nTRAIN diagnostic set:"
)

print(
    "  total:",
    len(
        train_examples
    ),
)

print(
    "  YES:",
    sum(
        example[
            "_is_yes"
        ]

        for example
        in train_examples
    ),
)

print(
    "  NO:",
    sum(
        not example[
            "_is_yes"
        ]

        for example
        in train_examples
    ),
)


print(
    "\nVALIDATION diagnostic set:"
)

print(
    "  total:",
    len(
        val_real_examples
    ),
)

print(
    "  YES:",
    sum(
        example[
            "_is_yes"
        ]

        for example
        in val_real_examples
    ),
)

print(
    "  NO:",
    sum(
        not example[
            "_is_yes"
        ]

        for example
        in val_real_examples
    ),
)


# ============================================================
# Release temporary model
# ============================================================

del temporary_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Load FRESH original Stage-5 for actual experiment
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING FRESH ORIGINAL STAGE-5"
)

print(
    "=" * 100
)


model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


# ============================================================
# Decision token IDs
# ============================================================

yes_ids = (
    model.tokenizer(
        " Yes",
        add_special_tokens=False,
    ).input_ids
)


no_ids = (
    model.tokenizer(
        " No",
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
        "Unable to tokenize Yes/No."
    )


YES_TOKEN = int(
    yes_ids[
        0
    ]
)


NO_TOKEN = int(
    no_ids[
        0
    ]
)


if (
    YES_TOKEN
    == NO_TOKEN
):

    raise RuntimeError(
        "Yes and No decision tokens identical."
    )


print(
    "\nDecision tokens:"
)

print(
    " YES:",
    yes_ids,
    "->",
    YES_TOKEN,
    model.tokenizer.convert_ids_to_tokens(
        [
            YES_TOKEN
        ]
    ),
)

print(
    " NO:",
    no_ids,
    "->",
    NO_TOKEN,
    model.tokenizer.convert_ids_to_tokens(
        [
            NO_TOKEN
        ]
    ),
)


# ============================================================
# Freeze everything
# ============================================================

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


if (
    len(
        layers
    )
    != 6
):

    raise RuntimeError(
        "Expected six encoder transformer layers."
    )


# ============================================================
# Unfreeze T5/T6
# ============================================================

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


# ============================================================
# Unfreeze projector
# ============================================================

for parameter in (
    model.projector.parameters()
):

    parameter.requires_grad = True


# ============================================================
# Unfreeze LoRA ONLY
# ============================================================

for name, parameter in (
    model.llm.named_parameters()
):

    if (
        "lora_"
        in name.lower()
    ):

        parameter.requires_grad = True


# ============================================================
# Parameter groups
# ============================================================

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


lora_named_parameters = [
    (
        name,
        parameter,
    )

    for name, parameter
    in model.llm.named_parameters()

    if (
        parameter.requires_grad
        and "lora_"
        in name.lower()
    )
]


lora_parameters = [
    parameter

    for _,
    parameter
    in lora_named_parameters
]


other_llm_trainable = [
    name

    for name, parameter
    in model.llm.named_parameters()

    if (
        parameter.requires_grad
        and "lora_"
        not in name.lower()
    )
]


if other_llm_trainable:

    raise RuntimeError(
        "Non-LoRA LLM weights unexpectedly trainable: "
        f"{other_llm_trainable[:10]}"
    )


# ============================================================
# Audit early encoder
# ============================================================

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
        "T1-T4 / early encoder unexpectedly trainable."
    )


print(
    "\n"
    + "=" * 100
)

print(
    "TRAINABLE PARAMETER AUDIT"
)

print(
    "=" * 100
)


print(
    "Early encoder:",
    f"{early_encoder_trainable:,}"
)

print(
    "T5/T6:",
    f"{sum(p.numel() for p in late_parameters):,}"
)

print(
    "Projector:",
    f"{sum(p.numel() for p in projector_parameters):,}"
)

print(
    "LoRA:",
    f"{sum(p.numel() for p in lora_parameters):,}"
)


# ============================================================
# Deterministic capacity / grounding test
#
# eval() disables dropout but NOT gradients.
# ============================================================

model.encoder.eval()

model.projector.eval()

model.llm.eval()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    [
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

        {
            "params":
                lora_parameters,

            "lr":
                LR_LORA,

            "weight_decay":
                0.0,
        },
    ]
)


# ============================================================
# Pairwise batch loss
# ============================================================

def pairwise_loss_batch(
    batch,
):

    (
        inputs_embeds,
        attention_mask,
    ) = (
        model.pad_and_apply_batch(
            batch
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


    pair_logits_list = []

    targets = []


    for index, example in enumerate(
        batch
    ):

        last_position = (
            int(
                attention_mask[
                    index
                ]
                .sum()
                .item()
            )
            - 1
        )


        logits = (
            outputs.logits[
                index,
                last_position,
                :
            ]
        )


        pair_logits_list.append(
            torch.stack(
                [
                    logits[
                        YES_TOKEN
                    ],

                    logits[
                        NO_TOKEN
                    ],
                ]
            )
        )


        targets.append(
            0
            if example[
                "_is_yes"
            ]
            else 1
        )


    pair_logits = (
        torch.stack(
            pair_logits_list,
            dim=0,
        )
        .float()
    )


    target_tensor = torch.tensor(
        targets,
        dtype=torch.long,
        device=device,
    )


    return F.cross_entropy(
        pair_logits,
        target_tensor,
    )


# ============================================================
# Balanced accuracy helper
# ============================================================

def balanced_accuracy_from_predictions(
    predictions,
    labels,
):

    yes_correct = 0
    yes_total = 0

    no_correct = 0
    no_total = 0


    for prediction, label in zip(
        predictions,
        labels,
    ):

        if label:

            yes_total += 1


            if prediction:

                yes_correct += 1


        else:

            no_total += 1


            if not prediction:

                no_correct += 1


    if (
        yes_total
        == 0
        or no_total
        == 0
    ):

        return None


    return (
        0.5
        * (
            yes_correct
            / yes_total
            +
            no_correct
            / no_total
        )
    )


# ============================================================
# Pairwise evaluation
# ============================================================

def evaluate_pairwise(
    examples,
):

    predictions = []

    gold_labels = []

    donor_labels = []

    p_correct_values = []

    margins = []


    with torch.no_grad():

        for example in examples:

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


            pair_logits = (
                torch.stack(
                    [
                        logits[
                            YES_TOKEN
                        ],

                        logits[
                            NO_TOKEN
                        ],
                    ]
                )
                .float()
            )


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


            prediction = (
                p_yes
                >= p_no
            )


            gold = (
                example[
                    "_is_yes"
                ]
            )


            donor = (
                example.get(
                    "_donor_is_yes"
                )
            )


            predictions.append(
                prediction
            )


            gold_labels.append(
                gold
            )


            donor_labels.append(
                donor
            )


            p_correct_values.append(
                p_yes
                if gold
                else p_no
            )


            margins.append(
                float(
                    (
                        pair_logits[
                            0
                        ]
                        -
                        pair_logits[
                            1
                        ]
                    ).item()
                )
            )


    original_ba = (
        balanced_accuracy_from_predictions(
            predictions,
            gold_labels,
        )
    )


    donor_ba = None


    if all(
        donor is not None
        for donor
        in donor_labels
    ):

        donor_ba = (
            balanced_accuracy_from_predictions(
                predictions,
                donor_labels,
            )
        )


    predicted_yes = sum(
        predictions
    )


    margin_tensor = torch.tensor(
        margins,
        dtype=torch.float32,
    )


    return {
        "balanced_accuracy_vs_original":
            original_ba,

        "balanced_accuracy_vs_donor":
            donor_ba,

        "predicted_yes":
            predicted_yes,

        "predicted_no":
            len(
                predictions
            )
            - predicted_yes,

        "mean_p_correct_original":
            sum(
                p_correct_values
            )
            / len(
                p_correct_values
            ),

        "margin_sd":
            float(
                margin_tensor.std().item()
            ),
    }


# ============================================================
# Generation evaluation
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
    examples,
):

    predictions = []

    gold_labels = []

    donor_labels = []

    valid = 0

    samples = []


    with torch.no_grad():

        for example in examples:

            generated = (
                model.generate(
                    [
                        example
                    ],
                    max_new_tokens=4,
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


            donor = (
                example.get(
                    "_donor_is_yes"
                )
            )


            if (
                prediction
                is not None
            ):

                valid += 1


            predictions.append(
                prediction
            )


            gold_labels.append(
                gold
            )


            donor_labels.append(
                donor
            )


            samples.append(
                {
                    "gold_is_yes":
                        gold,

                    "donor_is_yes":
                        donor,

                    "prediction":
                        prediction,

                    "signal_source":
                        example[
                            "_signal_source"
                        ],

                    "participant_id":
                        example[
                            "_participant_id"
                        ],

                    "window_start":
                        example[
                            "_window_start"
                        ],

                    "raw_generation":
                        generated,
                }
            )


    # Invalid generation counts as incorrect.

    original_predictions = [
        prediction

        if prediction
        is not None

        else (
            not gold
        )

        for prediction, gold
        in zip(
            predictions,
            gold_labels,
        )
    ]


    original_ba = (
        balanced_accuracy_from_predictions(
            original_predictions,
            gold_labels,
        )
    )


    donor_ba = None


    if all(
        donor is not None
        for donor
        in donor_labels
    ):

        donor_predictions = [
            prediction

            if prediction
            is not None

            else (
                not donor
            )

            for prediction, donor
            in zip(
                predictions,
                donor_labels,
            )
        ]


        donor_ba = (
            balanced_accuracy_from_predictions(
                donor_predictions,
                donor_labels,
            )
        )


    return {
        "valid_rate":
            valid
            / len(
                examples
            ),

        "balanced_accuracy_vs_original":
            original_ba,

        "balanced_accuracy_vs_donor":
            donor_ba,

        "samples":
            samples,
    }


# ============================================================
# Representation diagnostic on REAL validation
# ============================================================

def representation_stats(
    examples,
):

    encoder_vectors = []

    projector_vectors = []

    labels = []


    patch_size = (
        model.patch_size
        if hasattr(
            model,
            "patch_size",
        )
        else model.encoder.patch_size
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
        )


        feature_sd = float(
            matrix.std(
                dim=0
            )
            .mean()
            .item()
        )


        yes_indices = [
            index

            for index, label
            in enumerate(
                labels
            )

            if label
        ]


        no_indices = [
            index

            for index, label
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

            "yes_no_centroid_cosine":
                cosine,

            "yes_no_centroid_l2_per_sqrt_dim":
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
# Gradient helper
# ============================================================

def grad_norm(
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
# Best-state helpers
#
# Save only trainable modules/parameters, not the 1.2B model.
# ============================================================

def snapshot_trainable_state():

    return {
        "t5":
            {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in layers[
                    4
                ].state_dict().items()
            },

        "t6":
            {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in layers[
                    5
                ].state_dict().items()
            },

        "projector":
            {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.projector.state_dict().items()
            },

        "lora":
            {
                name:
                    parameter.detach()
                    .cpu()
                    .clone()

                for name, parameter
                in lora_named_parameters
            },
    }


def restore_trainable_state(
    state,
):

    layers[
        4
    ].load_state_dict(
        state[
            "t5"
        ],
        strict=True,
    )


    layers[
        5
    ].load_state_dict(
        state[
            "t6"
        ],
        strict=True,
    )


    model.projector.load_state_dict(
        state[
            "projector"
        ],
        strict=True,
    )


    with torch.no_grad():

        current_lora = dict(
            model.llm.named_parameters()
        )


        for name, value in (
            state[
                "lora"
            ].items()
        ):

            current_lora[
                name
            ].copy_(
                value.to(
                    current_lora[
                        name
                    ].device
                )
            )


# ============================================================
# Evaluation bundle
# ============================================================

def evaluate_all_pairwise():

    train_real = (
        evaluate_pairwise(
            train_examples
        )
    )


    val_real = (
        evaluate_pairwise(
            val_real_examples
        )
    )


    val_zero = (
        evaluate_pairwise(
            val_zero_examples
        )
    )


    val_shuffle = (
        evaluate_pairwise(
            val_shuffled_examples
        )
    )


    representation = (
        representation_stats(
            val_real_examples
        )
    )


    return {
        "train_real":
            train_real,

        "val_real":
            val_real,

        "val_zero":
            val_zero,

        "val_shuffle":
            val_shuffle,

        "val_real_representation":
            representation,
    }


# ============================================================
# Initial evaluation
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "BEFORE TRAINING"
)

print(
    "=" * 110
)


initial_metrics = (
    evaluate_all_pairwise()
)


print(
    "TRAIN real BA:",
    f"{100 * initial_metrics['train_real']['balanced_accuracy_vs_original']:.1f}%"
)


print(
    "VAL real BA:",
    f"{100 * initial_metrics['val_real']['balanced_accuracy_vs_original']:.1f}%"
)


print(
    "VAL zero BA:",
    f"{100 * initial_metrics['val_zero']['balanced_accuracy_vs_original']:.1f}%"
)


print(
    "VAL shuffled BA vs original:",
    f"{100 * initial_metrics['val_shuffle']['balanced_accuracy_vs_original']:.1f}%"
)


print(
    "VAL shuffled BA vs donor:",
    f"{100 * initial_metrics['val_shuffle']['balanced_accuracy_vs_donor']:.1f}%"
)


print(
    "VAL encoder SD:",
    f"{initial_metrics['val_real_representation']['encoder']['feature_sd']:.8f}"
)


# ============================================================
# Best checkpoint tracking
#
# Selection criterion = REAL validation BA only.
# Ablation performance is NOT used for early stopping.
# ============================================================

best_epoch = 0

best_val_real_ba = (
    initial_metrics[
        "val_real"
    ][
        "balanced_accuracy_vs_original"
    ]
)

best_state = (
    snapshot_trainable_state()
)


history = [
    {
        "epoch":
            0,

        "mean_sequence_loss":
            None,

        "mean_pairwise_loss":
            None,

        "mean_total_loss":
            None,

        "metrics":
            initial_metrics,
    }
]


first_gradient_audit = None


# ============================================================
# Training
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "TRAINING"
)

print(
    "=" * 110
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    # --------------------------------------------------------
    # Deterministic epoch shuffle
    # --------------------------------------------------------

    epoch_examples = list(
        train_examples
    )


    epoch_rng = random.Random(
        SEED
        + 1000
        + epoch
    )


    epoch_rng.shuffle(
        epoch_examples
    )


    batches = [
        epoch_examples[
            start:
            start
            + BATCH_SIZE
        ]

        for start
        in range(
            0,
            len(
                epoch_examples
            ),
            BATCH_SIZE,
        )
    ]


    sequence_losses = []

    pairwise_losses = []

    total_losses = []


    for batch in batches:

        optimizer.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Natural-language sequence objective
        # ----------------------------------------------------

        sequence_loss = (
            model.compute_loss(
                batch
            )
        )


        # ----------------------------------------------------
        # Explicit Yes-vs-No discrimination
        # ----------------------------------------------------

        pair_loss = (
            pairwise_loss_batch(
                batch
            )
        )


        total_loss = (
            SEQUENCE_WEIGHT
            * sequence_loss
            +
            PAIRWISE_WEIGHT
            * pair_loss
        )


        total_loss.backward()


        # ----------------------------------------------------
        # First-step raw gradient audit
        # ----------------------------------------------------

        if (
            first_gradient_audit
            is None
        ):

            first_gradient_audit = {
                "t5_t6":
                    grad_norm(
                        late_parameters
                    ),

                "projector":
                    grad_norm(
                        projector_parameters
                    ),

                "lora":
                    grad_norm(
                        lora_parameters
                    ),
            }


            print(
                "\nFirst gradients BEFORE clipping:"
            )

            print(
                "  T5/T6:",
                f"{first_gradient_audit['t5_t6']:.8e}"
            )

            print(
                "  Projector:",
                f"{first_gradient_audit['projector']:.8e}"
            )

            print(
                "  LoRA:",
                f"{first_gradient_audit['lora']:.8e}"
            )


        # ----------------------------------------------------
        # CRITICAL:
        # clip groups separately.
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            late_parameters,
            CLIP_T5_T6,
        )


        torch.nn.utils.clip_grad_norm_(
            projector_parameters,
            CLIP_PROJECTOR,
        )


        torch.nn.utils.clip_grad_norm_(
            lora_parameters,
            CLIP_LORA,
        )


        optimizer.step()


        sequence_losses.append(
            float(
                sequence_loss
                .detach()
                .float()
                .cpu()
                .item()
            )
        )


        pairwise_losses.append(
            float(
                pair_loss
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


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if epoch in EVAL_EPOCHS:

        metrics = (
            evaluate_all_pairwise()
        )


        mean_sequence_loss = (
            sum(
                sequence_losses
            )
            / len(
                sequence_losses
            )
        )


        mean_pairwise_loss = (
            sum(
                pairwise_losses
            )
            / len(
                pairwise_losses
            )
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

                "mean_sequence_loss":
                    mean_sequence_loss,

                "mean_pairwise_loss":
                    mean_pairwise_loss,

                "mean_total_loss":
                    mean_total_loss,

                "metrics":
                    metrics,
            }
        )


        train_ba = (
            metrics[
                "train_real"
            ][
                "balanced_accuracy_vs_original"
            ]
        )


        real_ba = (
            metrics[
                "val_real"
            ][
                "balanced_accuracy_vs_original"
            ]
        )


        zero_ba = (
            metrics[
                "val_zero"
            ][
                "balanced_accuracy_vs_original"
            ]
        )


        shuffle_original_ba = (
            metrics[
                "val_shuffle"
            ][
                "balanced_accuracy_vs_original"
            ]
        )


        shuffle_donor_ba = (
            metrics[
                "val_shuffle"
            ][
                "balanced_accuracy_vs_donor"
            ]
        )


        encoder_sd = (
            metrics[
                "val_real_representation"
            ][
                "encoder"
            ][
                "feature_sd"
            ]
        )


        print(
            f"Epoch {epoch:2d} | "
            f"seq={mean_sequence_loss:7.4f} | "
            f"pair={mean_pairwise_loss:7.4f} | "
            f"TRAIN={100 * train_ba:5.1f}% | "
            f"REAL={100 * real_ba:5.1f}% | "
            f"ZERO={100 * zero_ba:5.1f}% | "
            f"SHUF-orig={100 * shuffle_original_ba:5.1f}% | "
            f"SHUF-donor={100 * shuffle_donor_ba:5.1f}% | "
            f"EncSD={encoder_sd:.6f}"
        )


        # ----------------------------------------------------
        # Early-stopping state selected ONLY by real VAL BA.
        # ----------------------------------------------------

        if (
            real_ba
            >
            best_val_real_ba
            + 1e-12
        ):

            best_val_real_ba = (
                real_ba
            )


            best_epoch = (
                epoch
            )


            best_state = (
                snapshot_trainable_state()
            )


            print(
                f"  ✓ New best REAL validation BA "
                f"at epoch {epoch}."
            )


# ============================================================
# Restore best REAL-validation epoch
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "RESTORING BEST REAL-VALIDATION STATE"
)

print(
    "=" * 110
)


print(
    "Best epoch:",
    best_epoch,
)

print(
    "Best REAL validation BA:",
    f"{100 * best_val_real_ba:.1f}%"
)


restore_trainable_state(
    best_state
)


# ============================================================
# Final pairwise signal-necessity evaluation
# ============================================================

final_metrics = (
    evaluate_all_pairwise()
)


# ============================================================
# Final generation ablations
# ============================================================

print(
    "\nRunning final greedy generation "
    "on REAL / ZERO / SHUFFLED validation..."
)


generation_real = (
    evaluate_generation(
        val_real_examples
    )
)


generation_zero = (
    evaluate_generation(
        val_zero_examples
    )
)


generation_shuffle = (
    evaluate_generation(
        val_shuffled_examples
    )
)


# ============================================================
# Final summary
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "CGM SIGNAL-NECESSITY RESULT"
)

print(
    "=" * 120
)


train_ba = (
    final_metrics[
        "train_real"
    ][
        "balanced_accuracy_vs_original"
    ]
)


real_ba = (
    final_metrics[
        "val_real"
    ][
        "balanced_accuracy_vs_original"
    ]
)


zero_ba = (
    final_metrics[
        "val_zero"
    ][
        "balanced_accuracy_vs_original"
    ]
)


shuffle_original_ba = (
    final_metrics[
        "val_shuffle"
    ][
        "balanced_accuracy_vs_original"
    ]
)


shuffle_donor_ba = (
    final_metrics[
        "val_shuffle"
    ][
        "balanced_accuracy_vs_donor"
    ]
)


print(
    "Best epoch:",
    best_epoch
)


print(
    "\nPAIRWISE"
)


print(
    "Train REAL:",
    f"{100 * train_ba:.1f}%"
)


print(
    "Validation REAL:",
    f"{100 * real_ba:.1f}%"
)


print(
    "Validation ZERO:",
    f"{100 * zero_ba:.1f}%"
)


print(
    "Validation SHUFFLED vs original label:",
    f"{100 * shuffle_original_ba:.1f}%"
)


print(
    "Validation SHUFFLED vs donor-signal label:",
    f"{100 * shuffle_donor_ba:.1f}%"
)


print(
    "\nSignal necessity gaps:"
)


print(
    "REAL - ZERO:",
    f"{100 * (real_ba - zero_ba):+.1f} pp"
)


print(
    "REAL - SHUFFLED(original):",
    f"{100 * (real_ba - shuffle_original_ba):+.1f} pp"
)


print(
    "\nGENERATION"
)


print(
    "REAL valid:",
    f"{100 * generation_real['valid_rate']:.1f}%"
)


print(
    "REAL BA:",
    f"{100 * generation_real['balanced_accuracy_vs_original']:.1f}%"
)


print(
    "ZERO valid:",
    f"{100 * generation_zero['valid_rate']:.1f}%"
)


print(
    "ZERO BA:",
    f"{100 * generation_zero['balanced_accuracy_vs_original']:.1f}%"
)


print(
    "SHUFFLED valid:",
    f"{100 * generation_shuffle['valid_rate']:.1f}%"
)


print(
    "SHUFFLED BA vs original:",
    f"{100 * generation_shuffle['balanced_accuracy_vs_original']:.1f}%"
)


print(
    "SHUFFLED BA vs donor signal:",
    f"{100 * generation_shuffle['balanced_accuracy_vs_donor']:.1f}%"
)


# ============================================================
# Representation
# ============================================================

representation = (
    final_metrics[
        "val_real_representation"
    ]
)


print(
    "\nREPRESENTATION — REAL VALIDATION"
)


print(
    "Encoder feature SD:",
    f"{representation['encoder']['feature_sd']:.8f}"
)


print(
    "Encoder YES/NO cosine:",
    f"{representation['encoder']['yes_no_centroid_cosine']:.8f}"
)


print(
    "Encoder YES/NO L2/sqrt(dim):",
    f"{representation['encoder']['yes_no_centroid_l2_per_sqrt_dim']:.8f}"
)


print(
    "Projector feature SD:",
    f"{representation['projector']['feature_sd']:.8f}"
)


print(
    "Projector YES/NO cosine:",
    f"{representation['projector']['yes_no_centroid_cosine']:.8f}"
)


# ============================================================
# Representative generations
# ============================================================

def print_samples(
    title,
    result,
    n=6,
):

    print(
        "\n"
        + title
    )


    for item in (
        result[
            "samples"
        ][
            :n
        ]
    ):

        donor_text = ""


        if (
            item[
                "donor_is_yes"
            ]
            is not None
        ):

            donor_text = (
                " | donor="
                + (
                    "YES"
                    if item[
                        "donor_is_yes"
                    ]
                    else "NO"
                )
            )


        print(
            "gold="
            + (
                "YES"
                if item[
                    "gold_is_yes"
                ]
                else "NO"
            )
            + donor_text
            + " | "
            + repr(
                item[
                    "raw_generation"
                ]
            )
        )


print_samples(
    "REAL examples:",
    generation_real,
)


print_samples(
    "ZERO examples:",
    generation_zero,
)


print_samples(
    "SHUFFLED examples:",
    generation_shuffle,
)


# ============================================================
# Interpretation helper
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "INTERPRETATION HELPER"
)

print(
    "=" * 120
)


if (
    real_ba >= 0.75
    and zero_ba <= 0.60
    and shuffle_original_ba <= 0.60
    and shuffle_donor_ba >= 0.70
):

    print(
        "STRONG SIGNAL-DEPENDENCE PATTERN:"
    )

    print(
        "Real validation works, zero/shuffled signals "
        "break original-label performance, and predictions "
        "follow the shuffled donor waveform."
    )


elif (
    real_ba >= 0.75
    and zero_ba >= 0.70
):

    print(
        "LIKELY NON-SIGNAL SHORTCUT:"
    )

    print(
        "Real and zero-waveform validation are both strong."
    )


elif (
    train_ba >= 0.90
    and real_ba <= 0.60
):

    print(
        "TRAIN-SET MEMORIZATION WITHOUT "
        "HELD-OUT GENERALIZATION."
    )


elif (
    real_ba > zero_ba
    and real_ba > shuffle_original_ba
):

    print(
        "PARTIAL SIGNAL DEPENDENCE:"
    )

    print(
        "Real CGM performs better than the ablations, "
        "but the evidence is not yet decisive."
    )


else:

    print(
        "NO CLEAR SIGNAL-DEPENDENCE PATTERN YET."
    )


# ============================================================
# Serialize selections
# ============================================================

def serializable_example_metadata(
    example,
):

    return {
        "participant_id":
            example[
                "_participant_id"
            ],

        "window_start":
            example[
                "_window_start"
            ],

        "is_yes":
            example[
                "_is_yes"
            ],

        "window_max":
            example[
                "_window_max"
            ],
    }


# ============================================================
# Save
# ============================================================

results = {
    "experiment":
        "cgm_signal_necessity",

    "stage5_repo":
        STAGE5_REPO_ID,

    "seed":
        SEED,

    "task":
        TASK_ID,

    "normalization":
        {
            "type":
                "fixed_global",

            "mean":
                train_mean,

            "std":
                train_std,
        },

    "text_interface":
        {
            "per_example_numeric_metadata":
                False,

            "common_pre_prompt":
                COMMON_PRE_PROMPT,

            "common_time_series_text":
                COMMON_TS_TEXT,

            "common_post_prompt":
                COMMON_POST_PROMPT,
        },

    "train_per_class":
        TRAIN_PER_CLASS,

    "validation_per_class":
        VAL_PER_CLASS,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "best_epoch":
        best_epoch,

    "hyperparameters":
        {
            "lr_t5_t6":
                LR_T5_T6,

            "lr_projector":
                LR_PROJECTOR,

            "lr_lora":
                LR_LORA,

            "sequence_weight":
                SEQUENCE_WEIGHT,

            "pairwise_weight":
                PAIRWISE_WEIGHT,

            "clip_t5_t6":
                CLIP_T5_T6,

            "clip_projector":
                CLIP_PROJECTOR,

            "clip_lora":
                CLIP_LORA,
        },

    "participant_audit":
        {
            "selected_train_participants":
                sorted(
                    str(x)
                    for x
                    in train_participants
                ),

            "selected_validation_participants":
                sorted(
                    str(x)
                    for x
                    in val_participants
                ),

            "overlap":
                sorted(
                    str(x)
                    for x
                    in participant_overlap
                ),

            "test_participants_sealed":
                len(
                    manifest[
                        "test"
                    ]
                ),
        },

    "selected_train_examples":
        [
            serializable_example_metadata(
                example
            )

            for example
            in train_examples
        ],

    "selected_validation_examples":
        [
            serializable_example_metadata(
                example
            )

            for example
            in val_real_examples
        ],

    "first_gradient_audit":
        first_gradient_audit,

    "history":
        history,

    "final_pairwise":
        final_metrics,

    "final_generation":
        {
            "real":
                generation_real,

            "zero":
                generation_zero,

            "shuffled_opposite":
                generation_shuffle,
        },
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
    "\nSaved:"
)

print(
    RESULTS_PATH
)


print(
    "\nNo checkpoint was saved."
)

print(
    "Validation participants were used only "
    "for held-out evaluation / epoch selection."
)

print(
    "TEST participants remain sealed."
)


# ============================================================
# Cleanup
# ============================================================

del optimizer
del model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()