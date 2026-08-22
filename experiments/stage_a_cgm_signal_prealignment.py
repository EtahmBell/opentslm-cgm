"""
STAGE A — CGM SIGNAL PREALIGNMENT

Goal
----
Teach the Stage-5 sensor pathway a simple, transferable CGM fact:

    Did glucose exceed 180 mg/dL?

This stage deliberately does NOT train natural-language generation.

We first want to establish:

    CGM waveform
        ↓
    T5 / T6 / projector
        ↓
    useful, transferable Yes-vs-No representation

Only after this works do we move to Stage B language alignment.


MODEL
-----
Start from the ORIGINAL public Stage-5 ECG checkpoint:

    OpenTSLM/llama-3.2-1b-ecg-sp

Frozen:
    - patch embedding
    - position/input components
    - Transformer 1-4
    - entire LLM
    - entire Stage-5 LoRA

Trainable:
    - Transformer 5
    - Transformer 6
    - projector


INPUT
-----
Fixed global CGM normalization:

    (raw_mg_dl - training_mean) / training_std

The text is IDENTICAL for every example.

No per-window:
    - mean
    - SD
    - max
    - TIR
    - participant info
    - window information

Therefore the label-changing information exists ONLY
inside the waveform.


TASK
----
Binary:

    Question:
    Did glucose exceed 180 mg/dL?

    (a) Yes
    (b) No

True pairwise loss only:

    CE([logit("(a"), logit("(b")], gold_class)

The rest of the vocabulary is excluded from the loss.


TRAINING
--------
80 training windows:
    40 YES
    40 NO

Participant-aware sampling from TRAIN participants only.

Single-example optimizer updates.

Label order alternates:

    YES
    NO
    YES
    NO
    ...

Within each class, example order is reshuffled each epoch.

30 epochs.


VALIDATION
----------
40 held-out windows:
    20 YES
    20 NO

From VALIDATION participants only.

TRAIN / VALIDATION participant overlap must be zero.

TEST participants remain SEALED.


SIGNAL-NECESSITY ABLATIONS
--------------------------
At evaluation time, both TRAIN and VALIDATION are tested with:

1. REAL
   Correct waveform.

2. ZERO
   Waveform replaced with zeros.

3. OPPOSITE-LABEL SHUFFLE
   YES prompt receives a NO waveform.
   NO prompt receives a YES waveform.

For shuffled data we report:

    BA vs ORIGINAL label
    BA vs DONOR-SIGNAL label


Strong signal-learning pattern:

    Train REAL         high
    Train ZERO         ~50%
    Train SHUF-orig    low
    Train SHUF-donor   high

    Val REAL           > ~70-75%
    Val ZERO           ~50%
    Val SHUF-orig      low
    Val SHUF-donor     high


If:
    Train REAL high
    Val REAL ~50%

then the sensor pathway memorized training windows
without learning a transferable CGM concept.


CHECKPOINT
----------
If REAL validation BA rises above the untrained 50% baseline,
the best REAL-validation state is saved as:

    checkpoints/cgm_stage_a_above180_best.pt

Only T5, T6, projector weights + metadata are saved.

No patient data are included.

TEST participants remain sealed.
"""

import gc
import json
import math
import random
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
    / "cgm_stage_a_signal_prealignment.json"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_stage_a_above180_best.pt"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

TRAIN_PER_CLASS = 40
VAL_PER_CLASS = 20

EPOCHS = 30


# ------------------------------------------------------------
# Optimization
# ------------------------------------------------------------

LR_T5_T6 = 2e-4

LR_PROJECTOR = 1e-4

WEIGHT_DECAY = 1e-2


# Clip the two trainable groups independently.
CLIP_T5_T6 = 1.0

CLIP_PROJECTOR = 1.0


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

    manifest = json.load(
        f
    )


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


if (
    validation_manifest
    is None
):

    validation_manifest = (
        manifest.get(
            "val"
        )
    )


if (
    validation_manifest
    is None
):

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
    len(
        train_files
    ),
)

print(
    "Validation participants:",
    len(
        validation_files
    ),
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
# Temporary model for EOS / tokenizer-compatible data builder
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
# Build TRAIN / VALIDATION pools
#
# TEST SPLIT IS NEVER LOADED.
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
    or len(
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
    or len(
        val_no_pool
    )
    < VAL_PER_CLASS
):

    raise RuntimeError(
        "Not enough balanced validation examples."
    )


# ============================================================
# Participant-aware deterministic sampler
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

        progress = False


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


                progress = True


                if (
                    len(
                        selected
                    )
                    == n
                ):

                    break


        if (
            not progress
        ):

            raise RuntimeError(
                "Could not sample enough examples."
            )


    return selected


# ============================================================
# Select data
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
# Participant leakage audit
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


overlap = (
    train_participants
    & val_participants
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
        overlap
    ),
)


if overlap:

    raise RuntimeError(
        "TRAIN / VALIDATION participant leakage: "
        f"{sorted(overlap)}"
    )


# ============================================================
# INVARIANT TEXT INTERFACE
#
# No example-specific text.
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
    "The numeric signal uses the same fixed "
    "normalization for every CGM segment: "
    f"training-set mean {train_mean:.3f} mg/dL and "
    f"training-set standard deviation "
    f"{train_std:.3f} mg/dL."
)


COMMON_POST_PROMPT = (
    "\nQuestion: Did glucose exceed "
    "180 mg/dL during this segment?\n"
    "(a) Yes\n"
    "(b) No\n"
    "Answer:"
)


# ============================================================
# Convert to Stage-A examples
# ============================================================

def make_stage_a_example(
    base_example,
):

    signal = (
        base_example[
            "time_series"
        ][
            0
        ]
    )


    if (
        not torch.is_tensor(
            signal
        )
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

        # Not used by Stage-A loss,
        # but retained for compatibility.
        "answer":
            (
                "(a)"
                if is_yes
                else "(b)"
            )
            + EOS_TOKEN,

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


train_yes = [
    make_stage_a_example(
        example
    )

    for example
    in selected_train_yes
]


train_no = [
    make_stage_a_example(
        example
    )

    for example
    in selected_train_no
]


val_yes = [
    make_stage_a_example(
        example
    )

    for example
    in selected_val_yes
]


val_no = [
    make_stage_a_example(
        example
    )

    for example
    in selected_val_no
]


# ============================================================
# Verify all text is identical
# ============================================================

all_real_examples = (
    train_yes
    + train_no
    + val_yes
    + val_no
)


unique_text = {
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
    in all_real_examples
}


print(
    "\nUnique prompt/text combinations:",
    len(
        unique_text
    ),
)


if (
    len(
        unique_text
    )
    != 1
):

    raise RuntimeError(
        "Per-example text variation detected."
    )


print(
    "✓ Text is identical across all examples."
)


# ============================================================
# Fixed evaluation order
# ============================================================

def interleave(
    yes_examples,
    no_examples,
):

    if (
        len(
            yes_examples
        )
        != len(
            no_examples
        )
    ):

        raise RuntimeError(
            "Expected balanced class lists."
        )


    output = []


    for yes_example, no_example in zip(
        yes_examples,
        no_examples,
    ):

        output.append(
            yes_example
        )

        output.append(
            no_example
        )


    return output


train_real = (
    interleave(
        train_yes,
        train_no,
    )
)


val_real = (
    interleave(
        val_yes,
        val_no,
    )
)


# ============================================================
# Clone signal helper
# ============================================================

def clone_with_signal(
    example,
    signal,
    source,
    donor_is_yes=None,
):

    cloned = dict(
        example
    )


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


# ============================================================
# ZERO ablation
# ============================================================

def make_zero_examples(
    real_examples,
):

    output = []


    for example in real_examples:

        signal = (
            example[
                "time_series"
            ][
                0
            ]
        )


        output.append(
            clone_with_signal(
                example,
                torch.zeros_like(
                    signal
                ),
                source="zero",
            )
        )


    return output


train_zero = (
    make_zero_examples(
        train_real
    )
)


val_zero = (
    make_zero_examples(
        val_real
    )
)


# ============================================================
# OPPOSITE-LABEL shuffle
# ============================================================

def make_opposite_shuffle(
    yes_examples,
    no_examples,
    seed,
):

    rng = random.Random(
        seed
    )


    yes_donors = list(
        yes_examples
    )


    no_donors = list(
        no_examples
    )


    rng.shuffle(
        yes_donors
    )


    rng.shuffle(
        no_donors
    )


    originals = (
        interleave(
            yes_examples,
            no_examples,
        )
    )


    yes_index = 0
    no_index = 0

    output = []


    for original in originals:

        if (
            original[
                "_is_yes"
            ]
        ):

            donor = (
                no_donors[
                    no_index
                ]
            )

            no_index += 1


        else:

            donor = (
                yes_donors[
                    yes_index
                ]
            )

            yes_index += 1


        donor_signal = (
            donor[
                "time_series"
            ][
                0
            ]
        )


        output.append(
            clone_with_signal(
                original,
                donor_signal,
                source="opposite_label_shuffle",
                donor_is_yes=donor[
                    "_is_yes"
                ],
            )
        )


    return output


train_shuffle = (
    make_opposite_shuffle(
        train_yes,
        train_no,
        SEED + 901,
    )
)


val_shuffle = (
    make_opposite_shuffle(
        val_yes,
        val_no,
        SEED + 902,
    )
)


# ============================================================
# Shuffle audit
# ============================================================

for example in (
    train_shuffle
    + val_shuffle
):

    if (
        example[
            "_donor_is_yes"
        ]
        == example[
            "_is_yes"
        ]
    ):

        raise RuntimeError(
            "Opposite-label shuffle failed."
        )


print(
    "✓ All shuffled signals come from "
    "the opposite class."
)


# ============================================================
# Dataset summary
# ============================================================

print(
    "\nTRAIN:"
)

print(
    "  total:",
    len(
        train_real
    ),
)

print(
    "  YES:",
    len(
        train_yes
    ),
)

print(
    "  NO:",
    len(
        train_no
    ),
)


print(
    "\nVALIDATION:"
)

print(
    "  total:",
    len(
        val_real
    ),
)

print(
    "  YES:",
    len(
        val_yes
    ),
)

print(
    "  NO:",
    len(
        val_no
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
# Fresh ORIGINAL Stage-5
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
# Pairwise decision tokens
# ============================================================

a_ids = (
    model.tokenizer(
        "(a)",
        add_special_tokens=False,
    ).input_ids
)


b_ids = (
    model.tokenizer(
        "(b)",
        add_special_tokens=False,
    ).input_ids
)


if (
    len(
        a_ids
    )
    == 0
    or len(
        b_ids
    )
    == 0
):

    raise RuntimeError(
        "Unable to tokenize (a)/(b)."
    )


A_TOKEN = int(
    a_ids[
        0
    ]
)


B_TOKEN = int(
    b_ids[
        0
    ]
)


if (
    A_TOKEN
    == B_TOKEN
):

    raise RuntimeError(
        "(a)/(b) decision tokens are identical."
    )


print(
    "\nDecision tokens:"
)

print(
    " (a):",
    a_ids,
    "->",
    A_TOKEN,
    model.tokenizer.convert_ids_to_tokens(
        [
            A_TOKEN
        ]
    ),
)


print(
    " (b):",
    b_ids,
    "->",
    B_TOKEN,
    model.tokenizer.convert_ids_to_tokens(
        [
            B_TOKEN
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
        "Expected six encoder Transformer layers."
    )


# ============================================================
# Train ONLY T5 / T6
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
# Train projector
# ============================================================

for parameter in (
    model.projector.parameters()
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


lora_trainable = [
    name

    for name, parameter
    in model.llm.named_parameters()

    if parameter.requires_grad
]


if lora_trainable:

    raise RuntimeError(
        "LLM / LoRA must be frozen during Stage A."
    )


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
    "LLM / LoRA:",
    "0"
)


# ============================================================
# Disable dropout while preserving gradients
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
    ]
)


# ============================================================
# Get pair logits
# ============================================================

def get_pair_logits(
    example,
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


    full_logits = (
        outputs.logits[
            0,
            last_position,
            :
        ]
    )


    return torch.stack(
        [
            full_logits[
                A_TOKEN
            ],

            full_logits[
                B_TOKEN
            ],
        ]
    )


# ============================================================
# Pairwise loss
# ============================================================

def get_pairwise_loss(
    example,
):

    pair_logits = (
        get_pair_logits(
            example
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
# Balanced accuracy
# ============================================================

def balanced_accuracy(
    predictions,
    labels,
):

    yes_total = 0
    yes_correct = 0

    no_total = 0
    no_correct = 0


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
# Evaluation
# ============================================================

def evaluate_pairwise(
    examples,
):

    predictions = []

    original_labels = []

    donor_labels = []

    correct_probabilities = []

    margins = []


    with torch.no_grad():

        for example in examples:

            pair_logits = (
                get_pair_logits(
                    example
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


            prediction = (
                p_yes
                >= p_no
            )


            original_label = (
                example[
                    "_is_yes"
                ]
            )


            donor_label = (
                example.get(
                    "_donor_is_yes"
                )
            )


            predictions.append(
                prediction
            )


            original_labels.append(
                original_label
            )


            donor_labels.append(
                donor_label
            )


            correct_probabilities.append(
                p_yes
                if original_label
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
        balanced_accuracy(
            predictions,
            original_labels,
        )
    )


    donor_ba = None


    if all(
        label is not None
        for label
        in donor_labels
    ):

        donor_ba = (
            balanced_accuracy(
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
        "ba_vs_original":
            original_ba,

        "ba_vs_donor":
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
                correct_probabilities
            )
            / len(
                correct_probabilities
            ),

        "margin_sd":
            float(
                margin_tensor.std().item()
            ),
    }


# ============================================================
# Representation diagnostics
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
# Gradient norm
# ============================================================

def grad_norm(
    parameters,
):

    squared = 0.0


    for parameter in parameters:

        if (
            parameter.grad
            is None
        ):

            continue


        value = float(
            parameter.grad
            .detach()
            .float()
            .norm()
            .item()
        )


        squared += (
            value
            * value
        )


    return math.sqrt(
        squared
    )


# ============================================================
# Trainable-state snapshot
# ============================================================

def snapshot_state():

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
    }


def restore_state(
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


# ============================================================
# Full evaluation
# ============================================================

def evaluate_everything():

    return {
        "train_real":
            evaluate_pairwise(
                train_real
            ),

        "train_zero":
            evaluate_pairwise(
                train_zero
            ),

        "train_shuffle":
            evaluate_pairwise(
                train_shuffle
            ),

        "val_real":
            evaluate_pairwise(
                val_real
            ),

        "val_zero":
            evaluate_pairwise(
                val_zero
            ),

        "val_shuffle":
            evaluate_pairwise(
                val_shuffle
            ),

        "train_representation":
            representation_stats(
                train_real
            ),

        "val_representation":
            representation_stats(
                val_real
            ),
    }


# ============================================================
# Initial state
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "STAGE A — BEFORE TRAINING"
)

print(
    "=" * 120
)


initial_metrics = (
    evaluate_everything()
)


def print_eval(
    prefix,
    metrics,
):

    print(
        f"{prefix} REAL: "
        f"{100 * metrics[prefix.lower() + '_real']['ba_vs_original']:.1f}%"
    )

    print(
        f"{prefix} ZERO: "
        f"{100 * metrics[prefix.lower() + '_zero']['ba_vs_original']:.1f}%"
    )

    print(
        f"{prefix} SHUF vs original: "
        f"{100 * metrics[prefix.lower() + '_shuffle']['ba_vs_original']:.1f}%"
    )

    print(
        f"{prefix} SHUF vs donor: "
        f"{100 * metrics[prefix.lower() + '_shuffle']['ba_vs_donor']:.1f}%"
    )


print_eval(
    "TRAIN",
    initial_metrics,
)


print_eval(
    "VAL",
    initial_metrics,
)


print(
    "\nTRAIN encoder SD:",
    f"{initial_metrics['train_representation']['encoder']['feature_sd']:.8f}"
)


print(
    "VAL encoder SD:",
    f"{initial_metrics['val_representation']['encoder']['feature_sd']:.8f}"
)


# ============================================================
# Best state
#
# Selected ONLY using REAL validation BA.
# ABLATIONS are never used for selection.
# ============================================================

initial_val_ba = (
    initial_metrics[
        "val_real"
    ][
        "ba_vs_original"
    ]
)


best_val_ba = (
    initial_val_ba
)


best_epoch = 0

best_state = (
    snapshot_state()
)


history = [
    {
        "epoch":
            0,

        "mean_pairwise_loss":
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
    + "=" * 120
)

print(
    "STAGE A — TRAINING"
)

print(
    "=" * 120
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    # --------------------------------------------------------
    # Shuffle each class independently, then alternate.
    #
    # Batch size is effectively EXACTLY ONE.
    # --------------------------------------------------------

    epoch_yes = list(
        train_yes
    )


    epoch_no = list(
        train_no
    )


    rng_yes = random.Random(
        SEED
        + 1000
        + epoch
    )


    rng_no = random.Random(
        SEED
        + 2000
        + epoch
    )


    rng_yes.shuffle(
        epoch_yes
    )


    rng_no.shuffle(
        epoch_no
    )


    epoch_examples = (
        interleave(
            epoch_yes,
            epoch_no,
        )
    )


    losses = []


    for example in epoch_examples:

        optimizer.zero_grad(
            set_to_none=True
        )


        loss = (
            get_pairwise_loss(
                example
            )
        )


        loss.backward()


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

                "gold":
                    (
                        "YES"
                        if example[
                            "_is_yes"
                        ]
                        else "NO"
                    ),

                "loss":
                    float(
                        loss.detach()
                        .float()
                        .cpu()
                        .item()
                    ),
            }


            print(
                "\nFirst gradients BEFORE clipping:"
            )

            print(
                "  gold:",
                first_gradient_audit[
                    "gold"
                ]
            )

            print(
                "  loss:",
                f"{first_gradient_audit['loss']:.6f}"
            )

            print(
                "  T5/T6:",
                f"{first_gradient_audit['t5_t6']:.8e}"
            )

            print(
                "  projector:",
                f"{first_gradient_audit['projector']:.8e}"
            )


        # ----------------------------------------------------
        # Clip sensor groups independently.
        # ----------------------------------------------------

        torch.nn.utils.clip_grad_norm_(
            late_parameters,
            CLIP_T5_T6,
        )


        torch.nn.utils.clip_grad_norm_(
            projector_parameters,
            CLIP_PROJECTOR,
        )


        optimizer.step()


        losses.append(
            float(
                loss.detach()
                .float()
                .cpu()
                .item()
            )
        )


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        epoch
        in EVAL_EPOCHS
    ):

        metrics = (
            evaluate_everything()
        )


        mean_loss = (
            sum(
                losses
            )
            / len(
                losses
            )
        )


        history.append(
            {
                "epoch":
                    epoch,

                "mean_pairwise_loss":
                    mean_loss,

                "metrics":
                    metrics,
            }
        )


        train_real_ba = (
            metrics[
                "train_real"
            ][
                "ba_vs_original"
            ]
        )


        train_zero_ba = (
            metrics[
                "train_zero"
            ][
                "ba_vs_original"
            ]
        )


        train_shuffle_original = (
            metrics[
                "train_shuffle"
            ][
                "ba_vs_original"
            ]
        )


        train_shuffle_donor = (
            metrics[
                "train_shuffle"
            ][
                "ba_vs_donor"
            ]
        )


        val_real_ba = (
            metrics[
                "val_real"
            ][
                "ba_vs_original"
            ]
        )


        val_zero_ba = (
            metrics[
                "val_zero"
            ][
                "ba_vs_original"
            ]
        )


        val_shuffle_original = (
            metrics[
                "val_shuffle"
            ][
                "ba_vs_original"
            ]
        )


        val_shuffle_donor = (
            metrics[
                "val_shuffle"
            ][
                "ba_vs_donor"
            ]
        )


        train_enc_sd = (
            metrics[
                "train_representation"
            ][
                "encoder"
            ][
                "feature_sd"
            ]
        )


        val_enc_sd = (
            metrics[
                "val_representation"
            ][
                "encoder"
            ][
                "feature_sd"
            ]
        )


        print(
            f"Epoch {epoch:2d} | "
            f"loss={mean_loss:7.4f} | "
            f"TRAIN real={100 * train_real_ba:5.1f}% "
            f"zero={100 * train_zero_ba:5.1f}% "
            f"shufO={100 * train_shuffle_original:5.1f}% "
            f"shufD={100 * train_shuffle_donor:5.1f}% | "
            f"VAL real={100 * val_real_ba:5.1f}% "
            f"zero={100 * val_zero_ba:5.1f}% "
            f"shufO={100 * val_shuffle_original:5.1f}% "
            f"shufD={100 * val_shuffle_donor:5.1f}% | "
            f"EncSD train={train_enc_sd:.5f} "
            f"val={val_enc_sd:.5f}"
        )


        # ----------------------------------------------------
        # Select using ONLY REAL validation BA.
        # ----------------------------------------------------

        if (
            val_real_ba
            >
            best_val_ba
            + 1e-12
        ):

            best_val_ba = (
                val_real_ba
            )


            best_epoch = (
                epoch
            )


            best_state = (
                snapshot_state()
            )


            print(
                f"  ✓ New best REAL validation BA: "
                f"{100 * best_val_ba:.1f}%"
            )


# ============================================================
# Evaluate FINAL epoch BEFORE restoring best
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "FINAL-EPOCH DIAGNOSTIC"
)

print(
    "=" * 120
)


final_epoch_metrics = (
    evaluate_everything()
)


print_eval(
    "TRAIN",
    final_epoch_metrics,
)


print_eval(
    "VAL",
    final_epoch_metrics,
)


print(
    "\nFINAL epoch train encoder SD:",
    f"{final_epoch_metrics['train_representation']['encoder']['feature_sd']:.8f}"
)


print(
    "FINAL epoch val encoder SD:",
    f"{final_epoch_metrics['val_representation']['encoder']['feature_sd']:.8f}"
)


# ============================================================
# Restore best validation state
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "RESTORING BEST REAL-VALIDATION STATE"
)

print(
    "=" * 120
)


print(
    "Best epoch:",
    best_epoch,
)


print(
    "Best REAL validation BA:",
    f"{100 * best_val_ba:.1f}%"
)


restore_state(
    best_state
)


best_metrics = (
    evaluate_everything()
)


# ============================================================
# Final Stage-A summary
# ============================================================

train_real_ba = (
    best_metrics[
        "train_real"
    ][
        "ba_vs_original"
    ]
)


train_zero_ba = (
    best_metrics[
        "train_zero"
    ][
        "ba_vs_original"
    ]
)


train_shuffle_original = (
    best_metrics[
        "train_shuffle"
    ][
        "ba_vs_original"
    ]
)


train_shuffle_donor = (
    best_metrics[
        "train_shuffle"
    ][
        "ba_vs_donor"
    ]
)


val_real_ba = (
    best_metrics[
        "val_real"
    ][
        "ba_vs_original"
    ]
)


val_zero_ba = (
    best_metrics[
        "val_zero"
    ][
        "ba_vs_original"
    ]
)


val_shuffle_original = (
    best_metrics[
        "val_shuffle"
    ][
        "ba_vs_original"
    ]
)


val_shuffle_donor = (
    best_metrics[
        "val_shuffle"
    ][
        "ba_vs_donor"
    ]
)


print(
    "\n"
    + "=" * 120
)

print(
    "STAGE A — BEST-VALIDATION RESULT"
)

print(
    "=" * 120
)


print(
    "Best epoch:",
    best_epoch
)


print(
    "\nTRAIN"
)


print(
    "REAL:",
    f"{100 * train_real_ba:.1f}%"
)


print(
    "ZERO:",
    f"{100 * train_zero_ba:.1f}%"
)


print(
    "SHUFFLED vs original:",
    f"{100 * train_shuffle_original:.1f}%"
)


print(
    "SHUFFLED vs donor signal:",
    f"{100 * train_shuffle_donor:.1f}%"
)


print(
    "\nVALIDATION"
)


print(
    "REAL:",
    f"{100 * val_real_ba:.1f}%"
)


print(
    "ZERO:",
    f"{100 * val_zero_ba:.1f}%"
)


print(
    "SHUFFLED vs original:",
    f"{100 * val_shuffle_original:.1f}%"
)


print(
    "SHUFFLED vs donor signal:",
    f"{100 * val_shuffle_donor:.1f}%"
)


print(
    "\nVALIDATION SIGNAL GAPS"
)


print(
    "REAL - ZERO:",
    f"{100 * (val_real_ba - val_zero_ba):+.1f} pp"
)


print(
    "REAL - SHUFFLED(original):",
    f"{100 * (val_real_ba - val_shuffle_original):+.1f} pp"
)


# ============================================================
# Representation summary
# ============================================================

train_rep = (
    best_metrics[
        "train_representation"
    ]
)


val_rep = (
    best_metrics[
        "val_representation"
    ]
)


print(
    "\nREPRESENTATION"
)


print(
    "Train encoder SD:",
    f"{train_rep['encoder']['feature_sd']:.8f}"
)


print(
    "Train encoder Y/N cosine:",
    f"{train_rep['encoder']['yes_no_cosine']:.8f}"
)


print(
    "Train projector SD:",
    f"{train_rep['projector']['feature_sd']:.8f}"
)


print(
    "Val encoder SD:",
    f"{val_rep['encoder']['feature_sd']:.8f}"
)


print(
    "Val encoder Y/N cosine:",
    f"{val_rep['encoder']['yes_no_cosine']:.8f}"
)


print(
    "Val projector SD:",
    f"{val_rep['projector']['feature_sd']:.8f}"
)


# ============================================================
# Interpretation helper
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "STAGE A INTERPRETATION"
)

print(
    "=" * 120
)


stage_a_success = False


if (
    val_real_ba >= 0.75
    and val_zero_ba <= 0.60
    and val_shuffle_original <= 0.60
    and val_shuffle_donor >= 0.70
):

    stage_a_success = True


    print(
        "STRONG TRANSFERABLE SIGNAL LEARNING."
    )


    print(
        "Held-out validation performance depends on "
        "the real CGM waveform and follows the donor "
        "waveform under opposite-label shuffling."
    )


elif (
    val_real_ba >= 0.70
    and val_real_ba
    > val_zero_ba
    + 0.10
    and val_real_ba
    > val_shuffle_original
    + 0.10
):

    stage_a_success = True


    print(
        "PROMISING / PARTIAL TRANSFERABLE SIGNAL LEARNING."
    )


    print(
        "Real validation clearly outperforms waveform "
        "ablations, though donor-following may not yet "
        "be fully clean."
    )


elif (
    train_real_ba >= 0.85
    and val_real_ba <= 0.60
):

    print(
        "TRAINING-WINDOW SIGNAL MEMORIZATION "
        "WITHOUT HELD-OUT GENERALIZATION."
    )


    print(
        "The sensor pathway learned the training examples "
        "but did not acquire a transferable >180 concept."
    )


elif (
    train_real_ba <= 0.60
):

    print(
        "SENSOR PATH DID NOT LEARN EVEN THE "
        "TRAINING SET."
    )


    print(
        "Do not proceed to Stage B."
    )


else:

    print(
        "INTERMEDIATE RESULT."
    )


    print(
        "Inspect train-vs-validation and the "
        "real/zero/shuffled gaps."
    )


# ============================================================
# Save checkpoint ONLY if validation improved above baseline
# ============================================================

checkpoint_saved = False


if (
    best_val_ba
    >
    initial_val_ba
    + 1e-12
):

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        {
            "experiment":
                "cgm_stage_a_signal_prealignment",

            "base_repo":
                STAGE5_REPO_ID,

            "task":
                TASK_ID,

            "best_epoch":
                best_epoch,

            "best_validation_real_ba":
                best_val_ba,

            "normalization":
                {
                    "type":
                        "fixed_global",

                    "mean":
                        train_mean,

                    "std":
                        train_std,
                },

            "trainable_modules":
                {
                    "t5":
                        best_state[
                            "t5"
                        ],

                    "t6":
                        best_state[
                            "t6"
                        ],

                    "projector":
                        best_state[
                            "projector"
                        ],
                },
        },

        CHECKPOINT_PATH,
    )


    checkpoint_saved = True


    print(
        "\nSaved best Stage-A checkpoint:"
    )


    print(
        CHECKPOINT_PATH
    )


else:

    print(
        "\nNo Stage-A checkpoint saved because "
        "REAL validation BA never improved above "
        "the untrained baseline."
    )


# ============================================================
# Serialize metadata
# ============================================================

def example_metadata(
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


results = {
    "experiment":
        "cgm_stage_a_signal_prealignment",

    "seed":
        SEED,

    "stage5_repo":
        STAGE5_REPO_ID,

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
            "per_example_text_variation":
                False,

            "common_pre_prompt":
                COMMON_PRE_PROMPT,

            "common_time_series_text":
                COMMON_TS_TEXT,

            "common_post_prompt":
                COMMON_POST_PROMPT,
        },

    "training":
        {
            "train_per_class":
                TRAIN_PER_CLASS,

            "validation_per_class":
                VAL_PER_CLASS,

            "epochs":
                EPOCHS,

            "effective_batch_size":
                1,

            "loss":
                "true_pairwise_ab_only",

            "lr_t5_t6":
                LR_T5_T6,

            "lr_projector":
                LR_PROJECTOR,

            "weight_decay":
                WEIGHT_DECAY,

            "clip_t5_t6":
                CLIP_T5_T6,

            "clip_projector":
                CLIP_PROJECTOR,

            "lora_trainable":
                False,
        },

    "participant_audit":
        {
            "train_participants":
                sorted(
                    str(x)
                    for x
                    in train_participants
                ),

            "validation_participants":
                sorted(
                    str(x)
                    for x
                    in val_participants
                ),

            "overlap":
                sorted(
                    str(x)
                    for x
                    in overlap
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
            example_metadata(
                example
            )

            for example
            in train_real
        ],

    "selected_validation_examples":
        [
            example_metadata(
                example
            )

            for example
            in val_real
        ],

    "first_gradient_audit":
        first_gradient_audit,

    "initial_metrics":
        initial_metrics,

    "history":
        history,

    "final_epoch_metrics":
        final_epoch_metrics,

    "best_epoch":
        best_epoch,

    "best_validation_real_ba":
        best_val_ba,

    "best_metrics":
        best_metrics,

    "stage_a_success":
        stage_a_success,

    "checkpoint_saved":
        checkpoint_saved,

    "checkpoint_path":
        (
            str(
                CHECKPOINT_PATH
            )
            if checkpoint_saved
            else None
        ),
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
    "\nTEST participants remain sealed."
)


# ============================================================
# Cleanup
# ============================================================

del optimizer
del model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()