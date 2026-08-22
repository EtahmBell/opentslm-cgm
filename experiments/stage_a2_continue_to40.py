"""
STAGE A.2 CONTINUATION — EPOCH 20 -> 40

Loads:
    checkpoints/cgm_stage_a2_above180_best.pt

That checkpoint contains the best Stage-A.2 weights:
    - original Stage-5 T5
    - original Stage-5 T6
    - original Stage-5 projector

The rest of the model is freshly loaded from:
    OpenTSLM/llama-3.2-1b-ecg-sp

IMPORTANT
---------
No TSQA components are used.

Frozen:
    - Stage-5 patch / early encoder
    - Stage-5 T1-T4
    - base Llama-3.2-1B
    - Stage-5 ECG LoRA

Trainable:
    - Stage-5 T5
    - Stage-5 T6
    - Stage-5 projector

The previous checkpoint did NOT save optimizer state.
Therefore AdamW is reinitialized here.

This is a warm-start continuation from epoch-20 MODEL
weights, not an exact continuation of Adam moments.

Training:
    epoch 21 -> 40
    231 YES + 231 NO
    fixed global CGM normalization
    invariant text
    true (a)-vs-(b) pairwise CE
    batch size 1
    alternating YES / NO updates

Validation:
    49 YES + 49 NO
    participant-disjoint validation set

Signal ablations:
    REAL
    ZERO
    opposite-label SHUFFLE

TEST PARTICIPANTS REMAIN SEALED.
"""

import copy
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

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)


from src.cgm_multitask import (
    build_multitask_examples,
)

from opentslm import OpenTSLM


DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)


MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
)


INPUT_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_stage_a2_above180_best.pt"
)


OUTPUT_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_stage_a2_above180_epoch40_best.pt"
)


RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_stage_a2_continue_to40.json"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

STAGE5_REPO_ID = (
    "OpenTSLM/llama-3.2-1b-ecg-sp"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128

STRIDE = 128


TRAIN_PER_CLASS = 231

VAL_PER_CLASS = 49


TARGET_EPOCH = 40


LR_T5_T6 = 2e-4

LR_PROJECTOR = 1e-4


WEIGHT_DECAY = 1e-2


CLIP_T5_T6 = 1.0

CLIP_PROJECTOR = 1.0


EVAL_BATCH_SIZE = 16


EVAL_EPOCHS = {
    25,
    30,
    35,
    40,
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
        "Validation split missing."
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
# Load saved Stage-A.2 checkpoint
# ============================================================

if (
    not INPUT_CHECKPOINT_PATH.exists()
):

    raise FileNotFoundError(
        INPUT_CHECKPOINT_PATH
    )


print(
    "\nLoading Stage-A.2 checkpoint metadata..."
)


saved_checkpoint = torch.load(
    INPUT_CHECKPOINT_PATH,
    map_location="cpu",
)


BASE_EPOCH = int(
    saved_checkpoint[
        "best_epoch"
    ]
)


print(
    "Checkpoint:",
    INPUT_CHECKPOINT_PATH,
)


print(
    "Saved epoch:",
    BASE_EPOCH,
)


print(
    "Saved validation REAL BA:",
    f"{100 * saved_checkpoint['best_validation_real_ba']:.1f}%"
)


if (
    BASE_EPOCH
    >= TARGET_EPOCH
):

    raise RuntimeError(
        f"Checkpoint is already epoch {BASE_EPOCH}."
    )


if (
    saved_checkpoint[
        "base_repo"
    ]
    != STAGE5_REPO_ID
):

    raise RuntimeError(
        "Checkpoint base model mismatch."
    )


# ============================================================
# Temporary model for EOS token
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
# Build train and validation pools
#
# Same deterministic selection as Stage A.2.
# ============================================================

print(
    "\nBuilding TRAIN pool..."
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
    "Building VALIDATION pool..."
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
# Class split
# ============================================================

def split_by_class(
    examples,
):

    yes = [
        ex
        for ex in examples
        if (
            ex[
                "_answer_clean"
            ]
            == "(a)"
        )
    ]


    no = [
        ex
        for ex in examples
        if (
            ex[
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
    "\nAvailable TRAIN:"
)

print(
    " YES:",
    len(
        train_yes_pool
    ),
)

print(
    " NO:",
    len(
        train_no_pool
    ),
)


print(
    "\nAvailable VALIDATION:"
)

print(
    " YES:",
    len(
        val_yes_pool
    ),
)

print(
    " NO:",
    len(
        val_no_pool
    ),
)


# ============================================================
# Same participant-aware deterministic sampler
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


    participants = list(
        grouped.keys()
    )


    rng.shuffle(
        participants
    )


    for participant in participants:

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


        for participant in participants:

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
                "Unable to sample requested examples."
            )


    return selected


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
    ex.get(
        "_participant_id"
    )

    for ex
    in (
        selected_train_yes
        + selected_train_no
    )
}


val_participants = {
    ex.get(
        "_participant_id"
    )

    for ex
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
    " train participants:",
    len(
        train_participants
    ),
)

print(
    " validation participants:",
    len(
        val_participants
    ),
)

print(
    " overlap:",
    len(
        overlap
    ),
)


if overlap:

    raise RuntimeError(
        "Participant leakage detected."
    )


# ============================================================
# Same invariant prompt
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
# Convert examples
# ============================================================

def make_example(
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

        "_donor_is_yes":
            None,
    }


train_yes = [
    make_example(
        ex
    )
    for ex
    in selected_train_yes
]


train_no = [
    make_example(
        ex
    )
    for ex
    in selected_train_no
]


val_yes = [
    make_example(
        ex
    )
    for ex
    in selected_val_yes
]


val_no = [
    make_example(
        ex
    )
    for ex
    in selected_val_no
]


# ============================================================
# Text invariance audit
# ============================================================

all_examples = (
    train_yes
    + train_no
    + val_yes
    + val_no
)


unique_text = {
    (
        ex[
            "pre_prompt"
        ],
        ex[
            "time_series_text"
        ][
            0
        ],
        ex[
            "post_prompt"
        ],
    )

    for ex
    in all_examples
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
        "Text interface changed."
    )


print(
    "✓ Text interface matches Stage A.2."
)


# ============================================================
# Balanced order
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
            "Classes must be balanced."
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
# Signal ablations
# ============================================================

def clone_with_signal(
    example,
    signal,
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
        "_donor_is_yes"
    ] = donor_is_yes


    return cloned


def make_zero_examples(
    examples,
):

    output = []


    for example in examples:

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
            )
        )


    return output


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


        output.append(
            clone_with_signal(
                original,
                donor[
                    "time_series"
                ][
                    0
                ],
                donor_is_yes=(
                    donor[
                        "_is_yes"
                    ]
                ),
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


print(
    "\nTRAIN:",
    len(
        train_real
    ),
)


print(
    "VALIDATION:",
    len(
        val_real
    ),
)


# ============================================================
# Free temporary model
# ============================================================

del temporary_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Load fresh Stage-5 base model
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING FRESH ORIGINAL STAGE-5 BASE"
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
        "Expected six encoder layers."
    )


# ============================================================
# Load epoch-20 Stage-A.2 modules
# ============================================================

modules = (
    saved_checkpoint[
        "trainable_modules"
    ]
)


layers[
    4
].load_state_dict(
    modules[
        "stage5_t5"
    ],
    strict=True,
)


layers[
    5
].load_state_dict(
    modules[
        "stage5_t6"
    ],
    strict=True,
)


model.projector.load_state_dict(
    modules[
        "stage5_projector"
    ],
    strict=True,
)


print(
    "\n✓ Loaded Stage-A.2 epoch-20 "
    "T5/T6/projector weights."
)


print(
    "✓ No TSQA modules loaded."
)


# ============================================================
# Pairwise tokens
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


print(
    "\nDecision tokens:"
)

print(
    "(a):",
    a_ids,
    "->",
    A_TOKEN,
)


print(
    "(b):",
    b_ids,
    "->",
    B_TOKEN,
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


# ============================================================
# Unfreeze original Stage-5 T5/T6
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
# Unfreeze Stage-5 projector
# ============================================================

for parameter in (
    model.projector.parameters()
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


llm_trainable = sum(
    parameter.numel()

    for parameter
    in model.llm.parameters()

    if parameter.requires_grad
)


print(
    "\nTRAINABLE PARAMETER AUDIT"
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
    f"{llm_trainable:,}"
)


if (
    llm_trainable
    != 0
):

    raise RuntimeError(
        "LLM / LoRA unexpectedly trainable."
    )


# ============================================================
# Disable dropout
# ============================================================

model.encoder.eval()

model.projector.eval()

model.llm.eval()


# ============================================================
# Fresh optimizer
#
# NOTE:
# optimizer state was not stored in epoch-20 checkpoint.
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


print(
    "\nIMPORTANT:"
)

print(
    "AdamW optimizer state has been RESET."
)

print(
    "Model weights continue from epoch 20."
)


# ============================================================
# Pairwise loss
# ============================================================

def pairwise_loss_single(
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


    pair_logits = torch.stack(
        [
            full_logits[
                A_TOKEN
            ],

            full_logits[
                B_TOKEN
            ],
        ]
    ).float()


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
        pair_logits.unsqueeze(
            0
        ),
        target,
    )


# ============================================================
# BA helper
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


    return 0.5 * (
        yes_correct
        / yes_total
        +
        no_correct
        / no_total
    )


# ============================================================
# Batched evaluation
# ============================================================

def evaluate_pairwise(
    examples,
):

    predictions = []

    original_labels = []

    donor_labels = []


    with torch.no_grad():

        for start in range(
            0,
            len(
                examples
            ),
            EVAL_BATCH_SIZE,
        ):

            batch = (
                examples[
                    start:
                    start
                    + EVAL_BATCH_SIZE
                ]
            )


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


                prediction = bool(
                    logits[
                        A_TOKEN
                    ]
                    >= logits[
                        B_TOKEN
                    ]
                )


                predictions.append(
                    prediction
                )


                original_labels.append(
                    example[
                        "_is_yes"
                    ]
                )


                donor_labels.append(
                    example.get(
                        "_donor_is_yes"
                    )
                )


    result = {
        "ba_vs_original":
            balanced_accuracy(
                predictions,
                original_labels,
            ),
    }


    if all(
        donor is not None

        for donor
        in donor_labels
    ):

        result[
            "ba_vs_donor"
        ] = (
            balanced_accuracy(
                predictions,
                donor_labels,
            )
        )


    else:

        result[
            "ba_vs_donor"
        ] = None


    return result


# ============================================================
# Evaluation bundle
# ============================================================

def evaluate_bundle():

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
    }


# ============================================================
# Snapshot helpers
# ============================================================

def snapshot_model_state():

    return {
        "t5":
            copy.deepcopy(
                layers[
                    4
                ].state_dict()
            ),

        "t6":
            copy.deepcopy(
                layers[
                    5
                ].state_dict()
            ),

        "projector":
            copy.deepcopy(
                model.projector.state_dict()
            ),
    }


def restore_model_state(
    state,
):

    layers[
        4
    ].load_state_dict(
        state[
            "t5"
        ]
    )


    layers[
        5
    ].load_state_dict(
        state[
            "t6"
        ]
    )


    model.projector.load_state_dict(
        state[
            "projector"
        ]
    )


# ============================================================
# Epoch-20 sanity check
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "LOADED EPOCH-20 CHECKPOINT"
)

print(
    "=" * 120
)


loaded_metrics = (
    evaluate_bundle()
)


print(
    "TRAIN REAL:",
    f"{100 * loaded_metrics['train_real']['ba_vs_original']:.1f}%"
)


print(
    "VAL REAL:",
    f"{100 * loaded_metrics['val_real']['ba_vs_original']:.1f}%"
)


print(
    "VAL ZERO:",
    f"{100 * loaded_metrics['val_zero']['ba_vs_original']:.1f}%"
)


print(
    "VAL SHUFFLED vs original:",
    f"{100 * loaded_metrics['val_shuffle']['ba_vs_original']:.1f}%"
)


print(
    "VAL SHUFFLED vs donor:",
    f"{100 * loaded_metrics['val_shuffle']['ba_vs_donor']:.1f}%"
)


# Should roughly reproduce:
#
# train real ~80.3%
# val real ~67.3%
#
# Minor numerical differences are possible but large
# disagreement means stop and investigate.


# ============================================================
# Best tracking
# ============================================================

best_epoch = (
    BASE_EPOCH
)


best_val_ba = (
    loaded_metrics[
        "val_real"
    ][
        "ba_vs_original"
    ]
)


best_model_state = (
    snapshot_model_state()
)


best_optimizer_state = copy.deepcopy(
    optimizer.state_dict()
)


history = [
    {
        "epoch":
            BASE_EPOCH,

        "mean_loss":
            None,

        "metrics":
            loaded_metrics,
    }
]


# ============================================================
# Training: epoch 21 -> 40
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    f"CONTINUING STAGE A.2: "
    f"EPOCH {BASE_EPOCH + 1} -> {TARGET_EPOCH}"
)

print(
    "=" * 120
)


for epoch in range(
    BASE_EPOCH + 1,
    TARGET_EPOCH + 1,
):

    epoch_yes = list(
        train_yes
    )


    epoch_no = list(
        train_no
    )


    # Same epoch-dependent shuffle rule as
    # the original Stage A.2 script.
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
            pairwise_loss_single(
                example
            )
        )


        loss.backward()


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


    mean_loss = (
        sum(
            losses
        )
        / len(
            losses
        )
    )


    # Lightweight progress every epoch.
    print(
        f"Finished epoch {epoch:2d} "
        f"| mean loss={mean_loss:.4f}"
    )


    # --------------------------------------------------------
    # Full diagnostic only at 25/30/35/40.
    # --------------------------------------------------------

    if (
        epoch
        in EVAL_EPOCHS
    ):

        metrics = (
            evaluate_bundle()
        )


        train_real_ba = (
            metrics[
                "train_real"
            ][
                "ba_vs_original"
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


        val_shuf_original = (
            metrics[
                "val_shuffle"
            ][
                "ba_vs_original"
            ]
        )


        val_shuf_donor = (
            metrics[
                "val_shuffle"
            ][
                "ba_vs_donor"
            ]
        )


        print(
            "\n"
            f"Epoch {epoch:2d} DIAGNOSTIC | "
            f"loss={mean_loss:.4f} | "
            f"TRAIN real="
            f"{100 * train_real_ba:.1f}% | "
            f"VAL real="
            f"{100 * val_real_ba:.1f}% | "
            f"zero="
            f"{100 * val_zero_ba:.1f}% | "
            f"shufO="
            f"{100 * val_shuf_original:.1f}% | "
            f"shufD="
            f"{100 * val_shuf_donor:.1f}%"
        )


        history.append(
            {
                "epoch":
                    epoch,

                "mean_loss":
                    mean_loss,

                "metrics":
                    metrics,
            }
        )


        if (
            val_real_ba
            >
            best_val_ba
            + 1e-12
        ):

            best_epoch = (
                epoch
            )


            best_val_ba = (
                val_real_ba
            )


            best_model_state = (
                snapshot_model_state()
            )


            best_optimizer_state = (
                copy.deepcopy(
                    optimizer.state_dict()
                )
            )


            print(
                f"✓ New best validation REAL BA: "
                f"{100 * best_val_ba:.1f}%"
            )


# ============================================================
# Final epoch-40 state
# ============================================================

epoch40_model_state = (
    snapshot_model_state()
)


epoch40_optimizer_state = (
    copy.deepcopy(
        optimizer.state_dict()
    )
)


epoch40_metrics = (
    evaluate_bundle()
)


# ============================================================
# Restore best epoch
# ============================================================

restore_model_state(
    best_model_state
)


best_metrics = (
    evaluate_bundle()
)


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "STAGE A.2 — EPOCH 20→40 RESULT"
)

print(
    "=" * 120
)


print(
    "Best epoch:",
    best_epoch,
)


print(
    "Best validation REAL BA:",
    f"{100 * best_val_ba:.1f}%"
)


print(
    "\nBEST STATE:"
)


print(
    "TRAIN REAL:",
    f"{100 * best_metrics['train_real']['ba_vs_original']:.1f}%"
)


print(
    "TRAIN ZERO:",
    f"{100 * best_metrics['train_zero']['ba_vs_original']:.1f}%"
)


print(
    "TRAIN SHUFFLED original:",
    f"{100 * best_metrics['train_shuffle']['ba_vs_original']:.1f}%"
)


print(
    "TRAIN SHUFFLED donor:",
    f"{100 * best_metrics['train_shuffle']['ba_vs_donor']:.1f}%"
)


print(
    "\nVALIDATION REAL:",
    f"{100 * best_metrics['val_real']['ba_vs_original']:.1f}%"
)


print(
    "VALIDATION ZERO:",
    f"{100 * best_metrics['val_zero']['ba_vs_original']:.1f}%"
)


print(
    "VALIDATION SHUFFLED original:",
    f"{100 * best_metrics['val_shuffle']['ba_vs_original']:.1f}%"
)


print(
    "VALIDATION SHUFFLED donor:",
    f"{100 * best_metrics['val_shuffle']['ba_vs_donor']:.1f}%"
)


print(
    "\nSIGNAL GAPS:"
)


print(
    "REAL - ZERO:",
    f"{100 * (
        best_metrics['val_real']['ba_vs_original']
        -
        best_metrics['val_zero']['ba_vs_original']
    ):+.1f} pp"
)


print(
    "REAL - SHUFFLED(original):",
    f"{100 * (
        best_metrics['val_real']['ba_vs_original']
        -
        best_metrics['val_shuffle']['ba_vs_original']
    ):+.1f} pp"
)


# ============================================================
# Save BEST checkpoint
#
# This time optimizer state IS included, so a future
# continuation can be exact from this extension onward.
# ============================================================

OUTPUT_CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


torch.save(
    {
        "experiment":
            "cgm_stage_a2_continue_to40",

        "base_repo":
            STAGE5_REPO_ID,

        "tsqa_transplant":
            False,

        "warm_started_from":
            str(
                INPUT_CHECKPOINT_PATH
            ),

        "optimizer_reset_at_epoch":
            BASE_EPOCH,

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
                "stage5_t5":
                    best_model_state[
                        "t5"
                    ],

                "stage5_t6":
                    best_model_state[
                        "t6"
                    ],

                "stage5_projector":
                    best_model_state[
                        "projector"
                    ],
            },

        "optimizer_state":
            best_optimizer_state,

        "epoch40_state":
            {
                "stage5_t5":
                    epoch40_model_state[
                        "t5"
                    ],

                "stage5_t6":
                    epoch40_model_state[
                        "t6"
                    ],

                "stage5_projector":
                    epoch40_model_state[
                        "projector"
                    ],

                "optimizer_state":
                    epoch40_optimizer_state,

                "metrics":
                    epoch40_metrics,
            },
    },

    OUTPUT_CHECKPOINT_PATH,
)


print(
    "\nSaved continuation checkpoint:"
)

print(
    OUTPUT_CHECKPOINT_PATH
)


# ============================================================
# Save JSON diagnostic
# ============================================================

results = {
    "experiment":
        "cgm_stage_a2_continue_to40",

    "base_repo":
        STAGE5_REPO_ID,

    "tsqa_transplant":
        False,

    "base_epoch":
        BASE_EPOCH,

    "target_epoch":
        TARGET_EPOCH,

    "optimizer_reset_at_epoch20":
        True,

    "note":
        (
            "Epoch-20 checkpoint did not include AdamW "
            "optimizer moments; continuation therefore "
            "warm-started model weights with a fresh "
            "optimizer."
        ),

    "training":
        {
            "train_total":
                len(
                    train_real
                ),

            "validation_total":
                len(
                    val_real
                ),

            "updates_per_epoch":
                len(
                    train_real
                ),

            "additional_epochs":
                TARGET_EPOCH
                - BASE_EPOCH,

            "lr_t5_t6":
                LR_T5_T6,

            "lr_projector":
                LR_PROJECTOR,

            "batch_size":
                1,
        },

    "loaded_epoch20_metrics":
        loaded_metrics,

    "history":
        history,

    "best_epoch":
        best_epoch,

    "best_validation_real_ba":
        best_val_ba,

    "best_metrics":
        best_metrics,

    "epoch40_metrics":
        epoch40_metrics,

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
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