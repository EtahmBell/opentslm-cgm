"""
STAGE B.1 — LoRA LANGUAGE READOUT

Goal
----
Take the successful Stage-A CGM representation:

    Stage5 T1-T4
    +
    TSQA-initialized / CGM-trained T5-T6
    +
    CGM-trained Stage5 projector

and teach the Stage-5 language side to EXPRESS the already
learned CGM fact as:

    Yes.
    No.

IMPORTANT
---------
The COMPLETE sensor path is frozen.

Only the existing Stage-5 ECG LoRA parameters are trainable.

Therefore Stage B cannot create a new sensor representation.
It can only learn to read the existing Stage-A representation.

Stage-A source checkpoint:

    checkpoints/
    cgm_stage_a3_tsqa_t56_s5proj_best.pt

No TSQA LLM is used.


INPUT
-----
Same fixed global normalization used in Stage A.

Same invariant text metadata.

NO:
    per-window mean
    per-window SD
    max
    TIR
    participant ID
    window ID
    label-derived metadata


TARGET
------
Natural language:

    " Yes."
    " No."

Custom causal sequence loss is used.

We DO NOT call OpenTSLM.compute_loss().

Target tokenization uses:

    add_special_tokens=False

so the first answer token is explicitly predicted from the
ACTUAL final prompt position.


AUXILIARY LOSS
--------------
At the final prompt position:

    CE([logit(" Yes"), logit(" No")], gold)

This makes the natural-language decision explicit.


DATA
----
TRAIN:
    231 YES
    231 NO
    462 total

VALIDATION:
    49 YES
    49 NO
    98 total

TEST:
    SEALED


ABLATIONS
---------
Validation:

    REAL signal
    ZERO signal
    opposite-label SHUFFLED signal

Success means:

    REAL language BA high
    ZERO ~50%
    SHUFFLED-original low
    SHUFFLED-donor high

while Stage-A sensor weights remain EXACTLY unchanged.
"""

import copy
import gc
import json
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

from src.cgm_multitask import build_multitask_examples
from opentslm import OpenTSLM


DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
)

STAGE_A_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_stage_a3_tsqa_t56_s5proj_best.pt"
)

OUTPUT_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_stage_b1_lora_language_best.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_stage_b1_lora_language.json"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

STAGE5_REPO = (
    "OpenTSLM/llama-3.2-1b-ecg-sp"
)

TASK_ID = "above_180"

WINDOW_SIZE = 128
STRIDE = 128

TRAIN_PER_CLASS = 231
VAL_PER_CLASS = 49


# LoRA training should need substantially fewer epochs
# than the Stage-A sensor training.
EPOCHS = 8

BATCH_SIZE = 4

LR_LORA = 1e-5

WEIGHT_DECAY = 1e-2

CLIP_LORA = 1.0


SEQUENCE_WEIGHT = 1.0
PAIRWISE_WEIGHT = 1.0


EVAL_BATCH_SIZE = 16

MAX_GENERATION_TOKENS = 4


EVAL_EPOCHS = {
    0,
    1,
    2,
    3,
    4,
    6,
    8,
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
    manifest["normalization"]["mean"]
)

train_std = float(
    manifest["normalization"]["std"]
)


validation_manifest = (
    manifest.get("validation")
)

if validation_manifest is None:
    validation_manifest = (
        manifest.get("val")
    )

if validation_manifest is None:
    raise RuntimeError(
        "Validation split missing."
    )


train_files = [
    DATA_DIR / item["relative_path"]
    for item in manifest["train"]
]

validation_files = [
    DATA_DIR / item["relative_path"]
    for item in validation_manifest
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
    len(manifest["test"]),
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
# Load Stage-A checkpoint
# ============================================================

if not STAGE_A_CHECKPOINT_PATH.exists():

    raise FileNotFoundError(
        STAGE_A_CHECKPOINT_PATH
    )


stage_a_checkpoint = torch.load(
    STAGE_A_CHECKPOINT_PATH,
    map_location="cpu",
)


print(
    "\nStage-A source checkpoint:"
)

print(
    STAGE_A_CHECKPOINT_PATH
)

print(
    "Stage-A validation BA:",
    f"{100 * stage_a_checkpoint['best_validation_real_ba']:.1f}%"
)

print(
    "Stage-A best epoch:",
    stage_a_checkpoint["best_epoch"],
)


# ============================================================
# Load PUBLIC Stage-5 model
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "LOADING STAGE-5 BASE + STAGE-A SENSOR CHECKPOINT"
)

print(
    "=" * 110
)


model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO,
        device=device,
        enable_lora=True,
    )
)


EOS_TOKEN = (
    model.get_eos_token()
)

EOS_TOKEN_ID = (
    model.tokenizer.eos_token_id
)


layers = (
    model
    .encoder
    .encoder
    .layers
)


if len(layers) != 6:

    raise RuntimeError(
        "Expected six encoder layers."
    )


# ============================================================
# Load Stage-A T5/T6/projector
# ============================================================

stage_a_modules = (
    stage_a_checkpoint[
        "trainable_modules"
    ]
)


layers[4].load_state_dict(
    stage_a_modules["t5"],
    strict=True,
)

layers[5].load_state_dict(
    stage_a_modules["t6"],
    strict=True,
)

model.projector.load_state_dict(
    stage_a_modules["projector"],
    strict=True,
)


print(
    "✓ Loaded Stage-A hybrid T5/T6."
)

print(
    "✓ Loaded Stage-A Stage-5 projector."
)

print(
    "✓ Stage-5 LLM + ECG LoRA retained."
)

print(
    "✓ No TSQA LLM loaded."
)


# ============================================================
# Record which params are the pre-existing Stage-5 LoRA
#
# At initial load, OpenTSLM marks these trainable.
# ============================================================

initial_lora_names = [
    name

    for name, parameter
    in model.llm.named_parameters()

    if parameter.requires_grad
]


if len(initial_lora_names) == 0:

    raise RuntimeError(
        "No Stage-5 LoRA parameters detected."
    )


# ============================================================
# Freeze EVERYTHING
# ============================================================

for parameter in model.encoder.parameters():
    parameter.requires_grad = False

for parameter in model.projector.parameters():
    parameter.requires_grad = False

for parameter in model.llm.parameters():
    parameter.requires_grad = False


# ============================================================
# Re-enable ONLY existing Stage-5 LoRA params
# ============================================================

llm_named_parameters = dict(
    model.llm.named_parameters()
)


for name in initial_lora_names:

    llm_named_parameters[
        name
    ].requires_grad = True


lora_parameters = [
    llm_named_parameters[name]
    for name in initial_lora_names
]


# ============================================================
# Audit
# ============================================================

sensor_trainable = (
    sum(
        parameter.numel()
        for parameter
        in model.encoder.parameters()
        if parameter.requires_grad
    )
    +
    sum(
        parameter.numel()
        for parameter
        in model.projector.parameters()
        if parameter.requires_grad
    )
)


lora_trainable = sum(
    parameter.numel()
    for parameter
    in lora_parameters
)


other_llm_trainable = sum(
    parameter.numel()

    for name, parameter
    in model.llm.named_parameters()

    if (
        parameter.requires_grad
        and name
        not in initial_lora_names
    )
)


print(
    "\n"
    + "=" * 110
)

print(
    "TRAINABLE PARAMETER AUDIT"
)

print(
    "=" * 110
)


print(
    "Sensor trainable:",
    f"{sensor_trainable:,}"
)

print(
    "Stage-5 LoRA trainable:",
    f"{lora_trainable:,}"
)

print(
    "Other LLM trainable:",
    f"{other_llm_trainable:,}"
)


if sensor_trainable != 0:

    raise RuntimeError(
        "Sensor pathway must be completely frozen."
    )


if other_llm_trainable != 0:

    raise RuntimeError(
        "Non-LoRA LLM parameters unexpectedly trainable."
    )


# ============================================================
# Build data
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
        task_ids=[TASK_ID],
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
        task_ids=[TASK_ID],
    )
)


# ============================================================
# Split
# ============================================================

def split_by_class(
    examples,
):

    yes = [
        example
        for example in examples
        if example["_answer_clean"] == "(a)"
    ]

    no = [
        example
        for example in examples
        if example["_answer_clean"] == "(b)"
    ]

    return yes, no


train_yes_pool, train_no_pool = (
    split_by_class(
        train_pool
    )
)


val_yes_pool, val_no_pool = (
    split_by_class(
        validation_pool
    )
)


print(
    "\nTRAIN available:"
)

print(
    " YES:",
    len(train_yes_pool),
)

print(
    " NO:",
    len(train_no_pool),
)


print(
    "\nVALIDATION available:"
)

print(
    " YES:",
    len(val_yes_pool),
)

print(
    " NO:",
    len(val_no_pool),
)


# ============================================================
# Same participant-aware deterministic sampler
# ============================================================

def participant_round_robin_sample(
    examples,
    n,
    seed,
):

    rng = random.Random(seed)

    grouped = defaultdict(list)


    for example in examples:

        participant = example.get(
            "_participant_id"
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
            grouped[participant]
        )


    selected = []


    while len(selected) < n:

        progress = False


        for participant in participants:

            if grouped[participant]:

                selected.append(
                    grouped[
                        participant
                    ].pop()
                )

                progress = True


                if len(selected) == n:
                    break


        if not progress:

            raise RuntimeError(
                "Could not obtain requested sample."
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
# Participant audit
# ============================================================

train_participants = {
    example.get("_participant_id")
    for example
    in (
        selected_train_yes
        + selected_train_no
    )
}


val_participants = {
    example.get("_participant_id")
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
    " train:",
    len(train_participants),
)

print(
    " validation:",
    len(val_participants),
)

print(
    " overlap:",
    len(overlap),
)


if overlap:

    raise RuntimeError(
        "TRAIN/VALIDATION participant leakage."
    )


# ============================================================
# Stage-B invariant language interface
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
    "Answer Yes or No.\n"
    "Answer:"
)


# ============================================================
# Natural answer tokens
# ============================================================

YES_TARGET_TEXT = " Yes."
NO_TARGET_TEXT = " No."


YES_TARGET_IDS = (
    model.tokenizer(
        YES_TARGET_TEXT,
        add_special_tokens=False,
    ).input_ids
)


NO_TARGET_IDS = (
    model.tokenizer(
        NO_TARGET_TEXT,
        add_special_tokens=False,
    ).input_ids
)


YES_DECISION_IDS = (
    model.tokenizer(
        " Yes",
        add_special_tokens=False,
    ).input_ids
)


NO_DECISION_IDS = (
    model.tokenizer(
        " No",
        add_special_tokens=False,
    ).input_ids
)


if (
    len(YES_DECISION_IDS) != 1
    or len(NO_DECISION_IDS) != 1
):

    raise RuntimeError(
        "Expected single-token Yes/No decisions."
    )


YES_TOKEN = int(
    YES_DECISION_IDS[0]
)

NO_TOKEN = int(
    NO_DECISION_IDS[0]
)


print(
    "\nNatural-language targets:"
)

print(
    " YES:",
    YES_TARGET_TEXT,
    YES_TARGET_IDS,
)

print(
    " NO:",
    NO_TARGET_TEXT,
    NO_TARGET_IDS,
)

print(
    "Decision tokens:"
)

print(
    " YES:",
    YES_TOKEN,
)

print(
    " NO:",
    NO_TOKEN,
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
        ][0]
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


    return {
        "pre_prompt":
            COMMON_PRE_PROMPT,

        "time_series":
            [signal],

        "time_series_text":
            [COMMON_TS_TEXT],

        "post_prompt":
            COMMON_POST_PROMPT,

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
    make_example(example)
    for example
    in selected_train_yes
]

train_no = [
    make_example(example)
    for example
    in selected_train_no
]

val_yes = [
    make_example(example)
    for example
    in selected_val_yes
]

val_no = [
    make_example(example)
    for example
    in selected_val_no
]


# ============================================================
# Verify text invariance
# ============================================================

unique_text = {
    (
        example["pre_prompt"],
        example["time_series_text"][0],
        example["post_prompt"],
    )

    for example
    in (
        train_yes
        + train_no
        + val_yes
        + val_no
    )
}


print(
    "\nUnique prompt/text combinations:",
    len(unique_text),
)


if len(unique_text) != 1:

    raise RuntimeError(
        "Per-example text variation detected."
    )


print(
    "✓ All label information is confined "
    "to the CGM waveform."
)


# ============================================================
# Interleave
# ============================================================

def interleave(
    yes_examples,
    no_examples,
):

    if len(yes_examples) != len(
        no_examples
    ):

        raise RuntimeError(
            "Expected balanced classes."
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
# Ablations
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


def make_zero(
    examples,
):

    return [
        clone_with_signal(
            example,

            torch.zeros_like(
                example[
                    "time_series"
                ][0]
            ),
        )

        for example
        in examples
    ]


def make_opposite_shuffle(
    yes_examples,
    no_examples,
    seed,
):

    rng = random.Random(seed)

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

        if original[
            "_is_yes"
        ]:

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
                ][0],

                donor_is_yes=(
                    donor[
                        "_is_yes"
                    ]
                ),
            )
        )


    return output


train_zero = (
    make_zero(
        train_real
    )
)


val_zero = (
    make_zero(
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
    len(train_real),
)

print(
    "VALIDATION:",
    len(val_real),
)


# ============================================================
# Disable dropout but KEEP gradients
# ============================================================

model.encoder.eval()
model.projector.eval()
model.llm.eval()


# ============================================================
# Target helper
# ============================================================

def target_ids_for_example(
    example,
):

    ids = list(
        YES_TARGET_IDS
        if example["_is_yes"]
        else NO_TARGET_IDS
    )


    if (
        EOS_TOKEN_ID is not None
        and (
            len(ids) == 0
            or ids[-1] != EOS_TOKEN_ID
        )
    ):

        ids.append(
            EOS_TOKEN_ID
        )


    return ids


# ============================================================
# Custom language + pairwise loss
#
# CRITICAL:
# No BOS/special answer prefix is inserted.
# ============================================================

def loss_for_batch(
    batch,
):

    (
        prompt_embeds,
        prompt_mask,
    ) = (
        model.pad_and_apply_batch(
            batch
        )
    )


    prompt_lengths = (
        prompt_mask
        .sum(
            dim=1
        )
        .long()
    )


    if not torch.all(
        prompt_lengths
        == prompt_lengths[0]
    ):

        raise RuntimeError(
            "Prompt lengths unexpectedly differ "
            "inside a Stage-B batch."
        )


    prompt_length = int(
        prompt_lengths[
            0
        ].item()
    )


    # Remove padding after the real prompt.
    prompt_embeds = (
        prompt_embeds[
            :,
            :prompt_length,
            :
        ]
    )


    batch_size = len(
        batch
    )


    target_lists = [
        target_ids_for_example(
            example
        )

        for example
        in batch
    ]


    max_target_length = max(
        len(ids)
        for ids
        in target_lists
    )


    pad_token_id = (
        model.tokenizer.pad_token_id
    )


    if pad_token_id is None:

        pad_token_id = (
            EOS_TOKEN_ID
            if EOS_TOKEN_ID is not None
            else 0
        )


    target_ids = torch.full(
        (
            batch_size,
            max_target_length,
        ),
        fill_value=pad_token_id,
        dtype=torch.long,
        device=device,
    )


    target_mask = torch.zeros(
        (
            batch_size,
            max_target_length,
        ),
        dtype=torch.long,
        device=device,
    )


    for index, ids in enumerate(
        target_lists
    ):

        tensor_ids = torch.tensor(
            ids,
            dtype=torch.long,
            device=device,
        )


        target_ids[
            index,
            :len(ids)
        ] = tensor_ids


        target_mask[
            index,
            :len(ids)
        ] = 1


    embedding_layer = (
        model.llm.get_input_embeddings()
    )


    target_embeds = (
        embedding_layer(
            target_ids
        )
    )


    full_embeds = torch.cat(
        [
            prompt_embeds,
            target_embeds,
        ],
        dim=1,
    )


    prompt_attention = torch.ones(
        (
            batch_size,
            prompt_length,
        ),
        dtype=torch.long,
        device=device,
    )


    full_attention = torch.cat(
        [
            prompt_attention,
            target_mask,
        ],
        dim=1,
    )


    labels = torch.full(
        (
            batch_size,
            prompt_length
            + max_target_length,
        ),
        fill_value=-100,
        dtype=torch.long,
        device=device,
    )


    answer_labels = (
        target_ids.clone()
    )


    answer_labels[
        target_mask == 0
    ] = -100


    labels[
        :,
        prompt_length:
    ] = answer_labels


    outputs = (
        model.llm(
            inputs_embeds=
                full_embeds,

            attention_mask=
                full_attention,

            labels=
                labels,

            return_dict=True,
        )
    )


    sequence_loss = (
        outputs.loss.float()
    )


    # --------------------------------------------------------
    # Last PROMPT position predicts first answer token.
    # --------------------------------------------------------

    next_logits = (
        outputs.logits[
            :,
            prompt_length - 1,
            :
        ]
    )


    yes_logits = (
        next_logits[
            :,
            YES_TOKEN
        ]
    )


    no_logits = (
        next_logits[
            :,
            NO_TOKEN
        ]
    )


    pair_logits = torch.stack(
        [
            yes_logits,
            no_logits,
        ],
        dim=1,
    ).float()


    pair_targets = torch.tensor(
        [
            0
            if example["_is_yes"]
            else 1

            for example
            in batch
        ],
        dtype=torch.long,
        device=device,
    )


    pairwise_loss = (
        F.cross_entropy(
            pair_logits,
            pair_targets,
        )
    )


    total_loss = (
        SEQUENCE_WEIGHT
        * sequence_loss
        +
        PAIRWISE_WEIGHT
        * pairwise_loss
    )


    return (
        total_loss,
        sequence_loss,
        pairwise_loss,
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

            if prediction is True:
                yes_correct += 1


        else:

            no_total += 1

            if prediction is False:
                no_correct += 1


    return 0.5 * (
        yes_correct
        / yes_total
        +
        no_correct
        / no_total
    )


# ============================================================
# Pairwise natural-language evaluation
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
            len(examples),
            EVAL_BATCH_SIZE,
        ):

            batch = examples[
                start:
                start + EVAL_BATCH_SIZE
            ]


            (
                prompt_embeds,
                prompt_mask,
            ) = (
                model.pad_and_apply_batch(
                    batch
                )
            )


            outputs = (
                model.llm(
                    inputs_embeds=
                        prompt_embeds,

                    attention_mask=
                        prompt_mask,

                    return_dict=True,
                )
            )


            for index, example in enumerate(
                batch
            ):

                last_position = (
                    int(
                        prompt_mask[
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
                        YES_TOKEN
                    ]
                    >= logits[
                        NO_TOKEN
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


    return {
        "ba_vs_original":
            original_ba,

        "ba_vs_donor":
            donor_ba,

        "predicted_yes":
            int(
                sum(
                    prediction is True
                    for prediction
                    in predictions
                )
            ),

        "predicted_no":
            int(
                sum(
                    prediction is False
                    for prediction
                    in predictions
                )
            ),
    }


# ============================================================
# Manual greedy generation
# ============================================================

def greedy_generate_batch(
    batch,
):

    (
        prompt_embeds,
        prompt_mask,
    ) = (
        model.pad_and_apply_batch(
            batch
        )
    )


    lengths = (
        prompt_mask
        .sum(
            dim=1
        )
        .long()
    )


    if not torch.all(
        lengths
        == lengths[0]
    ):

        raise RuntimeError(
            "Generation batch prompt lengths differ."
        )


    prompt_length = int(
        lengths[
            0
        ].item()
    )


    current_embeds = (
        prompt_embeds[
            :,
            :prompt_length,
            :
        ]
    )


    current_mask = torch.ones(
        (
            len(batch),
            prompt_length,
        ),
        dtype=torch.long,
        device=device,
    )


    generated_ids = [
        []
        for _ in batch
    ]


    embedding_layer = (
        model.llm.get_input_embeddings()
    )


    with torch.no_grad():

        for _ in range(
            MAX_GENERATION_TOKENS
        ):

            outputs = (
                model.llm(
                    inputs_embeds=
                        current_embeds,

                    attention_mask=
                        current_mask,

                    return_dict=True,
                )
            )


            next_ids = (
                outputs.logits[
                    :,
                    -1,
                    :
                ]
                .argmax(
                    dim=-1
                )
            )


            for index, token_id in enumerate(
                next_ids.tolist()
            ):

                generated_ids[
                    index
                ].append(
                    token_id
                )


            next_embeds = (
                embedding_layer(
                    next_ids
                )
                .unsqueeze(
                    1
                )
            )


            current_embeds = torch.cat(
                [
                    current_embeds,
                    next_embeds,
                ],
                dim=1,
            )


            current_mask = torch.cat(
                [
                    current_mask,

                    torch.ones(
                        (
                            len(batch),
                            1,
                        ),
                        dtype=torch.long,
                        device=device,
                    ),
                ],
                dim=1,
            )


    return [
        model.tokenizer.decode(
            ids,
            skip_special_tokens=True,
        )

        for ids
        in generated_ids
    ]


def parse_yes_no(
    text,
):

    normalized = (
        text
        .strip()
        .lower()
    )


    if normalized.startswith(
        "yes"
    ):
        return True


    if normalized.startswith(
        "no"
    ):
        return False


    return None


def evaluate_generation(
    examples,
):

    predictions = []

    labels = []

    texts = []


    for start in range(
        0,
        len(examples),
        EVAL_BATCH_SIZE,
    ):

        batch = examples[
            start:
            start + EVAL_BATCH_SIZE
        ]


        generated = (
            greedy_generate_batch(
                batch
            )
        )


        for example, text in zip(
            batch,
            generated,
        ):

            prediction = (
                parse_yes_no(
                    text
                )
            )


            predictions.append(
                prediction
            )


            labels.append(
                example[
                    "_is_yes"
                ]
            )


            texts.append(
                text
            )


    valid = sum(
        prediction is not None
        for prediction
        in predictions
    )


    return {
        "ba":
            balanced_accuracy(
                predictions,
                labels,
            ),

        "valid_fraction":
            valid
            / len(
                predictions
            ),

        "sample_outputs":
            texts[:6],
    }


# ============================================================
# Snapshot / restore LoRA only
# ============================================================

def snapshot_lora():

    current = dict(
        model.llm.named_parameters()
    )


    return {
        name:
            current[
                name
            ]
            .detach()
            .cpu()
            .clone()

        for name
        in initial_lora_names
    }


def restore_lora(
    state,
):

    current = dict(
        model.llm.named_parameters()
    )


    for name, tensor in state.items():

        current[
            name
        ].data.copy_(
            tensor.to(
                current[
                    name
                ].device
            )
        )


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    [
        {
            "params":
                lora_parameters,

            "lr":
                LR_LORA,

            "weight_decay":
                WEIGHT_DECAY,
        },
    ]
)


# ============================================================
# Evaluation helper
# ============================================================

def evaluate_all(
    include_generation=True,
):

    metrics = {
        "train_real":
            evaluate_pairwise(
                train_real
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


    if include_generation:

        metrics[
            "val_generation_real"
        ] = (
            evaluate_generation(
                val_real
            )
        )


    return metrics


# ============================================================
# Initial diagnostic
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "STAGE B.1 — BEFORE LANGUAGE TRAINING"
)

print(
    "=" * 120
)


initial_metrics = (
    evaluate_all(
        include_generation=True
    )
)


print(
    "PAIRWISE"
)

print(
    "TRAIN REAL:",
    f"{100 * initial_metrics['train_real']['ba_vs_original']:.1f}%"
)

print(
    "VAL REAL:",
    f"{100 * initial_metrics['val_real']['ba_vs_original']:.1f}%"
)

print(
    "VAL ZERO:",
    f"{100 * initial_metrics['val_zero']['ba_vs_original']:.1f}%"
)

print(
    "VAL SHUFFLED original:",
    f"{100 * initial_metrics['val_shuffle']['ba_vs_original']:.1f}%"
)

print(
    "VAL SHUFFLED donor:",
    f"{100 * initial_metrics['val_shuffle']['ba_vs_donor']:.1f}%"
)


print(
    "\nGENERATION"
)

print(
    "VAL valid:",
    f"{100 * initial_metrics['val_generation_real']['valid_fraction']:.1f}%"
)

print(
    "VAL BA:",
    f"{100 * initial_metrics['val_generation_real']['ba']:.1f}%"
)

print(
    "Examples:"
)

for text in (
    initial_metrics[
        "val_generation_real"
    ][
        "sample_outputs"
    ]
):

    print(
        repr(text)
    )


# ============================================================
# Best state
#
# Selection uses ONLY REAL validation language performance.
#
# Ablations NEVER choose the epoch.
# ============================================================

best_epoch = 0

best_generation_ba = (
    initial_metrics[
        "val_generation_real"
    ][
        "ba"
    ]
)

best_pairwise_ba = (
    initial_metrics[
        "val_real"
    ][
        "ba_vs_original"
    ]
)

best_lora_state = (
    snapshot_lora()
)

best_optimizer_state = copy.deepcopy(
    optimizer.state_dict()
)


history = [
    {
        "epoch":
            0,

        "mean_total_loss":
            None,

        "mean_sequence_loss":
            None,

        "mean_pairwise_loss":
            None,

        "metrics":
            initial_metrics,
    }
]


# ============================================================
# Training
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "STAGE B.1 — TRAINING LORA ONLY"
)

print(
    "=" * 120
)


for epoch in range(
    1,
    EPOCHS + 1,
):

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


    total_losses = []
    sequence_losses = []
    pairwise_losses = []


    for start in range(
        0,
        len(epoch_examples),
        BATCH_SIZE,
    ):

        batch = (
            epoch_examples[
                start:
                start + BATCH_SIZE
            ]
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        (
            total_loss,
            sequence_loss,
            pairwise_loss,
        ) = (
            loss_for_batch(
                batch
            )
        )


        total_loss.backward()


        torch.nn.utils.clip_grad_norm_(
            lora_parameters,
            CLIP_LORA,
        )


        optimizer.step()


        total_losses.append(
            float(
                total_loss
                .detach()
                .cpu()
                .item()
            )
        )


        sequence_losses.append(
            float(
                sequence_loss
                .detach()
                .cpu()
                .item()
            )
        )


        pairwise_losses.append(
            float(
                pairwise_loss
                .detach()
                .cpu()
                .item()
            )
        )


    mean_total = (
        sum(total_losses)
        / len(total_losses)
    )

    mean_sequence = (
        sum(sequence_losses)
        / len(sequence_losses)
    )

    mean_pairwise = (
        sum(pairwise_losses)
        / len(pairwise_losses)
    )


    print(
        f"Finished epoch {epoch:2d} | "
        f"total={mean_total:.4f} | "
        f"seq={mean_sequence:.4f} | "
        f"pair={mean_pairwise:.4f}"
    )


    if epoch in EVAL_EPOCHS:

        metrics = (
            evaluate_all(
                include_generation=True
            )
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


        generation_ba = (
            metrics[
                "val_generation_real"
            ][
                "ba"
            ]
        )


        generation_valid = (
            metrics[
                "val_generation_real"
            ][
                "valid_fraction"
            ]
        )


        print(
            f"\nEpoch {epoch:2d} DIAGNOSTIC\n"
            f"PAIRWISE: "
            f"TRAIN={100 * train_real_ba:.1f}% | "
            f"VAL={100 * val_real_ba:.1f}% | "
            f"ZERO={100 * val_zero_ba:.1f}% | "
            f"SHUF-O={100 * val_shuffle_original:.1f}% | "
            f"SHUF-D={100 * val_shuffle_donor:.1f}%\n"
            f"GENERATION: "
            f"VAL BA={100 * generation_ba:.1f}% | "
            f"valid={100 * generation_valid:.1f}%"
        )


        print(
            "Generation examples:"
        )


        for text in (
            metrics[
                "val_generation_real"
            ][
                "sample_outputs"
            ]
        ):

            print(
                " ",
                repr(text)
            )


        history.append(
            {
                "epoch":
                    epoch,

                "mean_total_loss":
                    mean_total,

                "mean_sequence_loss":
                    mean_sequence,

                "mean_pairwise_loss":
                    mean_pairwise,

                "metrics":
                    metrics,
            }
        )


        # ----------------------------------------------------
        # Choose epoch ONLY from REAL validation behavior.
        #
        # Primary: generation BA
        # Tiebreak: pairwise BA
        # ----------------------------------------------------

        candidate = (
            generation_ba,
            val_real_ba,
        )


        current_best = (
            best_generation_ba,
            best_pairwise_ba,
        )


        if candidate > current_best:

            best_epoch = epoch

            best_generation_ba = (
                generation_ba
            )

            best_pairwise_ba = (
                val_real_ba
            )

            best_lora_state = (
                snapshot_lora()
            )

            best_optimizer_state = (
                copy.deepcopy(
                    optimizer.state_dict()
                )
            )


            print(
                "✓ New best REAL validation "
                "language state."
            )


# ============================================================
# Restore best
# ============================================================

restore_lora(
    best_lora_state
)


# ============================================================
# Full best-state evaluation
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "RESTORING BEST STAGE-B STATE"
)

print(
    "=" * 120
)


print(
    "Best epoch:",
    best_epoch,
)

print(
    "Best REAL generation BA:",
    f"{100 * best_generation_ba:.1f}%"
)

print(
    "Best REAL pairwise BA:",
    f"{100 * best_pairwise_ba:.1f}%"
)


best_metrics = {
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

    "generation_real":
        evaluate_generation(
            val_real
        ),

    "generation_zero":
        evaluate_generation(
            val_zero
        ),

    "generation_shuffle":
        evaluate_generation(
            val_shuffle
        ),
}


print(
    "\nPAIRWISE — BEST STATE"
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
    "\nVALIDATION"
)

print(
    "REAL:",
    f"{100 * best_metrics['val_real']['ba_vs_original']:.1f}%"
)

print(
    "ZERO:",
    f"{100 * best_metrics['val_zero']['ba_vs_original']:.1f}%"
)

print(
    "SHUFFLED original:",
    f"{100 * best_metrics['val_shuffle']['ba_vs_original']:.1f}%"
)

print(
    "SHUFFLED donor:",
    f"{100 * best_metrics['val_shuffle']['ba_vs_donor']:.1f}%"
)


print(
    "\nGENERATION"
)

print(
    "REAL valid:",
    f"{100 * best_metrics['generation_real']['valid_fraction']:.1f}%"
)

print(
    "REAL BA:",
    f"{100 * best_metrics['generation_real']['ba']:.1f}%"
)

print(
    "ZERO BA:",
    f"{100 * best_metrics['generation_zero']['ba']:.1f}%"
)

print(
    "SHUFFLED BA vs original:",
    f"{100 * best_metrics['generation_shuffle']['ba']:.1f}%"
)


print(
    "\nREAL examples:"
)

for text in (
    best_metrics[
        "generation_real"
    ][
        "sample_outputs"
    ]
):

    print(
        repr(text)
    )


# ============================================================
# Stage-B interpretation
# ============================================================

print(
    "\n"
    + "=" * 120
)

print(
    "STAGE B.1 INTERPRETATION"
)

print(
    "=" * 120
)


stage_b_success = (
    best_metrics[
        "generation_real"
    ][
        "ba"
    ]
    >= 0.70

    and best_metrics[
        "generation_real"
    ][
        "valid_fraction"
    ]
    >= 0.90

    and best_metrics[
        "val_zero"
    ][
        "ba_vs_original"
    ]
    <= 0.60

    and best_metrics[
        "val_real"
    ][
        "ba_vs_original"
    ]
    >= 0.70
)


if stage_b_success:

    print(
        "STAGE B.1 SUCCESS."
    )

    print(
        "The frozen Stage-A CGM representation "
        "can be read out in natural language while "
        "remaining dependent on the waveform."
    )


elif (
    best_metrics[
        "val_real"
    ][
        "ba_vs_original"
    ]
    >= 0.70
    and best_metrics[
        "generation_real"
    ][
        "ba"
    ]
    < 0.70
):

    print(
        "SIGNAL READOUT WORKS, BUT GREEDY "
        "LANGUAGE GENERATION IS NOT YET STABLE."
    )


else:

    print(
        "LANGUAGE READOUT DID NOT YET PRESERVE "
        "THE STAGE-A SIGNAL PERFORMANCE."
    )


# ============================================================
# Save self-contained Stage-B checkpoint
# ============================================================

sensor_state = {
    "t5":
        {
            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in layers[4]
            .state_dict()
            .items()
        },

    "t6":
        {
            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in layers[5]
            .state_dict()
            .items()
        },

    "projector":
        {
            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in model.projector
            .state_dict()
            .items()
        },
}


OUTPUT_CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


torch.save(
    {
        "experiment":
            "cgm_stage_b1_lora_language_readout",

        "base_repo":
            STAGE5_REPO,

        "stage_a_source":
            str(
                STAGE_A_CHECKPOINT_PATH
            ),

        "sensor_architecture":
            {
                "t1_t4":
                    "stage5",

                "t5_t6_initialization":
                    "tsqa_then_cgm_stage_a",

                "projector":
                    "stage5_then_cgm_stage_a",
            },

        "best_epoch":
            best_epoch,

        "best_generation_real_ba":
            best_generation_ba,

        "best_pairwise_real_ba":
            best_pairwise_ba,

        "sensor_state":
            sensor_state,

        "lora_state":
            best_lora_state,

        "optimizer_state":
            best_optimizer_state,

        "normalization":
            {
                "type":
                    "fixed_global",

                "mean":
                    train_mean,

                "std":
                    train_std,
            },
    },

    OUTPUT_CHECKPOINT_PATH,
)


print(
    "\nSaved Stage-B checkpoint:"
)

print(
    OUTPUT_CHECKPOINT_PATH
)


# ============================================================
# Save diagnostic
# ============================================================

results = {
    "experiment":
        "cgm_stage_b1_lora_language_readout",

    "stage_a_source":
        str(
            STAGE_A_CHECKPOINT_PATH
        ),

    "stage_a_validation_ba":
        float(
            stage_a_checkpoint[
                "best_validation_real_ba"
            ]
        ),

    "training":
        {
            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "lr_lora":
                LR_LORA,

            "sequence_weight":
                SEQUENCE_WEIGHT,

            "pairwise_weight":
                PAIRWISE_WEIGHT,

            "sensor_trainable":
                False,

            "lora_trainable_parameters":
                lora_trainable,

            "custom_answer_loss":
                True,

            "answer_add_special_tokens":
                False,
        },

    "initial_metrics":
        initial_metrics,

    "history":
        history,

    "best_epoch":
        best_epoch,

    "best_metrics":
        best_metrics,

    "stage_b_success":
        stage_b_success,

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