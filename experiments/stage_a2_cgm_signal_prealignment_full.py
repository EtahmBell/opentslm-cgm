"""
STAGE A.2 — FULL-DATA CGM SIGNAL PREALIGNMENT

Purpose
-------
Test whether the ORIGINAL public OpenTSLM Stage-5 ECG checkpoint
can learn a TRANSFERABLE CGM waveform concept:

    Did glucose exceed 180 mg/dL?

This is the scaled-up version of Stage A.

IMPORTANT
---------
This experiment starts FRESH from:

    OpenTSLM/llama-3.2-1b-ecg-sp

It does NOT load the previous 80-window Stage-A checkpoint.

There are NO TSQA-transplanted components.

MODEL
-----
Original public Stage-5 ECG SP checkpoint.

Frozen:
    - patch / early encoder
    - Transformer 1-4
    - base LLM
    - Stage-5 ECG LoRA

Trainable:
    - original Stage-5 Transformer 5
    - original Stage-5 Transformer 6
    - original Stage-5 projector

INPUT
-----
Fixed global CGM normalization:

    (raw_mg_dl - 142.143) / 37.232

Text is IDENTICAL for every example.

No per-window:
    - mean
    - standard deviation
    - maximum
    - TIR
    - participant identity
    - window index
    - other signal-derived information

TASK
----
Question:
    Did glucose exceed 180 mg/dL?

Answers:
    (a) Yes
    (b) No

Loss:
    TRUE binary pairwise CE only.

Only logits for the first discriminating tokens of
"(a)" and "(b)" participate in the loss.

DATA
----
Training pool:
    YES available: 272
    NO available: 231

Use:
    231 YES
    231 NO
    = 462 training windows

Validation pool:
    YES available: 59
    NO available: 49

Use:
    49 YES
    49 NO
    = 98 validation windows

TRAIN and VALIDATION participants remain completely separate.

TEST participants are NEVER loaded.

TRAINING
--------
Single-example optimizer updates.

Each epoch:
    shuffle YES examples
    shuffle NO examples
    interleave:
        YES
        NO
        YES
        NO
        ...

20 epochs.

VALIDATION ABLATIONS
--------------------
For both TRAIN and VALIDATION:

1. REAL
   Original waveform.

2. ZERO
   Same prompt but waveform = all zeros.

3. OPPOSITE-LABEL SHUFFLE
   YES example receives a NO waveform.
   NO example receives a YES waveform.

For shuffled signals report:

    BA vs original label
    BA vs donor waveform label

Strong transferable signal learning would look like:

    TRAIN REAL          high
    TRAIN ZERO          ~50%
    TRAIN SHUF-original low
    TRAIN SHUF-donor    high

    VAL REAL            clearly > chance
    VAL ZERO            ~50%
    VAL SHUF-original   substantially lower
    VAL SHUF-donor      high

No language-generation training occurs in Stage A.

TEST REMAINS SEALED.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cgm_multitask import build_multitask_examples
from opentslm import OpenTSLM


DATA_DIR = Path(r"C:\tslm\cgm\data")

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_stage_a2_full_signal_prealignment.json"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_stage_a2_above180_best.pt"
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


# Use essentially the full balanced pools.
TRAIN_PER_CLASS = 231
VAL_PER_CLASS = 49


# 462 updates / epoch × 20 epochs
# = 9,240 optimizer updates.
EPOCHS = 20


LR_T5_T6 = 2e-4
LR_PROJECTOR = 1e-4

WEIGHT_DECAY = 1e-2

CLIP_T5_T6 = 1.0
CLIP_PROJECTOR = 1.0


# Evaluation is batched for speed.
EVAL_BATCH_SIZE = 16
REP_BATCH_SIZE = 32


EVAL_EPOCHS = {
    0,
    1,
    2,
    5,
    10,
    15,
    20,
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


validation_manifest = manifest.get(
    "validation"
)

if validation_manifest is None:
    validation_manifest = manifest.get(
        "val"
    )

if validation_manifest is None:
    raise RuntimeError(
        "Validation split not found in manifest."
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
# Temporary model for EOS
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
# Build pools
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
        task_ids=[TASK_ID],
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
        task_ids=[TASK_ID],
    )
)


# ============================================================
# Class split
# ============================================================

def split_by_class(examples):

    yes = [
        ex
        for ex in examples
        if ex["_answer_clean"] == "(a)"
    ]

    no = [
        ex
        for ex in examples
        if ex["_answer_clean"] == "(b)"
    ]

    return yes, no


train_yes_pool, train_no_pool = (
    split_by_class(train_pool)
)

val_yes_pool, val_no_pool = (
    split_by_class(validation_pool)
)


print(
    "\nTRAIN available:"
)

print(
    "  YES:",
    len(train_yes_pool),
)

print(
    "  NO:",
    len(train_no_pool),
)


print(
    "\nVALIDATION available:"
)

print(
    "  YES:",
    len(val_yes_pool),
)

print(
    "  NO:",
    len(val_no_pool),
)


if len(train_yes_pool) < TRAIN_PER_CLASS:
    raise RuntimeError(
        "Not enough train YES examples."
    )

if len(train_no_pool) < TRAIN_PER_CLASS:
    raise RuntimeError(
        "Not enough train NO examples."
    )

if len(val_yes_pool) < VAL_PER_CLASS:
    raise RuntimeError(
        "Not enough validation YES examples."
    )

if len(val_no_pool) < VAL_PER_CLASS:
    raise RuntimeError(
        "Not enough validation NO examples."
    )


# ============================================================
# Participant-aware sampler
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

        grouped[participant].append(
            example
        )


    participants = list(
        grouped.keys()
    )

    rng.shuffle(participants)


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
                    grouped[participant].pop()
                )

                progress = True

                if len(selected) == n:
                    break


        if not progress:
            raise RuntimeError(
                "Could not obtain requested sample."
            )


    return selected


# ============================================================
# Select full balanced sets
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
    ex.get("_participant_id")
    for ex in (
        selected_train_yes
        + selected_train_no
    )
}

val_participants = {
    ex.get("_participant_id")
    for ex in (
        selected_val_yes
        + selected_val_no
    )
}

participant_overlap = (
    train_participants
    & val_participants
)


print(
    "\nParticipant split audit:"
)

print(
    "  selected train participants:",
    len(train_participants),
)

print(
    "  selected validation participants:",
    len(val_participants),
)

print(
    "  overlap:",
    len(participant_overlap),
)


if participant_overlap:

    raise RuntimeError(
        "TRAIN/VALIDATION participant leakage: "
        f"{sorted(participant_overlap)}"
    )


# ============================================================
# Invariant interface
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

def make_stage_a_example(
    base_example,
):

    signal = (
        base_example[
            "time_series"
        ][0]
    )

    if not torch.is_tensor(signal):
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
        base_example["_answer_clean"]
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
    make_stage_a_example(x)
    for x in selected_train_yes
]

train_no = [
    make_stage_a_example(x)
    for x in selected_train_no
]

val_yes = [
    make_stage_a_example(x)
    for x in selected_val_yes
]

val_no = [
    make_stage_a_example(x)
    for x in selected_val_no
]


# ============================================================
# Text invariance audit
# ============================================================

all_examples_for_audit = (
    train_yes
    + train_no
    + val_yes
    + val_no
)


unique_text_inputs = {
    (
        ex["pre_prompt"],
        ex["time_series_text"][0],
        ex["post_prompt"],
    )
    for ex in all_examples_for_audit
}


print(
    "\nUnique prompt/text combinations:",
    len(unique_text_inputs),
)


if len(unique_text_inputs) != 1:
    raise RuntimeError(
        "Per-example textual variation detected."
    )


print(
    "✓ Text is identical across all examples."
)


# ============================================================
# Interleave classes
# ============================================================

def interleave(
    yes_examples,
    no_examples,
):

    if len(yes_examples) != len(no_examples):
        raise RuntimeError(
            "Expected balanced classes."
        )

    output = []

    for y, n in zip(
        yes_examples,
        no_examples,
    ):
        output.append(y)
        output.append(n)

    return output


train_real = interleave(
    train_yes,
    train_no,
)

val_real = interleave(
    val_yes,
    val_no,
)


# ============================================================
# Signal clone
# ============================================================

def clone_with_signal(
    example,
    signal,
    source,
    donor_is_yes=None,
):

    cloned = dict(example)

    cloned["time_series"] = [
        signal
        .detach()
        .clone()
        .float()
    ]

    cloned["_signal_source"] = source

    cloned["_donor_is_yes"] = (
        donor_is_yes
    )

    return cloned


# ============================================================
# Zero ablation
# ============================================================

def make_zero_examples(
    examples,
):

    output = []

    for ex in examples:

        signal = ex[
            "time_series"
        ][0]

        output.append(
            clone_with_signal(
                ex,
                torch.zeros_like(signal),
                source="zero",
            )
        )

    return output


train_zero = make_zero_examples(
    train_real
)

val_zero = make_zero_examples(
    val_real
)


# ============================================================
# Opposite-label shuffle
# ============================================================

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

    rng.shuffle(yes_donors)
    rng.shuffle(no_donors)


    originals = interleave(
        yes_examples,
        no_examples,
    )


    yes_index = 0
    no_index = 0

    shuffled = []


    for original in originals:

        if original["_is_yes"]:

            donor = no_donors[
                no_index
            ]

            no_index += 1

        else:

            donor = yes_donors[
                yes_index
            ]

            yes_index += 1


        shuffled.append(
            clone_with_signal(
                original,
                donor["time_series"][0],
                source=(
                    "opposite_label_shuffle"
                ),
                donor_is_yes=(
                    donor["_is_yes"]
                ),
            )
        )


    return shuffled


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


for ex in (
    train_shuffle
    + val_shuffle
):

    if (
        ex["_donor_is_yes"]
        == ex["_is_yes"]
    ):
        raise RuntimeError(
            "Shuffle audit failed."
        )


print(
    "✓ Every shuffled waveform comes "
    "from the opposite class."
)


print(
    "\nTRAIN diagnostic set:"
)

print(
    "  total:",
    len(train_real),
)

print(
    "  YES:",
    len(train_yes),
)

print(
    "  NO:",
    len(train_no),
)


print(
    "\nVALIDATION diagnostic set:"
)

print(
    "  total:",
    len(val_real),
)

print(
    "  YES:",
    len(val_yes),
)

print(
    "  NO:",
    len(val_no),
)


# ============================================================
# Remove temporary model
# ============================================================

del temporary_model

gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()


# ============================================================
# Fresh original Stage-5
# ============================================================

print(
    "\n"
    + "=" * 100
)

print(
    "LOADING FRESH ORIGINAL PUBLIC STAGE-5"
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
# Decision tokens
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


if not a_ids or not b_ids:
    raise RuntimeError(
        "Could not tokenize (a)/(b)."
    )


A_TOKEN = int(a_ids[0])
B_TOKEN = int(b_ids[0])


if A_TOKEN == B_TOKEN:
    raise RuntimeError(
        "Decision tokens identical."
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
        [A_TOKEN]
    ),
)

print(
    " (b):",
    b_ids,
    "->",
    B_TOKEN,
    model.tokenizer.convert_ids_to_tokens(
        [B_TOKEN]
    ),
)


# ============================================================
# Freeze EVERYTHING
# ============================================================

for p in model.encoder.parameters():
    p.requires_grad = False

for p in model.projector.parameters():
    p.requires_grad = False

for p in model.llm.parameters():
    p.requires_grad = False


layers = (
    model.encoder
    .encoder
    .layers
)


if len(layers) != 6:
    raise RuntimeError(
        "Expected six encoder layers."
    )


# ============================================================
# Unfreeze original Stage-5 T5/T6
# ============================================================

for p in layers[4].parameters():
    p.requires_grad = True

for p in layers[5].parameters():
    p.requires_grad = True


# ============================================================
# Unfreeze original Stage-5 projector
# ============================================================

for p in model.projector.parameters():
    p.requires_grad = True


late_parameters = [
    p
    for layer in (
        layers[4],
        layers[5],
    )
    for p in layer.parameters()
    if p.requires_grad
]


projector_parameters = [
    p
    for p in model.projector.parameters()
    if p.requires_grad
]


# ============================================================
# Parameter audit
# ============================================================

early_encoder_trainable = sum(
    p.numel()
    for name, p
    in model.encoder.named_parameters()
    if (
        p.requires_grad
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


llm_trainable = sum(
    p.numel()
    for p in model.llm.parameters()
    if p.requires_grad
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
    "Original Stage-5 T5/T6:",
    f"{sum(p.numel() for p in late_parameters):,}"
)

print(
    "Original Stage-5 projector:",
    f"{sum(p.numel() for p in projector_parameters):,}"
)

print(
    "Base LLM + Stage-5 LoRA:",
    f"{llm_trainable:,}"
)


if early_encoder_trainable != 0:
    raise RuntimeError(
        "Early encoder unexpectedly trainable."
    )

if llm_trainable != 0:
    raise RuntimeError(
        "LLM / LoRA unexpectedly trainable."
    )


# ============================================================
# No dropout, gradients still enabled
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
# Pairwise training loss — SINGLE EXAMPLE
# ============================================================

def pairwise_loss_single(
    example,
):

    (
        inputs_embeds,
        attention_mask,
    ) = (
        model.pad_and_apply_batch(
            [example]
        )
    )


    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    )


    last_position = (
        int(
            attention_mask[0]
            .sum()
            .item()
        )
        - 1
    )


    logits = outputs.logits[
        0,
        last_position,
        :
    ]


    pair_logits = torch.stack(
        [
            logits[A_TOKEN],
            logits[B_TOKEN],
        ]
    ).float()


    target = torch.tensor(
        [
            0
            if example["_is_yes"]
            else 1
        ],
        dtype=torch.long,
        device=device,
    )


    return F.cross_entropy(
        pair_logits.unsqueeze(0),
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


    for pred, label in zip(
        predictions,
        labels,
    ):

        if label:

            yes_total += 1

            if pred:
                yes_correct += 1

        else:

            no_total += 1

            if not pred:
                no_correct += 1


    return 0.5 * (
        yes_correct / yes_total
        +
        no_correct / no_total
    )


# ============================================================
# Batched pairwise evaluation
# ============================================================

def evaluate_pairwise(
    examples,
):

    predictions = []
    original_labels = []
    donor_labels = []

    p_correct = []
    margins = []


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
                inputs_embeds,
                attention_mask,
            ) = (
                model.pad_and_apply_batch(
                    batch
                )
            )


            outputs = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
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


                logits = outputs.logits[
                    index,
                    last_position,
                    :
                ]


                pair_logits = torch.stack(
                    [
                        logits[A_TOKEN],
                        logits[B_TOKEN],
                    ]
                ).float()


                probs = torch.softmax(
                    pair_logits,
                    dim=0,
                )


                p_yes = float(
                    probs[0].item()
                )

                p_no = float(
                    probs[1].item()
                )


                prediction = (
                    p_yes >= p_no
                )

                original = (
                    example["_is_yes"]
                )

                donor = example.get(
                    "_donor_is_yes"
                )


                predictions.append(
                    prediction
                )

                original_labels.append(
                    original
                )

                donor_labels.append(
                    donor
                )


                p_correct.append(
                    p_yes
                    if original
                    else p_no
                )


                margins.append(
                    float(
                        (
                            pair_logits[0]
                            - pair_logits[1]
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
        x is not None
        for x in donor_labels
    ):

        donor_ba = (
            balanced_accuracy(
                predictions,
                donor_labels,
            )
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
            int(sum(predictions)),

        "predicted_no":
            int(
                len(predictions)
                - sum(predictions)
            ),

        "mean_p_correct_original":
            float(
                sum(p_correct)
                / len(p_correct)
            ),

        "margin_sd":
            float(
                margin_tensor.std().item()
            ),
    }


# ============================================================
# Batched representation diagnostic
# ============================================================

def representation_stats(
    examples,
):

    encoder_vectors = []
    projector_vectors = []
    labels = []


    with torch.no_grad():

        for start in range(
            0,
            len(examples),
            REP_BATCH_SIZE,
        ):

            batch = examples[
                start:
                start + REP_BATCH_SIZE
            ]


            signals = torch.stack(
                [
                    ex[
                        "time_series"
                    ][0]
                    .detach()
                    .clone()
                    .float()

                    for ex in batch
                ],
                dim=0,
            ).to(device)


            encoded = model.encoder(
                signals
            )


            projected = model.projector(
                encoded
            )


            for index, example in enumerate(
                batch
            ):

                encoder_vectors.append(
                    encoded[index]
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )

                projector_vectors.append(
                    projected[index]
                    .detach()
                    .float()
                    .cpu()
                    .reshape(-1)
                )

                labels.append(
                    example["_is_yes"]
                )


    def summarize(vectors):

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
            i
            for i, label
            in enumerate(labels)
            if label
        ]

        no_indices = [
            i
            for i, label
            in enumerate(labels)
            if not label
        ]


        yes_centroid = (
            matrix[yes_indices]
            .mean(dim=0)
        )

        no_centroid = (
            matrix[no_indices]
            .mean(dim=0)
        )


        cosine = float(
            F.cosine_similarity(
                yes_centroid.unsqueeze(0),
                no_centroid.unsqueeze(0),
            ).item()
        )


        l2 = float(
            torch.norm(
                yes_centroid
                - no_centroid
            ).item()
            / math.sqrt(
                matrix.shape[1]
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

def grad_norm(parameters):

    squared = 0.0

    for p in parameters:

        if p.grad is None:
            continue

        value = float(
            p.grad
            .detach()
            .float()
            .norm()
            .item()
        )

        squared += value * value

    return math.sqrt(squared)


# ============================================================
# State snapshot / restore
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


def restore_state(state):

    layers[4].load_state_dict(
        state["t5"],
        strict=True,
    )

    layers[5].load_state_dict(
        state["t6"],
        strict=True,
    )

    model.projector.load_state_dict(
        state["projector"],
        strict=True,
    )


# ============================================================
# Full diagnostic
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
# Pretty output
# ============================================================

def print_metrics(
    epoch,
    mean_loss,
    metrics,
):

    tr = metrics["train_real"]
    tz = metrics["train_zero"]
    ts = metrics["train_shuffle"]

    vr = metrics["val_real"]
    vz = metrics["val_zero"]
    vs = metrics["val_shuffle"]

    train_sd = (
        metrics[
            "train_representation"
        ][
            "encoder"
        ][
            "feature_sd"
        ]
    )

    val_sd = (
        metrics[
            "val_representation"
        ][
            "encoder"
        ][
            "feature_sd"
        ]
    )


    prefix = (
        f"Epoch {epoch:2d}"
        if epoch > 0
        else "Epoch  0"
    )


    loss_text = (
        f"loss={mean_loss:7.4f}"
        if mean_loss is not None
        else "loss=   -   "
    )


    print(
        f"{prefix} | "
        f"{loss_text} | "
        f"TRAIN real={100 * tr['ba_vs_original']:5.1f}% "
        f"zero={100 * tz['ba_vs_original']:5.1f}% "
        f"shufO={100 * ts['ba_vs_original']:5.1f}% "
        f"shufD={100 * ts['ba_vs_donor']:5.1f}% | "
        f"VAL real={100 * vr['ba_vs_original']:5.1f}% "
        f"zero={100 * vz['ba_vs_original']:5.1f}% "
        f"shufO={100 * vs['ba_vs_original']:5.1f}% "
        f"shufD={100 * vs['ba_vs_donor']:5.1f}% | "
        f"EncSD train={train_sd:.5f} "
        f"val={val_sd:.5f}"
    )


# ============================================================
# Initial evaluation
# ============================================================

print(
    "\n"
    + "=" * 130
)

print(
    "STAGE A.2 — BEFORE TRAINING"
)

print(
    "=" * 130
)


initial_metrics = (
    evaluate_everything()
)

print_metrics(
    0,
    None,
    initial_metrics,
)


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
    + "=" * 130
)

print(
    "STAGE A.2 — TRAINING"
)

print(
    "=" * 130
)


for epoch in range(
    1,
    EPOCHS + 1,
):

    # --------------------------------------------------------
    # SINGLE-EXAMPLE updates.
    #
    # Shuffle each class independently, then alternate.
    # --------------------------------------------------------

    epoch_yes = list(
        train_yes
    )

    epoch_no = list(
        train_no
    )


    rng_yes = random.Random(
        SEED + 1000 + epoch
    )

    rng_no = random.Random(
        SEED + 2000 + epoch
    )


    rng_yes.shuffle(
        epoch_yes
    )

    rng_no.shuffle(
        epoch_no
    )


    epoch_examples = interleave(
        epoch_yes,
        epoch_no,
    )


    losses = []


    for example in epoch_examples:

        optimizer.zero_grad(
            set_to_none=True
        )


        loss = pairwise_loss_single(
            example
        )


        loss.backward()


        if first_gradient_audit is None:

            first_gradient_audit = {
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

                "t5_t6":
                    grad_norm(
                        late_parameters
                    ),

                "projector":
                    grad_norm(
                        projector_parameters
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


        # Separate clipping.
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


    if epoch in EVAL_EPOCHS:

        mean_loss = (
            sum(losses)
            / len(losses)
        )


        metrics = (
            evaluate_everything()
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


        print_metrics(
            epoch,
            mean_loss,
            metrics,
        )


        val_real_ba = (
            metrics[
                "val_real"
            ][
                "ba_vs_original"
            ]
        )


        if (
            val_real_ba
            > best_val_ba
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
# Final epoch diagnostic
# ============================================================

print(
    "\n"
    + "=" * 130
)

print(
    "FINAL-EPOCH DIAGNOSTIC"
)

print(
    "=" * 130
)


final_epoch_metrics = (
    evaluate_everything()
)

print_metrics(
    EPOCHS,
    None,
    final_epoch_metrics,
)


# ============================================================
# Restore best validation state
# ============================================================

print(
    "\n"
    + "=" * 130
)

print(
    "RESTORING BEST REAL-VALIDATION STATE"
)

print(
    "=" * 130
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
# Final summary
# ============================================================

tr = best_metrics[
    "train_real"
]

tz = best_metrics[
    "train_zero"
]

ts = best_metrics[
    "train_shuffle"
]

vr = best_metrics[
    "val_real"
]

vz = best_metrics[
    "val_zero"
]

vs = best_metrics[
    "val_shuffle"
]


print(
    "\n"
    + "=" * 130
)

print(
    "STAGE A.2 — BEST-VALIDATION RESULT"
)

print(
    "=" * 130
)


print(
    "Best epoch:",
    best_epoch,
)


print(
    "\nTRAIN"
)

print(
    "REAL:",
    f"{100 * tr['ba_vs_original']:.1f}%"
)

print(
    "ZERO:",
    f"{100 * tz['ba_vs_original']:.1f}%"
)

print(
    "SHUFFLED vs original:",
    f"{100 * ts['ba_vs_original']:.1f}%"
)

print(
    "SHUFFLED vs donor:",
    f"{100 * ts['ba_vs_donor']:.1f}%"
)


print(
    "\nVALIDATION"
)

print(
    "REAL:",
    f"{100 * vr['ba_vs_original']:.1f}%"
)

print(
    "ZERO:",
    f"{100 * vz['ba_vs_original']:.1f}%"
)

print(
    "SHUFFLED vs original:",
    f"{100 * vs['ba_vs_original']:.1f}%"
)

print(
    "SHUFFLED vs donor:",
    f"{100 * vs['ba_vs_donor']:.1f}%"
)


print(
    "\nVALIDATION SIGNAL GAPS"
)

print(
    "REAL - ZERO:",
    f"{100 * (vr['ba_vs_original'] - vz['ba_vs_original']):+.1f} pp"
)

print(
    "REAL - SHUFFLED(original):",
    f"{100 * (vr['ba_vs_original'] - vs['ba_vs_original']):+.1f} pp"
)


# ============================================================
# Representation
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
    "Train projector Y/N cosine:",
    f"{train_rep['projector']['yes_no_cosine']:.8f}"
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

print(
    "Val projector Y/N cosine:",
    f"{val_rep['projector']['yes_no_cosine']:.8f}"
)


# ============================================================
# Interpretation
# ============================================================

print(
    "\n"
    + "=" * 130
)

print(
    "STAGE A.2 INTERPRETATION"
)

print(
    "=" * 130
)


stage_a_success = False


if (
    vr["ba_vs_original"] >= 0.75
    and vz["ba_vs_original"] <= 0.60
    and vs["ba_vs_original"] <= 0.60
    and vs["ba_vs_donor"] >= 0.70
):

    stage_a_success = True

    print(
        "STRONG TRANSFERABLE SIGNAL LEARNING."
    )

    print(
        "Real held-out CGM works, zero/shuffled "
        "waveforms break original-label performance, "
        "and shuffled predictions follow the donor signal."
    )


elif (
    vr["ba_vs_original"] >= 0.70
    and vr["ba_vs_original"]
    >= vz["ba_vs_original"] + 0.10
    and vr["ba_vs_original"]
    >= vs["ba_vs_original"] + 0.10
):

    stage_a_success = True

    print(
        "PROMISING TRANSFERABLE SIGNAL LEARNING."
    )


elif (
    tr["ba_vs_original"] >= 0.85
    and vr["ba_vs_original"] <= 0.60
):

    print(
        "SIGNAL MEMORIZATION WITHOUT "
        "HELD-OUT GENERALIZATION."
    )

    print(
        "If this persists with 462 training windows, "
        "sample size alone is unlikely to explain the issue."
    )


elif (
    tr["ba_vs_original"] <= 0.60
):

    print(
        "SENSOR PATH DID NOT LEARN "
        "THE FULL TRAINING SET."
    )


else:

    print(
        "INTERMEDIATE RESULT."
    )

    print(
        "Inspect the train/validation and "
        "real/zero/shuffled patterns."
    )


# ============================================================
# Save best checkpoint
# ============================================================

checkpoint_saved = False


if (
    best_val_ba
    > initial_val_ba
    + 1e-12
):

    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        {
            "experiment":
                "cgm_stage_a2_full_signal_prealignment",

            "base_repo":
                STAGE5_REPO_ID,

            "base_model_description":
                (
                    "Original public Stage-5 ECG SP; "
                    "no TSQA transplanted modules."
                ),

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
                    "stage5_t5":
                        best_state["t5"],

                    "stage5_t6":
                        best_state["t6"],

                    "stage5_projector":
                        best_state[
                            "projector"
                        ],
                },
        },

        CHECKPOINT_PATH,
    )


    checkpoint_saved = True


    print(
        "\nSaved best Stage-A.2 checkpoint:"
    )

    print(
        CHECKPOINT_PATH
    )


else:

    print(
        "\nNo checkpoint saved because validation "
        "never improved above the untrained baseline."
    )


# ============================================================
# Metadata serialization
# ============================================================

def example_metadata(ex):

    return {
        "participant_id":
            ex["_participant_id"],

        "window_start":
            ex["_window_start"],

        "is_yes":
            ex["_is_yes"],

        "window_max":
            ex["_window_max"],
    }


results = {
    "experiment":
        "cgm_stage_a2_full_signal_prealignment",

    "model":
        {
            "repo":
                STAGE5_REPO_ID,

            "base_llm":
                "meta-llama/Llama-3.2-1B",

            "initialization":
                "original_public_stage5",

            "tsqa_transplant":
                False,

            "trainable":
                [
                    "original_stage5_t5",
                    "original_stage5_t6",
                    "original_stage5_projector",
                ],

            "frozen":
                [
                    "stage5_patch_and_early_encoder",
                    "stage5_t1_t4",
                    "base_llm",
                    "stage5_ecg_lora",
                ],
        },

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

    "data":
        {
            "train_per_class":
                TRAIN_PER_CLASS,

            "validation_per_class":
                VAL_PER_CLASS,

            "train_total":
                len(train_real),

            "validation_total":
                len(val_real),

            "test_participants_sealed":
                len(manifest["test"]),
        },

    "training":
        {
            "epochs":
                EPOCHS,

            "effective_batch_size":
                1,

            "optimizer_updates_per_epoch":
                len(train_real),

            "total_optimizer_updates":
                len(train_real)
                * EPOCHS,

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
                    in participant_overlap
                ),
        },

    "selected_train_examples":
        [
            example_metadata(x)
            for x in train_real
        ],

    "selected_validation_examples":
        [
            example_metadata(x)
            for x in val_real
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
            str(CHECKPOINT_PATH)
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