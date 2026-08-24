"""
STAGE A.3 — TSQA LATE-SENSOR REPAIR COMPARISON

Scientific question
-------------------
Does replacing the signal-collapsed late sensor stack of the
PUBLIC Stage-5 ECG checkpoint with the corresponding earlier
TSQA sensor modules improve transferable CGM learning?

IMPORTANT:
    We do NOT transplant the TSQA LLM.
    We do NOT transplant TSQA LoRA.

The language model in every condition remains:

    Stage-5 ECG LLM + Stage-5 ECG LoRA

and remains FROZEN.


REFERENCE RESULT
----------------
Original public Stage-5:

    S5 T1-T4
    S5 T5-T6
    S5 projector

Full Stage-A.2:
    best held-out validation BA = 69.4%
    zero signal BA              = 50.0%

That reference has already been run and is NOT repeated here.


ARM 1
-----
tsqa_t56_s5proj

    Stage-5 patch
    Stage-5 T1-T4
    TSQA T5-T6
    Stage-5 projector
    Stage-5 frozen LLM/LoRA


ARM 2
-----
tsqa_t56_tsqaproj

    Stage-5 patch
    Stage-5 T1-T4
    TSQA T5-T6
    TSQA projector
    Stage-5 frozen LLM/LoRA


TRAINING
--------
Same Stage-A protocol:

    231 YES
    231 NO
    = 462 TRAIN windows

    49 YES
    49 NO
    = 98 VALIDATION windows

    fixed global normalization

    identical text across examples

    pairwise (a) vs (b) objective only

    batch size = 1

    T5/T6 + projector trainable

    all LLM/LoRA weights frozen

    20 epochs


ABLATIONS
---------
REAL
ZERO
OPPOSITE-LABEL SHUFFLED


USAGE
-----

Run only TSQA-projector arm:

    python .\\experiments\\stage_a3_tsqa_repair_comparison.py --arm tsqaproj

Run Stage-5-projector arm:

    python .\\experiments\\stage_a3_tsqa_repair_comparison.py --arm s5proj

Run both sequentially:

    python .\\experiments\\stage_a3_tsqa_repair_comparison.py --arm both


TEST PARTICIPANTS REMAIN SEALED.
"""

import argparse
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

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_stage_a3_tsqa_repair_comparison.json"
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

STAGE5_REPO = (
    "OpenTSLM/llama-3.2-1b-ecg-sp"
)

TSQA_REPO = (
    "OpenTSLM/llama-3.2-1b-tsqa-sp"
)


TASK_ID = "above_180"

WINDOW_SIZE = 128

STRIDE = 128


TRAIN_PER_CLASS = 231

VAL_PER_CLASS = 49


EPOCHS = 20


LR_T5_T6 = 2e-4

LR_PROJECTOR = 1e-4


WEIGHT_DECAY = 1e-2


CLIP_T5_T6 = 1.0

CLIP_PROJECTOR = 1.0


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


# Known reference results.
ORIGINAL_STAGE5_EPOCH20_BA = 0.673

ORIGINAL_STAGE5_BEST_EXPLORATORY_BA = 0.694


# ============================================================
# CLI
# ============================================================

parser = argparse.ArgumentParser()


parser.add_argument(
    "--arm",
    choices=[
        "s5proj",
        "tsqaproj",
        "both",
    ],
    default="both",
)


args = parser.parse_args()


if args.arm == "s5proj":

    ARMS = [
        "tsqa_t56_s5proj"
    ]


elif args.arm == "tsqaproj":

    ARMS = [
        "tsqa_t56_tsqaproj"
    ]


else:

    ARMS = [
        "tsqa_t56_s5proj",
        "tsqa_t56_tsqaproj",
    ]


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


if validation_manifest is None:

    validation_manifest = (
        manifest.get(
            "val"
        )
    )


if validation_manifest is None:

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
# Load TSQA DONOR and extract ONLY sensor modules
# ============================================================

print(
    "\n"
    + "=" * 110
)

print(
    "LOADING TSQA DONOR SENSOR MODULES"
)

print(
    "=" * 110
)


tsqa_donor = (
    OpenTSLM.load_pretrained(
        TSQA_REPO,
        device=device,
        enable_lora=True,
    )
)


tsqa_layers = (
    tsqa_donor
    .encoder
    .encoder
    .layers
)


if len(tsqa_layers) != 6:

    raise RuntimeError(
        "Expected six TSQA encoder layers."
    )


def cpu_state_dict(
    module,
):

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in module.state_dict().items()
    }


TSQA_T5_STATE = (
    cpu_state_dict(
        tsqa_layers[
            4
        ]
    )
)


TSQA_T6_STATE = (
    cpu_state_dict(
        tsqa_layers[
            5
        ]
    )
)


TSQA_PROJECTOR_STATE = (
    cpu_state_dict(
        tsqa_donor.projector
    )
)


EOS_TOKEN = (
    tsqa_donor.get_eos_token()
)


print(
    "✓ Captured TSQA T5."
)

print(
    "✓ Captured TSQA T6."
)

print(
    "✓ Captured TSQA projector."
)

print(
    "✓ TSQA LLM/LoRA will NOT be used."
)


del tsqa_donor

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Build dataset
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
# Split classes
# ============================================================

def split_by_class(
    examples,
):

    yes = [
        ex

        for ex
        in examples

        if (
            ex[
                "_answer_clean"
            ]
            == "(a)"
        )
    ]


    no = [
        ex

        for ex
        in examples

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
) = split_by_class(
    train_pool
)


(
    val_yes_pool,
    val_no_pool,
) = split_by_class(
    validation_pool
)


print(
    "\nTRAIN available:"
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
    "\nVALIDATION available:"
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
# Same participant-aware sampler as Stage A.2
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


    while len(selected) < n:

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


                if len(selected) == n:

                    break


        if not progress:

            raise RuntimeError(
                "Unable to obtain requested sample."
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


participant_overlap = (
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
        participant_overlap
    ),
)


if participant_overlap:

    raise RuntimeError(
        "TRAIN/VALIDATION participant leakage."
    )


# ============================================================
# Identical text interface
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
# Text invariance
# ============================================================

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
    in (
        train_yes
        + train_no
        + val_yes
        + val_no
    )
}


print(
    "\nUnique prompt/text combinations:",
    len(
        unique_text
    ),
)


if len(unique_text) != 1:

    raise RuntimeError(
        "Per-example textual variation detected."
    )


print(
    "✓ Text identical across every example."
)


# ============================================================
# Interleave
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


def make_zero(
    examples,
):

    return [
        clone_with_signal(
            ex,
            torch.zeros_like(
                ex[
                    "time_series"
                ][
                    0
                ]
            ),
        )

        for ex
        in examples
    ]


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
# Helpers
# ============================================================

def balanced_accuracy(
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


    return 0.5 * (
        yes_correct
        / yes_total
        +
        no_correct
        / no_total
    )


def grad_norm(
    parameters,
):

    total = 0.0


    for parameter in parameters:

        if parameter.grad is None:

            continue


        value = float(
            parameter.grad
            .detach()
            .float()
            .norm()
            .item()
        )


        total += (
            value
            * value
        )


    return math.sqrt(
        total
    )


def module_cpu_state(
    module,
):

    return {
        key:
            value.detach()
            .cpu()
            .clone()

        for key, value
        in module.state_dict().items()
    }


# ============================================================
# Run an arm
# ============================================================

def run_arm(
    arm_name,
):

    print(
        "\n\n"
        + "#" * 130
    )

    print(
        "STARTING:",
        arm_name
    )

    print(
        "#" * 130
    )


    random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )


    # --------------------------------------------------------
    # Fresh Stage-5 model.
    #
    # IMPORTANT:
    # LLM and LoRA are Stage-5, NOT TSQA.
    # --------------------------------------------------------

    model = (
        OpenTSLM.load_pretrained(
            STAGE5_REPO,
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


    if len(layers) != 6:

        raise RuntimeError(
            "Expected six Stage-5 encoder layers."
        )


    # --------------------------------------------------------
    # Transplant TSQA T5/T6.
    # --------------------------------------------------------

    layers[
        4
    ].load_state_dict(
        TSQA_T5_STATE,
        strict=True,
    )


    layers[
        5
    ].load_state_dict(
        TSQA_T6_STATE,
        strict=True,
    )


    if (
        arm_name
        == "tsqa_t56_tsqaproj"
    ):

        model.projector.load_state_dict(
            TSQA_PROJECTOR_STATE,
            strict=True,
        )


        projector_source = (
            "TSQA"
        )


    elif (
        arm_name
        == "tsqa_t56_s5proj"
    ):

        projector_source = (
            "Stage5"
        )


    else:

        raise ValueError(
            arm_name
        )


    print(
        "\nInitialization:"
    )

    print(
        " T1-T4 source: Stage5"
    )

    print(
        " T5-T6 source: TSQA"
    )

    print(
        " projector source:",
        projector_source,
    )

    print(
        " LLM source: Stage5"
    )

    print(
        " LoRA source: Stage5"
    )


    # --------------------------------------------------------
    # Pairwise tokens
    # --------------------------------------------------------

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
        " (a):",
        a_ids,
        "->",
        A_TOKEN,
    )

    print(
        " (b):",
        b_ids,
        "->",
        B_TOKEN,
    )


    # --------------------------------------------------------
    # Freeze everything.
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Train only transplanted T5/T6 + chosen projector.
    # --------------------------------------------------------

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


    early_trainable = sum(
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


    print(
        "\nTRAINABLE PARAMETER AUDIT"
    )

    print(
        " Early encoder:",
        f"{early_trainable:,}"
    )

    print(
        " T5/T6:",
        f"{sum(p.numel() for p in late_parameters):,}"
    )

    print(
        " Projector:",
        f"{sum(p.numel() for p in projector_parameters):,}"
    )

    print(
        " Stage5 LLM / LoRA:",
        f"{llm_trainable:,}"
    )


    if early_trainable != 0:

        raise RuntimeError(
            "Early encoder unexpectedly trainable."
        )


    if llm_trainable != 0:

        raise RuntimeError(
            "LLM/LoRA unexpectedly trainable."
        )


    model.encoder.eval()

    model.projector.eval()

    model.llm.eval()


    optimizer = (
        torch.optim.AdamW(
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
    )


    # ========================================================
    # Pairwise loss
    # ========================================================

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
                        A_TOKEN
                    ],

                    logits[
                        B_TOKEN
                    ],
                ]
            )
            .float()
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
            pair_logits.unsqueeze(
                0
            ),
            target,
        )


    # ========================================================
    # Evaluation
    # ========================================================

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


        original_ba = (
            balanced_accuracy(
                predictions,
                original_labels,
            )
        )


        donor_ba = None


        if all(
            donor is not None

            for donor
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
                        predictions
                    )
                ),

            "predicted_no":
                int(
                    len(
                        predictions
                    )
                    - sum(
                        predictions
                    )
                ),
        }


    # ========================================================
    # Representation diagnostic
    # ========================================================

    def representation_stats(
        examples,
    ):

        encoder_vectors = []

        projector_vectors = []

        labels = []


        with torch.no_grad():

            for start in range(
                0,
                len(
                    examples
                ),
                REP_BATCH_SIZE,
            ):

                batch = (
                    examples[
                        start:
                        start
                        + REP_BATCH_SIZE
                    ]
                )


                signals = (
                    torch.stack(
                        [
                            ex[
                                "time_series"
                            ][
                                0
                            ]
                            .detach()
                            .clone()
                            .float()

                            for ex
                            in batch
                        ],
                        dim=0,
                    )
                    .to(
                        device
                    )
                )


                encoded = (
                    model.encoder(
                        signals
                    )
                )


                projected = (
                    model.projector(
                        encoded
                    )
                )


                for index, example in enumerate(
                    batch
                ):

                    encoder_vectors.append(
                        encoded[
                            index
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
                            index
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

            matrix = (
                torch.stack(
                    vectors
                )
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


            return {
                "feature_sd":
                    feature_sd,

                "yes_no_cosine":
                    cosine,
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


    # ========================================================
    # Evaluation bundle
    # ========================================================

    def evaluate_all():

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


    # ========================================================
    # Snapshot
    # ========================================================

    def snapshot():

        return {
            "t5":
                module_cpu_state(
                    layers[
                        4
                    ]
                ),

            "t6":
                module_cpu_state(
                    layers[
                        5
                    ]
                ),

            "projector":
                module_cpu_state(
                    model.projector
                ),
        }


    # ========================================================
    # Initial diagnostic
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        arm_name,
        "— BEFORE TRAINING"
    )

    print(
        "=" * 120
    )


    initial = (
        evaluate_all()
    )


    initial_rep = (
        representation_stats(
            val_real
        )
    )


    print(
        "TRAIN REAL:",
        f"{100 * initial['train_real']['ba_vs_original']:.1f}%"
    )


    print(
        "VAL REAL:",
        f"{100 * initial['val_real']['ba_vs_original']:.1f}%"
    )


    print(
        "VAL ZERO:",
        f"{100 * initial['val_zero']['ba_vs_original']:.1f}%"
    )


    print(
        "VAL encoder SD:",
        f"{initial_rep['encoder']['feature_sd']:.8f}"
    )


    print(
        "VAL projector SD:",
        f"{initial_rep['projector']['feature_sd']:.8f}"
    )


    best_epoch = 0

    best_val_ba = (
        initial[
            "val_real"
        ][
            "ba_vs_original"
        ]
    )


    best_state = (
        snapshot()
    )


    best_optimizer_state = (
        optimizer.state_dict()
    )


    history = [
        {
            "epoch":
                0,

            "mean_loss":
                None,

            "metrics":
                initial,

            "representation":
                initial_rep,
        }
    ]


    first_gradient = None


    # ========================================================
    # Train
    # ========================================================

    print(
        "\nTRAINING..."
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


            if first_gradient is None:

                first_gradient = {
                    "t5_t6":
                        grad_norm(
                            late_parameters
                        ),

                    "projector":
                        grad_norm(
                            projector_parameters
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
                    "\nFirst gradients:"
                )

                print(
                    " T5/T6:",
                    f"{first_gradient['t5_t6']:.8e}"
                )

                print(
                    " Projector:",
                    f"{first_gradient['projector']:.8e}"
                )


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


        print(
            f"Finished epoch {epoch:2d} "
            f"| mean loss={mean_loss:.4f}"
        )


        if epoch in EVAL_EPOCHS:

            metrics = (
                evaluate_all()
            )


            representation = (
                representation_stats(
                    val_real
                )
            )


            train_ba = (
                metrics[
                    "train_real"
                ][
                    "ba_vs_original"
                ]
            )


            val_ba = (
                metrics[
                    "val_real"
                ][
                    "ba_vs_original"
                ]
            )


            zero_ba = (
                metrics[
                    "val_zero"
                ][
                    "ba_vs_original"
                ]
            )


            shuf_original = (
                metrics[
                    "val_shuffle"
                ][
                    "ba_vs_original"
                ]
            )


            shuf_donor = (
                metrics[
                    "val_shuffle"
                ][
                    "ba_vs_donor"
                ]
            )


            print(
                f"\nEpoch {epoch:2d} DIAGNOSTIC | "
                f"TRAIN={100 * train_ba:5.1f}% | "
                f"VAL={100 * val_ba:5.1f}% | "
                f"ZERO={100 * zero_ba:5.1f}% | "
                f"SHUF-O={100 * shuf_original:5.1f}% | "
                f"SHUF-D={100 * shuf_donor:5.1f}% | "
                f"EncSD="
                f"{representation['encoder']['feature_sd']:.5f}"
            )


            history.append(
                {
                    "epoch":
                        epoch,

                    "mean_loss":
                        mean_loss,

                    "metrics":
                        metrics,

                    "representation":
                        representation,
                }
            )


            if (
                val_ba
                > best_val_ba
                + 1e-12
            ):

                best_epoch = (
                    epoch
                )


                best_val_ba = (
                    val_ba
                )


                best_state = (
                    snapshot()
                )


                best_optimizer_state = (
                    {
                        key:
                            value

                        for key, value
                        in optimizer.state_dict().items()
                    }
                )


                print(
                    "✓ NEW BEST:",
                    f"{100 * best_val_ba:.1f}%"
                )


    # ========================================================
    # Restore best
    # ========================================================

    layers[
        4
    ].load_state_dict(
        best_state[
            "t5"
        ]
    )


    layers[
        5
    ].load_state_dict(
        best_state[
            "t6"
        ]
    )


    model.projector.load_state_dict(
        best_state[
            "projector"
        ]
    )


    best_metrics = (
        evaluate_all()
    )


    best_rep = (
        representation_stats(
            val_real
        )
    )


    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n"
        + "=" * 120
    )

    print(
        arm_name,
        "— BEST RESULT"
    )

    print(
        "=" * 120
    )


    print(
        "Best epoch:",
        best_epoch
    )


    print(
        "TRAIN REAL:",
        f"{100 * best_metrics['train_real']['ba_vs_original']:.1f}%"
    )


    print(
        "VAL REAL:",
        f"{100 * best_metrics['val_real']['ba_vs_original']:.1f}%"
    )


    print(
        "VAL ZERO:",
        f"{100 * best_metrics['val_zero']['ba_vs_original']:.1f}%"
    )


    print(
        "VAL SHUFFLED original:",
        f"{100 * best_metrics['val_shuffle']['ba_vs_original']:.1f}%"
    )


    print(
        "VAL SHUFFLED donor:",
        f"{100 * best_metrics['val_shuffle']['ba_vs_donor']:.1f}%"
    )


    print(
        "VAL encoder SD:",
        f"{best_rep['encoder']['feature_sd']:.8f}"
    )


    print(
        "VAL encoder Y/N cosine:",
        f"{best_rep['encoder']['yes_no_cosine']:.8f}"
    )


    print(
        "VAL projector SD:",
        f"{best_rep['projector']['feature_sd']:.8f}"
    )


    print(
        "\nREFERENCE:"
    )


    print(
        "Original Stage5 epoch-20 protocol:",
        f"{100 * ORIGINAL_STAGE5_EPOCH20_BA:.1f}%"
    )


    print(
        "Original Stage5 exploratory best:",
        f"{100 * ORIGINAL_STAGE5_BEST_EXPLORATORY_BA:.1f}%"
    )


    print(
        "Delta vs original epoch-20:",
        f"{100 * (
            best_val_ba
            - ORIGINAL_STAGE5_EPOCH20_BA
        ):+.1f} pp"
    )


    print(
        "Delta vs exploratory 69.4%:",
        f"{100 * (
            best_val_ba
            - ORIGINAL_STAGE5_BEST_EXPLORATORY_BA
        ):+.1f} pp"
    )


    # ========================================================
    # Save arm checkpoint
    # ========================================================

    checkpoint_path = (
        ROOT
        / "checkpoints"
        / (
            "cgm_stage_a3_"
            + arm_name
            + "_best.pt"
        )
    )


    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    torch.save(
        {
            "experiment":
                "cgm_stage_a3_tsqa_repair",

            "arm":
                arm_name,

            "base_model":
                STAGE5_REPO,

            "tsqa_donor":
                TSQA_REPO,

            "llm_source":
                "stage5",

            "lora_source":
                "stage5",

            "t1_t4_source":
                "stage5",

            "t5_t6_source":
                "tsqa",

            "projector_source":
                projector_source,

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
                best_state,

            "optimizer_state":
                best_optimizer_state,
        },

        checkpoint_path,
    )


    print(
        "\nSaved checkpoint:"
    )

    print(
        checkpoint_path
    )


    result = {
        "arm":
            arm_name,

        "projector_source":
            projector_source,

        "best_epoch":
            best_epoch,

        "best_validation_real_ba":
            best_val_ba,

        "delta_vs_original_epoch20":
            best_val_ba
            - ORIGINAL_STAGE5_EPOCH20_BA,

        "delta_vs_original_exploratory_best":
            best_val_ba
            - ORIGINAL_STAGE5_BEST_EXPLORATORY_BA,

        "initial":
            initial,

        "initial_representation":
            initial_rep,

        "first_gradient":
            first_gradient,

        "history":
            history,

        "best_metrics":
            best_metrics,

        "best_representation":
            best_rep,

        "checkpoint":
            str(
                checkpoint_path
            ),
    }


    del optimizer
    del model

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# Run selected arm(s)
# ============================================================

all_results = {}


for arm in ARMS:

    all_results[
        arm
    ] = (
        run_arm(
            arm
        )
    )


# ============================================================
# Final comparison
# ============================================================

print(
    "\n\n"
    + "=" * 130
)

print(
    "STAGE A.3 — TSQA REPAIR COMPARISON"
)

print(
    "=" * 130
)


print(
    "Original Stage5 epoch-20:",
    f"{100 * ORIGINAL_STAGE5_EPOCH20_BA:.1f}%"
)


print(
    "Original Stage5 exploratory best:",
    f"{100 * ORIGINAL_STAGE5_BEST_EXPLORATORY_BA:.1f}%"
)


for arm_name, result in (
    all_results.items()
):

    print(
        "\n",
        arm_name,
        sep="",
    )


    print(
        " Best epoch:",
        result[
            "best_epoch"
        ]
    )


    print(
        " Validation REAL:",
        f"{100 * result['best_validation_real_ba']:.1f}%"
    )


    print(
        " Delta vs 67.3:",
        f"{100 * result['delta_vs_original_epoch20']:+.1f} pp"
    )


    print(
        " Delta vs 69.4:",
        f"{100 * result['delta_vs_original_exploratory_best']:+.1f} pp"
    )


# ============================================================
# Save combined diagnostic
# ============================================================

existing_results = {}


if RESULTS_PATH.exists():

    try:

        with RESULTS_PATH.open(
            "r",
            encoding="utf-8",
        ) as f:

            existing_results = (
                json.load(
                    f
                )
            )

    except Exception:

        existing_results = {}


stored_arms = (
    existing_results.get(
        "results",
        {}
    )
)


stored_arms.update(
    all_results
)


output = {
    "experiment":
        "cgm_stage_a3_tsqa_repair_comparison",

    "reference":
        {
            "original_stage5_epoch20_ba":
                ORIGINAL_STAGE5_EPOCH20_BA,

            "original_stage5_exploratory_best_ba":
                ORIGINAL_STAGE5_BEST_EXPLORATORY_BA,
        },

    "architecture":
        {
            "base_model":
                STAGE5_REPO,

            "tsqa_donor":
                TSQA_REPO,

            "tsqa_llm_used":
                False,

            "stage5_llm_frozen":
                True,

            "stage5_lora_frozen":
                True,
        },

    "data":
        {
            "train_total":
                len(
                    train_real
                ),

            "validation_total":
                len(
                    val_real
                ),

            "test_participants_sealed":
                len(
                    manifest[
                        "test"
                    ]
                ),
        },

    "results":
        stored_arms,
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
    "\nSaved combined diagnostic:"
)

print(
    RESULTS_PATH
)


print(
    "\nTEST participants remain sealed."
)