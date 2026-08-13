"""
Stage-5 CGM class-token alignment.

Purpose
-------
Stage-5 full adaptation reached low sequence loss but collapsed to
predicting "(a)" for every example.

Token diagnostics showed:

    (a) -> [BOS, "(a", ")", EOS]
    (b) -> [BOS, "(b", ")", EOS]

Only one supervised answer token distinguishes the classes.

This experiment changes ONLY the loss objective:

    OLD:
        autoregressive loss over the full answer sequence

    NEW:
        full-vocabulary cross-entropy ONLY at the
        first class-discriminating answer token

Everything else stays the same:
    - original Stage-5 ECG checkpoint
    - seed 42
    - same 23 train participants
    - same 5 validation participants
    - same 6 sealed test participants
    - same normalization
    - same six CGM tasks
    - same 1200 balanced examples
    - same training order
    - same learning rates
    - encoder trainable
    - projector trainable
    - LoRA trainable
    - base Llama frozen

Primary metric:
    direct class-token balanced accuracy

Generation is printed only as a secondary diagnostic because
we are not explicitly training ")" / EOS formatting here.

TEST participants remain sealed.
"""

import json
import random
import re
import sys

from pathlib import Path

import torch
import torch.nn.functional as F

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


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
    balance_multitask_examples,
    sample_across_participants,
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

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_alignment_stage5_class_token_seed42.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_stage5_class_token_seed42_results.json"
)


ACTIVE_TASKS = [
    "above_160",
    "above_180",
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
    "tir_70_180_above_90",
]


WINDOW_SIZE = 128
STRIDE = 128

TRAIN_EXAMPLES_PER_CLASS = 100


ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4
LORA_LR = 1e-4

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0

PRINT_EVERY = 50


# For train-fit diagnostic:
TRAIN_DIAGNOSTIC_PER_CLASS = 10


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
# Directories
# ============================================================

CHECKPOINT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

RESULTS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
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


def manifest_files(
    split_name,
):

    return [
        DATA_DIR
        / item[
            "relative_path"
        ]

        for item
        in manifest[
            split_name
        ]
    ]


train_files = (
    manifest_files(
        "train"
    )
)

val_files = (
    manifest_files(
        "validation"
    )
)


print(
    "Train participants:",
    len(train_files),
)

print(
    "Validation participants:",
    len(val_files),
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
# Load ORIGINAL Stage-5 checkpoint
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "LOADING ORIGINAL STAGE-5 ECG CHECKPOINT"
)

print(
    "=" * 78
)


model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


EOS_TOKEN = (
    model.get_eos_token()
)


# ============================================================
# Configure trainable parameters
# ============================================================

# Freeze all LLM parameters first.
for parameter in (
    model.llm.parameters()
):

    parameter.requires_grad = False


# Re-enable ONLY LoRA.
for name, parameter in (
    model.llm.named_parameters()
):

    if "lora_" in name:

        parameter.requires_grad = True


# Encoder + projector train.
for parameter in (
    model.encoder.parameters()
):

    parameter.requires_grad = True


for parameter in (
    model.projector.parameters()
):

    parameter.requires_grad = True


encoder_parameters = list(
    model.encoder.parameters()
)

projector_parameters = list(
    model.projector.parameters()
)

lora_parameters = (
    model.get_lora_parameters()
)


base_llm_trainable = sum(
    p.numel()

    for name, p
    in model.llm.named_parameters()

    if (
        p.requires_grad
        and "lora_" not in name
    )
)


print(
    "\nTrainable encoder parameters:",
    f"{sum(p.numel() for p in encoder_parameters if p.requires_grad):,}"
)

print(
    "Trainable projector parameters:",
    f"{sum(p.numel() for p in projector_parameters if p.requires_grad):,}"
)

print(
    "Trainable LoRA parameters:",
    f"{sum(p.numel() for p in lora_parameters if p.requires_grad):,}"
)

print(
    "Trainable non-LoRA LLM parameters:",
    f"{base_llm_trainable:,}"
)


if base_llm_trainable != 0:

    raise RuntimeError(
        "Base Llama unexpectedly trainable."
    )


print(
    "\n✓ Base Llama frozen."
)

print(
    "✓ LoRA trainable."
)

print(
    "✓ Encoder trainable."
)

print(
    "✓ Projector trainable."
)


# ============================================================
# Build exact training set
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "BUILDING EXACT SEED-42 TRAINING SET"
)

print(
    "=" * 78
)


raw_train_examples = (
    build_multitask_examples(
        files=train_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=ACTIVE_TASKS,
    )
)


train_examples = (
    balance_multitask_examples(
        examples=raw_train_examples,
        task_ids=ACTIVE_TASKS,
        per_class=TRAIN_EXAMPLES_PER_CLASS,
        seed=SEED,
    )
)


if len(
    train_examples
) != 1200:

    raise RuntimeError(
        f"Expected 1200 examples, "
        f"got {len(train_examples)}"
    )


print(
    "\nFixed training examples:",
    len(train_examples),
)


# ============================================================
# Determine answer-token structure
# ============================================================

answer_a_values = {
    x[
        "answer"
    ]

    for x in train_examples

    if x[
        "_answer_clean"
    ] == "(a)"
}


answer_b_values = {
    x[
        "answer"
    ]

    for x in train_examples

    if x[
        "_answer_clean"
    ] == "(b)"
}


if len(answer_a_values) != 1:

    raise RuntimeError(
        f"Unexpected A answers: "
        f"{answer_a_values}"
    )


if len(answer_b_values) != 1:

    raise RuntimeError(
        f"Unexpected B answers: "
        f"{answer_b_values}"
    )


ANSWER_A = next(
    iter(
        answer_a_values
    )
)

ANSWER_B = next(
    iter(
        answer_b_values
    )
)


def tokenize_answer(
    answer,
):

    return (
        model.tokenizer(
            answer,
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


answer_a_ids = (
    tokenize_answer(
        ANSWER_A
    )
)

answer_b_ids = (
    tokenize_answer(
        ANSWER_B
    )
)


# ============================================================
# Find first token where A and B differ
# ============================================================

common_prefix_len = 0


while (
    common_prefix_len
    < min(
        len(answer_a_ids),
        len(answer_b_ids),
    )
    and
    answer_a_ids[
        common_prefix_len
    ].item()
    ==
    answer_b_ids[
        common_prefix_len
    ].item()
):

    common_prefix_len += 1


if common_prefix_len >= min(
    len(answer_a_ids),
    len(answer_b_ids),
):

    raise RuntimeError(
        "A and B answers have no "
        "class-discriminating token."
    )


common_prefix_ids = (
    answer_a_ids[
        :common_prefix_len
    ]
)


A_TOKEN_ID = int(
    answer_a_ids[
        common_prefix_len
    ].item()
)

B_TOKEN_ID = int(
    answer_b_ids[
        common_prefix_len
    ].item()
)


A_TOKEN = (
    model.tokenizer.convert_ids_to_tokens(
        [
            A_TOKEN_ID
        ]
    )[0]
)

B_TOKEN = (
    model.tokenizer.convert_ids_to_tokens(
        [
            B_TOKEN_ID
        ]
    )[0]
)


print(
    "\n"
    + "=" * 78
)

print(
    "CLASS TOKEN SETUP"
)

print(
    "=" * 78
)


print(
    "Answer A:",
    repr(
        ANSWER_A
    ),
)

print(
    "Answer B:",
    repr(
        ANSWER_B
    ),
)

print(
    "\nA token IDs:",
    answer_a_ids.tolist(),
)

print(
    "B token IDs:",
    answer_b_ids.tolist(),
)

print(
    "\nCommon prefix IDs:",
    common_prefix_ids.tolist(),
)

print(
    "Common prefix tokens:",
    model.tokenizer.convert_ids_to_tokens(
        common_prefix_ids.tolist()
    ),
)

print(
    "\nA decision token:",
    repr(
        A_TOKEN
    ),
    A_TOKEN_ID,
)

print(
    "B decision token:",
    repr(
        B_TOKEN
    ),
    B_TOKEN_ID,
)


# ============================================================
# Build validation set
# ============================================================

raw_val_examples = (
    build_multitask_examples(
        files=val_files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=ACTIVE_TASKS,
    )
)


def build_balanced_validation(
    examples,
    task_ids,
    seed,
):

    output = {}


    for task_index, task_id in enumerate(
        task_ids
    ):

        task_examples = [
            x
            for x in examples
            if x[
                "_task_id"
            ] == task_id
        ]


        yes_examples = [
            x
            for x in task_examples
            if x[
                "_answer_clean"
            ] == "(a)"
        ]


        no_examples = [
            x
            for x in task_examples
            if x[
                "_answer_clean"
            ] == "(b)"
        ]


        n_per_class = min(
            len(yes_examples),
            len(no_examples),
        )


        task_seed = (
            seed
            + 10000
            + task_index * 1000
        )


        selected_yes = (
            sample_across_participants(
                yes_examples,
                n=n_per_class,
                seed=task_seed,
            )
        )


        selected_no = (
            sample_across_participants(
                no_examples,
                n=n_per_class,
                seed=task_seed + 1,
            )
        )


        balanced = (
            selected_yes
            + selected_no
        )


        rng = random.Random(
            task_seed + 2
        )

        rng.shuffle(
            balanced
        )


        output[
            task_id
        ] = balanced


        print(
            f"{task_id:28s} | "
            f"YES={len(selected_yes):3d} | "
            f"NO={len(selected_no):3d} | "
            f"total={len(balanced):3d}"
        )


    return output


print(
    "\nBuilding validation set..."
)


val_by_task = (
    build_balanced_validation(
        raw_val_examples,
        ACTIVE_TASKS,
        SEED,
    )
)


# ============================================================
# Class-token logits
# ============================================================

def get_class_decision_logits(
    example,
):
    """
    Return full-vocabulary logits at the point where
    the answer first differs between "(a)" and "(b)".

    We reproduce the shared answer prefix from the
    original OpenTSLM answer sequence before asking
    the model to predict the class token.
    """

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


    # Batch size = 1.
    # Remove any possible trailing padding.
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


    # --------------------------------------------------------
    # Add the common answer prefix.
    #
    # In our observed tokenizer:
    #
    #   common prefix = BOS
    #
    # Then the next token is either "(a" or "(b".
    # --------------------------------------------------------

    if common_prefix_ids.numel() > 0:

        prefix_embeds = (
            model.llm
            .get_input_embeddings()(
                common_prefix_ids.unsqueeze(
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
                    common_prefix_ids.numel(),
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


    # Last position predicts the NEXT token:
    # our class-discriminating token.
    decision_logits = (
        outputs.logits[
            0,
            -1,
            :
        ]
        .float()
    )


    return decision_logits


# ============================================================
# Custom class-token loss
# ============================================================

def compute_class_token_loss(
    example,
):

    decision_logits = (
        get_class_decision_logits(
            example
        )
    )


    gold = (
        example[
            "_answer_clean"
        ]
    )


    if gold == "(a)":

        target_token_id = (
            A_TOKEN_ID
        )

    elif gold == "(b)":

        target_token_id = (
            B_TOKEN_ID
        )

    else:

        raise RuntimeError(
            f"Unexpected gold label: "
            f"{gold}"
        )


    target = torch.tensor(
        [
            target_token_id
        ],
        dtype=torch.long,
        device=
            decision_logits.device,
    )


    # --------------------------------------------------------
    # Full-vocabulary cross-entropy.
    #
    # This is still an autoregressive language-model objective.
    # We are simply masking supervision so ONLY the class-
    # discriminating answer token contributes to the loss.
    # --------------------------------------------------------

    loss = (
        F.cross_entropy(
            decision_logits.unsqueeze(
                0
            ),
            target,
        )
    )


    return loss


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    [
        {
            "params":
                encoder_parameters,

            "lr":
                ENCODER_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },

        {
            "params":
                projector_parameters,

            "lr":
                PROJECTOR_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },

        {
            "params":
                lora_parameters,

            "lr":
                LORA_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },
    ]
)


trainable_parameters = (
    encoder_parameters
    + projector_parameters
    + lora_parameters
)


print(
    "\nLearning rates:"
)

print(
    f"Encoder:   {ENCODER_LR:.2e}"
)

print(
    f"Projector: {PROJECTOR_LR:.2e}"
)

print(
    f"LoRA:      {LORA_LR:.2e}"
)


# ============================================================
# Initial class-token smoke test
# ============================================================

def get_pair_probability(
    example,
):

    logits = (
        get_class_decision_logits(
            example
        )
    )


    pair_logits = (
        torch.stack(
            [
                logits[
                    A_TOKEN_ID
                ],

                logits[
                    B_TOKEN_ID
                ],
            ]
        )
    )


    probs = (
        torch.softmax(
            pair_logits,
            dim=0,
        )
    )


    return (
        float(
            probs[
                0
            ].detach().cpu()
        ),

        float(
            probs[
                1
            ].detach().cpu()
        ),
    )


print(
    "\n"
    + "=" * 78
)

print(
    "PRE-TRAINING CLASS-TOKEN SMOKE TEST"
)

print(
    "=" * 78
)


smoke_example = (
    train_examples[
        0
    ]
)


model.eval()


with torch.no_grad():

    smoke_pa, smoke_pb = (
        get_pair_probability(
            smoke_example
        )
    )


print(
    "Task:",
    smoke_example[
        "_task_id"
    ],
)

print(
    "Gold:",
    smoke_example[
        "_answer_clean"
    ],
)

print(
    f"P((a) | a,b): "
    f"{smoke_pa:.4f}"
)

print(
    f"P((b) | a,b): "
    f"{smoke_pb:.4f}"
)


# ============================================================
# Training
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STAGE-5 CLASS-TOKEN CGM ALIGNMENT — SEED 42"
)

print(
    "=" * 78
)


model.encoder.train()
model.projector.train()
model.llm.train()


losses = []


for step, example in enumerate(
    train_examples,
    start=1,
):

    optimizer.zero_grad(
        set_to_none=True
    )


    loss = (
        compute_class_token_loss(
            example
        )
    )


    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            f"Non-finite loss at "
            f"step {step}: "
            f"{loss.item()}"
        )


    loss.backward()


    clip_grad_norm_(
        trainable_parameters,
        GRAD_CLIP_NORM,
    )


    optimizer.step()


    loss_value = float(
        loss.detach().cpu()
    )


    losses.append(
        loss_value
    )


    if (
        step == 1
        or step % PRINT_EVERY == 0
        or step == len(train_examples)
    ):

        recent = (
            losses[
                -PRINT_EVERY:
            ]
        )


        recent_mean = (
            sum(
                recent
            )
            / len(
                recent
            )
        )


        print(
            f"Step "
            f"{step:4d}/"
            f"{len(train_examples)} | "
            f"Loss "
            f"{loss_value:.4f} | "
            f"Recent mean "
            f"{recent_mean:.4f}"
        )


mean_loss = (
    sum(losses)
    / len(losses)
)


print(
    "\nMean class-token training loss:",
    f"{mean_loss:.4f}"
)


# ============================================================
# Direct class-token evaluation
# ============================================================

def evaluate_class_tokens(
    examples,
):

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    pred_yes = 0
    pred_no = 0

    p_correct_values = []


    model.eval()


    with torch.no_grad():

        for example in examples:

            logits = (
                get_class_decision_logits(
                    example
                )
            )


            pair_logits = (
                torch.stack(
                    [
                        logits[
                            A_TOKEN_ID
                        ],

                        logits[
                            B_TOKEN_ID
                        ],
                    ]
                )
            )


            probs = (
                torch.softmax(
                    pair_logits,
                    dim=0,
                )
            )


            p_a = float(
                probs[
                    0
                ].cpu()
            )

            p_b = float(
                probs[
                    1
                ].cpu()
            )


            pred = (
                "(a)"
                if p_a > p_b
                else "(b)"
            )


            gold = (
                example[
                    "_answer_clean"
                ]
            )


            if pred == "(a)":

                pred_yes += 1

            else:

                pred_no += 1


            if gold == "(a)":

                yes_total += 1

                p_correct_values.append(
                    p_a
                )

                if pred == gold:

                    yes_correct += 1

            else:

                no_total += 1

                p_correct_values.append(
                    p_b
                )

                if pred == gold:

                    no_correct += 1


    yes_accuracy = (
        yes_correct
        / yes_total
        if yes_total
        else 0.0
    )


    no_accuracy = (
        no_correct
        / no_total
        if no_total
        else 0.0
    )


    balanced_accuracy = (
        yes_accuracy
        + no_accuracy
    ) / 2.0


    mean_p_correct = (
        sum(
            p_correct_values
        )
        / len(
            p_correct_values
        )
    )


    return {
        "n":
            len(examples),

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "predicted_yes":
            pred_yes,

        "predicted_no":
            pred_no,

        "mean_probability_correct":
            mean_p_correct,
    }


# ============================================================
# Balanced training diagnostic subset
# ============================================================

train_diag_by_task = {}


for task_index, task_id in enumerate(
    ACTIVE_TASKS
):

    task_examples = [
        x
        for x in train_examples
        if x[
            "_task_id"
        ] == task_id
    ]


    yes_examples = [
        x
        for x in task_examples
        if x[
            "_answer_clean"
        ] == "(a)"
    ]


    no_examples = [
        x
        for x in task_examples
        if x[
            "_answer_clean"
        ] == "(b)"
    ]


    rng = random.Random(
        SEED
        + 50000
        + task_index
    )


    selected = (
        rng.sample(
            yes_examples,
            TRAIN_DIAGNOSTIC_PER_CLASS,
        )
        +
        rng.sample(
            no_examples,
            TRAIN_DIAGNOSTIC_PER_CLASS,
        )
    )


    train_diag_by_task[
        task_id
    ] = selected


# ============================================================
# Print evaluation
# ============================================================

def print_eval(
    split_name,
    examples_by_task,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        split_name
    )

    print(
        "=" * 78
    )


    results = {}


    for task_id in ACTIVE_TASKS:

        result = (
            evaluate_class_tokens(
                examples_by_task[
                    task_id
                ]
            )
        )


        results[
            task_id
        ] = result


        print(
            f"\n{task_id}"
        )

        print(
            f"  n:                 "
            f"{result['n']}"
        )

        print(
            f"  YES accuracy:       "
            f"{result['yes_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  NO accuracy:        "
            f"{result['no_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  balanced accuracy:  "
            f"{result['balanced_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  predicted YES:      "
            f"{result['predicted_yes']}"
        )

        print(
            f"  predicted NO:       "
            f"{result['predicted_no']}"
        )

        print(
            f"  mean P(correct):    "
            f"{result['mean_probability_correct']:.4f}"
        )


    macro = (
        sum(
            x[
                "balanced_accuracy"
            ]

            for x in results.values()
        )
        / len(results)
    )


    print(
        "\nMACRO BALANCED ACCURACY:",
        f"{macro * 100:.1f}%"
    )


    return {
        "tasks":
            results,

        "macro_balanced_accuracy":
            macro,
    }


train_diag_results = (
    print_eval(
        "TRAIN CLASS-TOKEN DIAGNOSTIC",
        train_diag_by_task,
    )
)


validation_results = (
    print_eval(
        "VALIDATION CLASS-TOKEN PERFORMANCE",
        val_by_task,
    )
)


# ============================================================
# Secondary generation diagnostic
# ============================================================

def parse_generation(
    text,
):

    text = (
        str(text)
        .strip()
        .lower()
    )


    match = re.search(
        r"\([ab]\)",
        text,
    )


    if match:

        return match.group(0)


    return None


print(
    "\n"
    + "=" * 78
)

print(
    "SECONDARY RAW GENERATION CHECK"
)

print(
    "=" * 78
)


model.eval()


for task_id in ACTIVE_TASKS:

    examples = (
        val_by_task[
            task_id
        ]
    )


    yes_example = next(
        x
        for x in examples
        if x[
            "_answer_clean"
        ] == "(a)"
    )


    no_example = next(
        x
        for x in examples
        if x[
            "_answer_clean"
        ] == "(b)"
    )


    with torch.no_grad():

        outputs = (
            model.generate(
                [
                    yes_example,
                    no_example,
                ],
                max_new_tokens=6,
                do_sample=False,
            )
        )


    print(
        f"\n{task_id}"
    )

    print(
        "  Gold YES raw:",
        repr(
            outputs[
                0
            ]
        ),
    )

    print(
        "  Gold NO raw: ",
        repr(
            outputs[
                1
            ]
        ),
    )


# ============================================================
# Comparison
# ============================================================

TSQA_SEED42_MACRO = 0.739

STAGE5_FULL_SEQUENCE_MACRO = 0.500

class_token_macro = (
    validation_results[
        "macro_balanced_accuracy"
    ]
)


print(
    "\n"
    + "=" * 78
)

print(
    "SEED-42 COMPARISON"
)

print(
    "=" * 78
)


print(
    f"TSQA mixed alignment:             "
    f"{TSQA_SEED42_MACRO * 100:.1f}%"
)

print(
    f"Stage-5 full sequence adaptation: "
    f"{STAGE5_FULL_SEQUENCE_MACRO * 100:.1f}%"
)

print(
    f"Stage-5 CLASS-TOKEN alignment:    "
    f"{class_token_macro * 100:.1f}%"
)


# ============================================================
# Save checkpoint
# ============================================================

model.store_to_file(
    str(
        CHECKPOINT_PATH
    )
)


# ============================================================
# Save results
# ============================================================

results = {

    "experiment":
        "stage5_class_token_seed42",

    "seed":
        SEED,

    "repo_id":
        STAGE5_REPO_ID,

    "training_examples":
        len(
            train_examples
        ),

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "answer_tokenization":
        {
            "answer_a":
                ANSWER_A,

            "answer_b":
                ANSWER_B,

            "answer_a_ids":
                answer_a_ids
                .cpu()
                .tolist(),

            "answer_b_ids":
                answer_b_ids
                .cpu()
                .tolist(),

            "common_prefix_ids":
                common_prefix_ids
                .cpu()
                .tolist(),

            "a_token_id":
                A_TOKEN_ID,

            "a_token":
                A_TOKEN,

            "b_token_id":
                B_TOKEN_ID,

            "b_token":
                B_TOKEN,
        },

    "learning_rates":
        {
            "encoder":
                ENCODER_LR,

            "projector":
                PROJECTOR_LR,

            "lora":
                LORA_LR,
        },

    "mean_training_loss":
        mean_loss,

    "loss_history":
        losses,

    "train_diagnostic":
        train_diag_results,

    "validation":
        validation_results,

    "reference_macros":
        {
            "tsqa_seed42":
                TSQA_SEED42_MACRO,

            "stage5_full_sequence_seed42":
                STAGE5_FULL_SEQUENCE_MACRO,
        },
}


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
    "\nSaved checkpoint:"
)

print(
    CHECKPOINT_PATH
)


print(
    "\nSaved results:"
)

print(
    RESULTS_PATH
)


print(
    "\nTEST participants remain sealed."
)