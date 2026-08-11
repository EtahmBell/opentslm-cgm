"""
TIR-first CGM curriculum experiment.

Goal
----
Test whether curriculum ordering prevents the TIR failure observed
during fully shuffled six-task multitask training.

IMPORTANT CONTROL
-----------------
This experiment uses the EXACT SAME balanced set of 1200 examples
as the previous six-task experiment:

    6 tasks
    x 100 YES
    x 100 NO
    = 1200 examples

The only major change is training order:

    Stage 1:
        200 TIR examples

    Stage 2:
        remaining 1000 examples from the other five tasks

Therefore total training examples / optimizer updates remain 1200.

This isolates curriculum ORDER much better than training TIR and then
training all 1200 examples again.

TEST participants remain sealed.
"""

import json
import random
import re
import sys

from pathlib import Path

import torch

from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW


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

REPO_ID = (
    "OpenTSLM/llama-3.2-1b-tsqa-sp"
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
    / "cgm_alignment_tir_first_curriculum_tsqa.pt"
)

INTERMEDIATE_CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_alignment_tir_first_stage1_tsqa.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_tir_first_curriculum_results.json"
)


ACTIVE_TASKS = [
    "above_160",
    "above_180",
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
    "tir_70_180_above_90",
]

TIR_TASK = (
    "tir_70_180_above_90"
)


WINDOW_SIZE = 128
STRIDE = 128

TRAIN_EXAMPLES_PER_CLASS = 100

ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4

WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0

EVAL_BATCH_SIZE = 4
MAX_NEW_TOKENS = 6

PRINT_EVERY = 25


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
# Paths
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
    len(manifest["test"]),
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
# Load original TSQA model
# ============================================================

print(
    "\nLoading original TSQA OpenTSLM..."
)

model = (
    OpenTSLM.load_pretrained(
        REPO_ID,
        device=device,
    )
)

model.eval()


EOS_TOKEN = (
    model.get_eos_token()
)


# ============================================================
# Build EXACT SAME 1200-example training set
# ============================================================

print(
    "\nBuilding multitask training pool..."
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


print(
    "\nTotal training examples:",
    len(train_examples),
)


# ============================================================
# Split SAME examples into curriculum stages
# ============================================================

tir_train_examples = [

    x
    for x in train_examples

    if x["_task_id"]
    == TIR_TASK
]


other_train_examples = [

    x
    for x in train_examples

    if x["_task_id"]
    != TIR_TASK
]


if len(tir_train_examples) != 200:

    raise RuntimeError(
        f"Expected 200 TIR examples, "
        f"found {len(tir_train_examples)}"
    )


if len(other_train_examples) != 1000:

    raise RuntimeError(
        f"Expected 1000 non-TIR examples, "
        f"found {len(other_train_examples)}"
    )


# Deterministic curriculum order.
stage1_rng = random.Random(
    SEED
)

stage1_rng.shuffle(
    tir_train_examples
)


stage2_rng = random.Random(
    SEED + 1
)

stage2_rng.shuffle(
    other_train_examples
)


print(
    "\nCurriculum:"
)

print(
    "Stage 1 — TIR examples:",
    len(tir_train_examples),
)

print(
    "Stage 2 — other five tasks:",
    len(other_train_examples),
)

print(
    "Total optimizer steps:",
    (
        len(tir_train_examples)
        + len(other_train_examples)
    ),
)


# ============================================================
# Validation data
# ============================================================

print(
    "\nBuilding validation examples..."
)


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


# ============================================================
# Reconstruct exact balanced validation strategy
# from the original six-task experiment
# ============================================================

def build_balanced_validation(
    examples,
    task_ids,
    seed=42,
):

    output = {}


    for task_index, task_id in enumerate(
        task_ids
    ):

        task_examples = [

            x
            for x in examples

            if x["_task_id"]
            == task_id
        ]


        yes_examples = [

            x
            for x in task_examples

            if x["_answer_clean"]
            == "(a)"
        ]


        no_examples = [

            x
            for x in task_examples

            if x["_answer_clean"]
            == "(b)"
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


val_by_task = (
    build_balanced_validation(
        examples=raw_val_examples,
        task_ids=ACTIVE_TASKS,
        seed=SEED,
    )
)


# ============================================================
# Prediction helpers
# ============================================================

def clean_prediction(
    prediction,
):

    text = (
        str(prediction)
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


def chunks(
    items,
    batch_size,
):

    for start in range(
        0,
        len(items),
        batch_size,
    ):

        yield items[
            start:
            start + batch_size
        ]


# ============================================================
# Evaluate one task
# ============================================================

def evaluate_task(
    examples,
):

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    invalid = 0


    with torch.no_grad():

        for batch in chunks(
            examples,
            EVAL_BATCH_SIZE,
        ):

            predictions = (
                model.generate(
                    batch,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            )


            for example, raw_pred in zip(
                batch,
                predictions,
            ):

                gold = (
                    example[
                        "_answer_clean"
                    ]
                )

                pred = (
                    clean_prediction(
                        raw_pred
                    )
                )


                if pred is None:
                    invalid += 1


                if gold == "(a)":

                    yes_total += 1

                    if pred == gold:
                        yes_correct += 1

                else:

                    no_total += 1

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


    return {
        "n":
            len(examples),

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "invalid":
            invalid,
    }


# ============================================================
# Evaluate all six tasks
# ============================================================

def evaluate_all(
    label,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        label
    )

    print(
        "=" * 78
    )


    results = {}


    for task_id in ACTIVE_TASKS:

        result = (
            evaluate_task(
                val_by_task[
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
            f"  invalid outputs:    "
            f"{result['invalid']}"
        )


    macro = (
        sum(
            x[
                "balanced_accuracy"
            ]

            for x
            in results.values()
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


# ============================================================
# Training setup
# ============================================================

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


encoder_parameters = list(
    model.encoder.parameters()
)

projector_parameters = list(
    model.projector.parameters()
)


trainable_parameters = (
    encoder_parameters
    + projector_parameters
)


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
    ]
)


# ============================================================
# Generic training-stage function
# ============================================================

def train_stage(
    examples,
    label,
):

    print(
        "\n"
        + "=" * 78
    )

    print(
        label
    )

    print(
        "=" * 78
    )


    model.encoder.train()
    model.projector.train()
    model.llm.eval()


    losses = []


    for step, example in enumerate(
        examples,
        start=1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )


        loss = (
            model.compute_loss(
                [example]
            )
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss "
                f"at step {step}: "
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
            or step == len(examples)
        ):

            recent = (
                losses[
                    -PRINT_EVERY:
                ]
            )

            recent_mean = (
                sum(recent)
                / len(recent)
            )


            print(
                f"Step "
                f"{step:4d}/"
                f"{len(examples)} | "
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
        "\nMean loss:",
        f"{mean_loss:.4f}"
    )


    return losses


# ============================================================
# STAGE 1 — TIR FIRST
# ============================================================

stage1_losses = (
    train_stage(
        examples=tir_train_examples,
        label=(
            "STAGE 1 — TIR-FIRST ALIGNMENT"
        ),
    )
)


model.eval()


stage1_results = (
    evaluate_all(
        "AFTER STAGE 1 — TIR ONLY"
    )
)


model.store_to_file(
    str(
        INTERMEDIATE_CHECKPOINT_PATH
    )
)


# ============================================================
# STAGE 2 — OTHER FIVE TASKS
# ============================================================

stage2_losses = (
    train_stage(
        examples=other_train_examples,
        label=(
            "STAGE 2 — REMAINING FIVE CGM TASKS"
        ),
    )
)


model.eval()


final_results = (
    evaluate_all(
        "FINAL TIR-FIRST CURRICULUM — VALIDATION"
    )
)


# ============================================================
# Final comparison
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "CURRICULUM SUMMARY"
)

print(
    "=" * 78
)


# Previous fully shuffled six-task results.
previous_multitask = {
    "above_160":
        0.806,

    "above_180":
        0.837,

    "sustained_above_180_15m":
        0.892,

    "rise_50_30m":
        0.707,

    "fall_50_30m":
        0.694,

    "tir_70_180_above_90":
        0.500,
}


print(
    f"{'Task':28s} "
    f"{'Mixed':>9s} "
    f"{'After TIR':>10s} "
    f"{'Final':>9s}"
)

print(
    "-" * 62
)


for task_id in ACTIVE_TASKS:

    mixed = (
        previous_multitask[
            task_id
        ]
    )

    stage1 = (
        stage1_results[
            "tasks"
        ][
            task_id
        ][
            "balanced_accuracy"
        ]
    )

    final = (
        final_results[
            "tasks"
        ][
            task_id
        ][
            "balanced_accuracy"
        ]
    )


    print(
        f"{task_id:28s} "
        f"{mixed * 100:8.1f}% "
        f"{stage1 * 100:9.1f}% "
        f"{final * 100:8.1f}%"
    )


previous_macro = (
    sum(
        previous_multitask.values()
    )
    / len(
        previous_multitask
    )
)


print(
    "-" * 62
)

print(
    f"{'MACRO':28s} "
    f"{previous_macro * 100:8.1f}% "
    f"{stage1_results['macro_balanced_accuracy'] * 100:9.1f}% "
    f"{final_results['macro_balanced_accuracy'] * 100:8.1f}%"
)


# ============================================================
# Save final checkpoint
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
        "tir_first_curriculum_order_only",

    "seed":
        SEED,

    "repo_id":
        REPO_ID,

    "train_participants":
        len(train_files),

    "validation_participants":
        len(val_files),

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "total_training_examples":
        len(train_examples),

    "stage1_tir_examples":
        len(tir_train_examples),

    "stage2_other_task_examples":
        len(other_train_examples),

    "normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "stage1_results":
        stage1_results,

    "final_results":
        final_results,

    "previous_mixed_results":
        previous_multitask,

    "stage1_losses":
        stage1_losses,

    "stage2_losses":
        stage2_losses,
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
    "\nSaved Stage-1 checkpoint:"
)

print(
    INTERMEDIATE_CHECKPOINT_PATH
)


print(
    "\nSaved final checkpoint:"
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