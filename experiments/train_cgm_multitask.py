"""
Train OpenTSLM on multiple deterministic CGM grounding tasks.

Current experiment
------------------
Starting checkpoint:
    OpenTSLM/llama-3.2-1b-tsqa-sp

Training:
    23 participants
    6 CGM tasks
    100 YES + 100 NO per task
    = 1200 examples total

Validation:
    5 completely unseen participants
    Each task is balanced independently.

Test:
    6 participants remain completely untouched.

This is still a proof-of-concept experiment using the TSQA checkpoint.
The final comparison will later restart from the Stage-5 checkpoint.
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


# ============================================================
# OpenTSLM
# ============================================================

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

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
)


CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "cgm_alignment_multitask_tsqa.pt"
)

RESULTS_PATH = (
    ARTIFACT_DIR
    / "cgm_multitask_tsqa_results.json"
)

PREDICTIONS_BEFORE_PATH = (
    ARTIFACT_DIR
    / "cgm_multitask_predictions_before.jsonl"
)

PREDICTIONS_AFTER_PATH = (
    ARTIFACT_DIR
    / "cgm_multitask_predictions_after.jsonl"
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

NUM_EPOCHS = 1

ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0

# OpenTSLM supports batches, but keep this modest
# for laptop/CPU evaluation.
EVAL_BATCH_SIZE = 4

MAX_NEW_TOKENS = 6

PRINT_EVERY = 50


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

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load manifest
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
    "Sealed TEST participants:",
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
# Load original OpenTSLM
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
# Build TRAIN pool
# ============================================================

print(
    "\nBuilding training examples..."
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


print(
    "Raw multitask training examples:",
    len(raw_train_examples),
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
    "\nBalanced training examples:",
    len(train_examples),
)


# ============================================================
# Build VALIDATION pool
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


print(
    "Raw validation examples:",
    len(raw_val_examples),
)


# ============================================================
# Independently balance each validation task
# ============================================================

def build_balanced_validation(
    examples,
    task_ids,
    seed=42,
):
    """
    Balance each task independently.

    We use every example from the minority class and sample
    the same number from the majority class.

    Returns
    -------
    dict:
        task_id -> balanced list of validation examples
    """

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


        if n_per_class == 0:

            raise RuntimeError(
                f"Validation task "
                f"{task_id} has only one class."
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
# Prediction parsing
# ============================================================

def clean_prediction(
    prediction,
):
    """
    Extract the first (a) or (b) answer from generated text.

    Anything else is treated as invalid.
    """

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


# ============================================================
# Batch helper
# ============================================================

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
    task_id,
    examples,
):
    """
    Evaluate one balanced validation task.
    """

    gold_yes = 0
    gold_no = 0

    correct_yes = 0
    correct_no = 0

    invalid = 0

    predictions_output = []


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

                gold = example[
                    "_answer_clean"
                ]

                pred = clean_prediction(
                    raw_pred
                )


                if gold == "(a)":

                    gold_yes += 1

                    if pred == gold:
                        correct_yes += 1


                elif gold == "(b)":

                    gold_no += 1

                    if pred == gold:
                        correct_no += 1


                if pred is None:
                    invalid += 1


                predictions_output.append(
                    {
                        "task_id":
                            task_id,

                        "participant_id":
                            example[
                                "_participant_id"
                            ],

                        "window_start":
                            example[
                                "_window_start"
                            ],

                        "metric_name":
                            example[
                                "_metric_name"
                            ],

                        "metric_value":
                            float(
                                example[
                                    "_metric_value"
                                ]
                            ),

                        "gold":
                            gold,

                        "prediction":
                            pred,

                        "raw_prediction":
                            str(
                                raw_pred
                            ),

                        "correct":
                            bool(
                                pred == gold
                            ),
                    }
                )


    yes_accuracy = (
        correct_yes
        / gold_yes
        if gold_yes
        else 0.0
    )

    no_accuracy = (
        correct_no
        / gold_no
        if gold_no
        else 0.0
    )


    total = (
        gold_yes
        + gold_no
    )

    correct = (
        correct_yes
        + correct_no
    )


    overall_accuracy = (
        correct
        / total
        if total
        else 0.0
    )


    balanced_accuracy = (
        yes_accuracy
        + no_accuracy
    ) / 2.0


    valid_output_rate = (
        (total - invalid)
        / total
        if total
        else 0.0
    )


    metrics = {
        "task_id":
            task_id,

        "n":
            total,

        "yes_n":
            gold_yes,

        "no_n":
            gold_no,

        "overall_accuracy":
            overall_accuracy,

        "yes_accuracy":
            yes_accuracy,

        "no_accuracy":
            no_accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "valid_output_rate":
            valid_output_rate,

        "invalid_outputs":
            invalid,
    }


    return (
        metrics,
        predictions_output,
    )


# ============================================================
# Evaluate ALL tasks
# ============================================================

def evaluate_all(
    label,
    predictions_path,
):
    """
    Evaluate all six tasks independently.

    The macro balanced accuracy gives every task equal weight.
    """

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


    all_metrics = {}

    all_predictions = []


    for task_id in ACTIVE_TASKS:

        task_examples = (
            val_by_task[
                task_id
            ]
        )


        metrics, task_predictions = (
            evaluate_task(
                task_id=task_id,
                examples=task_examples,
            )
        )


        all_metrics[
            task_id
        ] = metrics


        all_predictions.extend(
            task_predictions
        )


        print(
            f"\n{task_id}"
        )

        print(
            f"  n:                 "
            f"{metrics['n']}"
        )

        print(
            f"  accuracy:          "
            f"{metrics['overall_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  YES accuracy:      "
            f"{metrics['yes_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  NO accuracy:       "
            f"{metrics['no_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  balanced accuracy: "
            f"{metrics['balanced_accuracy'] * 100:5.1f}%"
        )

        print(
            f"  valid output rate: "
            f"{metrics['valid_output_rate'] * 100:5.1f}%"
        )


    macro_balanced_accuracy = (
        sum(
            metrics[
                "balanced_accuracy"
            ]

            for metrics
            in all_metrics.values()
        )
        / len(all_metrics)
    )


    macro_accuracy = (
        sum(
            metrics[
                "overall_accuracy"
            ]

            for metrics
            in all_metrics.values()
        )
        / len(all_metrics)
    )


    print(
        "\n"
        + "-" * 78
    )

    print(
        "MACRO BALANCED ACCURACY:",
        f"{macro_balanced_accuracy * 100:.1f}%"
    )

    print(
        "MACRO ACCURACY:",
        f"{macro_accuracy * 100:.1f}%"
    )


    with predictions_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for row in all_predictions:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


    return {
        "tasks":
            all_metrics,

        "macro_balanced_accuracy":
            macro_balanced_accuracy,

        "macro_accuracy":
            macro_accuracy,
    }


# ============================================================
# BASELINE
# ============================================================

baseline_results = (
    evaluate_all(
        label=(
            "ORIGINAL TSQA OPENTSLM "
            "— MULTITASK VALIDATION"
        ),
        predictions_path=(
            PREDICTIONS_BEFORE_PATH
        ),
    )
)


# ============================================================
# Training setup
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "MULTITASK CGM TRAINING"
)

print(
    "=" * 78
)


# SP keeps the LLM frozen.
# Explicitly enforce that here as well.
for parameter in model.llm.parameters():

    parameter.requires_grad = False


for parameter in model.encoder.parameters():

    parameter.requires_grad = True


for parameter in model.projector.parameters():

    parameter.requires_grad = True


# Train encoder/projector only.
model.encoder.train()
model.projector.train()

# Keep frozen LLM deterministic.
model.llm.eval()


encoder_parameters = list(
    model.encoder.parameters()
)

projector_parameters = list(
    model.projector.parameters()
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


trainable_parameters = (
    encoder_parameters
    + projector_parameters
)


print(
    "Training examples:",
    len(train_examples),
)

print(
    "Epochs:",
    NUM_EPOCHS,
)

print(
    "Encoder LR:",
    ENCODER_LR,
)

print(
    "Projector LR:",
    PROJECTOR_LR,
)


# ============================================================
# Training
# ============================================================

loss_history = []


for epoch in range(
    NUM_EPOCHS
):

    print(
        f"\nEpoch "
        f"{epoch + 1}/"
        f"{NUM_EPOCHS}"
    )


    epoch_examples = list(
        train_examples
    )


    epoch_rng = random.Random(
        SEED
        + epoch
    )

    epoch_rng.shuffle(
        epoch_examples
    )


    epoch_losses = []


    for step, example in enumerate(
        epoch_examples,
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


        epoch_losses.append(
            loss_value
        )

        loss_history.append(
            loss_value
        )


        if (
            step == 1
            or step % PRINT_EVERY == 0
            or step == len(epoch_examples)
        ):

            recent = (
                epoch_losses[
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
                f"{len(epoch_examples)} | "
                f"Loss "
                f"{loss_value:.4f} | "
                f"Recent mean "
                f"{recent_mean:.4f}"
            )


    epoch_mean = (
        sum(epoch_losses)
        / len(epoch_losses)
    )


    print(
        f"\nEpoch "
        f"{epoch + 1} "
        f"mean loss: "
        f"{epoch_mean:.4f}"
    )


# ============================================================
# AFTER TRAINING
# ============================================================

model.eval()


aligned_results = (
    evaluate_all(
        label=(
            "CGM-ALIGNED TSQA OPENTSLM "
            "— MULTITASK VALIDATION"
        ),
        predictions_path=(
            PREDICTIONS_AFTER_PATH
        ),
    )
)


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "BEFORE vs AFTER"
)

print(
    "=" * 78
)


print(
    f"{'Task':28s} "
    f"{'Before':>10s} "
    f"{'After':>10s} "
    f"{'Delta':>10s}"
)


print(
    "-" * 62
)


for task_id in ACTIVE_TASKS:

    before = (
        baseline_results[
            "tasks"
        ][
            task_id
        ][
            "balanced_accuracy"
        ]
    )

    after = (
        aligned_results[
            "tasks"
        ][
            task_id
        ][
            "balanced_accuracy"
        ]
    )

    delta = (
        after
        - before
    )


    print(
        f"{task_id:28s} "
        f"{before * 100:9.1f}% "
        f"{after * 100:9.1f}% "
        f"{delta * 100:+9.1f}"
    )


baseline_macro = (
    baseline_results[
        "macro_balanced_accuracy"
    ]
)

aligned_macro = (
    aligned_results[
        "macro_balanced_accuracy"
    ]
)


print(
    "-" * 62
)

print(
    f"{'MACRO':28s} "
    f"{baseline_macro * 100:9.1f}% "
    f"{aligned_macro * 100:9.1f}% "
    f"{(aligned_macro - baseline_macro) * 100:+9.1f}"
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
# Save experiment metadata/results
# ============================================================

results = {

    "experiment":
        "cgm_multitask_tsqa",

    "seed":
        SEED,

    "repo_id":
        REPO_ID,

    "device":
        device,

    "active_tasks":
        ACTIVE_TASKS,

    "window_size":
        WINDOW_SIZE,

    "stride":
        STRIDE,

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

    "training_examples":
        len(train_examples),

    "train_examples_per_class_per_task":
        TRAIN_EXAMPLES_PER_CLASS,

    "epochs":
        NUM_EPOCHS,

    "encoder_lr":
        ENCODER_LR,

    "projector_lr":
        PROJECTOR_LR,

    "weight_decay":
        WEIGHT_DECAY,

    "gradient_clip_norm":
        GRAD_CLIP_NORM,

    "normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "baseline":
        baseline_results,

    "aligned":
        aligned_results,

    "loss_history":
        loss_history,
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
    "\nSaved prediction files:"
)

print(
    PREDICTIONS_BEFORE_PATH
)

print(
    PREDICTIONS_AFTER_PATH
)


print(
    "\nTEST participants remain sealed."
)