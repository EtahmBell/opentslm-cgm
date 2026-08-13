"""
Stage-5 OpenTSLM CGM grounding experiment.

Purpose
-------
Repeat our six-task CGM alignment benchmark starting from the
OpenTSLM Stage-5 ECG Soft Prompt checkpoint instead of TSQA.

Official Stage-5 checkpoint:
    OpenTSLM/llama-3.2-1b-ecg-sp

We keep fixed:
    - same 23 training participants
    - same 5 validation participants
    - same sealed 6 test participants
    - same fixed train-derived normalization
    - same six CGM tasks
    - same 100 YES + 100 NO examples/task
    - same 1200 unique training examples
    - same validation examples
    - same learning rates
    - same one-epoch / 1200-step training recipe

We run three optimization/order seeds:
    42, 43, 44

IMPORTANT
---------
The exact selected training examples are fixed using DATASET_SEED=42.
Only optimization/random training order changes across run seeds.

For seed 42, we preserve the exact ordering returned by the original
balance_multitask_examples(seed=42), making it the closest Stage-5
counterpart to the original TSQA mixed experiment.

TEST participants remain sealed.
"""

import gc
import json
import random
import re
import statistics
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

STAGE5_REPO_ID = (
    "OpenTSLM/llama-3.2-1b-ecg-sp"
)

DATASET_SEED = 42

RUN_SEEDS = [
    42,
    43,
    44,
]


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


RESULTS_PATH = (
    ARTIFACT_DIR
    / "cgm_stage5_multiseed_results.json"
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

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0

EVAL_BATCH_SIZE = 4

MAX_NEW_TOKENS = 6

PRINT_EVERY = 50


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
# Prediction helpers
# ============================================================

def clean_prediction(
    prediction,
):

    text = (
        str(
            prediction
        )
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
# Evaluation
# ============================================================

def evaluate_examples(
    model,
    examples,
):

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    pred_yes = 0
    pred_no = 0

    invalid = 0


    model.eval()


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


                if pred == "(a)":

                    pred_yes += 1

                elif pred == "(b)":

                    pred_no += 1

                else:

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


    accuracy = (
        yes_correct
        + no_correct
    ) / len(
        examples
    )


    valid_rate = (
        (
            len(examples)
            - invalid
        )
        / len(examples)
    )


    return {
        "n":
            len(examples),

        "accuracy":
            accuracy,

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

        "invalid":
            invalid,

        "valid_output_rate":
            valid_rate,
    }


def evaluate_all(
    model,
    val_by_task,
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
            evaluate_examples(
                model,
                val_by_task[
                    task_id
                ],
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
            f"  valid output rate:  "
            f"{result['valid_output_rate'] * 100:5.1f}%"
        )

        print(
            f"  invalid outputs:    "
            f"{result['invalid']}"
        )


    macro = (
        sum(
            result[
                "balanced_accuracy"
            ]

            for result
            in results.values()
        )
        / len(results)
    )


    macro_valid = (
        sum(
            result[
                "valid_output_rate"
            ]

            for result
            in results.values()
        )
        / len(results)
    )


    print(
        "\nMACRO BALANCED ACCURACY:",
        f"{macro * 100:.1f}%"
    )

    print(
        "MACRO VALID OUTPUT RATE:",
        f"{macro_valid * 100:.1f}%"
    )


    return {
        "tasks":
            results,

        "macro_balanced_accuracy":
            macro,

        "macro_valid_output_rate":
            macro_valid,
    }


# ============================================================
# Load Stage 5 once to:
#
# 1. verify checkpoint works
# 2. get EOS token
# 3. build datasets
# 4. obtain Stage-5 zero-shot baseline
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "LOADING STAGE-5 ECG CHECKPOINT"
)

print(
    "=" * 78
)


random.seed(
    DATASET_SEED
)

torch.manual_seed(
    DATASET_SEED
)


baseline_model = (
    OpenTSLM.load_pretrained(
        STAGE5_REPO_ID,
        device=device,
        enable_lora=True,
    )
)


baseline_model.eval()


EOS_TOKEN = (
    baseline_model.get_eos_token()
)


# ============================================================
# Build raw training examples
# ============================================================

print(
    "\nBuilding Stage-5 CGM training examples..."
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


# ============================================================
# Fix the exact training example selection ONCE
#
# All three run seeds see exactly the same 1200 examples.
# ============================================================

base_train_examples = (
    balance_multitask_examples(
        examples=raw_train_examples,
        task_ids=ACTIVE_TASKS,
        per_class=TRAIN_EXAMPLES_PER_CLASS,
        seed=DATASET_SEED,
    )
)


if len(
    base_train_examples
) != 1200:

    raise RuntimeError(
        f"Expected exactly 1200 training examples, "
        f"found {len(base_train_examples)}"
    )


print(
    "\nFixed training set:",
    len(base_train_examples),
)


# ============================================================
# Build validation pool
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
# Reconstruct fixed validation sets
# ============================================================

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
            len(
                yes_examples
            ),
            len(
                no_examples
            ),
        )


        if n_per_class == 0:

            raise RuntimeError(
                f"{task_id} has only one "
                "validation class."
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
        raw_val_examples,
        ACTIVE_TASKS,
        DATASET_SEED,
    )
)


# ============================================================
# One-example Stage-5 / CGM compatibility smoke test
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STAGE-5 + CGM SMOKE TEST"
)

print(
    "=" * 78
)


smoke_example = (
    base_train_examples[
        0
    ]
)


with torch.no_grad():

    smoke_output = (
        baseline_model.generate(
            [
                smoke_example
            ],
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
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
    "Stage-5 raw output:",
    smoke_output[
        0
    ],
)


# ============================================================
# Stage-5 zero-shot CGM benchmark
# ============================================================

zero_shot_results = (
    evaluate_all(
        baseline_model,
        val_by_task,
        (
            "STAGE-5 ZERO-SHOT CGM — "
            "BEFORE CGM ALIGNMENT"
        ),
    )
)


# Free this model before running three training jobs.
del baseline_model

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ============================================================
# Training setup helper
# ============================================================

def configure_for_cgm_training(
    model,
):

    # Freeze LLM.
    for parameter in (
        model.llm.parameters()
    ):

        parameter.requires_grad = False


    # Train encoder.
    for parameter in (
        model.encoder.parameters()
    ):

        parameter.requires_grad = True


    # Train projector.
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


    return (
        optimizer,
        trainable_parameters,
    )

def store_stage5_model(
    model,
    path,
):
    """
    Save encoder/projector plus the pretrained Stage-5 LoRA adapters.

    CGM training keeps LoRA frozen, but OpenTSLM's checkpoint saver
    only serializes LoRA parameters whose requires_grad=True.

    Temporarily mark only LoRA parameters trainable for serialization,
    then restore their frozen state.
    """

    lora_parameters = []

    for name, parameter in model.llm.named_parameters():

        if "lora_" in name:

            lora_parameters.append(
                (
                    parameter,
                    parameter.requires_grad,
                )
            )

            parameter.requires_grad = True


    try:

        model.store_to_file(
            str(path)
        )

    finally:

        for parameter, original_requires_grad in lora_parameters:

            parameter.requires_grad = (
                original_requires_grad
            )

# ============================================================
# Train one Stage-5 run
# ============================================================

def train_seed(
    run_seed,
):

    print(
        "\n\n"
        + "#" * 78
    )

    print(
        f"STAGE-5 CGM ALIGNMENT — "
        f"SEED {run_seed}"
    )

    print(
        "#" * 78
    )


    random.seed(
        run_seed
    )

    torch.manual_seed(
        run_seed
    )


    # --------------------------------------------------------
    # Fresh Stage-5 checkpoint for every run.
    # --------------------------------------------------------

    print(
        "\nReloading fresh Stage-5 checkpoint..."
    )


    model = (
        OpenTSLM.load_pretrained(
            STAGE5_REPO_ID,
            device=device,
            enable_lora=True,
        )
    )


    (
        optimizer,
        trainable_parameters,
    ) = configure_for_cgm_training(
        model
    )


    # --------------------------------------------------------
    # Exact same 1200 examples.
    #
    # Seed 42:
    # preserve the ordering returned by the original
    # balance_multitask_examples(seed=42).
    #
    # Seeds 43 and 44:
    # shuffle the SAME 1200 examples.
    # --------------------------------------------------------

    train_examples = list(
        base_train_examples
    )


    if run_seed != DATASET_SEED:

        order_rng = random.Random(
            run_seed
        )

        order_rng.shuffle(
            train_examples
        )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model.encoder.train()
    model.projector.train()
    model.llm.eval()


    losses = []


    for step, example in enumerate(
        train_examples,
        start=1,
    ):

        optimizer.zero_grad(
            set_to_none=True
        )


        loss = (
            model.compute_loss(
                [
                    example
                ]
            )
        )


        if not torch.isfinite(
            loss
        ):

            raise RuntimeError(
                f"Non-finite loss "
                f"for seed {run_seed}, "
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
                f"Seed {run_seed} | "
                f"Step "
                f"{step:4d}/"
                f"{len(train_examples)} | "
                f"Loss "
                f"{loss_value:.4f} | "
                f"Recent mean "
                f"{recent_mean:.4f}"
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
        "\nMean training loss:",
        f"{mean_loss:.4f}"
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()


    validation = (
        evaluate_all(
            model,
            val_by_task,
            (
                f"STAGE-5 CGM ALIGNED "
                f"— SEED {run_seed}"
            ),
        )
    )


    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / (
            "cgm_alignment_"
            f"stage5_mixed_seed{run_seed}.pt"
        )
    )


    store_stage5_model(
        model,
        checkpoint_path,
    )


    print(
        "\nSaved checkpoint:"
    )

    print(
        checkpoint_path
    )


    result = {
        "seed":
            run_seed,

        "mean_training_loss":
            mean_loss,

        "loss_history":
            losses,

        "validation":
            validation,

        "checkpoint":
            str(
                checkpoint_path
            ),
    }


    del model
    del optimizer

    gc.collect()


    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return result


# ============================================================
# Run seeds
# ============================================================

seed_results = {}


for seed in RUN_SEEDS:

    seed_results[
        str(seed)
    ] = train_seed(
        seed
    )


# ============================================================
# Aggregate across seeds
# ============================================================

print(
    "\n\n"
    + "=" * 92
)

print(
    "STAGE-5 THREE-SEED SUMMARY"
)

print(
    "=" * 92
)


aggregate_tasks = {}


print(
    f"{'Task':28s} "
    f"{'Seed42':>9s} "
    f"{'Seed43':>9s} "
    f"{'Seed44':>9s} "
    f"{'Mean':>9s} "
    f"{'SD':>9s}"
)

print(
    "-" * 86
)


for task_id in ACTIVE_TASKS:

    values = []


    for run_seed in RUN_SEEDS:

        value = (
            seed_results[
                str(
                    run_seed
                )
            ][
                "validation"
            ][
                "tasks"
            ][
                task_id
            ][
                "balanced_accuracy"
            ]
        )


        values.append(
            value
        )


    mean_value = (
        statistics.mean(
            values
        )
    )


    sd_value = (
        statistics.stdev(
            values
        )
    )


    aggregate_tasks[
        task_id
    ] = {
        "values":
            values,

        "mean":
            mean_value,

        "sd":
            sd_value,
    }


    print(
        f"{task_id:28s} "
        f"{values[0] * 100:8.1f}% "
        f"{values[1] * 100:8.1f}% "
        f"{values[2] * 100:8.1f}% "
        f"{mean_value * 100:8.1f}% "
        f"{sd_value * 100:8.1f}%"
    )


macro_values = [

    seed_results[
        str(
            run_seed
        )
    ][
        "validation"
    ][
        "macro_balanced_accuracy"
    ]

    for run_seed
    in RUN_SEEDS
]


macro_mean = (
    statistics.mean(
        macro_values
    )
)

macro_sd = (
    statistics.stdev(
        macro_values
    )
)


print(
    "-" * 86
)

print(
    f"{'MACRO':28s} "
    f"{macro_values[0] * 100:8.1f}% "
    f"{macro_values[1] * 100:8.1f}% "
    f"{macro_values[2] * 100:8.1f}% "
    f"{macro_mean * 100:8.1f}% "
    f"{macro_sd * 100:8.1f}%"
)


# ============================================================
# Compare with our original TSQA seed-42 result
# ============================================================

TSQA_SEED42 = {
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


TSQA_MACRO = (
    sum(
        TSQA_SEED42.values()
    )
    / len(
        TSQA_SEED42
    )
)


print(
    "\n"
    + "=" * 78
)

print(
    "TSQA vs STAGE-5 COMPARISON"
)

print(
    "=" * 78
)


print(
    f"{'Task':28s} "
    f"{'TSQA-42':>10s} "
    f"{'S5 mean':>10s} "
    f"{'S5 SD':>9s}"
)

print(
    "-" * 64
)


for task_id in ACTIVE_TASKS:

    tsqa = (
        TSQA_SEED42[
            task_id
        ]
    )

    stage5_mean = (
        aggregate_tasks[
            task_id
        ][
            "mean"
        ]
    )

    stage5_sd = (
        aggregate_tasks[
            task_id
        ][
            "sd"
        ]
    )


    print(
        f"{task_id:28s} "
        f"{tsqa * 100:9.1f}% "
        f"{stage5_mean * 100:9.1f}% "
        f"{stage5_sd * 100:8.1f}%"
    )


print(
    "-" * 64
)

print(
    f"{'MACRO':28s} "
    f"{TSQA_MACRO * 100:9.1f}% "
    f"{macro_mean * 100:9.1f}% "
    f"{macro_sd * 100:8.1f}%"
)


# ============================================================
# Save complete experiment
# ============================================================

results = {

    "experiment":
        "stage5_cgm_multiseed",

    "stage5_repo_id":
        STAGE5_REPO_ID,

    "dataset_seed":
        DATASET_SEED,

    "run_seeds":
        RUN_SEEDS,

    "train_participants":
        len(
            train_files
        ),

    "validation_participants":
        len(
            val_files
        ),

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "training_examples":
        len(
            base_train_examples
        ),

    "normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "zero_shot_stage5":
        zero_shot_results,

    "runs":
        seed_results,

    "aggregate":
        {
            "tasks":
                aggregate_tasks,

            "macro_values":
                macro_values,

            "macro_mean":
                macro_mean,

            "macro_sd":
                macro_sd,
        },

    "tsqa_seed42_reference":
        {
            "tasks":
                TSQA_SEED42,

            "macro":
                TSQA_MACRO,
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
    "\nSaved results:"
)

print(
    RESULTS_PATH
)


print(
    "\nTEST participants remain sealed."
)