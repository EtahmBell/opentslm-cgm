"""
TIR warm-up + rehearsal CGM alignment experiment.

Question
--------
Can we prevent catastrophic forgetting of TIR by:

1. introducing TIR first, then
2. continuing to rehearse TIR while learning the other five tasks?

IMPORTANT CONTROL
-----------------
Uses the SAME exact 1200 balanced training examples as the original
fully mixed six-task experiment.

No examples are duplicated.
No examples are removed.

Training order only:

    Stage 1:
        100 TIR examples

    Stage 2:
        1000 examples from the other five tasks
        + remaining 100 TIR examples

        Rehearsal schedule:
            10 non-TIR examples
            1 TIR example
            repeat

Total optimizer steps:
    100 + 1100 = 1200

TEST participants remain sealed.
"""

import json
import random
import re
import sys

from collections import Counter
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

PREVIOUS_MIXED_RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_multitask_tsqa_results.json"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
)

ARTIFACT_DIR = (
    ROOT
    / "artifacts"
)

WARMUP_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "cgm_alignment_tir_rehearsal_warmup_tsqa.pt"
)

FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "cgm_alignment_tir_rehearsal_tsqa.pt"
)

RESULTS_PATH = (
    ARTIFACT_DIR
    / "cgm_tir_rehearsal_results.json"
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

# Of the 200 total TIR training examples:
# 100 appear first,
# 100 are saved for rehearsal.
TIR_WARMUP_EXAMPLES = 100

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
# Load TSQA checkpoint
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
# Build exact same balanced 1200-example dataset
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


if len(train_examples) != 1200:

    raise RuntimeError(
        f"Expected 1200 training examples, "
        f"found {len(train_examples)}"
    )


# ============================================================
# Separate TIR from other tasks
# ============================================================

tir_examples = [

    x
    for x in train_examples

    if x["_task_id"]
    == TIR_TASK
]


other_examples = [

    x
    for x in train_examples

    if x["_task_id"]
    != TIR_TASK
]


if len(tir_examples) != 200:

    raise RuntimeError(
        f"Expected 200 TIR examples, "
        f"found {len(tir_examples)}"
    )


if len(other_examples) != 1000:

    raise RuntimeError(
        f"Expected 1000 non-TIR examples, "
        f"found {len(other_examples)}"
    )


# ============================================================
# Preserve deterministic order
# ============================================================

# train_examples was already deterministically shuffled
# by balance_multitask_examples(seed=42).
#
# We keep the relative order of examples as they appear there.

tir_warmup = (
    tir_examples[
        :TIR_WARMUP_EXAMPLES
    ]
)

tir_rehearsal = (
    tir_examples[
        TIR_WARMUP_EXAMPLES:
    ]
)


print(
    "\nTIR warm-up examples:",
    len(tir_warmup),
)

print(
    "TIR rehearsal examples:",
    len(tir_rehearsal),
)

print(
    "Other-task examples:",
    len(other_examples),
)


# ============================================================
# Build rehearsal schedule
# ============================================================

def build_rehearsal_schedule(
    other_examples,
    rehearsal_examples,
):
    """
    Evenly interleave rehearsal examples.

    Current experiment:
        1000 other examples
        100 TIR rehearsal examples

    Therefore:
        10 other examples
        1 TIR example
        repeat
    """

    if len(rehearsal_examples) == 0:

        return list(
            other_examples
        )


    if (
        len(other_examples)
        % len(rehearsal_examples)
        != 0
    ):

        raise RuntimeError(
            "For this controlled experiment, "
            "other examples must divide evenly "
            "by rehearsal examples."
        )


    others_per_rehearsal = (
        len(other_examples)
        // len(rehearsal_examples)
    )


    schedule = []

    other_index = 0


    for rehearsal_example in (
        rehearsal_examples
    ):

        next_other_index = (
            other_index
            + others_per_rehearsal
        )


        schedule.extend(
            other_examples[
                other_index:
                next_other_index
            ]
        )


        schedule.append(
            rehearsal_example
        )


        other_index = (
            next_other_index
        )


    # Safety check.
    if (
        other_index
        < len(other_examples)
    ):

        schedule.extend(
            other_examples[
                other_index:
            ]
        )


    return schedule


stage2_schedule = (
    build_rehearsal_schedule(
        other_examples=other_examples,
        rehearsal_examples=tir_rehearsal,
    )
)


print(
    "\nStage 1 steps:",
    len(tir_warmup),
)

print(
    "Stage 2 steps:",
    len(stage2_schedule),
)

print(
    "Total optimizer steps:",
    (
        len(tir_warmup)
        + len(stage2_schedule)
    ),
)


# ============================================================
# Verify EXACT training-example equality
# ============================================================

def example_key(
    example,
):

    return (
        example[
            "_task_id"
        ],
        example[
            "_participant_id"
        ],
        example[
            "_window_start"
        ],
        example[
            "_answer_clean"
        ],
    )


original_counter = Counter(
    example_key(x)
    for x
    in train_examples
)


curriculum_counter = Counter(
    example_key(x)
    for x
    in (
        tir_warmup
        + stage2_schedule
    )
)


if (
    original_counter
    != curriculum_counter
):

    raise RuntimeError(
        "Curriculum does NOT contain "
        "the exact same training examples."
    )


if (
    len(tir_warmup)
    + len(stage2_schedule)
    != 1200
):

    raise RuntimeError(
        "Curriculum does not contain "
        "exactly 1200 steps."
    )


print(
    "\n✓ Curriculum contains the exact same "
    "1200 examples exactly once."
)

print(
    "✓ Only training ORDER has changed."
)


# ============================================================
# Build validation data
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
# Reconstruct same balanced validation sets
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


        if n_per_class == 0:

            raise RuntimeError(
                f"{task_id} has only "
                "one validation class."
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
        SEED,
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
# Evaluation
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
            result[
                "balanced_accuracy"
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
# Generic training function
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
# STAGE 1 — TIR WARM-UP
# ============================================================

stage1_losses = (
    train_stage(
        tir_warmup,
        "STAGE 1 — 100-EXAMPLE TIR WARM-UP",
    )
)


model.eval()


stage1_results = (
    evaluate_all(
        "AFTER TIR WARM-UP — VALIDATION"
    )
)


model.store_to_file(
    str(
        WARMUP_CHECKPOINT_PATH
    )
)


# ============================================================
# STAGE 2 — REHEARSAL
# ============================================================

stage2_losses = (
    train_stage(
        stage2_schedule,
        (
            "STAGE 2 — FIVE TASKS "
            "+ CONTINUED TIR REHEARSAL"
        ),
    )
)


model.eval()


final_results = (
    evaluate_all(
        "FINAL TIR-REHEARSAL MODEL — VALIDATION"
    )
)


# ============================================================
# Load previous fully mixed result for comparison
# ============================================================

previous_mixed = None


if PREVIOUS_MIXED_RESULTS_PATH.exists():

    with PREVIOUS_MIXED_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        previous_payload = (
            json.load(f)
        )


    previous_mixed = (
        previous_payload[
            "aligned"
        ]
    )


# ============================================================
# Summary
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "TIR REHEARSAL CURRICULUM SUMMARY"
)

print(
    "=" * 78
)


if previous_mixed is not None:

    print(
        f"{'Task':28s} "
        f"{'Mixed':>9s} "
        f"{'Warm-up':>9s} "
        f"{'Final':>9s}"
    )

else:

    print(
        f"{'Task':28s} "
        f"{'Warm-up':>9s} "
        f"{'Final':>9s}"
    )


print(
    "-" * 62
)


for task_id in ACTIVE_TASKS:

    warmup = (
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


    if previous_mixed is not None:

        mixed = (
            previous_mixed[
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
            f"{warmup * 100:8.1f}% "
            f"{final * 100:8.1f}%"
        )


    else:

        print(
            f"{task_id:28s} "
            f"{warmup * 100:8.1f}% "
            f"{final * 100:8.1f}%"
        )


print(
    "-" * 62
)


if previous_mixed is not None:

    mixed_macro = (
        previous_mixed[
            "macro_balanced_accuracy"
        ]
    )


    print(
        f"{'MACRO':28s} "
        f"{mixed_macro * 100:8.1f}% "
        f"{stage1_results['macro_balanced_accuracy'] * 100:8.1f}% "
        f"{final_results['macro_balanced_accuracy'] * 100:8.1f}%"
    )


else:

    print(
        f"{'MACRO':28s} "
        f"{stage1_results['macro_balanced_accuracy'] * 100:8.1f}% "
        f"{final_results['macro_balanced_accuracy'] * 100:8.1f}%"
    )


# ============================================================
# Save final checkpoint
# ============================================================

model.store_to_file(
    str(
        FINAL_CHECKPOINT_PATH
    )
)


# ============================================================
# Save results
# ============================================================

results = {

    "experiment":
        "tir_warmup_plus_rehearsal",

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

    "total_optimizer_steps":
        (
            len(tir_warmup)
            + len(stage2_schedule)
        ),

    "tir_warmup_examples":
        len(tir_warmup),

    "tir_rehearsal_examples":
        len(tir_rehearsal),

    "other_task_examples":
        len(other_examples),

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

    "stage1_losses":
        stage1_losses,

    "stage2_losses":
        stage2_losses,
}


if previous_mixed is not None:

    results[
        "previous_mixed_results"
    ] = previous_mixed


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
    "\nSaved warm-up checkpoint:"
)

print(
    WARMUP_CHECKPOINT_PATH
)


print(
    "\nSaved final checkpoint:"
)

print(
    FINAL_CHECKPOINT_PATH
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