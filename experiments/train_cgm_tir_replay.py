"""
TIR replay / rehearsal experiment.

Purpose
-------
Test whether explicit replay can prevent catastrophic forgetting of TIR
while OpenTSLM learns five additional CGM concepts.

Stage 1
-------
Train on all 200 balanced TIR examples:
    100 YES
    100 NO

Stage 2
-------
Train on:
    1000 examples from the other five tasks
    +
    100 replay updates sampled from the Stage-1 TIR examples:
        50 YES
        50 NO

Replay schedule:
    10 other-task examples
    1 TIR replay example
    repeat

Total optimizer steps:
    200 + 1100 = 1300

We evaluate TIR:
    after Stage 1
    after 250 other-task updates
    after 500
    after 750
    after 1000

Final evaluation covers all six CGM tasks.

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

MIXED_RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_multitask_tsqa_results.json"
)

TIR_FIRST_RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "cgm_tir_first_curriculum_results.json"
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
    / "cgm_tir_replay_stage1_tsqa.pt"
)

FINAL_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "cgm_tir_replay_final_tsqa.pt"
)

RESULTS_PATH = (
    ARTIFACT_DIR
    / "cgm_tir_replay_results.json"
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

# Replay buffer:
# 50 TIR YES + 50 TIR NO
REPLAY_PER_CLASS = 50

ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4

WEIGHT_DECAY = 1e-2

GRAD_CLIP_NORM = 1.0

EVAL_BATCH_SIZE = 4

MAX_NEW_TOKENS = 6

PRINT_EVERY = 25


# Evaluate TIR after this many
# OTHER-TASK updates during Stage 2.
FORGETTING_MILESTONES = {
    250,
    500,
    750,
    1000,
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
# Load original TSQA OpenTSLM
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
# Build exact same 1200-example base dataset
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


base_train_examples = (
    balance_multitask_examples(
        examples=raw_train_examples,
        task_ids=ACTIVE_TASKS,
        per_class=TRAIN_EXAMPLES_PER_CLASS,
        seed=SEED,
    )
)


if len(base_train_examples) != 1200:

    raise RuntimeError(
        f"Expected 1200 base training examples, "
        f"found {len(base_train_examples)}"
    )


print(
    "\nBase balanced training examples:",
    len(base_train_examples),
)


# ============================================================
# Separate TIR and other five tasks
# ============================================================

tir_examples = [

    x
    for x in base_train_examples

    if x[
        "_task_id"
    ] == TIR_TASK
]


other_examples = [

    x
    for x in base_train_examples

    if x[
        "_task_id"
    ] != TIR_TASK
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
# Verify TIR is exactly balanced
# ============================================================

tir_yes = [

    x
    for x in tir_examples

    if x[
        "_answer_clean"
    ] == "(a)"
]


tir_no = [

    x
    for x in tir_examples

    if x[
        "_answer_clean"
    ] == "(b)"
]


if (
    len(tir_yes) != 100
    or len(tir_no) != 100
):

    raise RuntimeError(
        "Expected TIR warm-up pool to contain "
        "100 YES and 100 NO examples."
    )


print(
    "\nStage-1 TIR pool:"
)

print(
    "YES:",
    len(tir_yes),
)

print(
    "NO: ",
    len(tir_no),
)


# ============================================================
# Stage 1:
# all 200 TIR examples
# ============================================================

tir_warmup = list(
    tir_examples
)


warmup_rng = random.Random(
    SEED + 100
)

warmup_rng.shuffle(
    tir_warmup
)


# ============================================================
# Build balanced replay buffer
# ============================================================

replay_yes = (
    sample_across_participants(
        tir_yes,
        n=REPLAY_PER_CLASS,
        seed=SEED + 200,
    )
)


replay_no = (
    sample_across_participants(
        tir_no,
        n=REPLAY_PER_CLASS,
        seed=SEED + 201,
    )
)


replay_examples = (
    replay_yes
    + replay_no
)


replay_rng = random.Random(
    SEED + 202
)

replay_rng.shuffle(
    replay_examples
)


print(
    "\nReplay buffer:"
)

print(
    "YES:",
    len(replay_yes),
)

print(
    "NO: ",
    len(replay_no),
)

print(
    "Total:",
    len(replay_examples),
)


# ============================================================
# Replay participant diversity
# ============================================================

replay_yes_participants = set(
    x[
        "_participant_id"
    ]
    for x in replay_yes
)

replay_no_participants = set(
    x[
        "_participant_id"
    ]
    for x in replay_no
)


print(
    "Replay YES participants:",
    len(
        replay_yes_participants
    ),
)

print(
    "Replay NO participants:",
    len(
        replay_no_participants
    ),
)


# ============================================================
# Shuffle other five tasks
# ============================================================

other_examples = list(
    other_examples
)


other_rng = random.Random(
    SEED + 300
)

other_rng.shuffle(
    other_examples
)


# ============================================================
# Build Stage-2 replay schedule
# ============================================================

def build_replay_schedule(
    other_examples,
    replay_examples,
):
    """
    Interleave:
        10 other examples
        1 replay example

    For this experiment:
        1000 other
        100 replay
    """

    if (
        len(other_examples)
        % len(replay_examples)
        != 0
    ):

        raise RuntimeError(
            "Other-task examples must divide "
            "evenly by replay examples."
        )


    others_per_replay = (
        len(other_examples)
        // len(replay_examples)
    )


    schedule = []

    other_index = 0


    for replay_example in (
        replay_examples
    ):

        next_index = (
            other_index
            + others_per_replay
        )


        for example in (
            other_examples[
                other_index:
                next_index
            ]
        ):

            schedule.append(
                (
                    example,
                    False,
                )
            )


        schedule.append(
            (
                replay_example,
                True,
            )
        )


        other_index = (
            next_index
        )


    return (
        schedule,
        others_per_replay,
    )


(
    stage2_schedule,
    others_per_replay,
) = build_replay_schedule(
    other_examples,
    replay_examples,
)


print(
    "\nStage-2 replay interval:"
)

print(
    f"{others_per_replay} other updates "
    "per 1 TIR replay update"
)

print(
    "Stage 1 steps:",
    len(tir_warmup),
)

print(
    "Stage 2 other-task steps:",
    len(other_examples),
)

print(
    "Stage 2 replay steps:",
    len(replay_examples),
)

print(
    "Stage 2 total steps:",
    len(stage2_schedule),
)

print(
    "TOTAL optimizer steps:",
    (
        len(tir_warmup)
        + len(stage2_schedule)
    ),
)


# ============================================================
# Audit Stage-2 schedule
# ============================================================

stage2_labels = Counter()

stage2_tasks = Counter()


for example, is_replay in (
    stage2_schedule
):

    stage2_labels[
        example[
            "_answer_clean"
        ]
    ] += 1

    stage2_tasks[
        example[
            "_task_id"
        ]
    ] += 1


print(
    "\nStage-2 class counts:"
)

print(
    "YES:",
    stage2_labels["(a)"],
)

print(
    "NO: ",
    stage2_labels["(b)"],
)


print(
    "\nStage-2 task counts:"
)

for task_id in ACTIVE_TASKS:

    print(
        f"{task_id:28s}",
        stage2_tasks[
            task_id
        ],
    )


# ============================================================
# Build validation examples
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
# Exact same balanced validation strategy
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
    examples,
):

    yes_total = 0
    no_total = 0

    yes_correct = 0
    no_correct = 0

    pred_yes = 0
    pred_no = 0

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

        "invalid":
            invalid,
    }


def print_task_result(
    label,
    result,
):

    print(
        f"\n{label}"
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
        f"  invalid outputs:    "
        f"{result['invalid']}"
    )


def evaluate_tir(
    label,
):

    result = (
        evaluate_examples(
            val_by_task[
                TIR_TASK
            ]
        )
    )


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


    print_task_result(
        "TIR",
        result,
    )


    return result


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
            evaluate_examples(
                val_by_task[
                    task_id
                ]
            )
        )


        results[
            task_id
        ] = result


        print_task_result(
            task_id,
            result,
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
# One optimization step
# ============================================================

def optimization_step(
    example,
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
            f"Non-finite loss: "
            f"{loss.item()}"
        )


    loss.backward()


    clip_grad_norm_(
        trainable_parameters,
        GRAD_CLIP_NORM,
    )


    optimizer.step()


    return float(
        loss.detach().cpu()
    )


# ============================================================
# STAGE 1 — full balanced TIR warm-up
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STAGE 1 — 200-EXAMPLE BALANCED TIR WARM-UP"
)

print(
    "=" * 78
)


model.encoder.train()
model.projector.train()
model.llm.eval()


stage1_losses = []


for step, example in enumerate(
    tir_warmup,
    start=1,
):

    loss_value = (
        optimization_step(
            example
        )
    )


    stage1_losses.append(
        loss_value
    )


    if (
        step == 1
        or step % PRINT_EVERY == 0
        or step == len(tir_warmup)
    ):

        recent = (
            stage1_losses[
                -PRINT_EVERY:
            ]
        )


        print(
            f"Step "
            f"{step:3d}/"
            f"{len(tir_warmup)} | "
            f"Loss "
            f"{loss_value:.4f} | "
            f"Recent mean "
            f"{sum(recent) / len(recent):.4f}"
        )


print(
    "\nStage-1 mean loss:",
    f"{sum(stage1_losses) / len(stage1_losses):.4f}"
)


# ============================================================
# Evaluate immediately after TIR learning
# ============================================================

model.eval()


tir_after_warmup = (
    evaluate_tir(
        "TIR AFTER STAGE 1 — BEFORE OTHER TASKS"
    )
)


model.store_to_file(
    str(
        WARMUP_CHECKPOINT_PATH
    )
)


# ============================================================
# STAGE 2 — other tasks + TIR replay
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "STAGE 2 — OTHER TASKS + BALANCED TIR REPLAY"
)

print(
    "=" * 78
)


model.encoder.train()
model.projector.train()
model.llm.eval()


stage2_losses = []

other_updates = 0
replay_updates = 0


forgetting_curve = {
    "0": tir_after_warmup,
}


for step, (
    example,
    is_replay,
) in enumerate(
    stage2_schedule,
    start=1,
):

    loss_value = (
        optimization_step(
            example
        )
    )


    stage2_losses.append(
        loss_value
    )


    if is_replay:

        replay_updates += 1

    else:

        other_updates += 1


    if (
        step == 1
        or step % PRINT_EVERY == 0
        or step == len(stage2_schedule)
    ):

        recent = (
            stage2_losses[
                -PRINT_EVERY:
            ]
        )


        print(
            f"Step "
            f"{step:4d}/"
            f"{len(stage2_schedule)} | "
            f"other={other_updates:4d} | "
            f"replay={replay_updates:3d} | "
            f"Loss "
            f"{loss_value:.4f} | "
            f"Recent mean "
            f"{sum(recent) / len(recent):.4f}"
        )


    # --------------------------------------------------------
    # Evaluate after each milestone.
    #
    # We trigger AFTER a replay step so, for example,
    # the 250-other-update checkpoint also includes the
    # associated 25 replay updates.
    # --------------------------------------------------------

    if (
        is_replay
        and other_updates
        in FORGETTING_MILESTONES
    ):

        model.eval()


        result = (
            evaluate_tir(
                (
                    f"TIR AFTER "
                    f"{other_updates} OTHER UPDATES "
                    f"+ {replay_updates} REPLAY UPDATES"
                )
            )
        )


        forgetting_curve[
            str(
                other_updates
            )
        ] = result


        # Return to training mode.
        model.encoder.train()
        model.projector.train()
        model.llm.eval()


print(
    "\nStage-2 mean loss:",
    f"{sum(stage2_losses) / len(stage2_losses):.4f}"
)


# ============================================================
# Final six-task validation
# ============================================================

model.eval()


final_results = (
    evaluate_all(
        "FINAL TIR-REPLAY MODEL — VALIDATION"
    )
)


# ============================================================
# Forgetting curve summary
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "TIR FORGETTING / RETENTION CURVE"
)

print(
    "=" * 78
)


for milestone in [
    "0",
    "250",
    "500",
    "750",
    "1000",
]:

    if milestone not in (
        forgetting_curve
    ):

        continue


    result = (
        forgetting_curve[
            milestone
        ]
    )


    if milestone == "0":

        label = (
            "After TIR warm-up"
        )

    else:

        label = (
            f"After "
            f"{milestone} "
            f"other updates"
        )


    print(
        f"{label:30s} "
        f"{result['balanced_accuracy'] * 100:6.1f}%"
    )


# ============================================================
# Load previous results for comparison
# ============================================================

mixed_results = None
tir_first_results = None


if MIXED_RESULTS_PATH.exists():

    with MIXED_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(f)


    mixed_results = (
        payload[
            "aligned"
        ]
    )


if TIR_FIRST_RESULTS_PATH.exists():

    with TIR_FIRST_RESULTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:

        payload = json.load(f)


    tir_first_results = (
        payload[
            "final_results"
        ]
    )


# ============================================================
# Final comparison
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "FINAL STRATEGY COMPARISON"
)

print(
    "=" * 78
)


if (
    mixed_results is not None
):

    print(
        f"{'Task':28s} "
        f"{'Mixed':>9s} "
        f"{'Replay':>9s} "
        f"{'Delta':>9s}"
    )

    print(
        "-" * 62
    )


    for task_id in ACTIVE_TASKS:

        mixed = (
            mixed_results[
                "tasks"
            ][
                task_id
            ][
                "balanced_accuracy"
            ]
        )

        replay = (
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
            f"{replay * 100:8.1f}% "
            f"{(replay - mixed) * 100:+8.1f}"
        )


    mixed_macro = (
        mixed_results[
            "macro_balanced_accuracy"
        ]
    )

    replay_macro = (
        final_results[
            "macro_balanced_accuracy"
        ]
    )


    print(
        "-" * 62
    )

    print(
        f"{'MACRO':28s} "
        f"{mixed_macro * 100:8.1f}% "
        f"{replay_macro * 100:8.1f}% "
        f"{(replay_macro - mixed_macro) * 100:+8.1f}"
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
# Save experiment results
# ============================================================

results = {

    "experiment":
        "tir_replay",

    "seed":
        SEED,

    "repo_id":
        REPO_ID,

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

    "base_unique_training_examples":
        len(
            base_train_examples
        ),

    "stage1_tir_steps":
        len(
            tir_warmup
        ),

    "stage2_other_steps":
        len(
            other_examples
        ),

    "stage2_replay_steps":
        len(
            replay_examples
        ),

    "total_optimizer_steps":
        (
            len(
                tir_warmup
            )
            + len(
                stage2_schedule
            )
        ),

    "replay_per_class":
        REPLAY_PER_CLASS,

    "other_updates_per_replay":
        others_per_replay,

    "normalization":
        {
            "mean":
                train_mean,

            "std":
                train_std,
        },

    "tir_after_warmup":
        tir_after_warmup,

    "forgetting_curve":
        forgetting_curve,

    "final_results":
        final_results,

    "stage1_losses":
        stage1_losses,

    "stage2_losses":
        stage2_losses,
}


if mixed_results is not None:

    results[
        "previous_mixed_results"
    ] = mixed_results


if tir_first_results is not None:

    results[
        "previous_tir_first_results"
    ] = tir_first_results


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