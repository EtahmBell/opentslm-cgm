"""
Evaluate our CGM smoke-test checkpoint on participant 1001 windows
that were NEVER used for training and do not overlap the training windows.

This is still NOT a scientific test of generalization because all windows
come from the same participant.

It answers one question:

Did our 8-example fine-tuning merely memorize those 8 windows,
or did it learn something that transfers to other CGM windows?
"""

import json
import random

import torch

from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

REPO_ID = "OpenTSLM/llama-3.2-1b-tsqa-sp"

CHECKPOINT_PATH = (
    r"C:\tslm\OpenTSLM\cgm_alignment_smoke_checkpoint.pt"
)

CGM_PATH = r"C:\tslm\cgm\data\1001_DEX.json"

WINDOW_SIZE = 128
STRIDE = 64
THRESHOLD = 160

NUM_TRAIN_EXAMPLES = 8

random.seed(42)
torch.manual_seed(42)


# ============================================================
# Load CGM
# ============================================================

with open(CGM_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

glucose = [
    record["blood_glucose"]["value"]
    for record in data["body"]["cgm"]
    if record.get("blood_glucose", {}).get("value") is not None
]

glucose = torch.tensor(
    glucose,
    dtype=torch.float32,
)

print("Participant:", data["header"]["patient_id"])
print("Total readings:", len(glucose))


# ============================================================
# We need the EOS token to construct answers, so load model.
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nLoading ORIGINAL pretrained model...")

model = OpenTSLM.load_pretrained(
    REPO_ID,
    device=device,
)

EOS = model.get_eos_token()


# ============================================================
# Build exactly the same 43 candidate windows used during
# training.
# ============================================================

examples = []

for start in range(
    0,
    len(glucose) - WINDOW_SIZE + 1,
    STRIDE,
):

    raw_window = glucose[
        start:start + WINDOW_SIZE
    ]

    mean = raw_window.mean()
    std = raw_window.std()

    normalized = (
        raw_window - mean
    ) / (std + 1e-8)

    max_glucose = raw_window.max().item()

    answer = (
        "(a)"
        if max_glucose > THRESHOLD
        else "(b)"
    )

    question = f"""
The following time series contains continuous glucose monitor
measurements sampled approximately every 5 minutes.

Did the glucose concentration ever exceed {THRESHOLD} mg/dL
during this interval?

Choose one answer:
(a) Yes
(b) No

Only answer (a) or (b).
""".strip()

    description = (
        "This is a continuous glucose monitor time series. "
        f"The original measurements have mean "
        f"{mean.item():.2f} mg/dL and standard deviation "
        f"{std.item():.2f} mg/dL."
    )

    examples.append(
        {
            "pre_prompt":
                question,

            "time_series_text":
                [description],

            "time_series":
                [normalized.clone()],

            "post_prompt":
                "Predict the CGM Threshold Answer:",

            "answer":
                answer + EOS,

            "_answer_clean":
                answer,

            "_max_glucose":
                max_glucose,

            "_start":
                start,
        }
    )


print("\nTotal candidate windows:", len(examples))


# ============================================================
# Reconstruct the SAME 8 training examples used by the smoke
# script.
#
# Same random seed + same selection procedure = same samples.
# ============================================================

positive = [
    x for x in examples
    if x["_answer_clean"] == "(a)"
]

negative = [
    x for x in examples
    if x["_answer_clean"] == "(b)"
]

random.shuffle(positive)
random.shuffle(negative)

half = NUM_TRAIN_EXAMPLES // 2

train_examples = (
    positive[:half]
    + negative[:half]
)

random.shuffle(train_examples)

train_starts = {
    x["_start"]
    for x in train_examples
}

print(
    "Training window starts:",
    sorted(train_starts),
)


# ============================================================
# Exclude ANY candidate that overlaps a training window.
#
# Window A overlaps Window B when:
#
#   A_start < B_end
#       AND
#   B_start < A_end
# ============================================================

def windows_overlap(start_a, start_b):

    end_a = start_a + WINDOW_SIZE
    end_b = start_b + WINDOW_SIZE

    return (
        start_a < end_b
        and start_b < end_a
    )


unseen_examples = []

for example in examples:

    start = example["_start"]

    overlaps_training = any(
        windows_overlap(
            start,
            train_start,
        )
        for train_start in train_starts
    )

    if not overlaps_training:
        unseen_examples.append(example)


print(
    "Non-overlapping unseen windows:",
    len(unseen_examples),
)


# ============================================================
# Balance the held-out test set.
#
# This prevents accuracy from looking good simply because one
# answer occurs much more often.
# ============================================================

unseen_positive = [
    x for x in unseen_examples
    if x["_answer_clean"] == "(a)"
]

unseen_negative = [
    x for x in unseen_examples
    if x["_answer_clean"] == "(b)"
]

n_per_class = min(
    len(unseen_positive),
    len(unseen_negative),
)

test_examples = (
    unseen_positive[:n_per_class]
    + unseen_negative[:n_per_class]
)

test_rng = random.Random(123)
test_rng.shuffle(test_examples)

print(
    "Held-out YES:",
    sum(
        x["_answer_clean"] == "(a)"
        for x in test_examples
    )
)

print(
    "Held-out NO:",
    sum(
        x["_answer_clean"] == "(b)"
        for x in test_examples
    )
)

print(
    "Final held-out test size:",
    len(test_examples),
)


# ============================================================
# Format samples for OpenTSLM
# ============================================================

def model_sample(example):

    return {
        "pre_prompt":
            example["pre_prompt"],

        "time_series_text":
            example["time_series_text"],

        "time_series":
            example["time_series"],

        "post_prompt":
            example["post_prompt"],

        "answer":
            example["answer"],
    }


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    examples_to_evaluate,
    label,
):

    correct = 0

    yes_correct = 0
    yes_total = 0

    no_correct = 0
    no_total = 0

    print("\n" + "=" * 65)
    print(label)
    print("=" * 65)

    for i, example in enumerate(
        examples_to_evaluate
    ):

        prediction = model.generate(
            [model_sample(example)],
            max_new_tokens=5,
        )[0].strip()

        gold = example["_answer_clean"]

        is_correct = (
            prediction.startswith(gold)
        )

        if gold == "(a)":
            yes_total += 1

            if is_correct:
                yes_correct += 1

        else:
            no_total += 1

            if is_correct:
                no_correct += 1

        if is_correct:
            correct += 1

        print(
            f"{i + 1:02d} | "
            f"start={example['_start']:4d} | "
            f"max={example['_max_glucose']:6.1f} | "
            f"gold={gold} | "
            f"pred={repr(prediction)} | "
            f"{'✓' if is_correct else '✗'}"
        )

    total = len(
        examples_to_evaluate
    )

    accuracy = correct / total

    print(
        f"\nOverall: "
        f"{correct}/{total} "
        f"({accuracy * 100:.1f}%)"
    )

    if yes_total:
        print(
            f"YES accuracy: "
            f"{yes_correct}/{yes_total} "
            f"({yes_correct / yes_total * 100:.1f}%)"
        )

    if no_total:
        print(
            f"NO accuracy: "
            f"{no_correct}/{no_total} "
            f"({no_correct / no_total * 100:.1f}%)"
        )

    return accuracy


# ============================================================
# 1. Evaluate ORIGINAL OpenTSLM before CGM alignment
# ============================================================

model.eval()

baseline_accuracy = evaluate(
    test_examples,
    "ORIGINAL OPENTSLM — HELD-OUT CGM",
)


# ============================================================
# 2. Load our CGM-trained encoder/projector
# ============================================================

print(
    "\nLoading CGM alignment checkpoint..."
)

model.load_from_file(
    CHECKPOINT_PATH
)

model.eval()


# ============================================================
# Confirm it still memorized the 8 training windows
# ============================================================

train_accuracy = evaluate(
    train_examples,
    "CGM-TRAINED MODEL — TRAINING WINDOWS",
)


# ============================================================
# 3. Evaluate CGM-trained model on non-overlapping windows
# ============================================================

unseen_accuracy = evaluate(
    test_examples,
    "CGM-TRAINED MODEL — HELD-OUT CGM",
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(
    f"Original OpenTSLM held-out: "
    f"{baseline_accuracy * 100:.1f}%"
)

print(
    f"CGM model training windows: "
    f"{train_accuracy * 100:.1f}%"
)

print(
    f"CGM model held-out:         "
    f"{unseen_accuracy * 100:.1f}%"
)