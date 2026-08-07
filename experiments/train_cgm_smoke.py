"""
Tiny CGM alignment training experiment.

PURPOSE:
    Prove that we can fine-tune an existing OpenTSLM checkpoint
    on AI-READI CGM -> language alignment examples.

IMPORTANT:
    This is NOT a scientific evaluation.
    It intentionally overfits one participant as a pipeline test.
"""

import json
import random

import torch
from torch.optim import AdamW

from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

REPO_ID = "OpenTSLM/llama-3.2-1b-tsqa-sp"

CGM_PATH = r"C:\tslm\cgm\data\1001_DEX.json"

WINDOW_SIZE = 128
STRIDE = 64

# For this stricter smoke test, every example asks
# the SAME question. Only the CGM signal changes.
THRESHOLDS = [160]

# Balanced 4 YES / 4 NO examples.
NUM_TRAIN_EXAMPLES = 8

# Each example will be seen about 10 times.
TRAINING_STEPS = 80

ENCODER_LR = 2e-4
PROJECTOR_LR = 1e-4
WEIGHT_DECAY = 1e-2

random.seed(42)
torch.manual_seed(42)


# ============================================================
# Load raw CGM
# ============================================================

with open(CGM_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

glucose = [
    record["blood_glucose"]["value"]
    for record in data["body"]["cgm"]
    if record.get("blood_glucose", {}).get("value") is not None
]

glucose = torch.tensor(glucose, dtype=torch.float32)

print("Participant:", data["header"]["patient_id"])
print("Total readings:", len(glucose))


# ============================================================
# Load pretrained OpenTSLM
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("\nLoading model on:", device)

model = OpenTSLM.load_pretrained(
    REPO_ID,
    device=device,
)

EOS = model.get_eos_token()


# ============================================================
# Build CGM alignment examples
# ============================================================

examples = []

for start in range(
    0,
    len(glucose) - WINDOW_SIZE + 1,
    STRIDE,
):
    raw_window = glucose[start:start + WINDOW_SIZE]

    mean = raw_window.mean()
    std = raw_window.std()

    # Same general normalization style used by TSQA.
    normalized = (
        raw_window - mean
    ) / (std + 1e-8)

    max_glucose = raw_window.max().item()

    for threshold in THRESHOLDS:

        answer_yes = max_glucose > threshold

        question = f"""
The following time series contains continuous glucose monitor
measurements sampled approximately every 5 minutes.

Did the glucose concentration ever exceed {threshold} mg/dL
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

        answer = "(a)" if answer_yes else "(b)"

        examples.append(
            {
                "pre_prompt": question,

                "time_series_text": [
                    description
                ],

                "time_series": [
                    normalized.clone()
                ],

                "post_prompt":
                    "Predict the CGM Threshold Answer:",

                "answer":
                    answer + EOS,

                # Metadata used only for our evaluation/debugging.
                "_answer_clean":
                    answer,

                "_threshold":
                    threshold,

                "_max_glucose":
                    max_glucose,

                "_start":
                    start,
            }
        )


print("\nGenerated examples:", len(examples))


# ============================================================
# Balance YES / NO examples
# ============================================================

positive = [
    x for x in examples
    if x["_answer_clean"] == "(a)"
]

negative = [
    x for x in examples
    if x["_answer_clean"] == "(b)"
]

print("YES examples:", len(positive))
print("NO examples:", len(negative))


random.shuffle(positive)
random.shuffle(negative)

half = NUM_TRAIN_EXAMPLES // 2

train_examples = (
    positive[:half]
    + negative[:half]
)

random.shuffle(train_examples)

print(
    "Training examples:",
    len(train_examples)
)

print(
    "Training YES:",
    sum(
        x["_answer_clean"] == "(a)"
        for x in train_examples
    )
)

print(
    "Training NO:",
    sum(
        x["_answer_clean"] == "(b)"
        for x in train_examples
    )
)


# ============================================================
# Convert our metadata-rich example into the format OpenTSLM
# expects.
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
# Evaluate all training examples
# ============================================================

def evaluate(examples_to_evaluate, label):
    correct = 0

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    for i, example in enumerate(examples_to_evaluate):

        prediction = model.generate(
            [model_sample(example)],
            max_new_tokens=10,
        )[0].strip()

        gold = example["_answer_clean"]

        is_correct = prediction.startswith(gold)

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

    accuracy = (
        correct / len(examples_to_evaluate)
    )

    print(
        f"\nAccuracy: "
        f"{correct}/{len(examples_to_evaluate)} "
        f"({accuracy * 100:.1f}%)"
    )

    return accuracy


# ============================================================
# Predictions BEFORE training
# ============================================================

model.encoder.eval()
model.projector.eval()
model.llm.eval()

before_accuracy = evaluate(
    train_examples,
    "BEFORE TRAINING",
)


# ============================================================
# Set up optimizer
# ============================================================

# Llama is frozen by the OpenTSLM soft-prompt implementation.
#
# We train:
#   1. the time-series encoder
#   2. the projector mapping time-series representations into
#      the LLM embedding space.

optimizer = AdamW(
    [
        {
            "params":
                model.encoder.parameters(),

            "lr":
                ENCODER_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },

        {
            "params":
                model.projector.parameters(),

            "lr":
                PROJECTOR_LR,

            "weight_decay":
                WEIGHT_DECAY,
        },
    ]
)


# Keep the frozen language model deterministic.
model.llm.eval()

model.encoder.train()
model.projector.train()


# ============================================================
# Tiny overfit training loop
# ============================================================

print("\n" + "=" * 60)
print("TRAINING")
print("=" * 60)

for step in range(TRAINING_STEPS):

    # Cycle repeatedly through the same 8 examples.
    example = train_examples[
        step % len(train_examples)
    ]

    batch = [
        model_sample(example)
    ]

    optimizer.zero_grad()

    loss = model.compute_loss(batch)

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        list(model.encoder.parameters())
        + list(model.projector.parameters()),
        max_norm=1.0,
    )

    optimizer.step()

    print(
        f"Step {step + 1:02d}/"
        f"{TRAINING_STEPS} | "
        f"gold={example['_answer_clean']} | "
        f"max={example['_max_glucose']:.1f} | "
        f"loss={loss.item():.4f}"
    )


# ============================================================
# Predictions AFTER training
# ============================================================

model.encoder.eval()
model.projector.eval()
model.llm.eval()

after_accuracy = evaluate(
    train_examples,
    "AFTER TRAINING",
)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(
    f"Before training accuracy: "
    f"{before_accuracy * 100:.1f}%"
)

print(
    f"After training accuracy:  "
    f"{after_accuracy * 100:.1f}%"
)


# ============================================================
# Save trained checkpoint
# ============================================================

OUTPUT_PATH = (
    "cgm_alignment_smoke_checkpoint.pt"
)

model.store_to_file(
    OUTPUT_PATH
)

print(
    "\nSaved checkpoint:",
    OUTPUT_PATH
)