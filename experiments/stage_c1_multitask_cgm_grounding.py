"""
STAGE C.1 — MULTI-TASK CGM SENSOR GROUNDING

Goal
----
Test whether the repaired OpenTSLM sensor initialization discovered in Stage A
generalizes beyond the single >180 mg/dL task.

Starting architecture (FRESH; no Stage-A or B.1 CGM checkpoint loaded):
    Stage-5 patch + T1-T4          frozen
    TSQA T5-T6                    transplanted, trainable
    Stage-5 projector             trainable
    Stage-5 LLM + ECG LoRA        frozen

Tasks
-----
    above_160
    above_180
    sustained_above_180_15m
    rise_50_30m
    fall_50_30m
    tir_70_180_above_90

Design
------
- Fixed global CGM normalization from training participants.
- 128-reading windows.
- 100 YES + 100 NO training examples per task (when available), matching
  the earlier TSQA six-task feasibility experiment.
- All balanced validation examples are used for each task.
- Participant-separated train/validation split.
- Test participants remain sealed.
- Task-specific prompt, but text is invariant within a task and contains
  no per-window statistics or label-derived metadata.
- True pairwise next-token loss on "(a)" versus "(b)".
- Single-example optimizer updates to preserve the successful Stage-A
  training formulation.
- Best checkpoint selected ONLY by REAL validation macro balanced accuracy;
  minimum per-task REAL BA is used only as a tie-breaker.
- ZERO and opposite-label SHUFFLED signals are diagnostics only and never
  select the checkpoint.

Why 8 epochs?
-------------
With 6 tasks × 200 examples = 1,200 single-example updates per epoch,
8 epochs is about 9,600 optimizer updates, close to the ~9,240 updates
used in the matched 20-epoch Stage-A.3 run.

IMPORTANT HYPERPARAMETER NOTE
-----------------------------
The exact Stage-A.3 learning-rate constants were not present in the saved
console logs/report used to reconstruct this script. The defaults below are
reasonable, but for the cleanest methodological continuity you should compare
LR_T56, LR_PROJECTOR, WEIGHT_DECAY, CLIP_T56, and CLIP_PROJECTOR against:

    experiments/stage_a3_tsqa_repair_comparison.py

before the definitive run.

PowerShell:
    Select-String -Path .\experiments\stage_a3_tsqa_repair_comparison.py `
      -Pattern "LR_|WEIGHT_DECAY|CLIP"

If those values differ, edit the constants below to match Stage A.3 exactly.
"""

import copy
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cgm_multitask import build_multitask_examples
from opentslm import OpenTSLM

DATA_DIR = Path(r"C:\tslm\cgm\data")
MANIFEST_PATH = ROOT / "artifacts" / "cgm_alignment_manifest.json"
OUTPUT_CHECKPOINT_PATH = ROOT / "checkpoints" / "cgm_stage_c1_multitask_sensor_best.pt"
RESULTS_PATH = ROOT / "artifacts" / "cgm_stage_c1_multitask_sensor.json"

SEED = 42
STAGE5_REPO = "OpenTSLM/llama-3.2-1b-ecg-sp"
TSQA_REPO = "OpenTSLM/llama-3.2-1b-tsqa-sp"

WINDOW_SIZE = 128
STRIDE = 128

TASK_IDS = [
    "above_160",
    "above_180",
    "sustained_above_180_15m",
    "rise_50_30m",
    "fall_50_30m",
    "tir_70_180_above_90",
]

TRAIN_PER_CLASS_PER_TASK = 100
EPOCHS = 8
EVAL_EPOCHS = {0, 1, 2, 4, 6, 8}
EVAL_BATCH_SIZE = 16

# IMPORTANT: match these to Stage A.3 if its values differ.
LR_T56 = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
CLIP_T56 = 1.0
CLIP_PROJECTOR = 1.0

TSQA_REFERENCE_BA = {
    "above_160": 0.806,
    "above_180": 0.837,
    "sustained_above_180_15m": 0.892,
    "rise_50_30m": 0.707,
    "fall_50_30m": 0.694,
    "tir_70_180_above_90": 0.500,
}

TASK_QUESTIONS = {
    "above_160": "Did glucose exceed 160 mg/dL during this segment?",
    "above_180": "Did glucose exceed 180 mg/dL during this segment?",
    "sustained_above_180_15m": (
        "Was glucose above 180 mg/dL for at least "
        "15 consecutive minutes during this segment?"
    ),
    "rise_50_30m": (
        "Did glucose rise by more than 50 mg/dL over any "
        "approximately 30-minute interval during this segment?"
    ),
    "fall_50_30m": (
        "Did glucose fall by more than 50 mg/dL over any "
        "approximately 30-minute interval during this segment?"
    ),
    "tir_70_180_above_90": (
        "Was more than 90% of this segment within 70 to 180 mg/dL?"
    ),
}

random.seed(SEED)
torch.manual_seed(SEED)

with MANIFEST_PATH.open("r", encoding="utf-8") as f:
    manifest = json.load(f)

train_mean = float(manifest["normalization"]["mean"])
train_std = float(manifest["normalization"]["std"])

validation_manifest = manifest.get("validation") or manifest.get("val")
if validation_manifest is None:
    raise RuntimeError("Validation split missing from manifest.")

train_files = [DATA_DIR / item["relative_path"] for item in manifest["train"]]
validation_files = [DATA_DIR / item["relative_path"] for item in validation_manifest]

print("Train participants:", len(train_files))
print("Validation participants:", len(validation_files))
print("TEST participants sealed:", len(manifest["test"]))
print(f"Fixed normalization mean: {train_mean:.3f}")
print(f"Fixed normalization SD: {train_std:.3f}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("\nDevice:", device)

print("\n" + "=" * 110)
print("LOADING TSQA DONOR T5/T6")
print("=" * 110)

tsqa_donor = OpenTSLM.load_pretrained(
    TSQA_REPO,
    device=device,
    enable_lora=True,
)

tsqa_layers = tsqa_donor.encoder.encoder.layers
if len(tsqa_layers) != 6:
    raise RuntimeError(f"Expected six TSQA encoder layers; found {len(tsqa_layers)}.")

tsqa_t5_state = {
    k: v.detach().cpu().clone()
    for k, v in tsqa_layers[4].state_dict().items()
}
tsqa_t6_state = {
    k: v.detach().cpu().clone()
    for k, v in tsqa_layers[5].state_dict().items()
}

print("✓ Captured TSQA T5.")
print("✓ Captured TSQA T6.")
print("✓ TSQA projector will NOT be used.")
print("✓ TSQA LLM/LoRA will NOT be used.")

del tsqa_donor
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("\n" + "=" * 110)
print("LOADING FRESH STAGE-5 + TRANSPLANTING TSQA T5/T6")
print("=" * 110)

model = OpenTSLM.load_pretrained(
    STAGE5_REPO,
    device=device,
    enable_lora=True,
)

EOS_TOKEN = model.get_eos_token()
layers = model.encoder.encoder.layers

if len(layers) != 6:
    raise RuntimeError(f"Expected six Stage-5 encoder layers; found {len(layers)}.")

layers[4].load_state_dict(tsqa_t5_state, strict=True)
layers[5].load_state_dict(tsqa_t6_state, strict=True)

print("✓ Stage-5 patch/T1-T4 retained.")
print("✓ TSQA T5/T6 transplanted.")
print("✓ Stage-5 projector retained.")
print("✓ Stage-5 LLM + ECG LoRA retained.")
print("✓ NO Stage-A or Stage-B CGM checkpoint loaded.")

for p in model.encoder.parameters():
    p.requires_grad = False
for p in model.projector.parameters():
    p.requires_grad = False
for p in model.llm.parameters():
    p.requires_grad = False

for p in layers[4].parameters():
    p.requires_grad = True
for p in layers[5].parameters():
    p.requires_grad = True
for p in model.projector.parameters():
    p.requires_grad = True

t56_parameters = list(layers[4].parameters()) + list(layers[5].parameters())
projector_parameters = list(model.projector.parameters())

early_trainable = sum(
    p.numel()
    for layer in layers[:4]
    for p in layer.parameters()
    if p.requires_grad
)
t56_trainable = sum(p.numel() for p in t56_parameters if p.requires_grad)
projector_trainable = sum(p.numel() for p in projector_parameters if p.requires_grad)
llm_trainable = sum(p.numel() for p in model.llm.parameters() if p.requires_grad)

print("\n" + "=" * 110)
print("TRAINABLE PARAMETER AUDIT")
print("=" * 110)
print("Early encoder:", f"{early_trainable:,}")
print("T5/T6:", f"{t56_trainable:,}")
print("Projector:", f"{projector_trainable:,}")
print("Stage-5 LLM / LoRA:", f"{llm_trainable:,}")

if early_trainable != 0:
    raise RuntimeError("T1-T4 must remain frozen.")
if t56_trainable == 0:
    raise RuntimeError("T5/T6 unexpectedly frozen.")
if projector_trainable == 0:
    raise RuntimeError("Projector unexpectedly frozen.")
if llm_trainable != 0:
    raise RuntimeError("Stage-5 LLM/LoRA must remain frozen.")

a_ids = model.tokenizer("(a)", add_special_tokens=False).input_ids
b_ids = model.tokenizer("(b)", add_special_tokens=False).input_ids
if not a_ids or not b_ids:
    raise RuntimeError("Could not tokenize (a)/(b).")

A_TOKEN_ID = int(a_ids[0])
B_TOKEN_ID = int(b_ids[0])

print("\nDecision tokens:")
print(" (a):", a_ids, "->", A_TOKEN_ID)
print(" (b):", b_ids, "->", B_TOKEN_ID)

COMMON_PRE_PROMPT = (
    "Analyze the continuous glucose monitor signal and answer the factual question.\n"
)

COMMON_TS_TEXT = (
    "This is a continuous glucose monitor (CGM) signal measured in mg/dL. "
    f"It contains {WINDOW_SIZE} readings sampled approximately every 5 minutes. "
    "The numeric signal uses the same fixed normalization for every segment: "
    f"training-set mean {train_mean:.3f} mg/dL and training-set standard deviation "
    f"{train_std:.3f} mg/dL."
)

def task_post_prompt(task_id):
    return (
        f"\nQuestion: {TASK_QUESTIONS[task_id]}\n"
        "(a) Yes\n"
        "(b) No\n"
        "Answer:"
    )

def build_task_pool(files, task_id):
    base_pool = build_multitask_examples(
        files=files,
        normalization_mean=train_mean,
        normalization_std=train_std,
        eos_token=EOS_TOKEN,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        task_ids=[task_id],
    )

    converted = []

    for base in base_pool:
        clean_answer = base["_answer_clean"]
        if clean_answer not in {"(a)", "(b)"}:
            raise RuntimeError(
                f"Unexpected answer {clean_answer!r} for task {task_id}."
            )

        signal = base["time_series"][0]
        if not torch.is_tensor(signal):
            signal = torch.tensor(signal, dtype=torch.float32)

        converted.append(
            {
                "pre_prompt": COMMON_PRE_PROMPT,
                "time_series": [signal.detach().clone().float()],
                "time_series_text": [COMMON_TS_TEXT],
                "post_prompt": task_post_prompt(task_id),
                "answer": clean_answer + EOS_TOKEN,
                "_answer_clean": clean_answer,
                "_is_yes": clean_answer == "(a)",
                "_task_id": task_id,
                "_participant_id": base.get("_participant_id"),
                "_window_start": base.get("_window_start"),
                "_window_max": base.get("_window_max"),
                "_donor_is_yes": None,
            }
        )

    return converted

print("\nBuilding task pools...")

train_pools = {}
val_pools = {}

for task_id in TASK_IDS:
    print(" ", task_id)
    train_pools[task_id] = build_task_pool(train_files, task_id)
    val_pools[task_id] = build_task_pool(validation_files, task_id)

def split_by_class(examples):
    yes = [x for x in examples if x["_is_yes"]]
    no = [x for x in examples if not x["_is_yes"]]
    return yes, no

def participant_round_robin_sample(examples, n, seed):
    rng = random.Random(seed)
    grouped = defaultdict(list)

    for example in examples:
        participant = example.get("_participant_id")
        if participant is None:
            raise RuntimeError("Participant ID missing.")
        grouped[participant].append(example)

    participants = list(grouped.keys())
    rng.shuffle(participants)

    for participant in participants:
        rng.shuffle(grouped[participant])

    selected = []

    while len(selected) < n:
        progress = False

        for participant in participants:
            if grouped[participant]:
                selected.append(grouped[participant].pop())
                progress = True

                if len(selected) == n:
                    break

        if not progress:
            raise RuntimeError(f"Could not obtain requested sample n={n}.")

    return selected

selected_train = {}
selected_val = {}

print("\n" + "=" * 110)
print("TASK POOL AUDIT")
print("=" * 110)

for task_index, task_id in enumerate(TASK_IDS):
    train_yes_pool, train_no_pool = split_by_class(train_pools[task_id])
    val_yes_pool, val_no_pool = split_by_class(val_pools[task_id])

    train_n = min(
        TRAIN_PER_CLASS_PER_TASK,
        len(train_yes_pool),
        len(train_no_pool),
    )
    val_n = min(len(val_yes_pool), len(val_no_pool))

    if train_n == 0 or val_n == 0:
        raise RuntimeError(f"Task {task_id} lacks a balanced pool.")

    train_yes = participant_round_robin_sample(
        train_yes_pool,
        train_n,
        SEED + 1000 + task_index * 10 + 1,
    )
    train_no = participant_round_robin_sample(
        train_no_pool,
        train_n,
        SEED + 1000 + task_index * 10 + 2,
    )
    val_yes = participant_round_robin_sample(
        val_yes_pool,
        val_n,
        SEED + 2000 + task_index * 10 + 1,
    )
    val_no = participant_round_robin_sample(
        val_no_pool,
        val_n,
        SEED + 2000 + task_index * 10 + 2,
    )

    selected_train[task_id] = {"yes": train_yes, "no": train_no}
    selected_val[task_id] = {"yes": val_yes, "no": val_no}

    print(
        f"{task_id:32s} | "
        f"TRAIN pool Y/N={len(train_yes_pool):4d}/{len(train_no_pool):4d} "
        f"-> selected {train_n:3d}/class | "
        f"VAL pool Y/N={len(val_yes_pool):3d}/{len(val_no_pool):3d} "
        f"-> selected {val_n:3d}/class"
    )

all_train_examples = [
    example
    for task_id in TASK_IDS
    for class_name in ("yes", "no")
    for example in selected_train[task_id][class_name]
]

all_val_examples = [
    example
    for task_id in TASK_IDS
    for class_name in ("yes", "no")
    for example in selected_val[task_id][class_name]
]

train_participants = {x["_participant_id"] for x in all_train_examples}
val_participants = {x["_participant_id"] for x in all_val_examples}
overlap = train_participants & val_participants

print("\nParticipant split audit:")
print(" train:", len(train_participants))
print(" validation:", len(val_participants))
print(" overlap:", len(overlap))

if overlap:
    raise RuntimeError(f"TRAIN/VALIDATION leakage: {sorted(overlap)}")

for task_id in TASK_IDS:
    task_examples = (
        selected_train[task_id]["yes"]
        + selected_train[task_id]["no"]
        + selected_val[task_id]["yes"]
        + selected_val[task_id]["no"]
    )

    unique_text = {
        (
            x["pre_prompt"],
            x["time_series_text"][0],
            x["post_prompt"],
        )
        for x in task_examples
    }

    if len(unique_text) != 1:
        raise RuntimeError(
            f"Per-example text variation detected for task {task_id}."
        )

print("✓ Text is invariant within every task.")
print("✓ No per-window statistics are included in text.")

def interleave(yes_examples, no_examples):
    if len(yes_examples) != len(no_examples):
        raise RuntimeError("Expected balanced yes/no lists.")

    output = []
    for yes_example, no_example in zip(yes_examples, no_examples):
        output.append(yes_example)
        output.append(no_example)
    return output

def clone_with_signal(example, signal, donor_is_yes=None):
    cloned = dict(example)
    cloned["time_series"] = [signal.detach().clone().float()]
    cloned["_donor_is_yes"] = donor_is_yes
    return cloned

def make_zero(examples):
    return [
        clone_with_signal(
            example,
            torch.zeros_like(example["time_series"][0]),
        )
        for example in examples
    ]

def make_opposite_shuffle(yes_examples, no_examples, seed):
    rng = random.Random(seed)
    yes_donors = list(yes_examples)
    no_donors = list(no_examples)
    rng.shuffle(yes_donors)
    rng.shuffle(no_donors)

    originals = interleave(yes_examples, no_examples)
    yes_index = 0
    no_index = 0
    output = []

    for original in originals:
        if original["_is_yes"]:
            donor = no_donors[no_index]
            no_index += 1
        else:
            donor = yes_donors[yes_index]
            yes_index += 1

        output.append(
            clone_with_signal(
                original,
                donor["time_series"][0],
                donor_is_yes=donor["_is_yes"],
            )
        )

    return output

train_real = {}
val_real = {}
val_zero = {}
val_shuffle = {}

for task_index, task_id in enumerate(TASK_IDS):
    train_real[task_id] = interleave(
        selected_train[task_id]["yes"],
        selected_train[task_id]["no"],
    )
    val_real[task_id] = interleave(
        selected_val[task_id]["yes"],
        selected_val[task_id]["no"],
    )
    val_zero[task_id] = make_zero(val_real[task_id])
    val_shuffle[task_id] = make_opposite_shuffle(
        selected_val[task_id]["yes"],
        selected_val[task_id]["no"],
        seed=SEED + 9000 + task_index,
    )

total_train_examples = sum(len(train_real[t]) for t in TASK_IDS)
total_val_examples = sum(len(val_real[t]) for t in TASK_IDS)

print("\nSelected multi-task dataset:")
print(" TRAIN total:", total_train_examples)
print(" VAL total:", total_val_examples)
print(" Planned optimizer updates:", total_train_examples * EPOCHS)

model.encoder.eval()
model.projector.eval()
model.llm.eval()

def pairwise_loss_for_example(example):
    inputs_embeds, attention_mask = model.pad_and_apply_batch([example])

    outputs = model.llm(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        return_dict=True,
    )

    last_prompt_index = int(attention_mask[0].sum().item()) - 1
    next_logits = outputs.logits[0, last_prompt_index, :]

    pair_logits = torch.stack(
        [next_logits[A_TOKEN_ID], next_logits[B_TOKEN_ID]]
    )

    target = torch.tensor(
        [0 if example["_is_yes"] else 1],
        dtype=torch.long,
        device=device,
    )

    return F.cross_entropy(
        pair_logits.float().unsqueeze(0),
        target,
    )

def balanced_accuracy(predictions, labels):
    yes_total = yes_correct = no_total = no_correct = 0

    for prediction, label in zip(predictions, labels):
        if label:
            yes_total += 1
            yes_correct += int(prediction is True)
        else:
            no_total += 1
            no_correct += int(prediction is False)

    if yes_total == 0 or no_total == 0:
        raise RuntimeError("Balanced accuracy requires both classes.")

    yes_acc = yes_correct / yes_total
    no_acc = no_correct / no_total

    return {
        "balanced_accuracy": 0.5 * (yes_acc + no_acc),
        "yes_accuracy": yes_acc,
        "no_accuracy": no_acc,
    }

def evaluate_examples(examples):
    predictions = []
    original_labels = []
    donor_labels = []

    with torch.no_grad():
        for start in range(0, len(examples), EVAL_BATCH_SIZE):
            batch = examples[start:start + EVAL_BATCH_SIZE]
            inputs_embeds, attention_mask = model.pad_and_apply_batch(batch)

            outputs = model.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
            )

            for index, example in enumerate(batch):
                last_prompt_index = int(attention_mask[index].sum().item()) - 1
                next_logits = outputs.logits[index, last_prompt_index, :]

                prediction = bool(
                    next_logits[A_TOKEN_ID] >= next_logits[B_TOKEN_ID]
                )

                predictions.append(prediction)
                original_labels.append(example["_is_yes"])
                donor_labels.append(example.get("_donor_is_yes"))

    original_metrics = balanced_accuracy(predictions, original_labels)

    donor_metrics = None
    if all(label is not None for label in donor_labels):
        donor_metrics = balanced_accuracy(predictions, donor_labels)

    return {
        "n": len(examples),
        "ba_vs_original": original_metrics["balanced_accuracy"],
        "yes_accuracy": original_metrics["yes_accuracy"],
        "no_accuracy": original_metrics["no_accuracy"],
        "ba_vs_donor": (
            None
            if donor_metrics is None
            else donor_metrics["balanced_accuracy"]
        ),
        "predicted_yes": sum(x is True for x in predictions),
        "predicted_no": sum(x is False for x in predictions),
    }

def evaluate_all_tasks():
    task_metrics = {}

    for task_id in TASK_IDS:
        task_metrics[task_id] = {
            "train_real": evaluate_examples(train_real[task_id]),
            "val_real": evaluate_examples(val_real[task_id]),
            "val_zero": evaluate_examples(val_zero[task_id]),
            "val_shuffle": evaluate_examples(val_shuffle[task_id]),
        }

    real_bas = [
        task_metrics[t]["val_real"]["ba_vs_original"]
        for t in TASK_IDS
    ]
    zero_bas = [
        task_metrics[t]["val_zero"]["ba_vs_original"]
        for t in TASK_IDS
    ]
    shuffle_original_bas = [
        task_metrics[t]["val_shuffle"]["ba_vs_original"]
        for t in TASK_IDS
    ]
    shuffle_donor_bas = [
        task_metrics[t]["val_shuffle"]["ba_vs_donor"]
        for t in TASK_IDS
    ]

    return {
        "tasks": task_metrics,
        "summary": {
            "macro_val_real_ba": sum(real_bas) / len(real_bas),
            "macro_val_zero_ba": sum(zero_bas) / len(zero_bas),
            "macro_val_shuffle_original_ba": (
                sum(shuffle_original_bas) / len(shuffle_original_bas)
            ),
            "macro_val_shuffle_donor_ba": (
                sum(shuffle_donor_bas) / len(shuffle_donor_bas)
            ),
            "min_val_real_ba": min(real_bas),
            "tasks_above_60": sum(x >= 0.60 for x in real_bas),
            "tasks_above_chance": sum(x > 0.50 for x in real_bas),
        },
    }

def print_metrics(metrics, heading):
    print("\n" + "=" * 132)
    print(heading)
    print("=" * 132)
    print(
        f"{'Task':32s} | {'Train':>7s} | {'Val real':>8s} | "
        f"{'Zero':>7s} | {'Shuf-O':>7s} | {'Shuf-D':>7s} | {'TSQA ref':>8s}"
    )
    print("-" * 132)

    for task_id in TASK_IDS:
        row = metrics["tasks"][task_id]
        print(
            f"{task_id:32s} | "
            f"{100 * row['train_real']['ba_vs_original']:6.1f}% | "
            f"{100 * row['val_real']['ba_vs_original']:7.1f}% | "
            f"{100 * row['val_zero']['ba_vs_original']:6.1f}% | "
            f"{100 * row['val_shuffle']['ba_vs_original']:6.1f}% | "
            f"{100 * row['val_shuffle']['ba_vs_donor']:6.1f}% | "
            f"{100 * TSQA_REFERENCE_BA[task_id]:7.1f}%"
        )

    s = metrics["summary"]
    print("-" * 132)
    print(
        "MACRO "
        f"| REAL={100 * s['macro_val_real_ba']:.1f}% "
        f"| ZERO={100 * s['macro_val_zero_ba']:.1f}% "
        f"| SHUF-O={100 * s['macro_val_shuffle_original_ba']:.1f}% "
        f"| SHUF-D={100 * s['macro_val_shuffle_donor_ba']:.1f}% "
        f"| MIN TASK={100 * s['min_val_real_ba']:.1f}% "
        f"| >60%={s['tasks_above_60']}/{len(TASK_IDS)}"
    )

optimizer = torch.optim.AdamW(
    [
        {
            "params": t56_parameters,
            "lr": LR_T56,
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": projector_parameters,
            "lr": LR_PROJECTOR,
            "weight_decay": WEIGHT_DECAY,
        },
    ]
)

print("\nOptimizer configuration:")
print(" LR_T56:", LR_T56)
print(" LR_PROJECTOR:", LR_PROJECTOR)
print(" WEIGHT_DECAY:", WEIGHT_DECAY)
print(" CLIP_T56:", CLIP_T56)
print(" CLIP_PROJECTOR:", CLIP_PROJECTOR)

def snapshot_sensor_state():
    return {
        "t5": {
            k: v.detach().cpu().clone()
            for k, v in layers[4].state_dict().items()
        },
        "t6": {
            k: v.detach().cpu().clone()
            for k, v in layers[5].state_dict().items()
        },
        "projector": {
            k: v.detach().cpu().clone()
            for k, v in model.projector.state_dict().items()
        },
    }

def restore_sensor_state(state):
    layers[4].load_state_dict(state["t5"], strict=True)
    layers[5].load_state_dict(state["t6"], strict=True)
    model.projector.load_state_dict(state["projector"], strict=True)

initial_metrics = evaluate_all_tasks()
print_metrics(
    initial_metrics,
    "STAGE C.1 — BEFORE MULTI-TASK CGM TRAINING",
)

best_epoch = 0
best_metrics = copy.deepcopy(initial_metrics)
best_sensor_state = snapshot_sensor_state()
best_optimizer_state = copy.deepcopy(optimizer.state_dict())
best_macro = initial_metrics["summary"]["macro_val_real_ba"]
best_min_task = initial_metrics["summary"]["min_val_real_ba"]

history = [
    {
        "epoch": 0,
        "mean_loss": None,
        "metrics": initial_metrics,
    }
]

print("\n" + "=" * 132)
print("STAGE C.1 — MULTI-TASK SENSOR TRAINING")
print("=" * 132)

for epoch in range(1, EPOCHS + 1):
    epoch_examples = []
    for task_id in TASK_IDS:
        epoch_examples.extend(train_real[task_id])

    rng = random.Random(SEED + 10000 + epoch)
    rng.shuffle(epoch_examples)

    losses = []

    for example in epoch_examples:
        optimizer.zero_grad(set_to_none=True)
        loss = pairwise_loss_for_example(example)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(t56_parameters, CLIP_T56)
        torch.nn.utils.clip_grad_norm_(projector_parameters, CLIP_PROJECTOR)

        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    mean_loss = sum(losses) / len(losses)

    print(
        f"Finished epoch {epoch:2d} | "
        f"mean pairwise loss={mean_loss:.4f}"
    )

    if epoch in EVAL_EPOCHS:
        metrics = evaluate_all_tasks()
        print_metrics(metrics, f"STAGE C.1 — EPOCH {epoch}")

        history.append(
            {
                "epoch": epoch,
                "mean_loss": mean_loss,
                "metrics": metrics,
            }
        )

        macro_real = metrics["summary"]["macro_val_real_ba"]
        min_task_real = metrics["summary"]["min_val_real_ba"]

        if (macro_real, min_task_real) > (best_macro, best_min_task):
            best_epoch = epoch
            best_macro = macro_real
            best_min_task = min_task_real
            best_metrics = copy.deepcopy(metrics)
            best_sensor_state = snapshot_sensor_state()
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            print("✓ NEW BEST REAL VALIDATION STATE")

restore_sensor_state(best_sensor_state)

print_metrics(
    best_metrics,
    f"STAGE C.1 — BEST STATE (EPOCH {best_epoch})",
)

summary = best_metrics["summary"]
real_minus_zero = (
    summary["macro_val_real_ba"]
    - summary["macro_val_zero_ba"]
)
real_minus_shuffle = (
    summary["macro_val_real_ba"]
    - summary["macro_val_shuffle_original_ba"]
)

print("\n" + "=" * 132)
print("STAGE C.1 INTERPRETATION")
print("=" * 132)
print("Best epoch:", best_epoch)
print("Macro REAL BA:", f"{100 * summary['macro_val_real_ba']:.1f}%")
print("Macro ZERO BA:", f"{100 * summary['macro_val_zero_ba']:.1f}%")
print(
    "Macro SHUFFLED-original BA:",
    f"{100 * summary['macro_val_shuffle_original_ba']:.1f}%",
)
print("Macro REAL - ZERO:", f"{100 * real_minus_zero:+.1f} pp")
print(
    "Macro REAL - SHUFFLED-original:",
    f"{100 * real_minus_shuffle:+.1f} pp",
)
print(
    "Tasks >=60% REAL BA:",
    f"{summary['tasks_above_60']}/{len(TASK_IDS)}",
)

OUTPUT_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "experiment": "cgm_stage_c1_multitask_sensor_grounding",
        "base_repo": STAGE5_REPO,
        "donor_repo": TSQA_REPO,
        "architecture": {
            "patch_t1_t4": "stage5_frozen",
            "t5_t6_initialization": "tsqa_trainable",
            "projector": "stage5_trainable",
            "llm": "stage5_frozen",
            "lora": "stage5_frozen",
        },
        "task_ids": TASK_IDS,
        "best_epoch": best_epoch,
        "best_macro_val_real_ba": best_macro,
        "best_min_task_val_real_ba": best_min_task,
        "sensor_state": best_sensor_state,
        "optimizer_state": best_optimizer_state,
        "normalization": {
            "type": "fixed_global",
            "mean": train_mean,
            "std": train_std,
        },
        "training": {
            "train_per_class_per_task": TRAIN_PER_CLASS_PER_TASK,
            "epochs": EPOCHS,
            "lr_t56": LR_T56,
            "lr_projector": LR_PROJECTOR,
            "weight_decay": WEIGHT_DECAY,
            "clip_t56": CLIP_T56,
            "clip_projector": CLIP_PROJECTOR,
            "single_example_updates": True,
            "pairwise_objective": True,
        },
        "best_metrics": best_metrics,
    },
    OUTPUT_CHECKPOINT_PATH,
)

print("\nSaved Stage-C.1 checkpoint:")
print(OUTPUT_CHECKPOINT_PATH)

results = {
    "experiment": "cgm_stage_c1_multitask_sensor_grounding",
    "base_repo": STAGE5_REPO,
    "donor_repo": TSQA_REPO,
    "tasks": TASK_IDS,
    "test_participants_sealed": len(manifest["test"]),
    "normalization": {
        "mean": train_mean,
        "std": train_std,
    },
    "training": {
        "train_per_class_per_task": TRAIN_PER_CLASS_PER_TASK,
        "epochs": EPOCHS,
        "planned_updates": total_train_examples * EPOCHS,
        "lr_t56": LR_T56,
        "lr_projector": LR_PROJECTOR,
        "weight_decay": WEIGHT_DECAY,
        "clip_t56": CLIP_T56,
        "clip_projector": CLIP_PROJECTOR,
    },
    "initial_metrics": initial_metrics,
    "history": history,
    "best_epoch": best_epoch,
    "best_metrics": best_metrics,
    "tsqa_reference_ba": TSQA_REFERENCE_BA,
}

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

with RESULTS_PATH.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nSaved diagnostic:")
print(RESULTS_PATH)
print("\nTEST participants remain sealed.")

del optimizer
del model
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()
