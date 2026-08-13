"""
Diagnose Stage-5 answer-token behavior.

Question
--------
The Stage-5 full-adaptation model reached low teacher-forced
training loss (~0.30), but model.generate() predicts "(a)"
for every training and validation example.

This script asks:

    At the token position where "(a)" and "(b)" first differ,
    does the model internally prefer the correct class?

We:

1. Load the trained Stage-5 full-adaptation checkpoint.
2. Reconstruct the exact balanced seed-42 training set.
3. Inspect how the tokenizer encodes the actual answer strings.
4. Find the first token where the "(a)" and "(b)" answers differ.
5. For balanced samples from every task, measure:
       P(a-token) vs P(b-token)
   at that exact decision point.
6. Also score the full "(a)" and "(b)" candidate answers.

Interpretation
--------------
If gold-NO examples internally favor b, but generate() returns "(a)":
    -> decoding / generation mismatch.

If gold-NO examples internally favor a:
    -> the model/objective itself has collapsed toward "(a)".

TEST participants remain sealed.
"""

import json
import random
import sys

from pathlib import Path

import torch
import torch.nn.functional as F


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
)

from opentslm import OpenTSLM


# ============================================================
# Configuration
# ============================================================

SEED = 42

STAGE5_REPO_ID = (
    "OpenTSLM/llama-3.2-1b-ecg-sp"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "cgm_alignment_stage5_full_adaptation_seed42.pt"
)

RESULTS_PATH = (
    ROOT
    / "artifacts"
    / "stage5_answer_token_diagnostic.json"
)

DATA_DIR = Path(
    r"C:\tslm\cgm\data"
)

MANIFEST_PATH = (
    ROOT
    / "artifacts"
    / "cgm_alignment_manifest.json"
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


# Small diagnostic:
# 10 YES + 10 NO from each task
# = 120 examples total.
SAMPLES_PER_CLASS_PER_TASK = 10


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


print(
    "Train participants:",
    len(train_files),
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
# Load Stage-5 model
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "LOADING STAGE-5 FULL-ADAPTATION MODEL"
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


print(
    "\nLoading trained CGM checkpoint:"
)

print(
    CHECKPOINT_PATH
)


model.load_from_file(
    str(
        CHECKPOINT_PATH
    )
)


model.eval()


print(
    "\n✓ Full-adaptation checkpoint loaded."
)


EOS_TOKEN = (
    model.get_eos_token()
)


# ============================================================
# Reconstruct exact training set
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "RECONSTRUCTING EXACT TRAINING SET"
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
        f"found {len(train_examples)}"
    )


print(
    "\nExact balanced training examples:",
    len(train_examples),
)


# ============================================================
# Determine the EXACT answer strings
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


if len(
    answer_a_values
) != 1:

    raise RuntimeError(
        "Expected exactly one raw answer "
        "string for class (a), found: "
        f"{answer_a_values}"
    )


if len(
    answer_b_values
) != 1:

    raise RuntimeError(
        "Expected exactly one raw answer "
        "string for class (b), found: "
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


print(
    "\nRaw answer A repr:",
    repr(
        ANSWER_A
    ),
)

print(
    "Raw answer B repr:",
    repr(
        ANSWER_B
    ),
)


# ============================================================
# Tokenize answers EXACTLY like compute_loss()
#
# compute_loss() uses tokenizer(...) with its normal defaults,
# so we intentionally do the same here.
# ============================================================

def tokenize_answer(
    answer,
):

    encoded = (
        model.tokenizer(
            answer,
            return_tensors="pt",
            truncation=True,
        )
    )


    return (
        encoded.input_ids[
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


answer_a_tokens = (
    model.tokenizer.convert_ids_to_tokens(
        answer_a_ids.tolist()
    )
)

answer_b_tokens = (
    model.tokenizer.convert_ids_to_tokens(
        answer_b_ids.tolist()
    )
)


print(
    "\n"
    + "=" * 78
)

print(
    "ANSWER TOKENIZATION"
)

print(
    "=" * 78
)


print(
    "\n(a) token IDs:"
)

print(
    answer_a_ids.tolist()
)

print(
    "(a) tokens:"
)

print(
    answer_a_tokens
)


print(
    "\n(b) token IDs:"
)

print(
    answer_b_ids.tolist()
)

print(
    "(b) tokens:"
)

print(
    answer_b_tokens
)


# ============================================================
# Find longest common prefix
# ============================================================

common_prefix_len = 0


max_common = min(
    len(
        answer_a_ids
    ),
    len(
        answer_b_ids
    ),
)


while (
    common_prefix_len
    < max_common
    and answer_a_ids[
        common_prefix_len
    ].item()
    == answer_b_ids[
        common_prefix_len
    ].item()
):

    common_prefix_len += 1


if common_prefix_len >= max_common:

    raise RuntimeError(
        "Could not find a divergent token "
        "between answers A and B."
    )


a_class_token_id = (
    answer_a_ids[
        common_prefix_len
    ].item()
)

b_class_token_id = (
    answer_b_ids[
        common_prefix_len
    ].item()
)


a_class_token = (
    model.tokenizer.convert_ids_to_tokens(
        [
            a_class_token_id
        ]
    )[0]
)

b_class_token = (
    model.tokenizer.convert_ids_to_tokens(
        [
            b_class_token_id
        ]
    )[0]
)


common_prefix_ids = (
    answer_a_ids[
        :common_prefix_len
    ]
)


print(
    "\nCommon answer-prefix length:",
    common_prefix_len,
)


print(
    "Common prefix tokens:"
)

print(
    model.tokenizer.convert_ids_to_tokens(
        common_prefix_ids.tolist()
    )
)


print(
    "\nFIRST CLASS-DISCRIMINATING TOKENS:"
)

print(
    f"(a) token: "
    f"{repr(a_class_token)} "
    f"| id={a_class_token_id}"
)

print(
    f"(b) token: "
    f"{repr(b_class_token)} "
    f"| id={b_class_token_id}"
)


# ============================================================
# Score one complete candidate answer
# ============================================================

def score_candidate_answer(
    example,
    candidate_ids,
):
    """
    Compute teacher-forced log probability of an answer sequence
    conditioned on the multimodal CGM prompt.

    Returns:
        total_logprob
        per_token_logprobs
        logits
        prompt_length
    """

    with torch.no_grad():

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


        prompt_length = (
            prompt_embeds.shape[
                1
            ]
        )


        answer_embeds = (
            model.llm
            .get_input_embeddings()(
                candidate_ids.unsqueeze(
                    0
                )
            )
        )


        answer_embeds = (
            answer_embeds.to(
                prompt_embeds.dtype
            )
        )


        combined_embeds = (
            torch.cat(
                [
                    prompt_embeds,
                    answer_embeds,
                ],
                dim=1,
            )
        )


        answer_mask = (
            torch.ones(
                (
                    1,
                    candidate_ids.numel(),
                ),
                dtype=
                    prompt_mask.dtype,
                device=
                    prompt_mask.device,
            )
        )


        combined_mask = (
            torch.cat(
                [
                    prompt_mask,
                    answer_mask,
                ],
                dim=1,
            )
        )


        outputs = (
            model.llm(
                inputs_embeds=
                    combined_embeds,

                attention_mask=
                    combined_mask,

                return_dict=True,
            )
        )


        logits = (
            outputs.logits[
                0
            ]
            .float()
        )


        # ----------------------------------------------------
        # Causal LM:
        #
        # token 0 of the answer is predicted by the
        # LAST prompt position.
        #
        # token 1 is predicted by answer token 0.
        # etc.
        # ----------------------------------------------------

        prediction_positions = (
            torch.arange(
                prompt_length - 1,
                prompt_length
                + candidate_ids.numel()
                - 1,
                device=device,
            )
        )


        relevant_logits = (
            logits[
                prediction_positions
            ]
        )


        log_probs = (
            F.log_softmax(
                relevant_logits,
                dim=-1,
            )
        )


        token_logprobs = (
            log_probs.gather(
                1,
                candidate_ids.unsqueeze(
                    1
                ),
            )
            .squeeze(
                1
            )
        )


        total_logprob = (
            token_logprobs
            .sum()
            .item()
        )


        return (
            total_logprob,
            token_logprobs,
            logits,
            prompt_length,
        )


# ============================================================
# Analyze one example
# ============================================================

def diagnose_example(
    example,
):

    # Score candidate A.
    (
        score_a,
        token_logprobs_a,
        logits_a,
        prompt_length,
    ) = (
        score_candidate_answer(
            example,
            answer_a_ids,
        )
    )


    # Score candidate B.
    (
        score_b,
        token_logprobs_b,
        _,
        _,
    ) = (
        score_candidate_answer(
            example,
            answer_b_ids,
        )
    )


    # --------------------------------------------------------
    # At the divergence point, both candidate contexts are
    # identical up to the class-decision token.
    #
    # Position that predicts answer token j:
    #
    #     prompt_length + j - 1
    # --------------------------------------------------------

    decision_position = (
        prompt_length
        + common_prefix_len
        - 1
    )


    decision_logits = (
        logits_a[
            decision_position
        ]
    )


    a_logit = (
        decision_logits[
            a_class_token_id
        ].item()
    )

    b_logit = (
        decision_logits[
            b_class_token_id
        ].item()
    )


    pair_logits = (
        torch.tensor(
            [
                a_logit,
                b_logit,
            ],
            dtype=torch.float32,
        )
    )


    pair_probs = (
        torch.softmax(
            pair_logits,
            dim=0,
        )
    )


    p_a = (
        pair_probs[
            0
        ].item()
    )

    p_b = (
        pair_probs[
            1
        ].item()
    )


    token_preference = (
        "(a)"
        if p_a > p_b
        else "(b)"
    )


    full_answer_preference = (
        "(a)"
        if score_a > score_b
        else "(b)"
    )


    return {
        "task_id":
            example[
                "_task_id"
            ],

        "participant_id":
            example.get(
                "_participant_id"
            ),

        "window_start":
            example.get(
                "_window_start"
            ),

        "gold":
            example[
                "_answer_clean"
            ],

        "p_a_given_a_or_b":
            p_a,

        "p_b_given_a_or_b":
            p_b,

        "a_minus_b_logit":
            (
                a_logit
                - b_logit
            ),

        "token_preference":
            token_preference,

        "full_answer_logprob_a":
            score_a,

        "full_answer_logprob_b":
            score_b,

        "full_answer_preference":
            full_answer_preference,

        "a_token_logprobs":
            token_logprobs_a
            .cpu()
            .tolist(),

        "b_token_logprobs":
            token_logprobs_b
            .cpu()
            .tolist(),
    }


# ============================================================
# Select balanced diagnostic subset
# ============================================================

diagnostic_examples = []


print(
    "\n"
    + "=" * 78
)

print(
    "BUILDING BALANCED DIAGNOSTIC SUBSET"
)

print(
    "=" * 78
)


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
        + task_index * 1000
        + 777
    )


    selected_yes = (
        rng.sample(
            yes_examples,
            SAMPLES_PER_CLASS_PER_TASK,
        )
    )


    selected_no = (
        rng.sample(
            no_examples,
            SAMPLES_PER_CLASS_PER_TASK,
        )
    )


    diagnostic_examples.extend(
        selected_yes
    )

    diagnostic_examples.extend(
        selected_no
    )


    print(
        f"{task_id:28s} | "
        f"YES={len(selected_yes):2d} | "
        f"NO={len(selected_no):2d}"
    )


print(
    "\nDiagnostic examples:",
    len(
        diagnostic_examples
    ),
)


# ============================================================
# Run diagnostic
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "CLASS-TOKEN DIAGNOSTIC"
)

print(
    "=" * 78
)


records = []


for index, example in enumerate(
    diagnostic_examples,
    start=1,
):

    record = (
        diagnose_example(
            example
        )
    )


    records.append(
        record
    )


    if (
        index == 1
        or index % 20 == 0
        or index
        == len(
            diagnostic_examples
        )
    ):

        print(
            f"Processed "
            f"{index}/"
            f"{len(diagnostic_examples)}"
        )


# ============================================================
# Aggregate by task
# ============================================================

task_results = {}


print(
    "\n"
    + "=" * 78
)

print(
    "RESULTS BY TASK"
)

print(
    "=" * 78
)


for task_id in ACTIVE_TASKS:

    task_records = [

        x
        for x in records

        if x[
            "task_id"
        ] == task_id
    ]


    yes_records = [

        x
        for x in task_records

        if x[
            "gold"
        ] == "(a)"
    ]


    no_records = [

        x
        for x in task_records

        if x[
            "gold"
        ] == "(b)"
    ]


    # --------------------------------------------------------
    # Divergence-token classification
    # --------------------------------------------------------

    token_correct = sum(

        x[
            "token_preference"
        ]
        == x[
            "gold"
        ]

        for x in task_records
    )


    token_accuracy = (
        token_correct
        / len(
            task_records
        )
    )


    # --------------------------------------------------------
    # Full-answer preference
    # --------------------------------------------------------

    full_correct = sum(

        x[
            "full_answer_preference"
        ]
        == x[
            "gold"
        ]

        for x in task_records
    )


    full_accuracy = (
        full_correct
        / len(
            task_records
        )
    )


    mean_yes_p_a = (
        sum(
            x[
                "p_a_given_a_or_b"
            ]

            for x in yes_records
        )
        / len(
            yes_records
        )
    )


    mean_no_p_b = (
        sum(
            x[
                "p_b_given_a_or_b"
            ]

            for x in no_records
        )
        / len(
            no_records
        )
    )


    no_prefers_b = sum(

        x[
            "token_preference"
        ]
        == "(b)"

        for x in no_records
    )


    yes_prefers_a = sum(

        x[
            "token_preference"
        ]
        == "(a)"

        for x in yes_records
    )


    task_results[
        task_id
    ] = {
        "n":
            len(
                task_records
            ),

        "token_accuracy":
            token_accuracy,

        "full_answer_accuracy":
            full_accuracy,

        "mean_p_a_on_gold_yes":
            mean_yes_p_a,

        "mean_p_b_on_gold_no":
            mean_no_p_b,

        "gold_yes_prefers_a":
            yes_prefers_a,

        "gold_no_prefers_b":
            no_prefers_b,
    }


    print(
        f"\n{task_id}"
    )

    print(
        f"  n:                         "
        f"{len(task_records)}"
    )

    print(
        f"  class-token accuracy:      "
        f"{token_accuracy * 100:5.1f}%"
    )

    print(
        f"  full-answer accuracy:      "
        f"{full_accuracy * 100:5.1f}%"
    )

    print(
        f"  Gold YES preferring (a):   "
        f"{yes_prefers_a}/"
        f"{len(yes_records)}"
    )

    print(
        f"  Gold NO preferring (b):    "
        f"{no_prefers_b}/"
        f"{len(no_records)}"
    )

    print(
        f"  Mean P(a) on gold YES:     "
        f"{mean_yes_p_a:.4f}"
    )

    print(
        f"  Mean P(b) on gold NO:      "
        f"{mean_no_p_b:.4f}"
    )


# ============================================================
# Overall
# ============================================================

overall_token_accuracy = (
    sum(
        x[
            "token_preference"
        ]
        == x[
            "gold"
        ]

        for x in records
    )
    / len(
        records
    )
)


overall_full_accuracy = (
    sum(
        x[
            "full_answer_preference"
        ]
        == x[
            "gold"
        ]

        for x in records
    )
    / len(
        records
    )
)


all_yes = [

    x
    for x in records

    if x[
        "gold"
    ] == "(a)"
]


all_no = [

    x
    for x in records

    if x[
        "gold"
    ] == "(b)"
]


overall_mean_p_a_yes = (
    sum(
        x[
            "p_a_given_a_or_b"
        ]

        for x in all_yes
    )
    / len(
        all_yes
    )
)


overall_mean_p_b_no = (
    sum(
        x[
            "p_b_given_a_or_b"
        ]

        for x in all_no
    )
    / len(
        all_no
    )
)


no_prefers_b_total = sum(

    x[
        "token_preference"
    ]
    == "(b)"

    for x in all_no
)


print(
    "\n"
    + "=" * 78
)

print(
    "OVERALL ANSWER-TOKEN DIAGNOSTIC"
)

print(
    "=" * 78
)


print(
    "Examples:",
    len(
        records
    ),
)

print(
    "Class-token accuracy:",
    f"{overall_token_accuracy * 100:.1f}%"
)

print(
    "Full-answer preference accuracy:",
    f"{overall_full_accuracy * 100:.1f}%"
)

print(
    "Mean P(a) on gold YES:",
    f"{overall_mean_p_a_yes:.4f}"
)

print(
    "Mean P(b) on gold NO:",
    f"{overall_mean_p_b_no:.4f}"
)

print(
    "Gold-NO examples internally preferring (b):",
    f"{no_prefers_b_total}/"
    f"{len(all_no)}"
)


# ============================================================
# Print representative GOLD-NO cases
# ============================================================

print(
    "\n"
    + "=" * 78
)

print(
    "REPRESENTATIVE GOLD-NO EXAMPLES"
)

print(
    "=" * 78
)


for record in (
    all_no[
        :12
    ]
):

    print(
        f"\nTask: "
        f"{record['task_id']}"
    )

    print(
        f"Participant: "
        f"{record['participant_id']}"
    )

    print(
        f"Window: "
        f"{record['window_start']}"
    )

    print(
        "Gold: (b)"
    )

    print(
        f"P(a | a,b): "
        f"{record['p_a_given_a_or_b']:.4f}"
    )

    print(
        f"P(b | a,b): "
        f"{record['p_b_given_a_or_b']:.4f}"
    )

    print(
        f"Token preference: "
        f"{record['token_preference']}"
    )

    print(
        f"Full-answer preference: "
        f"{record['full_answer_preference']}"
    )


# ============================================================
# Save results
# ============================================================

results = {
    "experiment":
        "stage5_answer_token_diagnostic",

    "seed":
        SEED,

    "repo_id":
        STAGE5_REPO_ID,

    "checkpoint":
        str(
            CHECKPOINT_PATH
        ),

    "test_participants_sealed":
        len(
            manifest[
                "test"
            ]
        ),

    "answer_a":
        {
            "raw":
                ANSWER_A,

            "token_ids":
                answer_a_ids
                .cpu()
                .tolist(),

            "tokens":
                answer_a_tokens,
        },

    "answer_b":
        {
            "raw":
                ANSWER_B,

            "token_ids":
                answer_b_ids
                .cpu()
                .tolist(),

            "tokens":
                answer_b_tokens,
        },

    "common_prefix_len":
        common_prefix_len,

    "class_tokens":
        {
            "a_id":
                a_class_token_id,

            "a_token":
                a_class_token,

            "b_id":
                b_class_token_id,

            "b_token":
                b_class_token,
        },

    "n_examples":
        len(
            records
        ),

    "overall":
        {
            "token_accuracy":
                overall_token_accuracy,

            "full_answer_accuracy":
                overall_full_accuracy,

            "mean_p_a_gold_yes":
                overall_mean_p_a_yes,

            "mean_p_b_gold_no":
                overall_mean_p_b_no,

            "gold_no_prefers_b":
                no_prefers_b_total,

            "gold_no_total":
                len(
                    all_no
                ),
        },

    "tasks":
        task_results,

    "records":
        records,
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
    "\nSaved diagnostic:"
)

print(
    RESULTS_PATH
)


print(
    "\nTEST participants remain sealed."
)