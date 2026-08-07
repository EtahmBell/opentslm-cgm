"""
Controlled OpenTSLM test.

Test A:
    Run one untouched official TSQA test example.

Test B:
    Keep the EXACT same TSQA question / prompt / text,
    but replace only its time series with a normalized CGM window
    of exactly the same length.

If A and B both generate answers:
    our CGM signal can pass through the pretrained encoder.

If A works but B is blank:
    the CGM signal distribution is causing the issue.

This avoids changing multiple things at once.
"""

import copy
import json

import torch

from opentslm import OpenTSLM
from opentslm.model_config import PATCH_SIZE
from opentslm.time_series_datasets.TSQADataset import TSQADataset
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


REPO_ID = "OpenTSLM/llama-3.2-1b-tsqa-sp"
CGM_PATH = r"C:\tslm\cgm\data\1001_DEX.json"


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model = OpenTSLM.load_pretrained(
    REPO_ID,
    device=device,
)


# ------------------------------------------------------------
# Load ONE real TSQA example
# ------------------------------------------------------------

print("\nLoading TSQA dataset...")

dataset = TSQADataset(
    "test",
    EOS_TOKEN=model.get_eos_token(),
)

official_sample = copy.deepcopy(dataset[0])

official_series = official_sample["time_series"][0]

print("\nOfficial sample")
print("------------------------------")
print("Question:")
print(official_sample["pre_prompt"])

print("\nPost prompt:")
print(official_sample["post_prompt"])

print("\nTime-series text:")
print(official_sample["time_series_text"])

print("\nSeries length:")
print(len(official_series))


# ------------------------------------------------------------
# TEST A:
# untouched official sample
# ------------------------------------------------------------

batch = extend_time_series_to_match_patch_size_and_aggregate(
    [copy.deepcopy(official_sample)],
    patch_size=PATCH_SIZE,
)

prediction = model.generate(
    batch,
    max_new_tokens=200,
)[0]

print("\n========================================")
print("TEST A — ORIGINAL TSQA SERIES")
print("========================================")
print("Prediction:", repr(prediction))


# ------------------------------------------------------------
# Load CGM
# ------------------------------------------------------------

with open(CGM_PATH, "r", encoding="utf-8") as f:
    cgm_json = json.load(f)

values = [
    record["blood_glucose"]["value"]
    for record in cgm_json["body"]["cgm"]
    if record.get("blood_glucose", {}).get("value") is not None
]

cgm = torch.tensor(values, dtype=torch.float32)


# ------------------------------------------------------------
# Make CGM exactly the same length as the TSQA example
# ------------------------------------------------------------

L = len(official_series)

cgm_window = cgm[:L]

mean = cgm_window.mean()
std = cgm_window.std()

cgm_normalized = (
    cgm_window - mean
) / (std + 1e-8)

print("\nCGM window")
print("------------------------------")
print("Length:", len(cgm_window))
print("Original mean:", round(mean.item(), 2))
print("Original std:", round(std.item(), 2))
print(
    "Normalized mean:",
    round(cgm_normalized.mean().item(), 4),
)
print(
    "Normalized std:",
    round(cgm_normalized.std().item(), 4),
)


# ------------------------------------------------------------
# TEST B:
# Change ONLY the time series.
#
# Question, text, post-prompt, etc. remain exactly those
# from the official TSQA example.
# ------------------------------------------------------------

cgm_sample = copy.deepcopy(official_sample)

cgm_sample["time_series"] = [
    cgm_normalized.tolist()
]

batch = extend_time_series_to_match_patch_size_and_aggregate(
    [cgm_sample],
    patch_size=PATCH_SIZE,
)

prediction = model.generate(
    batch,
    max_new_tokens=200,
)[0]

print("\n========================================")
print("TEST B — CGM, SAME TSQA PROMPT")
print("========================================")
print("Prediction:", repr(prediction))