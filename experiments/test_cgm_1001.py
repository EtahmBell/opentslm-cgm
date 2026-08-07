"""
First OpenTSLM inference test on a real AI-READI CGM trace.

This is only a smoke test:
- Load participant AIREADI-1001's Dexcom G6 measurements.
- Normalize the CGM trace using the same procedure as TSQADataset.
- Package the trace into OpenTSLM's expected prompt format.
- Ask one machine-verifiable question.
"""

import json
from pathlib import Path

import torch

from opentslm import OpenTSLM
from opentslm.model_config import PATCH_SIZE
from opentslm.prompt.prompt_with_answer import PromptWithAnswer
from opentslm.prompt.text_prompt import TextPrompt
from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.util import (
    extend_time_series_to_match_patch_size_and_aggregate,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

REPO_ID = "OpenTSLM/llama-3.2-1b-tsqa-sp"

CGM_PATH = Path(
    r"C:\tslm\cgm\data\1001_DEX.json"
)

QUESTION = """
Given the following definitions:
Constant volatility: The time series shows relatively consistent fluctuation magnitude throughout the period.
Increased volatility: The time series shows a rise in the magnitude of fluctuation over time.
Decreased volatility: The time series shows a reduction in the magnitude of fluctuations over time.

Select one of the following answers that best describes the provided time series:
(a) This time series has a constant volatility.
(b) This time series has an increased volatility.
(c) This time series has a decreased volatility.

Only answer (a), (b), or (c).
""".strip()

# We already computed the ground truth independently.
GOLD_ANSWER = "(a)"


# ---------------------------------------------------------------------
# Load CGM
# ---------------------------------------------------------------------

with CGM_PATH.open("r", encoding="utf-8") as file:
    data = json.load(file)

records = data["body"]["cgm"]

glucose_values = [
    record["blood_glucose"]["value"]
    for record in records
    if record.get("blood_glucose", {}).get("value") is not None
]

series = torch.tensor(
    glucose_values,
    dtype=torch.float32,
)

print("Participant:", data["header"]["patient_id"])
print("Number of readings:", len(series))


# ---------------------------------------------------------------------
# Normalize like OpenTSLM's TSQA dataset
# ---------------------------------------------------------------------

original_mean = series.mean()
original_std = series.std()

normalized_series = (
    series - original_mean
) / (original_std + 1e-8)

print(
    f"Original mean: {original_mean.item():.2f} mg/dL"
)

print(
    f"Original std: {original_std.item():.2f} mg/dL"
)

print(
    f"Original min: {series.min().item():.0f} mg/dL"
)

print(
    f"Original max: {series.max().item():.0f} mg/dL"
)


# ---------------------------------------------------------------------
# Load pretrained OpenTSLM
# ---------------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nLoading OpenTSLM on:", device)

model = OpenTSLM.load_pretrained(
    REPO_ID,
    device=device,
)


# ---------------------------------------------------------------------
# Construct an OpenTSLM sample
# ---------------------------------------------------------------------

# TSQA normally gives the language model the original mean and
# standard deviation while feeding the normalized raw signal through
# the time-series encoder.
#
# IMPORTANT:
# We intentionally do NOT give the model the minimum or maximum here,
# because that would directly reveal the answer to our question.

time_series_description = (
    "This is a continuous glucose monitor time series sampled "
    "approximately every 5 minutes. "
    f"The original glucose measurements have mean "
    f"{original_mean.item():.2f} mg/dL and standard deviation "
    f"{original_std.item():.2f} mg/dL."
)

sample = PromptWithAnswer(
    pre_prompt=TextPrompt(
        QUESTION
    ),
    text_time_series_prompt_list=[
        TextTimeSeriesPrompt(
            time_series_description,
            normalized_series.tolist(),
        )
    ],
    post_prompt=TextPrompt(
    "Predict the Volatility Answer:"
    ),
    answer=(
        GOLD_ANSWER
        + model.get_eos_token()
    ),
).to_dict()


# ---------------------------------------------------------------------
# Pad the time series to OpenTSLM's required patch size
# ---------------------------------------------------------------------

batch = (
    extend_time_series_to_match_patch_size_and_aggregate(
        [sample],
        patch_size=PATCH_SIZE,
    )
)


# ---------------------------------------------------------------------
# Run inference
# ---------------------------------------------------------------------

print("\nRunning inference...")

predictions = model.generate(
    batch,
    max_new_tokens=200,
)

prediction = predictions[0]


print("\n" + "=" * 60)
print("CGM ALIGNMENT TEST")
print("=" * 60)

print("\nQuestion:")
print(QUESTION)

print("\nGround truth:")
print(GOLD_ANSWER)

print("\nOpenTSLM prediction:")
print("Prediction repr:", repr(prediction))