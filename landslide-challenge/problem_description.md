# LandslideRisk-48: Landslide Event Classification at Unmonitored Sites

## Overview

Rainfall-triggered landslides kill thousands of people each year, yet most hillslopes have no history of recorded failures to learn from. Your task is a **tabular binary classification** problem: given one daily reading from a hillslope monitoring site, estimate the **probability that a landslide event was recorded at that site within 48 hours of the reading** (`landslide_risk_48h = 1`). Each reading contains static terrain properties (elevation, slope, aspect, soil type), surface conditions (soil moisture, vegetation), air temperature, and nested rainfall-accumulation windows (1h, 6h, 24h, 72h).

The central challenge is **transfer to new sites**. The data is split by site: **every site in `test.csv` is absent from `train.csv`**. A model has to learn how terrain, soil and rainfall combine into failure risk in general, not memorise which sites tend to fail. Two further difficulties:

- **Imbalance and calibration.** Only about 11% of readings are positive, and the metric rewards calibrated probabilities rather than hard labels.
- **Imperfect sensors.** About 5% of the sensor and weather values are missing. Rainfall is zero-inflated and heavy-tailed, and the extreme values are genuine storm readings, not errors.

## Evaluation

Submissions are scored with **binary log loss** (lower is better). Predictions are clipped to `[1e-15, 1 - 1e-15]` before scoring.

~~~python
import numpy as np

def evaluate(y_true, y_pred):
    p = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
~~~

Log loss punishes confident mistakes heavily, so **well-calibrated probabilities** matter more than ranking alone. Predicting the training base rate for every row scores about 0.37.

## Dataset

All files are in `public/`.

| File | Description |
|------|-------------|
| `train.csv` | Labelled daily readings from the training sites, 2024-01-01 to 2025-12-31 |
| `test.csv` | Unlabelled daily readings from 12 held-out sites over the same period |
| `sample_submission.csv` | Required submission format |

Each site has at most one reading per day. Rows are shuffled.

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Unique row identifier (carries no signal) |
| `location_id` | int | Monitoring site (test sites never appear in train) |
| `timestamp` | datetime | Reading time, `YYYY-MM-DD HH:MM:SS` |
| `elevation_m` | float | Elevation in metres (constant per site) |
| `slope_deg` | float | Slope angle in degrees (constant per site) |
| `aspect_deg` | float | Slope direction, 0–360 degrees (constant per site) |
| `soil_moisture` | float | Volumetric soil moisture, 0–1 (may be missing) |
| `soil_type` | string | One of `sandy`, `loam`, `clay`, `silt`, `rocky`, `peat` (constant per site) |
| `vegetation_index` | float | Vegetation cover, 0–1 (may be missing) |
| `rainfall_1h_mm` | float | Rainfall in the last 1 hour, mm (may be missing) |
| `rainfall_6h_mm` | float | Rainfall in the last 6 hours, mm (may be missing) |
| `rainfall_24h_mm` | float | Rainfall in the last 24 hours, mm (may be missing) |
| `rainfall_72h_mm` | float | Rainfall in the last 72 hours, mm (may be missing) |
| `temperature_c` | float | Air temperature, °C (may be missing) |
| `landslide_risk_48h` | int (0/1) | **Target**: 1 if a landslide event was recorded at the site within 48 hours of the reading. `train.csv` only |

When all four values are present, `rainfall_72h_mm ≥ rainfall_24h_mm ≥ rainfall_6h_mm ≥ rainfall_1h_mm`.

The source dataset also has a `previous_events_30d` column. It is not provided because it is computed from past target values, and across consecutive days of a test site it would reveal the test labels.

Features for a reading may use that reading and **earlier** readings from the same site. They must not use readings taken later at the same site.

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Row identifier from `test.csv` |
| `landslide_risk_48h` | float | Predicted probability that `landslide_risk_48h = 1`, in [0, 1] |

**Requirements**
- Must contain exactly one row per `record_id` in `test.csv` (same row count as `test.csv`), in any order.
- Include a header row.
- Every probability must be a number between 0 and 1. Missing values, duplicate IDs, or unknown IDs make the submission invalid.
