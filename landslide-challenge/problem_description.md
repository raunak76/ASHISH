# LandslideRisk-48: 48-Hour Landslide Forecasting at Seen and Unseen Sites

## Overview

Rainfall-triggered landslides kill thousands of people each year. Early-warning systems have to decide, from the readings a monitoring station sends today, how likely a slope is to fail in the next two days. Your task is to predict the **probability that a landslide occurs at a monitoring location within the next 48 hours**, using static terrain properties (elevation, slope, aspect, soil type), slowly varying surface conditions (soil moisture, vegetation), temperature, and nested rainfall-accumulation windows (1h, 6h, 24h, 72h).

Three things make this harder than an ordinary binary classifier:

1. **Forecasting forward in time.** All training rows are dated before 2025-09-01. All test rows are dated from 2025-09-01 to 2025-12-31. Event rates are strongly seasonal: about 7% in winter months and up to about 17% at the monsoon peak.
2. **Unseen locations.** The test set covers 44 locations. **6 of them never appear in `train.csv`**, so a model that memorises individual sites will not generalise to them. The metric weights these locations equally with the familiar ones.
3. **Imperfect sensors.** About 5% of the sensor and weather values are missing.

## Evaluation

Submissions are scored with **location-balanced log loss** (lower is better). Log loss is computed separately on the test rows from locations that appear in `train.csv` (seen) and on the rows from the 6 test-only locations (unseen), and the two values are averaged. Predictions are clipped to `[1e-15, 1 - 1e-15]` before scoring.

~~~python
import numpy as np

def log_loss(y, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

def evaluate(y_true, y_pred, is_unseen_location):
    seen = ~is_unseen_location
    return 0.5 * log_loss(y_true[seen], y_pred[seen]) + \
           0.5 * log_loss(y_true[is_unseen_location], y_pred[is_unseen_location])
~~~

Because log loss punishes confident mistakes heavily, **well-calibrated probabilities** matter more than ranking alone. Predicting the training base rate for every row scores about 0.34.

## Dataset

All files are in `public/`.

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 23,142 | Labelled daily readings from 38 locations, 2024-01-01 to 2025-08-31 |
| `test.csv` | 5,368 | Unlabelled daily readings from 44 locations, 2025-09-01 to 2025-12-31 |
| `sample_submission.csv` | 5,368 | Required submission format |

Each location has at most one reading per day. Rows are not sorted.

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Unique row identifier (carries no signal) |
| `location_id` | int | Monitoring location |
| `timestamp` | datetime | Observation time, `YYYY-MM-DD HH:MM:SS` |
| `elevation_m` | float | Elevation in metres (constant per location) |
| `slope_deg` | float | Slope angle in degrees (constant per location) |
| `aspect_deg` | float | Slope direction, 0–360 degrees (constant per location) |
| `soil_moisture` | float | Volumetric soil moisture, 0–1 (may be missing) |
| `soil_type` | string | One of `sandy`, `loam`, `clay`, `silt`, `rocky`, `peat` (constant per location) |
| `vegetation_index` | float | Vegetation cover, 0–1 (may be missing) |
| `rainfall_1h_mm` | float | Rainfall in the last 1 hour, mm (may be missing) |
| `rainfall_6h_mm` | float | Rainfall in the last 6 hours, mm (may be missing) |
| `rainfall_24h_mm` | float | Rainfall in the last 24 hours, mm (may be missing) |
| `rainfall_72h_mm` | float | Rainfall in the last 72 hours, mm (may be missing) |
| `temperature_c` | float | Air temperature, °C (may be missing) |
| `landslide_risk_48h` | int (0/1) | **Target**: 1 if a landslide occurred at the location within 48 hours of the reading. `train.csv` only |

When all four values are present, `rainfall_72h_mm ≥ rainfall_24h_mm ≥ rainfall_6h_mm ≥ rainfall_1h_mm`.

The source dataset also has a `previous_events_30d` column. It is not provided here because it is computed from past target values, and across consecutive test days it would reveal the test labels.

A forecast for a given row may use that row's readings and **earlier** readings from the same location, which are available at prediction time. It must not use readings from later rows.

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Row identifier from `test.csv` |
| `landslide_risk_48h` | float | Predicted probability of a landslide within 48 hours, in [0, 1] |

**Requirements**
- Must contain exactly 5,368 rows, one per `record_id` in `test.csv`, in any order.
- Include a header row.
- Every probability must be a number between 0 and 1. Missing values, duplicate IDs, or unknown IDs make the submission invalid.
