# LandslideRisk-48: Daily Inspection-Site Recommendation for Unmonitored Hillslopes

## Overview

A regional geohazard team runs a field-inspection crew that can visit **3 hillslope monitoring sites per day**. Each morning, every site sends one reading: static terrain properties (elevation, slope, aspect, soil type), surface conditions (soil moisture, vegetation), air temperature, and nested rainfall-accumulation windows (1h, 6h, 24h, 72h). Your task is to **recommend, for each day, the 3 sites the crew should inspect**, ranked so that the sites where a landslide event occurs within 48 hours of the reading come first.

Each day is one recommendation query, and the candidates are that day's readings from the 12 sites in `test.csv`. A site is **relevant** for a day if `landslide_risk_48h = 1` for its reading that day.

What makes this hard:

- **The candidate sites are new.** The data is split by site, and none of the 12 test sites appear in `train.csv`. A recommender has to learn how terrain, soil and rainfall combine into failure risk in general, not memorise which sites tend to fail.
- **Relevance is sparse.** About 11% of readings are relevant. On roughly 20% of test days no site is relevant, and the remaining days have about 1.9 relevant sites out of 12 on average.
- **Imperfect sensors.** About 5% of the sensor and weather values are missing. Rainfall is zero-inflated and heavy-tailed, and the extreme values are genuine storm readings, not errors.

## Evaluation

Submissions are scored with **MAP@3** (mean average precision at 3, higher is better, range 0–1), averaged over the test days that have at least one relevant site. Days with no relevant site are not scored, but they must still be in the submission.

~~~python
def average_precision_at_3(recommended, relevant):
    # recommended: list of 3 distinct location_ids, best first
    # relevant: set of location_ids with landslide_risk_48h == 1 that day (non-empty)
    hits, score = 0, 0.0
    for rank, site in enumerate(recommended[:3], start=1):
        if site in relevant:
            hits += 1
            score += hits / rank
    return score / min(3, len(relevant))

def evaluate(submission, truth):
    days = [d for d in truth if truth[d]]          # days with >= 1 relevant site
    return sum(average_precision_at_3(submission[d], truth[d]) for d in days) / len(days)
~~~

For reference, a random order scores about 0.15, and ranking each day's sites by `rainfall_72h_mm` alone scores about 0.34.

## Dataset

All files are in `public/`.

| File | Description |
|------|-------------|
| `train.csv` | Daily readings with relevance labels from the training sites, 2024-01-01 to 2025-12-31 |
| `test.csv` | Daily readings without labels from 12 held-out sites over the same days |
| `sample_submission.csv` | Required submission format, one row per test day |

Every site has exactly one reading per day, so every test day has 12 candidate sites. Rows are shuffled.

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Unique row identifier (carries no signal) |
| `query_id` | string | Query day, `YYYY-MM-DD` (the date part of `timestamp`) |
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
| `landslide_risk_48h` | int (0/1) | **Relevance label**: 1 if a landslide event occurred at the site within 48 hours of the reading. `train.csv` only |

When all four values are present, `rainfall_72h_mm ≥ rainfall_24h_mm ≥ rainfall_6h_mm ≥ rainfall_1h_mm`.

The source dataset also has a `previous_events_30d` column. It is not provided because it is computed from past relevance labels, and across consecutive days of a test site it would reveal the test labels.

Features for a reading may use that reading and **earlier** readings from the same site. They must not use readings taken later at the same site.

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | Query day from `test.csv`, `YYYY-MM-DD` |
| `rec_1` | int | `location_id` of the most recommended site that day |
| `rec_2` | int | Second recommended `location_id` |
| `rec_3` | int | Third recommended `location_id` |

**Requirements**
- Must contain exactly one row per distinct `query_id` in `test.csv` (731 rows), in any order.
- Include a header row.
- `rec_1`, `rec_2` and `rec_3` must be three different `location_id` values that appear in `test.csv` for that `query_id`. Missing, duplicate or unknown values make the submission invalid.
