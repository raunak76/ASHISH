# LandslideRisk-48: Daily Inspection-Site Recommendation for Unmonitored Hillslopes

## Overview

A regional geohazard team runs a field-inspection crew that can visit **3 hillslope monitoring sites per day**. Each morning, every site sends one reading: static terrain properties (elevation, slope, aspect, soil type), surface conditions (soil moisture, vegetation), air temperature, and nested rainfall-accumulation windows (1h, 6h, 24h, 72h). Your task is to **recommend, for each day, the 3 sites the crew should inspect**, ranked so that the sites where a landslide event occurs within 48 hours of the reading come first.

Each day is one recommendation query, and the candidates are the 12 test sites, each described by its reading that day in `test_readings.csv`. A site is **relevant** for a day if `landslide_risk_48h = 1` for its reading that day.

What makes this hard:

- **The candidate sites are new.** The data is split by site, and none of the 12 test sites appear in `train.csv`. A recommender has to learn how terrain, soil and rainfall combine into failure risk in general, not memorise which sites tend to fail.
- **Relevance is sparse.** About 11% of readings are relevant. The test queries are the 581 days on which at least one test site is relevant, with about 1.9 relevant sites out of 12 on average.
- **Imperfect sensors.** About 5% of the sensor and weather values are missing. Rainfall is zero-inflated and heavy-tailed, and the extreme values are genuine storm readings, not errors.

## Evaluation

Submissions are scored with **MAP@3** (mean average precision at 3, higher is better, range 0–1), averaged over all queries in `test.csv`. Every test query has at least one relevant site.

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
    # truth: {query_id: set of relevant location_ids}; submission: {query_id: [3 location_ids]}
    return sum(average_precision_at_3(submission[q], truth[q]) for q in truth) / len(truth)
~~~

For reference, a random order scores about 0.17, and ranking each day's sites by `rainfall_72h_mm` alone scores about 0.38.

## Dataset

All files are in `public/`. Readings files have one row per site per day; query files have one row per day.

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 29,240 | Readings with relevance labels from 40 training sites, every day from 2024-01-01 to 2025-12-31 |
| `train_queries.csv` | 731 | One row per training day: `query_id`, `date`, `candidate_sites` (the 40 training sites, space-separated) and `location_ids` (the relevant ones, space-separated; empty if none) |
| `test_readings.csv` | 8,772 | Readings without labels from the 12 test sites, every day over the same period (including days that are not queries, so each site's history is complete) |
| `test.csv` | 581 | One row per test query: `query_id`, `date` and `candidate_sites` (the 12 test sites, space-separated) |
| `sample_submission.csv` | 581 | Required submission format |

`train.csv` is sorted by `query_id` and `location_id`; `test_readings.csv` is shuffled.

### Readings columns (`train.csv`, `test_readings.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Unique row identifier (carries no signal) |
| `query_id` | string | Query identifier: `train_YYYY-MM-DD` in train files, `test_YYYY-MM-DD` in test files, where the date is the date part of `timestamp` |
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
| `query_id` | string | Query identifier from `test.csv`, e.g. `test_2024-01-03` |
| `location_ids` | string | The 3 recommended `location_id`s, space-separated, best first (e.g. `1012 1001 1048`) |

**Requirements**
- Must contain exactly one row per `query_id` in `test.csv` (581 rows), in any order.
- Include a header row.
- `location_ids` should list three different integers from that query's `candidate_sites`, best first. Only the first three distinct IDs are scored, and an ID that is not a relevant site for that day (including one outside the candidates) counts as a miss.
- Missing `query_id` rows, duplicate `query_id` rows, or non-integer tokens make the submission invalid.
