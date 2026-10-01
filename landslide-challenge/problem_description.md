# LandslideRisk-48: Value-Aware Inspection-Site Recommendation for Unmonitored Hillslopes

## Overview

A regional geohazard team runs one field crew. Each morning, every hillslope monitoring site sends one reading: static terrain properties (elevation, slope, aspect, soil type), surface conditions (soil moisture, vegetation), air temperature, and nested rainfall-accumulation windows (1h, 6h, 24h, 72h). Your task is to **recommend, for each day, which sites the crew should inspect**: any number of the 12 candidate sites, from none to all of them.

This is a **value-aware recommendation** problem. Every recommended site costs crew time, and a recommendation only creates value if the site turns out to be **relevant** that day, meaning a landslide event occurs there within 48 hours of the reading:

- Inspecting a site costs its `inspection_hours`: 1 hour on site plus access time that grows with elevation and slope (1.5–4.5 hours, given for every site).
- Inspecting a relevant site is worth **15 crew-hours** (the value of an early warning).

Recommending a site pays off only when `15 × P(landslide) > inspection_hours`. A good recommender therefore needs **calibrated probabilities**, recommends more sites on dangerous days and **none on quiet days**, and accounts for the fact that the steepest sites are both the riskiest and the most expensive to reach. A fixed top-K list cannot express this.

What makes this hard:

- **The candidate sites are new.** The data is split by site, and none of the 12 test sites appear in the training files. The model has to learn how terrain, soil and rainfall combine into failure risk in general, not memorise which sites tend to fail.
- **Asymmetric costs.** A missed landslide costs far more than a wasted inspection, but most site-days are quiet: about 11% of readings are relevant, and on about 20% of days none of the 12 test sites is relevant.
- **Imperfect sensors.** About 5% of the sensor and weather values are missing. Rainfall is zero-inflated and heavy-tailed, and the extreme values are genuine storm readings, not errors.

## Evaluation

Submissions are scored with **normalised net inspection value** (higher is better, range 0–1). The net value of a plan is the value of the relevant sites it inspects minus the hours of every inspection, summed over all test days. It is divided by the net value of the perfect plan, which inspects exactly the relevant sites, and floored at 0, the value of inspecting nothing.

~~~python
CATCH_VALUE = 15.0

def evaluate(plan, truth, inspection_hours):
    # plan:  {query_id: set of location_ids to inspect (may be empty)}
    # truth: {query_id: set of relevant location_ids (may be empty)}
    # inspection_hours: {location_id: hours}
    net = sum(
        (CATCH_VALUE if site in truth[q] else 0.0) - inspection_hours[site]
        for q in truth for site in plan[q] if site in inspection_hours  # non-candidates ignored
    )
    best = sum(CATCH_VALUE - inspection_hours[site] for q in truth for site in truth[q])
    return max(0.0, net / best)
~~~

For reference, inspecting nothing scores 0, inspecting every site every day scores 0, and inspecting the top 3 sites by `rainfall_72h_mm` every day scores about 0.01.

## Dataset

All files are in `public/`. Query files have one row per day; readings files have one row per site per day.

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 731 | Training queries, one per day from 2024-01-01 to 2025-12-31: `query_id`, `date`, `candidate_sites` (the 40 training sites) and the target `location_ids` (the relevant sites, or `none`) |
| `test.csv` | 731 | Test queries, one per day over the same period: `query_id`, `date` and `candidate_sites` (the 12 test sites) |
| `train_readings.csv` | 29,240 | Readings of the 40 training sites, every day |
| `test_readings.csv` | 8,772 | Readings of the 12 test sites, every day |
| `sample_submission.csv` | 731 | Required submission format |

A training reading is relevant when its `location_id` is listed in `location_ids` for its `query_id` in `train.csv`. `train_readings.csv` is sorted by `query_id` and `location_id`; `test_readings.csv` is shuffled.

### Query columns (`train.csv`, `test.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | Query identifier: `train_YYYY-MM-DD` in train files, `test_YYYY-MM-DD` in test files |
| `date` | string | Query day, `YYYY-MM-DD` |
| `candidate_sites` | string | Space-separated `location_id`s of that day's candidate sites |
| `location_ids` | string | **Target**: space-separated `location_id`s of the relevant sites, or `none`. `train.csv` only |

### Readings columns (`train_readings.csv`, `test_readings.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Unique row identifier (carries no signal) |
| `query_id` | string | Query the reading belongs to (`train_` or `test_` plus the date part of `timestamp`) |
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
| `inspection_hours` | float | Crew-hours to inspect the site: `1 + elevation_m / 1500 + slope_deg / 30`, rounded to 0.25 (constant per site) |

When all four values are present, `rainfall_72h_mm ≥ rainfall_24h_mm ≥ rainfall_6h_mm ≥ rainfall_1h_mm`.

The source dataset also has a `previous_events_30d` column. It is not provided because it is computed from past relevance labels, and across consecutive days of a test site it would reveal the test labels.

Features for a reading may use that reading and **earlier** readings from the same site. They must not use readings taken later at the same site.

## Submission

Submit a CSV file with the following format:

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | Query identifier from `test.csv`, e.g. `test_2024-01-03` |
| `location_ids` | string | Space-separated `location_id`s of the sites to inspect that day (e.g. `1012 1048`), or `none` to inspect nothing |

**Requirements**
- Must contain exactly one row per `query_id` in `test.csv` (731 rows), in any order.
- Include a header row.
- Each listed site is inspected once; repeated IDs count once, and IDs that are not among that day's `candidate_sites` are ignored.
- Missing `query_id` rows, duplicate `query_id` rows, or tokens that are neither integers nor `none` make the submission invalid.
