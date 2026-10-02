# LandslideRisk-48: Budget-Constrained Weekly Inspection-Site Recommendation for Unmonitored Hillslopes

## Overview

A regional geohazard team has one field crew with **12 hours per week** for slope inspections. Each morning, every hillslope monitoring site sends one reading: static terrain properties (elevation, slope, aspect, soil type), surface conditions (soil moisture, vegetation), air temperature, and nested rainfall-accumulation windows (1h, 6h, 24h, 72h). For each week, your task is to **recommend which sites the crew should inspect on which days**, choosing among 12 candidate sites × 7 days.

This is a **value-aware recommendation problem under a rolling multi-day budget**:

- Inspecting a site costs its `inspection_hours`: 1 hour on site plus access time that grows with elevation and slope (1.5–4.5 hours).
- Inspecting a site on a day it is **relevant**, meaning a landslide event occurs there within 48 hours of that day's reading, is worth **15 crew-hours** (the value of an early warning).
- The week's plan is executed **in date order**. An inspection that no longer fits in the week's remaining hours is skipped.

Twelve hours buy only three or four inspections, while a typical week has about ten relevant site-days. Every hour spent on a Monday is unavailable for a storm on Thursday. A good plan has to:

- estimate **calibrated** probabilities at sites it has never seen,
- weigh expected value against each site's cost, and
- decide how much budget to keep for later, potentially more valuable days.

A per-day threshold such as "inspect if `15 × p > hours`" spends the budget too early, and a fixed top-K list ignores both risk levels and costs.

**Plans must be causal.** The decision to inspect a site on a given day may use only readings up to and including that day (plus anything learned from the training files). Readings from later days of the same week must not influence earlier decisions.

## Technical focus

The challenge couples two problems that standard landslide-warning and inspection-planning methods treat separately or not at all:

1. **Generalisation to unseen sites.** Every test site is absent from training, so risk must transfer from terrain, soil and rainfall alone, with probabilities that stay calibrated on new sites. Site-specific thresholds and per-site models do not apply.
2. **Sequential budget dynamics.** The weekly hours are a shared resource that is consumed in date order and cannot be recovered. Each decision is a stochastic online-knapsack step: inspecting today removes the option of a possibly more valuable inspection later in the week, and the decision cannot look ahead.

Classical approaches miss the second part:

- **Value-of-information (VOI) analysis** ranks candidate investigations once by expected value per cost, assuming the full candidate set is known in advance. Here the candidates arrive day by day, and a whole-week ranking would use future readings.
- **Greedy cost–benefit rules** inspect whenever `15 × p > inspection_hours`. They ignore the option value of unspent hours and run the budget dry on the first worthwhile days.

On site-grouped validation of the reference solution, the greedy cost–benefit rule scores about 0.17–0.19 and a fixed top-1-per-day list about 0.21. A causal rule that accounts for the shared budget scores about 0.27–0.29.

What makes this hard:

- **The candidate sites are new.** The data is split by site, and none of the 12 test sites appear in the training files.
- **Asymmetric costs and a shared budget.** Most site-days are quiet (about 11% of readings are relevant), but missing a landslide costs far more than a wasted inspection, and every inspection competes for the same weekly hours.
- **Imperfect sensors.** About 5% of the sensor and weather values are missing. Rainfall is zero-inflated and heavy-tailed, and the extreme values are genuine storm readings, not errors.

## Evaluation

Submissions are scored with **normalised net inspection value** (higher is better, range 0–1). For each week, the listed inspections are executed in date order (in listed order within a day). Each executed inspection costs its `inspection_hours` and earns 15 if the site is relevant that day. An inspection is skipped if it does not fit in the week's remaining 12 hours. The total net value over all weeks is divided by that of the reference plan, which inspects exactly the relevant site-days in date order under the same budget. The result is clipped to [0, 1].

~~~python
CATCH_VALUE, WEEKLY_BUDGET = 15.0, 12.0

def run_week(plan, relevant, inspection_hours):
    # plan: list of (date, location_id) in execution order; relevant: set of (date, location_id)
    left, net = WEEKLY_BUDGET, 0.0
    for day, site in plan:
        hours = inspection_hours[site]
        if hours <= left:
            left -= hours
            net += (CATCH_VALUE if (day, site) in relevant else 0.0) - hours
    return net

def evaluate(plans, truth, inspection_hours):
    # plans / truth: {query_id: list of (date, location_id) sorted by date}
    net = sum(run_week(plans[q], set(truth[q]), inspection_hours) for q in truth)
    best = sum(run_week(truth[q], set(truth[q]), inspection_hours) for q in truth)
    return min(1.0, max(0.0, net / best))
~~~

Repeated site-days, sites that are not candidates, and dates outside the query's week are skipped without cost. For reference, an empty plan scores 0, and inspecting every site every day scores 0.

## Dataset

All files are in `public/`. Query files have one row per week; readings files have one row per site per day.

| File | Rows | Description |
|------|------|-------------|
| `train.csv` | 105 | Training queries, one per week from 2024-01-01 to 2025-12-31: `query_id`, `week_start`, `candidate_sites` (the 40 training sites) and the target `site_days` |
| `test.csv` | 105 | Test queries for the same weeks: `query_id`, `week_start` and `candidate_sites` (the 12 test sites) |
| `train_readings.csv` | 29,240 | Daily readings of the 40 training sites |
| `test_readings.csv` | 8,772 | Daily readings of the 12 test sites |
| `sample_submission.csv` | 105 | Required submission format |

Weeks run Monday to Sunday and are named after their Monday. The last week (starting 2025-12-29) has only three days. Every week contains at least one relevant site-day. A training reading is relevant when its `location_id@date` appears in `site_days` for its week in `train.csv`. `train_readings.csv` is sorted by date and `location_id`; `test_readings.csv` is shuffled.

### Query columns (`train.csv`, `test.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `query_id` | string | Query identifier: `train_week_YYYY-MM-DD` in train files, `test_week_YYYY-MM-DD` in test files (the Monday of the week) |
| `week_start` | string | Monday of the week, `YYYY-MM-DD` |
| `candidate_sites` | string | Space-separated `location_id`s of the candidate sites |
| `site_days` | string | **Target**: space-separated `location_id@YYYY-MM-DD` tokens for every relevant site-day of the week, in date order. `train.csv` only |

### Readings columns (`train_readings.csv`, `test_readings.csv`)

| Column | Type | Description |
|--------|------|-------------|
| `record_id` | int | Unique row identifier (carries no signal) |
| `query_id` | string | Week query the reading belongs to |
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
| `query_id` | string | Query identifier from `test.csv`, e.g. `test_week_2024-01-08` |
| `site_days` | string | Space-separated `location_id@YYYY-MM-DD` tokens to inspect that week, e.g. `1037@2024-01-13 1046@2024-01-14`. Leave empty to inspect nothing |

**Requirements**
- Must contain exactly one row per `query_id` in `test.csv` (105 rows), in any order.
- Include a header row.
- You may list more site-days than the budget allows. They are executed in date order, and those that no longer fit are skipped.
- Missing `query_id` rows, duplicate `query_id` rows, or tokens that are not `integer@YYYY-MM-DD` make the submission invalid.
