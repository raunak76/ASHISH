# Rubrics (10 criteria)

## 1. REQUIRED · TRAINING
**Criterion:** Uses a time-forward validation scheme in which every validation row is dated after every training row it is evaluated against (e.g. a date cutoff or expanding-window split). Does not rely on random K-fold over the full training period to select models.

**Why:** The test period (Sep–Dec 2025) lies entirely after the training data, and readings from neighbouring days at the same site are strongly correlated. Random K-fold leaks near-duplicate days into validation and overstates performance.

## 2. REQUIRED · TRAINING
**Criterion:** Estimates performance on unseen locations by holding out whole `location_id` groups during validation (group-wise split), and reports the seen-location and unseen-location scores separately.

**Why:** 6 of the 44 test locations never appear in `train.csv`, and they carry half the weight of the metric. Without a group hold-out there is no estimate of the part of the score that is hardest to get right.

## 3. REQUIRED · FEATURE_ENGINEERING
**Criterion:** Predictions for the 6 test-only locations do not depend on identity memorisation. For example, `location_id` is excluded, or any per-location encoding has an explicit fallback for unseen IDs instead of being passed to the model as a raw numeric feature.

**Why:** A raw `location_id` feature or a per-site target encoding without a fallback gives arbitrary outputs for IDs the model has never seen, which is exactly the generalisation the challenge tests.

## 4. REQUIRED · DATA_HANDLING
**Criterion:** Does not build features for a row from readings taken later in time at the same location (e.g. next-day rainfall, centred or backward-shifted rolling windows, or per-location statistics computed over the test period that include future rows).

**Why:** Test rows are consecutive daily readings, so future rainfall and moisture are physically available in `test.csv` but would not be available to a real early-warning system issuing a 48-hour forecast.

## 5. REQUIRED · DATA_HANDLING
**Criterion:** Handles the ~5% missing values in `soil_moisture`, `vegetation_index`, the four rainfall columns and `temperature_c` without dropping any `test.csv` rows. The submission contains exactly 5,368 unique `record_id`s with probabilities in [0, 1].

**Why:** Dropping or mis-imputing rows with missing sensor values either breaks the submission format or silently produces invalid predictions for those rows.

## 6. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Derives features from the nested rainfall windows, such as non-overlapping increments (6h−1h, 24h−6h, 72h−24h) or intensity ratios (e.g. 24h/72h), instead of using only the four cumulative totals.

**Why:** The windows are nested, so the raw totals are highly collinear. Increments separate recent intense rain from antecedent saturation, the two mechanisms behind rainfall-triggered slope failure.

## 7. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Builds antecedent-condition features from each location's earlier readings in timestamp order, such as lagged `rainfall_72h_mm` or `soil_moisture`, day-over-day changes, or trailing multi-day rainfall sums.

**Why:** Each row only covers the last 72 hours of rain, but slope stability depends on longer wetting history. Trailing per-location features recover that history without using the target.

## 8. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Encodes cyclic quantities (`aspect_deg` and the seasonal position of `timestamp`, e.g. day of year) with sine/cosine or an equivalent representation, so that 359° and 1°, or 31 December and 1 January, are treated as close.

**Why:** Event rates move from about 7% in winter to about 17% at the monsoon peak, and the test period crosses this seasonal decline. Raw angles and month numbers introduce artificial discontinuities.

## 9. RECOMMENDED · MODELING
**Criterion:** Produces calibrated probabilities, e.g. trained with a log-loss objective or calibrated afterwards (Platt/isotonic on a time-forward hold-out), and bounds final predictions away from exactly 0 and 1.

**Why:** The metric is log loss. A single confident wrong prediction at 0 or 1 costs up to about 34.5 for that row, and an uncalibrated ranking model can score worse than the constant base-rate baseline.

## 10. RECOMMENDED · MODELING
**Criterion:** Achieves a validation location-balanced log loss clearly better than the constant base-rate predictor (≈11% positive rate) on a time-forward split, and the gap between training and validation log loss is not large.

**Why:** Beating the base rate shows that the model extracts real signal from terrain, moisture and rainfall. A large train/validation gap means the model memorises site-specific noise that will not transfer to the test-only locations.
