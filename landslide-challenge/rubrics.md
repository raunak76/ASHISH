# Rubrics (10 criteria)

## 1. REQUIRED · TRAINING
**Criterion:** Validates with folds grouped by `location_id` (e.g. GroupKFold or leave-sites-out), so that no site contributes rows to both the fitting and the validation part of a fold. Does not select models with plain random K-fold over rows.

**Why:** Every test site is unseen, and readings from neighbouring days at one site are nearly identical. Random row-level K-fold leaks site identity into validation and overstates performance on new sites.

## 2. REQUIRED · FEATURE_ENGINEERING
**Criterion:** Predictions for test sites do not depend on site identity. For example, `location_id` is excluded, or any per-site encoding has an explicit fallback for unseen IDs instead of being passed to the model as a raw numeric feature.

**Why:** None of the test `location_id` values appear in `train.csv`. A raw ID feature or a per-site target encoding without a fallback produces arbitrary outputs for them.

## 3. REQUIRED · DATA_HANDLING
**Criterion:** Does not compute features for a reading from readings taken later at the same site (e.g. next-day rainfall, centred or backward-shifted rolling windows, or per-site statistics computed over the whole series including later rows).

**Why:** Each site's readings form a daily series. Later readings are available in the files but are not part of the conditions at the time of a reading, and the description explicitly forbids them.

## 4. REQUIRED · DATA_HANDLING
**Criterion:** Handles the ~5% missing values in `soil_moisture`, `vegetation_index`, the four rainfall columns and `temperature_c` without dropping any `test.csv` rows. The submission has exactly one row per test `record_id` with probabilities in [0, 1].

**Why:** Dropping or mis-imputing rows with missing sensor values either breaks the submission format or silently produces invalid predictions for those rows.

## 5. REQUIRED · DATA_HANDLING
**Criterion:** Does not treat heavy-tailed rainfall readings as errors to be removed or capped at a low percentile in a way that erases storm events (e.g. dropping rows above the 95th percentile of `rainfall_72h_mm`).

**Why:** Rainfall is zero-inflated with a long right tail, and the largest totals are the storm events that trigger most landslides. Generic outlier removal throws away the most informative positive rows.

## 6. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Derives features from the nested rainfall windows, such as non-overlapping increments (6h−1h, 24h−6h, 72h−24h) or intensity ratios (e.g. 24h/72h), instead of using only the four cumulative totals.

**Why:** The windows are nested, so the raw totals are highly collinear. Increments separate recent intense rain from antecedent saturation, the two mechanisms behind rainfall-triggered slope failure.

## 7. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Builds antecedent-wetness features from each site's earlier readings in timestamp order, such as lagged `rainfall_72h_mm` or `soil_moisture`, day-over-day changes, or trailing multi-day rainfall sums.

**Why:** A single reading only covers the last 72 hours of rain, but slope stability depends on a longer wetting history. Trailing per-site features recover that history without using the target.

## 8. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Encodes cyclic quantities (`aspect_deg` and seasonal position from `timestamp`, e.g. day of year) with sine/cosine or an equivalent representation, so that 359° and 1°, or 31 December and 1 January, are treated as close.

**Why:** The positive rate varies seasonally from about 7% in winter to about 17% at the monsoon peak. Raw angles and month numbers introduce artificial discontinuities.

## 9. RECOMMENDED · MODELING
**Criterion:** Produces calibrated probabilities, e.g. trained with a log-loss objective or calibrated afterwards (Platt/isotonic using site-grouped out-of-fold predictions), and bounds final predictions away from exactly 0 and 1.

**Why:** The metric is log loss. A single confident wrong prediction at 0 or 1 costs up to about 34.5 for that row, and an uncalibrated ranking model can score worse than the constant base-rate baseline.

## 10. RECOMMENDED · MODELING
**Criterion:** Achieves a site-grouped validation log loss clearly better than the constant base-rate predictor (≈11% positive rate), and the gap between training and validation log loss is not large.

**Why:** Beating the base rate shows that the model extracts real signal from terrain, moisture and rainfall. A large train/validation gap means the model memorises site-specific noise that will not transfer to new sites.
