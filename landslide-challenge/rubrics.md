# Rubrics (10 criteria)

## 1. REQUIRED · TRAINING
**Criterion:** Validates with folds grouped by `location_id` (e.g. GroupKFold or leave-sites-out), so that no site contributes rows to both the fitting and the validation part of a fold, and computes the normalised net-value metric on the validation-fold sites. Does not select models or thresholds with random row-level K-fold.

**Why:** Every test site is unseen, and readings from neighbouring days at one site are nearly identical. Row-level K-fold leaks site identity into validation and overstates both accuracy and calibration on new sites.

## 2. REQUIRED · MODELING
**Criterion:** The decision of whether to recommend a site uses both its estimated landslide probability and its own `inspection_hours`, e.g. recommend when `15 × p > inspection_hours` or an equivalent threshold tuned on site-grouped validation. Does not recommend a fixed number of sites every day.

**Why:** The metric rewards inspecting a site only when its expected catch value exceeds its cost. On validation, a fixed top-3 list scores about 0.17 and a cost-blind threshold about 0.20, against about 0.22 for the cost-aware rule.

## 3. REQUIRED · MODELING
**Criterion:** Produces calibrated probabilities for unseen sites (e.g. a log-loss objective, or Platt/isotonic calibration fitted on site-grouped out-of-fold predictions) and checks calibration, e.g. mean predicted vs. observed positive rate or a scan of the threshold multiplier around 1.

**Why:** The decision threshold is an absolute probability. On site-grouped validation, doubling well-calibrated probabilities drops the score from about 0.22 to about 0.05, even though the ranking of sites is unchanged.

## 4. REQUIRED · FEATURE_ENGINEERING
**Criterion:** Probabilities for test sites do not depend on site identity. For example, `location_id` is excluded, or any per-site encoding has an explicit fallback for unseen IDs instead of being passed to the model as a raw numeric feature.

**Why:** None of the test `location_id` values appear in the training files. A raw ID feature or a per-site target encoding without a fallback produces arbitrary probabilities for them.

## 5. REQUIRED · DATA_HANDLING
**Criterion:** Does not compute features for a reading from readings taken later at the same site (e.g. next-day rainfall, centred or backward-shifted rolling windows, or per-site statistics computed over the whole series including later rows).

**Why:** Each site's readings form a daily series. Later readings are in the files but are not part of the conditions on the query day, and the description explicitly forbids them.

## 6. REQUIRED · CODE_QUALITY
**Criterion:** The submission has exactly one row for each of the 731 `query_id` values in `test.csv`, uses `none` for days with no recommended site, and lists only IDs from that day's `candidate_sites`.

**Why:** About 20% of test days have no relevant site, and the best plan on many quiet days is to inspect nothing. A missing query makes the submission invalid.

## 7. REQUIRED · DATA_HANDLING
**Criterion:** Handles the ~5% missing values in `soil_moisture`, `vegetation_index`, the four rainfall columns and `temperature_c` so that every candidate site still gets a probability. Does not silently skip candidates with missing readings.

**Why:** A site with a missing reading can still be the one that fails. Skipping it removes it from that day's plan.

## 8. RECOMMENDED · DATA_HANDLING
**Criterion:** Does not treat heavy-tailed rainfall readings as errors to be removed or capped at a low percentile in a way that erases storm events (e.g. dropping rows above the 95th percentile of `rainfall_72h_mm`).

**Why:** Rainfall is zero-inflated with a long right tail, and the largest totals are the storm events behind most relevant sites. Generic outlier removal throws away the most informative rows.

## 9. RECOMMENDED · MODELING
**Criterion:** Reports the net-value score of simple plans under the same site-grouped validation (inspect nothing, a fixed top-K list, a cost-blind probability threshold) and shows that the final plan beats all of them.

**Why:** Inspecting nothing scores 0 by construction. Comparing against fixed top-K and cost-blind rules shows that the gain comes from calibration and cost-awareness, not from the model alone.

## 10. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Derives features from the nested rainfall windows (e.g. 6h−1h, 24h−6h, 72h−24h increments or intensity ratios) or from each site's earlier readings (lagged rainfall or soil moisture, trailing multi-day sums), computed in timestamp order per site.

**Why:** The windows are nested and collinear, and a single reading covers only 72 hours of rain. Increments and trailing history separate recent intense rain from longer antecedent saturation.
