# Rubrics (10 criteria)

## 1. REQUIRED · TRAINING
**Criterion:** Validates with folds grouped by `location_id` (e.g. GroupKFold or leave-sites-out), so that no site contributes rows to both the fitting and the validation part of a fold, and scores candidate plans with the weekly net-value metric on the validation-fold sites. Does not select models or thresholds with random row-level K-fold.

**Why:** Every test site is unseen, and readings from neighbouring days at one site are nearly identical. Row-level K-fold leaks site identity into validation and overstates both accuracy and calibration on new sites.

## 2. REQUIRED · MODELING
**Criterion:** The plan is causal. Whether a site is inspected on a given day depends only on readings up to that day and on rules or parameters fixed from the training data. Readings from later days of the same week do not influence earlier decisions (e.g. no sorting of a whole week's site-days by predicted value before choosing).

**Why:** The description requires causal plans. A whole-week ranking uses future rainfall and is not available to a real crew, even though the grader cannot detect it.

## 3. REQUIRED · MODELING
**Criterion:** The decision rule accounts for the shared weekly budget, e.g. a threshold on expected return per hour, `(15 × p − hours) / hours`, tuned on site-grouped validation, or a rule that depends on the hours left. It does not simply inspect whenever `15 × p > hours`, or a fixed number of sites per day.

**Why:** On site-grouped validation, the myopic rule scores about 0.17–0.19 and a fixed top-1 per day about 0.21, while a budget-aware threshold scores about 0.27–0.29. Spending early in the week leaves no hours for later storms.

## 4. REQUIRED · MODELING
**Criterion:** Produces calibrated probabilities for unseen sites (e.g. a log-loss objective, or Platt/isotonic calibration fitted on site-grouped out-of-fold predictions) and checks calibration, e.g. mean predicted vs. observed positive rate.

**Why:** The value of an inspection, `15 × p − hours`, depends on the absolute probability, not only on the ranking of sites.

## 5. REQUIRED · FEATURE_ENGINEERING
**Criterion:** Probabilities for test sites do not depend on site identity. For example, `location_id` is excluded, or any per-site encoding has an explicit fallback for unseen IDs.

**Why:** None of the test `location_id` values appear in the training files. A raw ID feature or a per-site target encoding without a fallback produces arbitrary probabilities for them.

## 6. REQUIRED · DATA_HANDLING
**Criterion:** Does not compute features for a reading from readings taken later at the same site (e.g. next-day rainfall, centred or backward-shifted rolling windows, or per-site statistics computed over the whole series including later rows).

**Why:** Later readings are in the files but are not part of the conditions on the day of the decision, and the description explicitly forbids them.

## 7. REQUIRED · CODE_QUALITY
**Criterion:** The submission has exactly one row for each of the 105 `query_id` values in `test.csv`, and every token is `location_id@YYYY-MM-DD` with a candidate site and a date inside that week. Weeks with nothing worth inspecting are left empty.

**Why:** A missing week or a malformed token makes the submission invalid, and tokens outside the week are silently ignored.

## 8. REQUIRED · DATA_HANDLING
**Criterion:** Handles the ~5% missing values in `soil_moisture`, `vegetation_index`, the four rainfall columns and `temperature_c` so that every candidate site-day still gets a probability, without dropping readings.

**Why:** A site with a missing reading can still be the one that fails. Dropping it removes it from that day's options.

## 9. RECOMMENDED · MODELING
**Criterion:** Reports the score of simple plans under the same site-grouped validation (empty plan, fixed top-K per day, the myopic `15 × p > hours` rule) and shows that the final plan beats all of them.

**Why:** Comparing against these plans shows that the gain comes from budget-aware decisions, not from the probability model alone.

## 10. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Derives features from the nested rainfall windows (e.g. 6h−1h, 24h−6h, 72h−24h increments) or from each site's earlier readings (lagged rainfall or soil moisture, trailing multi-day sums), computed in timestamp order per site, and does not cap or remove heavy-tailed rainfall values.

**Why:** The windows are nested and collinear, and the largest rainfall totals are the storm events behind most relevant site-days.
