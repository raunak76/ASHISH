# Rubrics (10 criteria)

## 1. REQUIRED · TRAINING
**Criterion:** Validates with folds grouped by `location_id` (e.g. GroupKFold or leave-sites-out), so that no site contributes rows to both the fitting and the validation part of a fold, and computes MAP@3 per day among the validation-fold sites (e.g. from `train_queries.csv` restricted to those sites). Does not select models with random row-level K-fold.

**Why:** Every candidate site in the test set is unseen, and readings from neighbouring days at one site are nearly identical. Row-level K-fold leaks site identity into validation and overstates performance on new sites.

## 2. REQUIRED · FEATURE_ENGINEERING
**Criterion:** Scores for test sites do not depend on site identity. For example, `location_id` is excluded, or any per-site encoding has an explicit fallback for unseen IDs instead of being passed to the model as a raw numeric feature.

**Why:** None of the test `location_id` values appear in `train.csv`. A raw ID feature or a per-site target encoding without a fallback produces arbitrary scores for them.

## 3. REQUIRED · DATA_HANDLING
**Criterion:** Does not compute features for a reading from readings taken later at the same site (e.g. next-day rainfall, centred or backward-shifted rolling windows, or per-site statistics computed over the whole series including later rows).

**Why:** Each site's readings form a daily series. Later readings are available in the files but are not part of the conditions on the query day, and the description explicitly forbids them.

## 4. REQUIRED · CODE_QUALITY
**Criterion:** The submission has exactly one row for each of the 581 `query_id` values in `test.csv`, and each `location_ids` value lists three distinct, space-separated IDs taken from that query's `candidate_location_ids`, ordered best first.

**Why:** Readings in `test_readings.csv` cover all 731 days, but only the days in `test.csv` are queries. Missing a query makes the submission invalid, and recommending a repeated or non-candidate site wastes one of only three slots.

## 5. REQUIRED · DATA_HANDLING
**Criterion:** Handles the ~5% missing values in `soil_moisture`, `vegetation_index`, the four rainfall columns and `temperature_c` so that every candidate site still gets a score. Does not silently drop candidates with missing readings from the ranking.

**Why:** Dropping a site with a missing reading removes it from that day's candidate list and can push a relevant site out of the top 3.

## 6. REQUIRED · DATA_HANDLING
**Criterion:** Does not treat heavy-tailed rainfall readings as errors to be removed or capped at a low percentile in a way that erases storm events (e.g. dropping rows above the 95th percentile of `rainfall_72h_mm`).

**Why:** Rainfall is zero-inflated with a long right tail, and the largest totals are the storm events behind most relevant sites. Generic outlier removal throws away the most informative rows.

## 7. RECOMMENDED · MODELING
**Criterion:** Reports MAP@3 for a random order and for a simple domain heuristic (e.g. ranking by `rainfall_72h_mm`) under the same site-grouped validation, and shows that the final recommender beats both.

**Why:** Random order scores about 0.17 and the rainfall heuristic about 0.38. A learned model that does not clearly beat a one-column heuristic is not extracting signal from terrain and soil.

## 8. RECOMMENDED · FEATURE_ENGINEERING
**Criterion:** Derives features from the nested rainfall windows (e.g. 6h−1h, 24h−6h, 72h−24h increments or intensity ratios) or from each site's earlier readings (lagged rainfall or soil moisture, trailing multi-day sums), computed in timestamp order per site.

**Why:** The windows are nested and collinear, and a single reading covers only 72 hours of rain. Increments and trailing history separate recent intense rain from longer antecedent saturation.

## 9. RECOMMENDED · MODELING
**Criterion:** Compares at least one pointwise scorer (a per-reading probability model) with a within-day ranking approach (e.g. a pairwise or listwise objective grouped by `query_id`, or rank-averaging within days), and picks the final recommender by site-grouped MAP@3.

**Why:** Only the order of sites within a day matters to the metric. Comparing objectives shows whether optimising the ordering directly helps on new sites or overfits site-specific patterns.

## 10. RECOMMENDED · COMMUNICATION
**Criterion:** Explains how ties and low-information days are handled (e.g. days where all sites have near-zero scores) and states the tie-break rule used to choose three sites.

**Why:** Many days are dry at every site, so scores within a day can be nearly identical. A deterministic, explained tie-break makes the recommendations reproducible.
