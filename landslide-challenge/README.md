# LandslideRisk-48: Shipd Eris challenge

Dataset: **LandslideRisk-48: Short-Term Landslide Probability from Rainfall & Terrain**

## History

| Version | Format | Domain Routing result |
|---|---|---|
| 1 | Time-forward probability prediction | ❌ "Forecasting", not accepting submissions |
| 2 | Site-split binary classification | ❌ "Tabular", not accepting submissions |
| 3 | Daily top-3 site recommendation, `test.csv` mein har din ki 12 rows | ✅ Domain pass, ❌ "public test IDs are not unique" |
| 4 | Same recommendation, `test.csv` mein har din ki ek row | ✅ Evaluator pass, ❌ answers mein `candidate_location_ids` tha jo `test.csv` mein bhi tha |
| 5 | answers mein sirf target; candidates column ka naam `candidate_sites`; train/test query IDs alag | ❌ "Train and test feature columns must match" |
| 6 (current) | `train.csv` = `test.csv` ke columns + target `location_ids`; readings alag `train_readings.csv` / `test_readings.csv` mein (same columns, label nahi) | Checks dobara chalana hai |

Recommendation kyun: Shipd Discord announcement (09/04/2026) mein khule domains: NLP, Computer Vision, Object Detection, **Recommendation**, Sequence to Sequence, Prompt Engineering, RAG, Fine-Tuning, From Scratch, LLM Evaluation. Iske alawa Shipd Eris ke public accepted challenges mein kai Recommendation/Ranking challenges hain, jaise "Coastal Sensor Signature Recommendation" (sensor data, MAP@5), "Biocatalytic Product Recommendation" aur "Biomedical Concept Evidence Ranking". Source: github.com/OmerFarukMerey/project-eris-shipd-csofm-solutions.

## Form ke fields mein kya bharna hai

| Shipd field | Kya daalna hai |
|---|---|
| Difficulty | Hard |
| Compute Tier | CPU |
| Challenge Title | `LandslideRisk-48: Daily Inspection-Site Recommendation for Unmonitored Hillslopes` |
| Problem Description | `problem_description.md` ka poora text |
| Tags | `feature-engineering` |
| Grade Direction | **Maximize** |
| Min Score / Max Score | `0` / `1` |
| Grading Script | `grade.py` |
| prepare.py | `prepare.py` |
| Rubrics | `rubrics.md` (abhi platform pe rubrics band hain) |
| Solution | `solution.ipynb` |

## Kya verify kiya gaya hai (Shipd ke asli raw files pe)

Raw files: `train (1).csv` (38,012 rows, 52 sites, labelled), `test (1).csv` (10,800 rows, 2026, labels nahi hain, isliye use nahi hota), `sample_submission (1).csv` (ignore hota hai).

- `prepare.py`: deterministic hai (do baar chalaya, output byte-for-byte same). Public files:
  - `train.csv`: 723 queries (`query_id, date, candidate_sites, location_ids`)
  - `test.csv`: 581 queries (`query_id, date, candidate_sites`). Columns `train.csv` jaise hi hain, bas target nahi hai.
  - `train_readings.csv`: 29,240 readings (40 sites); `test_readings.csv`: 8,772 readings (12 sites). Dono ke columns same hain aur label column nahi hai.
  - `sample_submission.csv`: 581 rows (`query_id, location_ids`)
  - Private `answers.csv`: sirf `query_id, location_ids`
  - Query IDs: train mein `train_YYYY-MM-DD`, test mein `test_YYYY-MM-DD`, taaki dono kabhi overlap na hon.
- Shipd ke ab tak dikhe saare rules ka simulation (`shipd_sim.py`): 17/17 pass. Reproducible output, unique IDs, train/test columns match, answers ka koi column public test mein nahi, IDs disjoint, koi NaN nahi, readings columns same, aur sahi jawab pe 1.0 (full, public aur private).
- `previous_events_30d` hata diya (past labels ka rolling sum hai, test labels leak karta).
- `grade.py`: sahi jawab ko hi submission banao toh 1.0 (poore answers pe, aur Shipd ke public/private hisson pe alag-alag bhi). Random ≈ 0.17, rainfall_72h heuristic ≈ 0.38, sample submission = 0.14, sab ulta = 0.0.
  - Lenient hai: rows ka order, extra rows, 1–2 ya 3 se zyada IDs, aur NaN cell pe score deta hai.
  - Error sirf in pe: missing column, duplicate query, missing query, text ya decimal ID.
- `solution.ipynb`: ~25 second mein chalta hai. Test MAP@3 = **0.5283**.

| Ranker (3-fold GroupKFold by site) | MAP@3 |
|---|---|
| Random order | 0.1447 |
| Heuristic: rainfall_72h | 0.3229 |
| Heuristic: slope × moisture × rain72 | 0.4214 |
| Logistic regression, raw (final) | **0.4720** |
| LightGBM, raw | 0.4493 |
| LightGBM, raw + location_id | 0.4453 |
| Logistic regression, engineered | 0.4651 |
| LightGBM, engineered | 0.4695 |
| LambdaRank, engineered | 0.4496 |
| Blend of 3 (rank average) | 0.4670 |
