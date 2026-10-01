# LandslideRisk-48: Shipd Eris challenge

Dataset: **LandslideRisk-48: Short-Term Landslide Probability from Rainfall & Terrain**

## History

| Version | Format | Domain Routing result |
|---|---|---|
| 1 | Time-forward probability prediction | ❌ "Forecasting", not accepting submissions |
| 2 | Site-split binary classification | ❌ "Tabular", not accepting submissions |
| 3 | Daily top-3 site recommendation, `test.csv` mein har din ki 12 rows | ✅ Domain pass, ❌ "public test IDs are not unique" |
| 4 (current) | Same recommendation, `test.csv` mein har din ki **ek row** | Checks dobara chalana hai |

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

## Kya verify kiya gaya hai

- `prepare.py`: deterministic hai (do baar chalaya, output byte-for-byte same). Raw file ka naam `train (1).csv` jaisa ho tab bhi chalta hai. Public files:
  - `train.csv` (23,392 readings, 32 sites) aur `train_queries.csv` (731 din)
  - `test_readings.csv` (8,772 readings, 12 sites × 731 din, poori history)
  - `test.csv` (584 queries, har din ki ek row, unique `query_id`): sirf wo din jinmein kam se kam ek relevant site hai
  - `sample_submission.csv` (584 rows: `query_id, location_ids`)
  - Private `answers.csv`: `query_id, location_ids, candidate_location_ids`. Submission aur answers dono mein target column ka naam `location_ids` hai, taaki Shipd checker target pehchaan sake.
- `previous_events_30d` hata diya (past labels ka rolling sum hai, test labels leak karta).
- `grade.py`: perfect = 1.0, sabse ulta = 0.0, random ≈ 0.15, rainfall_72h heuristic = 0.34, sample submission = 0.098. Missing column, galat row count, duplicate day, NaN, text, sirf 2 IDs, ek hi site do baar, unknown site, galat query_id, aur decimal ID pe error aata hai.
- `solution.ipynb`: ~25 second mein chalta hai. Test MAP@3 = **0.5207**.

| Ranker (3-fold GroupKFold by site) | MAP@3 |
|---|---|
| Random order | 0.1714 |
| Heuristic: rainfall_72h | 0.3564 |
| Heuristic: slope × moisture × rain72 | 0.4031 |
| Logistic regression, raw (final) | **0.4579** |
| LightGBM, raw | 0.4259 |
| LightGBM, raw + location_id | 0.4208 |
| Logistic regression, engineered | 0.4564 |
| LightGBM, engineered | 0.4383 |
| LambdaRank, engineered | 0.4201 |
| Blend of 3 (rank average) | 0.4427 |
