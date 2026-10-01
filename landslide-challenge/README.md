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
| 6 | `train.csv` = `test.csv` ke columns + target `location_ids`; readings alag `train_readings.csv` / `test_readings.csv` mein (same columns, label nahi) | ✅ 8/8 checks pass, ❌ novelty score 5 se kam |
| 7 (current) | **Value-aware recommendation**: har din kitni bhi sites (ya `none`), har site ki `inspection_hours` cost, relevant site catch karne pe 15 ghante ki value | Checks dobara chalana hai |

Recommendation kyun: Shipd Discord announcement (09/04/2026) mein khule domains: NLP, Computer Vision, Object Detection, **Recommendation**, Sequence to Sequence, Prompt Engineering, RAG, Fine-Tuning, From Scratch, LLM Evaluation. Iske alawa Shipd Eris ke public accepted challenges mein kai Recommendation/Ranking challenges hain, jaise "Coastal Sensor Signature Recommendation" (sensor data, MAP@5), "Biocatalytic Product Recommendation" aur "Biomedical Concept Evidence Ranking". Source: github.com/OmerFarukMerey/project-eris-shipd-csofm-solutions.

## Form ke fields mein kya bharna hai

| Shipd field | Kya daalna hai |
|---|---|
| Difficulty | Hard |
| Compute Tier | CPU |
| Challenge Title | `LandslideRisk-48: Value-Aware Inspection-Site Recommendation for Unmonitored Hillslopes` |
| Problem Description | `problem_description.md` ka poora text |
| Tags | `feature-engineering` |
| Grade Direction | Maximize |
| Min Score / Max Score | `0` / `1` |
| Grading Script | `grade.py` |
| prepare.py | `prepare.py` |
| Rubrics | `rubrics.md` (abhi platform pe rubrics band hain) |
| Solution | `solution.ipynb` |

## Kya verify kiya gaya hai (Shipd ke asli raw files pe)

Raw files: `train (1).csv` (38,012 rows, 52 sites, labelled), `test (1).csv` (10,800 rows, 2026, labels nahi hain, isliye use nahi hota), `sample_submission (1).csv` (ignore hota hai).

- Version 6 ne Shipd ke saare 8/8 checks pass kar liye the, lekin novelty score 5 se kam tha. Novelty review ne "asymmetric inspection costs" suggest kiya, isliye version 7 bana.
- Do aur ideas test karke chhod diye, kyunki is data pe unse score nahi badhta tha: (a) daily hours budget ke saath knapsack (ranking jitna hi score, 0.592 = 0.592), (b) nayi sites ke liye few-shot support window (koi gain nahi). Cost-aware decision rule wala idea sach mein farak dalta hai.
- `prepare.py`: deterministic hai. Public files:
  - `train.csv`: 731 queries (`query_id, date, candidate_sites, location_ids`; relevant site na ho toh `none`)
  - `test.csv`: 731 queries (`query_id, date, candidate_sites`)
  - `train_readings.csv`: 29,240 readings (40 sites); `test_readings.csv`: 8,772 readings (12 sites). Dono mein `inspection_hours` column hai, dono ke columns same hain, aur label column nahi hai.
  - `sample_submission.csv`: 731 rows
  - Private `answers.csv`: sirf `query_id, location_ids`
- `shipd_sim.py`: Shipd ke ab tak dikhe saare rules ka simulation, 16/16 pass. `grade.py` ki hardcoded costs bhi `inspection_hours` se exactly match karti hain.
- `grade.py`: sahi jawab = 1.0 (full, public aur private split pe). Kuch inspect na karna / sab inspect karna / random = 0. Top-3 by rainfall = 0.011, aadhe hits wala perfect = 0.486.
- `solution.ipynb`: ~11 second mein chalta hai. Test score **0.2322**.

| Policy (3-fold GroupKFold by site) | Net value score |
|---|---|
| Kuch inspect mat karo / sab inspect karo | 0.0000 |
| Top-1 by rainfall_72h roz | 0.0668 |
| Top-3 by LR roz (fixed K) | 0.1720 |
| 15·p > average hours (site cost ignore) | 0.2007 |
| 15·p > hours, LR raw | 0.2215 |
| 15·p > hours, LightGBM | 0.2107 |
| 15·p > hours, blend LR + LightGBM | 0.2234 |
| Final: blend, alpha = 0.9 | **0.2282** |
