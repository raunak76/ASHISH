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
| 7 | Value-aware: har din kitni bhi sites (ya `none`), har site ki `inspection_hours` cost, catch pe 15 ghante | ❌ answers mein `none` ko missing maana; novelty 3/10 |
| 8 (final) | **Har query = 1 hafta, crew ke paas 12 ghante/hafta** (rolling multi-day budget). Target `site_days` = `location_id@date` tokens; har hafte kam se kam 1 event, isliye koi khaali answer nahi. "Technical focus" section VOI/greedy se fark batata hai | ✅ 8/8 checks pass, ❌ novelty abhi bhi 5 se kam |

Recommendation kyun: Shipd Discord announcement (09/04/2026) mein khule domains: NLP, Computer Vision, Object Detection, **Recommendation**, Sequence to Sequence, Prompt Engineering, RAG, Fine-Tuning, From Scratch, LLM Evaluation. Iske alawa Shipd Eris ke public accepted challenges mein kai Recommendation/Ranking challenges hain, jaise "Coastal Sensor Signature Recommendation" (sensor data, MAP@5), "Biocatalytic Product Recommendation" aur "Biomedical Concept Evidence Ranking". Source: github.com/OmerFarukMerey/project-eris-shipd-csofm-solutions.

## Form ke fields mein kya bharna hai

| Shipd field | Kya daalna hai |
|---|---|
| Difficulty | Hard |
| Compute Tier | CPU |
| Challenge Title | `LandslideRisk-48: Budget-Constrained Weekly Inspection-Site Recommendation for Unmonitored Hillslopes` |
| Problem Description | `problem_description.md` ka poora text |
| Tags | `feature-engineering` |
| Grade Direction | Maximize |
| Min Score / Max Score | `0` / `1` |
| Grading Script | `grade.py` |
| prepare.py | `prepare.py` |
| Rubrics | `rubrics.md` (abhi platform pe rubrics band hain) |
| Solution | `solution.ipynb` |

## Kya verify kiya gaya hai (Shipd ke asli raw files pe)

- Version 7 pe novelty 3/10 aaya. Reviewer ne suggest kiya: "rolling multi-day budget constraint that standard threshold-based solutions cannot directly address". Saath hi Shipd ne `none` ko missing target maana. Version 8 dono theek karta hai.
- Pehle test karke chhode gaye ideas: daily budget knapsack (ranking jitna hi score), few-shot support window (koi gain nahi).
- `prepare.py`: deterministic hai. Public files:
  - `train.csv` / `test.csv`: 105 weekly queries (`query_id, week_start, candidate_sites` [+ `site_days`])
  - `train_readings.csv`: 29,240 readings; `test_readings.csv`: 8,772 readings. Dono mein `inspection_hours` hai, label nahi.
  - Private `answers.csv`: `query_id, site_days`. Har hafte kam se kam 2 relevant site-days hain, koi khaali value nahi.
- `shipd_sim.py`: 16/16 simulated rules pass. Sahi jawab = 1.0 (full, public aur private split pe). Empty plan / sab inspect / sample = 0.
- `solution.ipynb`: ~20 second mein chalta hai, causal plan banata hai. Test score **0.3191**.

| Causal policy (3-fold GroupKFold by site) | Weekly net value |
|---|---|
| Kuch inspect mat karo | 0.0000 |
| Roz top-1 | 0.2109 |
| Myopic 15·p > hours (models ke hisaab se) | 0.168–0.195 |
| Budget-aware (return/hour > tau), LR raw | 0.2805 |
| **Budget-aware, LR engineered, tau = 1.2 (final)** | **0.2906** |
| Budget-aware, LightGBM | 0.2672 |

## Novelty ke liye test kiye gaye aur chhode gaye ideas

Ye sab Shipd ke asli raw data pe measure kiye gaye. Inmein se koi bhi best strategy nahi badalta, isliye inhe jodna sirf dikhawa hota:

| Idea | Nateeja |
|---|---|
| Daily hours budget (knapsack) | Ranking jitna hi score (0.592 vs 0.592) |
| Nayi sites ke liye few-shot labelled window | Adaptation se koi gain nahi |
| Soil moisture sirf training sites pe (partial observability) | AUC 0.7956 → 0.7931; soil moisture rainfall history se R² 0.90 tak predict ho jaata hai |
| Weekly budget carryover | Balance-aware policy: validation pe +0.01, test pe 0 |
| Adaptive inspection (pichhle inspection ka result agle faislon ka input) | Pichhle 1–7 din ke outcome jodne se AUC +0.0002: aaj ke sensors dene ke baad label site ke past outcomes se independent hai, latent state hai hi nahi |

Matlab is dataset ka label aaj ke features ka formula hai, plus noise. Weekly budget hi ek twist hai jo asli farak dalta hai (myopic 0.17–0.19 → budget-aware 0.27–0.29).
