# LandslideRisk-48: Shipd Eris challenge

Dataset: **LandslideRisk-48: Short-Term Landslide Probability from Rainfall & Terrain**

Pehla version time-forward forecasting tha, lekin Shipd ki "Domain Routing" check ne use "Forecasting" mein daala, jo domain abhi band hai. Ab ye **tabular binary classification** hai, jismein test ki saari sites train mein kabhi nahi aati (site ke hisaab se split).

## Form ke fields mein kya bharna hai

| Shipd field | Kya daalna hai |
|---|---|
| Difficulty | Hard |
| Compute Tier | CPU |
| Challenge Title | `LandslideRisk-48: Landslide Event Classification at Unmonitored Sites` |
| Problem Description | `problem_description.md` ka poora text |
| Tags | `feature-engineering` |
| Grade Direction | Minimize |
| Min Score / Max Score | `0` / `34.54` |
| Grading Script | `grade.py` |
| prepare.py | `prepare.py` |
| Rubrics | `rubrics.md` (abhi platform pe rubrics band hain, check skip ho raha hai) |
| Solution | `solution.ipynb` |

## Kya verify kiya gaya hai

- `prepare.py`: do baar chalaya, output byte-for-byte same aaya. Raw file ka naam `train (1).csv` jaisa ho tab bhi chalta hai.
- Split: 12 sites test mein jaati hain, event rate ke hisaab se stratified (sites ko rate se sort karke 12 bands banaye, har band se 1 site). Mere data pe train = 23,392 rows (32 sites), test = 8,772 rows (12 sites), dono mein koi common site nahi.
- `previous_events_30d` hata diya: ye past target ka rolling sum hai, jo test labels leak karta.
- Same date pe doosri sites ka event rate use karne se test pe koi fayda nahi milta (AUC ≈ 0.53), toh cross-site shortcut nahi hai.
- `grade.py`: perfect ≈ 0, sab 0.5 = 0.693, sample submission = 0.374, sab ulta = 34.54. Har invalid submission pe error aata hai.
- `solution.ipynb`: ~17 second mein chalta hai. Test log loss **0.3027** (baseline 0.3744).

| Model (GroupKFold by site, out-of-fold) | Log loss |
|---|---|
| Constant prior | 0.3374 |
| Logistic regression, raw | 0.2869 |
| LightGBM, raw | 0.2917 |
| LightGBM, raw + location_id | 0.2923 |
| Logistic regression, engineered | 0.2868 |
| LightGBM, engineered | 0.2908 |
| Blend 50/50 (final) | 0.2869 |
