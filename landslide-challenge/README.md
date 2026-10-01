# LandslideRisk-48: Shipd Eris challenge

Dataset: **LandslideRisk-48: Short-Term...** (Shipd → My Datasets)

## Form ke fields mein kya bharna hai

| Shipd field | Kya daalna hai |
|---|---|
| Dataset | `LandslideRisk-48` select karo |
| Domain | Tabular |
| Difficulty | Hard |
| Challenge Title | `LandslideRisk-48: 48-Hour Landslide Forecasting at Seen and Unseen Sites` |
| Problem Description | `problem_description.md` ka poora text |
| Prepare script | `prepare.py` |
| Grading script | `grade.py` |
| Config | Abhi baaki hai (form ka format dekhna hai) |
| Rubrics | `rubrics.md` (10 criteria) |
| Solution | `solution.ipynb` |

## Kya verify kiya gaya hai

- `prepare.py` ko do baar chalaya, aur dono baar saari files byte-for-byte same aayi (deterministic).
- Split: train = 23,142 rows (38 locations, 2025-09-01 se pehle); test = 5,368 rows (44 locations, 2025-09-01 se 2025-12-31 tak). 6 locations sirf test mein hain.
- `previous_events_30d` hata diya. Ye pichhle 30 din ke target ka rolling sum hai: test mein agle din ki value +1 ho jaye toh label 3004/3004 baar 1 tha. Ise rakhte toh labels leak ho jaate.
- `grade.py`: perfect submission ≈ 0, sab 0.5 = 0.693, sample submission = 0.339. Missing column, galat row count, duplicate IDs, NaN, text, [0, 1] se bahar ki values aur unknown IDs pe error aata hai. Rows ka order aur extra columns se fark nahi padta.
- `solution.ipynb` ~11 second mein poora chalta hai. Test score **0.2823** (baseline 0.3392).

| Model (time-forward + unseen-location validation) | Balanced log loss |
|---|---|
| Constant prior | 0.4217 |
| Logistic regression, raw | 0.3494 |
| LightGBM, raw | 0.3498 |
| LightGBM, raw + location_id | 0.3500 |
| Logistic regression, engineered | 0.3487 |
| LightGBM, engineered | 0.3487 |
| Blend 50/50 (final) | **0.3473** |

## Khud chala ke dekhna ho toh

```bash
pip install pandas numpy scikit-learn lightgbm nbconvert ipykernel
python -c "from pathlib import Path; from prepare import prepare; prepare(Path('raw'), Path('dataset/public'), Path('dataset/private'))"
jupyter nbconvert --to notebook --execute solution.ipynb --output executed.ipynb
python -c "import pandas as pd; from grade import grade; print(grade(pd.read_csv('working/submission.csv'), pd.read_csv('dataset/private/answers.csv')))"
```

`raw/` folder mein Shipd dataset ki `train.csv` honi chahiye.
