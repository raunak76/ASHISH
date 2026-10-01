from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "landslide_risk_48h"
ID = "record_id"
CUTOFF = "2025-09-01 00:00:00"  # test period: timestamp >= CUTOFF
N_UNSEEN_LOCATIONS = 6
SEED = 48

# previous_events_30d is a rolling 30-day sum of past target values. Across
# consecutive daily test rows its day-to-day change reveals the test labels,
# so it is removed from every public file.
DROP_COLUMNS = ["previous_events_30d"]


def _load_labelled(raw: Path) -> pd.DataFrame:
    """Collect every labelled row from the raw CSVs (file names may vary, e.g. 'train (1).csv')."""
    frames = []
    for path in sorted(raw.rglob("*.csv")):
        if path.name.lower().startswith("sample_submission"):
            continue
        # Timestamps stay as ISO strings so the cutoff comparison is exact and
        # the output is byte-identical between runs.
        df = pd.read_csv(path, dtype={"timestamp": str})
        if TARGET in df.columns:
            frames.append(df[df[TARGET].notna()])
    if not frames:
        raise FileNotFoundError(f"No CSV with a '{TARGET}' column found in {raw}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(ID)
    df[TARGET] = df[TARGET].astype(int)
    return df


def prepare(raw: Path, public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    df = _load_labelled(raw)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.sort_values(ID, kind="mergesort").reset_index(drop=True)

    # Hold out whole locations: they appear only in the test period.
    locations = np.sort(df["location_id"].unique())
    rng = np.random.default_rng(SEED)
    unseen = set(rng.choice(locations, N_UNSEEN_LOCATIONS, replace=False).tolist())
    is_unseen = df["location_id"].isin(unseen)
    is_test_period = df["timestamp"] >= CUTOFF

    train = df[~is_test_period & ~is_unseen]
    test = df[is_test_period].sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train.to_csv(public / "train.csv", index=False)
    test.drop(columns=[TARGET]).to_csv(public / "test.csv", index=False)

    sample = pd.DataFrame({ID: test[ID], TARGET: round(float(train[TARGET].mean()), 4)})
    sample.to_csv(public / "sample_submission.csv", index=False)

    answers = pd.DataFrame({
        ID: test[ID],
        TARGET: test[TARGET],
        "unseen_location": test["location_id"].isin(unseen).astype(int),
    })
    answers.to_csv(private / "answers.csv", index=False)
