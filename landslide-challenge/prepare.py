from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "landslide_risk_48h"
ID = "record_id"
N_TEST_LOCATIONS = 12
SEED = 48

# previous_events_30d is a rolling 30-day sum of past target values. Across
# consecutive daily rows of a test site its day-to-day change reveals the test
# labels, so it is removed from every public file.
DROP_COLUMNS = ["previous_events_30d"]


def _load_labelled(raw: Path) -> pd.DataFrame:
    """Collect every labelled row from the raw CSVs (file names may vary, e.g. 'train (1).csv')."""
    frames = []
    for path in sorted(raw.rglob("*.csv")):
        if path.name.lower().startswith("sample_submission"):
            continue
        df = pd.read_csv(path, dtype={"timestamp": str})
        if TARGET in df.columns:
            frames.append(df[df[TARGET].notna()])
    if not frames:
        raise FileNotFoundError(f"No CSV with a '{TARGET}' column found in {raw}")
    df = pd.concat(frames, ignore_index=True).drop_duplicates(ID)
    df[TARGET] = df[TARGET].astype(int)
    return df


def _pick_test_locations(df: pd.DataFrame) -> set:
    """Stratified pick: sort sites by event rate, cut into equal bands, draw one site per band."""
    rates = df.groupby("location_id")[TARGET].mean().reset_index()
    rates = rates.sort_values([TARGET, "location_id"], kind="mergesort")
    rng = np.random.default_rng(SEED)
    bands = np.array_split(rates["location_id"].to_numpy(), N_TEST_LOCATIONS)
    return {int(rng.choice(band)) for band in bands}


def prepare(raw: Path, public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    df = _load_labelled(raw)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df = df.sort_values(ID, kind="mergesort").reset_index(drop=True)

    # Whole sites go to test: no test location appears in train.
    test_locations = _pick_test_locations(df)
    is_test = df["location_id"].isin(test_locations)
    train = df[~is_test]
    test = df[is_test].sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    train.to_csv(public / "train.csv", index=False)
    test.drop(columns=[TARGET]).to_csv(public / "test.csv", index=False)

    sample = pd.DataFrame({ID: test[ID], TARGET: round(float(train[TARGET].mean()), 4)})
    sample.to_csv(public / "sample_submission.csv", index=False)

    test[[ID, TARGET]].to_csv(private / "answers.csv", index=False)
