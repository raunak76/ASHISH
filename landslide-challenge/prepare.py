from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "landslide_risk_48h"
ID = "record_id"
QUERY = "query_id"
K = 3
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


def _join(ids) -> str:
    return "|".join(str(i) for i in sorted(ids))


def prepare(raw: Path, public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    df = _load_labelled(raw)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    # One query per calendar day: rank that day's sites by landslide likelihood.
    df.insert(1, QUERY, df["timestamp"].str[:10])
    df = df.sort_values([QUERY, "location_id"], kind="mergesort").reset_index(drop=True)

    # Whole sites go to test: no test location appears in train.
    test_locations = _pick_test_locations(df)
    is_test = df["location_id"].isin(test_locations)
    train = df[~is_test]
    test = df[is_test]

    train.to_csv(public / "train.csv", index=False)
    test.drop(columns=[TARGET]).sample(frac=1.0, random_state=SEED).to_csv(public / "test.csv", index=False)

    days = test.groupby(QUERY)
    sample = days["location_id"].apply(lambda s: sorted(s)[:K]).reset_index()
    for k in range(K):
        sample[f"rec_{k + 1}"] = sample["location_id"].str[k]
    sample.drop(columns=["location_id"]).to_csv(public / "sample_submission.csv", index=False)

    answers = pd.DataFrame({
        QUERY: list(days.groups),
        "relevant": days.apply(lambda g: _join(g.loc[g[TARGET] == 1, "location_id"])).to_numpy(),
        "candidates": days["location_id"].apply(_join).to_numpy(),
    })
    answers.to_csv(private / "answers.csv", index=False)
