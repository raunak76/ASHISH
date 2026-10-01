from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "landslide_risk_48h"
ID = "record_id"
QUERY = "query_id"
DATE = "date"
CANDIDATES = "candidate_sites"
TARGET_COL = "location_ids"  # space-separated site IDs
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
    return " ".join(str(i) for i in sorted(ids))


def _queries(df: pd.DataFrame, with_labels: bool) -> pd.DataFrame:
    """One row per query day: its candidate sites and, if requested, the relevant ones."""
    days = df.groupby(QUERY, sort=True)
    candidates = days["location_id"].agg(_join)
    out = pd.DataFrame({
        QUERY: candidates.index,
        DATE: days[DATE].first().reindex(candidates.index).to_numpy(),
        CANDIDATES: candidates.to_numpy(),
    })
    if with_labels:
        relevant = df[df[TARGET] == 1].groupby(QUERY)["location_id"].agg(_join)
        out[TARGET_COL] = out[QUERY].map(relevant).fillna("")
    return out


def prepare(raw: Path, public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    df = _load_labelled(raw)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[DATE] = df["timestamp"].str[:10]
    df = df.sort_values([DATE, "location_id"], kind="mergesort").reset_index(drop=True)

    # Whole sites go to test: no test location appears in train.
    test_locations = _pick_test_locations(df)
    is_test = df["location_id"].isin(test_locations)

    # One query per day and split ("train_YYYY-MM-DD" / "test_YYYY-MM-DD"), so train
    # and test query IDs never overlap.
    df.insert(1, QUERY, is_test.map({True: "test_", False: "train_"}) + df[DATE])
    train = df[~is_test]
    test = df[is_test]

    # Queries (one row per day). train.csv and test.csv share the same columns and
    # train.csv adds the target (the relevant sites). Only days with at least one
    # relevant site are queries: other days cannot be scored by MAP@3, and knowing
    # that a day has an event does not change the order of sites within it.
    train_queries = _queries(train, with_labels=True)
    train_queries = train_queries[train_queries[TARGET_COL] != ""]
    train_queries.to_csv(public / "train.csv", index=False)
    answers = _queries(test, with_labels=True)
    answers = answers[answers[TARGET_COL] != ""].reset_index(drop=True)
    answers[[QUERY, DATE, CANDIDATES]].to_csv(public / "test.csv", index=False)

    # Site readings (one row per site and day), identical columns for train and test
    # and no label column: relevance comes only from train.csv. Readings cover every
    # day so that each site's history is complete, including days that are not queries.
    readings = [c for c in df.columns if c not in (DATE, TARGET)]
    train[readings].to_csv(public / "train_readings.csv", index=False)
    test[readings].sample(frac=1.0, random_state=SEED).to_csv(public / "test_readings.csv", index=False)

    sample = answers[[QUERY]].copy()
    sample[TARGET_COL] = answers[CANDIDATES].str.split().str[:K].str.join(" ")
    sample.to_csv(public / "sample_submission.csv", index=False)

    # Only the target goes into answers: any other column that also appears in the
    # public test file is flagged as a leaked target.
    answers[[QUERY, TARGET_COL]].to_csv(private / "answers.csv", index=False)
