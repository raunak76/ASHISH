from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "landslide_risk_48h"
ID = "record_id"
QUERY = "query_id"
DATE = "date"
WEEK = "week_start"
COST = "inspection_hours"
CANDIDATES = "candidate_sites"
TARGET_COL = "site_days"  # space-separated "location_id@YYYY-MM-DD" tokens
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


def _inspection_hours(df: pd.DataFrame) -> pd.Series:
    """Crew-hours to reach and inspect a site: 1 h on site plus access time that grows
    with elevation and slope, rounded to 0.25 h."""
    hours = 1 + df["elevation_m"] / 1500 + df["slope_deg"] / 30
    return (np.round(hours * 4) / 4).clip(1.0, 4.5)


def _pick_test_locations(df: pd.DataFrame) -> set:
    """Stratified pick: sort sites by event rate, cut into equal bands, draw one site per band."""
    rates = df.groupby("location_id")[TARGET].mean().reset_index()
    rates = rates.sort_values([TARGET, "location_id"], kind="mergesort")
    rng = np.random.default_rng(SEED)
    bands = np.array_split(rates["location_id"].to_numpy(), N_TEST_LOCATIONS)
    return {int(rng.choice(band)) for band in bands}


def _queries(df: pd.DataFrame, with_labels: bool) -> pd.DataFrame:
    """One row per query week: its candidate sites and, with labels, the relevant site-days."""
    weeks = df.groupby(QUERY, sort=True)
    candidates = weeks["location_id"].agg(lambda s: " ".join(str(i) for i in sorted(set(s))))
    out = pd.DataFrame({
        QUERY: candidates.index,
        WEEK: weeks[WEEK].first().reindex(candidates.index).to_numpy(),
        CANDIDATES: candidates.to_numpy(),
    })
    if with_labels:
        rel = df[df[TARGET] == 1].sort_values([DATE, "location_id"])
        tokens = (rel["location_id"].astype(str) + "@" + rel[DATE]).groupby(rel[QUERY]).agg(" ".join)
        out[TARGET_COL] = out[QUERY].map(tokens)
    return out


def prepare(raw: Path, public: Path, private: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)

    df = _load_labelled(raw)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df[COST] = _inspection_hours(df)
    df[DATE] = df["timestamp"].str[:10]
    day = pd.to_datetime(df[DATE])
    df[WEEK] = (day - pd.to_timedelta(day.dt.dayofweek, unit="D")).dt.strftime("%Y-%m-%d")
    df = df.sort_values([DATE, "location_id"], kind="mergesort").reset_index(drop=True)

    # Whole sites go to test: no test location appears in train.
    test_locations = _pick_test_locations(df)
    is_test = df["location_id"].isin(test_locations)

    # One query per Monday-to-Sunday week and split ("train_week_YYYY-MM-DD" /
    # "test_week_YYYY-MM-DD", dated by the Monday), so train and test IDs never overlap.
    df.insert(1, QUERY, is_test.map({True: "test_week_", False: "train_week_"}) + df[WEEK])
    train = df[~is_test]
    test = df[is_test]

    # Queries: train.csv and test.csv share the same columns; train.csv adds the target.
    # Every week has at least one relevant site-day, so the target is never empty.
    train_queries = _queries(train, with_labels=True)
    answers = _queries(test, with_labels=True)
    if train_queries[TARGET_COL].isna().any() or answers[TARGET_COL].isna().any():
        raise ValueError("Found a week without any relevant site-day")
    train_queries.to_csv(public / "train.csv", index=False)
    answers[[QUERY, WEEK, CANDIDATES]].to_csv(public / "test.csv", index=False)

    # Site readings (one row per site and day), identical columns for train and test
    # and no label column: relevance comes only from train.csv.
    readings = [c for c in df.columns if c not in (DATE, WEEK, TARGET)]
    train[readings].to_csv(public / "train_readings.csv", index=False)
    test[readings].sample(frac=1.0, random_state=SEED).to_csv(public / "test_readings.csv", index=False)

    # Format example only: one arbitrary candidate site on the Monday of each week.
    rng = np.random.default_rng(SEED)
    sites = np.array(sorted(test_locations))
    sample = answers[[QUERY]].copy()
    sample[TARGET_COL] = [f"{s}@{w}" for s, w in zip(rng.choice(sites, size=len(sample)), answers[WEEK])]
    sample.to_csv(public / "sample_submission.csv", index=False)

    # Only the target goes into answers: any other column that also appears in the
    # public test file is flagged as a leaked target.
    answers[[QUERY, TARGET_COL]].to_csv(private / "answers.csv", index=False)
