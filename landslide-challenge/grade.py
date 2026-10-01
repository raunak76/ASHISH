import pandas as pd

QUERY = "query_id"
TARGET_COL = "location_ids"
K = 3


def _ids(cell) -> list:
    """Space-separated location_ids -> ordered list without duplicates."""
    if pd.isna(cell):
        return []
    out = []
    for token in str(cell).split():
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"location_ids must be space-separated integers, got {token!r}")
        if not value.is_integer():
            raise ValueError(f"location_ids must be integers, got {token!r}")
        if int(value) not in out:
            out.append(int(value))
    return out


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    MAP@3 (higher is better). For each query in `answers`:
        AP@3 = sum over ranks k <= 3 holding a relevant site of precision@k, divided by min(3, #relevant)
    submission.location_ids: recommended location_ids, space-separated, best first (only the first 3 count;
    IDs that are not relevant simply count as misses).
    answers.location_ids: the relevant location_ids, space-separated.
    `answers` may be any subset of the test queries (e.g. a public or private split).
    Raise only for truly invalid submissions.
    """
    missing = {QUERY, TARGET_COL} - set(submission.columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")
    sub = submission[[QUERY, TARGET_COL]].copy()
    sub[QUERY] = sub[QUERY].astype(str).str.strip()
    if sub[QUERY].duplicated().any():
        raise ValueError("Submission contains duplicate query_id values")
    recs = dict(zip(sub[QUERY], sub[TARGET_COL]))

    queries = answers[QUERY].astype(str).str.strip()
    absent = [q for q in queries if q not in recs]
    if absent:
        raise ValueError(f"Submission is missing {len(absent)} query_id values, e.g. {absent[0]}")

    total, n_scored = 0.0, 0
    for q, rel_cell in zip(queries, answers[TARGET_COL]):
        relevant = set(_ids(rel_cell))
        if not relevant:
            continue
        hits, ap = 0, 0.0
        for rank, site in enumerate(_ids(recs[q])[:K], start=1):
            if site in relevant:
                hits += 1
                ap += hits / rank
        total += ap / min(K, len(relevant))
        n_scored += 1
    return total / n_scored if n_scored else 0.0
