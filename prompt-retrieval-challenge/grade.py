import pandas as pd

QUERY = "query_id"
TARGET_COL = "prompt_ids"
K = 10


def _ids(cell) -> list:
    """Space-separated prompt IDs -> distinct IDs in listed order."""
    if pd.isna(cell):
        return []
    out = []
    for token in str(cell).split():
        if token not in out:
            out.append(token)
    return out


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    MRR@10 (higher is better, 0-1): for each query, 1 / rank of the correct prompt among the
    first 10 distinct prompt IDs listed, or 0 if it is not there.
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
    ranked = dict(zip(sub[QUERY], sub[TARGET_COL]))

    queries = answers[QUERY].astype(str).str.strip()
    absent = [q for q in queries if q not in ranked]
    if absent:
        raise ValueError(f"Submission is missing {len(absent)} query_id values, e.g. {absent[0]}")
    if len(queries) == 0:
        return 0.0

    total = 0.0
    for q, truth in zip(queries, answers[TARGET_COL]):
        correct = str(truth).strip()
        top = _ids(ranked[q])[:K]
        if correct in top:
            total += 1.0 / (top.index(correct) + 1)
    return total / len(queries)
