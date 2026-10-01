import pandas as pd

QUERY = "query_id"
TARGET_COL = "location_ids"
CANDIDATES = "candidate_location_ids"
K = 3


def _ids(cell) -> list:
    if pd.isna(cell):
        return []
    return [int(float(x)) if float(x).is_integer() else float(x) for x in str(cell).split()]


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    MAP@3 over query days that have at least one relevant site (higher is better):
        AP@3 = sum over ranks k with a relevant site of precision@k, divided by min(3, #relevant)
    submission.location_ids: 3 distinct space-separated location_ids, best first.
    answers.location_ids: space-separated relevant location_ids (empty if none).
    Raise only for truly invalid submissions.
    """
    missing = {QUERY, TARGET_COL} - set(submission.columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")
    if len(submission) != len(answers):
        raise ValueError(f"Expected {len(answers)} rows, got {len(submission)}")
    sub = submission[[QUERY, TARGET_COL]].copy()
    sub[QUERY] = sub[QUERY].astype(str).str.strip()
    if sub[QUERY].duplicated().any():
        raise ValueError("Submission contains duplicate query_id values")
    ans = answers.copy()
    ans[QUERY] = ans[QUERY].astype(str).str.strip()
    if set(sub[QUERY]) != set(ans[QUERY]):
        raise ValueError("Submission query_id values do not match test.csv")
    recs = dict(zip(sub[QUERY], sub[TARGET_COL]))

    total, n_scored = 0.0, 0
    for q, rel_cell, cand_cell in zip(ans[QUERY], ans[TARGET_COL], ans[CANDIDATES]):
        try:
            recommended = _ids(recs[q])
        except ValueError:
            raise ValueError(f"Query {q}: location_ids must be space-separated integers")
        if len(recommended) != K or len(set(recommended)) != K or not set(recommended) <= set(_ids(cand_cell)):
            raise ValueError(f"Query {q}: location_ids must list {K} distinct candidate sites for that day")
        relevant = set(_ids(rel_cell))
        if not relevant:
            continue
        hits, ap = 0, 0.0
        for rank, site in enumerate(recommended, start=1):
            if site in relevant:
                hits += 1
                ap += hits / rank
        total += ap / min(K, len(relevant))
        n_scored += 1
    return total / n_scored if n_scored else 0.0
