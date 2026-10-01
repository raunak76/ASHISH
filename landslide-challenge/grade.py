import pandas as pd

QUERY = "query_id"
K = 3
REC_COLUMNS = [f"rec_{k + 1}" for k in range(K)]


def _ids(cell) -> set:
    if pd.isna(cell) or str(cell) == "":
        return set()
    return {int(x) for x in str(cell).split("|")}


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    MAP@3 over query days that have at least one relevant site (higher is better):
        AP@3 = sum over ranks k with a relevant site of precision@k, divided by min(3, #relevant)
    Raise only for truly invalid submissions.
    """
    missing = {QUERY, *REC_COLUMNS} - set(submission.columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")
    if len(submission) != len(answers):
        raise ValueError(f"Expected {len(answers)} rows, got {len(submission)}")
    sub = submission.copy()
    sub[QUERY] = sub[QUERY].astype(str).str.strip()
    if sub[QUERY].duplicated().any():
        raise ValueError("Submission contains duplicate query_id values")
    if set(sub[QUERY]) != set(answers[QUERY].astype(str)):
        raise ValueError("Submission query_id values do not match test.csv")
    recs = sub[REC_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if recs.isna().any().any() or (recs % 1 != 0).any().any():
        raise ValueError("rec_1..rec_3 must be integer location_id values")
    sub[REC_COLUMNS] = recs.astype(int)
    sub = sub.set_index(QUERY)

    total, n_scored = 0.0, 0
    for _, row in answers.iterrows():
        recommended = sub.loc[str(row[QUERY]), REC_COLUMNS].tolist()
        candidates = _ids(row["candidates"])
        if len(set(recommended)) != K or not set(recommended) <= candidates:
            raise ValueError(f"Query {row[QUERY]}: recommendations must be {K} distinct sites from that day's test.csv rows")
        relevant = _ids(row["relevant"])
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
