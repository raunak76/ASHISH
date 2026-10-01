import pandas as pd

QUERY = "query_id"
TARGET_COL = "location_ids"
NONE = "none"

# Value, in crew-hours, of inspecting a site within 48 hours before a landslide there.
CATCH_VALUE = 15.0

# Inspection cost (crew-hours) of each test site, as in the inspection_hours column:
# 1 + elevation_m / 1500 + slope_deg / 30, rounded to 0.25 h. IDs that are not test
# sites are ignored.
INSPECTION_HOURS = {
    1001: 4.0, 1002: 2.5, 1006: 3.0, 1011: 4.0, 1012: 4.5, 1023: 4.0,
    1028: 2.75, 1037: 2.75, 1044: 3.75, 1046: 2.25, 1048: 3.25, 1051: 3.5,
}


def _ids(cell) -> list:
    """Space-separated location_ids (or "none") -> list of distinct ints."""
    if pd.isna(cell):
        return []
    text = str(cell).strip()
    if text == "" or text.lower() == NONE:
        return []
    out = []
    for token in text.split():
        try:
            value = float(token)
        except ValueError:
            raise ValueError(f"location_ids must be space-separated integers or 'none', got {token!r}")
        if not value.is_integer():
            raise ValueError(f"location_ids must be integers, got {token!r}")
        if int(value) not in out:
            out.append(int(value))
    return out


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Normalised net inspection value (higher is better, 0-1):
        net = sum over inspected sites of (CATCH_VALUE if the site is relevant that day) - inspection_hours
        score = max(0, net / net of the perfect plan that inspects exactly the relevant sites)
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

    net, best = 0.0, 0.0
    for q, rel_cell in zip(queries, answers[TARGET_COL]):
        relevant = set(_ids(rel_cell))
        for site in _ids(recs[q]):
            if site in INSPECTION_HOURS:
                net += (CATCH_VALUE if site in relevant else 0.0) - INSPECTION_HOURS[site]
        best += sum(CATCH_VALUE - INSPECTION_HOURS[s] for s in relevant if s in INSPECTION_HOURS)
    if best <= 0:
        return 1.0 if net >= 0 else 0.0
    return max(0.0, net / best)
