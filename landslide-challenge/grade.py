from datetime import date, timedelta

import pandas as pd

QUERY = "query_id"
TARGET_COL = "site_days"

# Value, in crew-hours, of inspecting a site on a day it is relevant (a landslide there
# within 48 hours of that day's reading).
CATCH_VALUE = 15.0
# Crew-hours available for inspections in each Monday-to-Sunday week.
WEEKLY_BUDGET = 12.0

# Inspection cost (crew-hours) of each test site, as in the inspection_hours column:
# 1 + elevation_m / 1500 + slope_deg / 30, rounded to 0.25 h. Other IDs are ignored.
INSPECTION_HOURS = {
    1001: 4.0, 1002: 2.5, 1006: 3.0, 1011: 4.0, 1012: 4.5, 1023: 4.0,
    1028: 2.75, 1037: 2.75, 1044: 3.75, 1046: 2.25, 1048: 3.25, 1051: 3.5,
}


def _week_start(query_id: str) -> date:
    return date.fromisoformat(query_id[-10:])


def _tokens(cell) -> list:
    """Space-separated "location_id@YYYY-MM-DD" tokens -> list of (date, site), in listed order."""
    if pd.isna(cell):
        return []
    out = []
    for token in str(cell).split():
        site, sep, day = token.partition("@")
        try:
            value = float(site)
            when = date.fromisoformat(day)
        except ValueError:
            raise ValueError(f"site_days tokens must look like '1012@2024-01-03', got {token!r}")
        if not sep or not value.is_integer():
            raise ValueError(f"site_days tokens must look like '1012@2024-01-03', got {token!r}")
        out.append((when, int(value)))
    return out


def _execute(cell, week: date, relevant: set) -> float:
    """Run a week's plan in date order (listed order within a day) against the weekly budget;
    return its net value. Repeats, non-test sites and dates outside the week are skipped."""
    plan = sorted(enumerate(_tokens(cell)), key=lambda t: (t[1][0], t[0]))
    left, net, seen = WEEKLY_BUDGET, 0.0, set()
    for _, (when, site) in plan:
        if (when, site) in seen or site not in INSPECTION_HOURS or not week <= when < week + timedelta(days=7):
            continue
        seen.add((when, site))
        hours = INSPECTION_HOURS[site]
        if hours <= left + 1e-9:
            left -= hours
            net += (CATCH_VALUE if (when, site) in relevant else 0.0) - hours
    return net


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Normalised net inspection value under a weekly crew budget (higher is better, 0-1).
    Each week's plan is executed in date order; an inspection is skipped if it no longer fits
    in the week's remaining hours. Net value = CATCH_VALUE per inspected relevant site-day
    minus the hours of every executed inspection. The reference plan inspects exactly the
    relevant site-days, in date order, under the same budget.
        score = min(1, max(0, total net / total reference net))
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
    plans = dict(zip(sub[QUERY], sub[TARGET_COL]))

    queries = answers[QUERY].astype(str).str.strip()
    absent = [q for q in queries if q not in plans]
    if absent:
        raise ValueError(f"Submission is missing {len(absent)} query_id values, e.g. {absent[0]}")

    net, best = 0.0, 0.0
    for q, rel_cell in zip(queries, answers[TARGET_COL]):
        week = _week_start(q)
        relevant = set(_tokens(rel_cell))
        net += _execute(plans[q], week, relevant)
        best += _execute(rel_cell, week, relevant)
    if best <= 0:
        return 1.0 if net >= 0 else 0.0
    return min(1.0, max(0.0, net / best))
