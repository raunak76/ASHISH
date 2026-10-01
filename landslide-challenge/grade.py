import numpy as np
import pandas as pd

TARGET = "landslide_risk_48h"
ID = "record_id"
EPS = 1e-15


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    """
    Location-balanced log loss (lower is better):
        0.5 * LogLoss(rows at seen locations) + 0.5 * LogLoss(rows at unseen locations)
    Predictions are clipped to [1e-15, 1 - 1e-15].
    Raise only for truly invalid submissions.
    """
    missing = {ID, TARGET} - set(submission.columns)
    if missing:
        raise ValueError(f"Submission is missing columns: {sorted(missing)}")
    if len(submission) != len(answers):
        raise ValueError(f"Expected {len(answers)} rows, got {len(submission)}")
    if submission[ID].duplicated().any():
        raise ValueError("Submission contains duplicate record_id values")

    sub = submission[[ID, TARGET]].copy()
    sub[ID] = pd.to_numeric(sub[ID], errors="coerce")
    merged = answers.merge(sub, on=ID, how="left", suffixes=("_true", "_pred"))
    pred = pd.to_numeric(merged[f"{TARGET}_pred"], errors="coerce")
    if pred.isna().any():
        raise ValueError("Submission has missing, non-numeric or unknown record_id predictions")
    if ((pred < 0) | (pred > 1)).any():
        raise ValueError("Predictions must be probabilities in [0, 1]")

    y = merged[f"{TARGET}_true"].to_numpy(dtype=float)
    p = np.clip(pred.to_numpy(dtype=float), EPS, 1 - EPS)
    unseen = merged["unseen_location"].to_numpy() == 1

    if unseen.all() or not unseen.any():
        return _log_loss(y, p)
    return 0.5 * _log_loss(y[~unseen], p[~unseen]) + 0.5 * _log_loss(y[unseen], p[unseen])
