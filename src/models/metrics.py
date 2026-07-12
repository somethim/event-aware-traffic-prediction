"""Evaluation metrics for the thesis criteria: accuracy, reliability, response time."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def mape(y_true, y_pred, eps: float = 1e-6) -> float:
    """Mean absolute percentage error (%). eps guards against dividing by zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), eps))) * 100)


def regression_metrics(y_true, y_pred) -> dict:
    """Accuracy + reliability summary."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAPE": mape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        # Reliability is captured by the worst-case and tail error below.
        "max_abs_error": float(err.max()),
        "p95_abs_error": float(np.percentile(err, 95)),
        "n": int(len(y_true)),
    }


def timed_predict(model, x):
    """Return predictions plus latency in ms per 1000 rows, the 'response time' criterion."""
    t0 = time.perf_counter()
    preds = model.predict(x)
    dt = time.perf_counter() - t0
    per_1k = dt / max(len(x), 1) * 1000 * 1000
    return preds, per_1k
