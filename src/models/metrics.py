"""Evaluation metrics for the thesis criteria: accuracy, reliability, response time."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def wape(y_true, y_pred) -> float:
    """Weighted absolute percentage error (%), defined even for individual zero flows."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom * 100) if denom else float("nan")


def regression_metrics(y_true, y_pred) -> dict:
    """Accuracy + reliability summary."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "WAPE": wape(y_true, y_pred),
        "R2": float(r2_score(y_true, y_pred)),
        # Reliability is captured by the worst-case and tail error below.
        "p95_abs_error": float(np.percentile(err, 95)),
        "n": int(len(y_true)),
    }


def timed_predict(model, x, repeats: int = 5):
    """Warm up, then report repeated inference timing instead of a one-off measurement."""
    model.predict(x.iloc[: min(len(x), 1000)])
    samples = []
    preds = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        preds = model.predict(x)
        samples.append((time.perf_counter() - t0) / max(len(x), 1) * 1_000_000)
    return preds, {
        "inference_ms_per_1k_mean": float(np.mean(samples)),
        "inference_ms_per_1k_std": float(np.std(samples, ddof=1)) if repeats > 1 else 0.0,
        "repeats": repeats,
        "batch_size": int(len(x)),
    }
