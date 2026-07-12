"""Shared train/evaluate/save routine for the traffic model runs (A and A+).

Both runs are identical except for their feature set and output names, so the mechanics
live here once. Keeping them in sync is exactly what makes the A vs A+ comparison fair.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from ..models.metrics import regression_metrics, timed_predict
from ..models.traffic_model import build_model
from .prepare import load_dataset


def train_eval_traffic(
    cfg: dict,
    feats: list[str],
    model_file: Path,
    pred_file: Path,
    pred_col: str,
    require_cols: list[str] | None = None,
) -> dict:
    """Fit the configured traffic model on `feats`, evaluate on the held-out test split,
    persist the model and the test-set predictions, and return the metrics."""
    df = load_dataset()
    for col in require_cols or []:
        if col not in df.columns:
            raise RuntimeError(f"required column {col!r} missing — did an earlier stage run?")

    train, test = df[df["is_train"]], df[~df["is_train"]]

    model = build_model(cfg)  # same builder for A and A+ -> only the feature set differs
    model.fit(train[feats], train["flow"])

    preds, latency = timed_predict(model, test[feats])
    metrics = regression_metrics(test["flow"], preds)
    metrics["inference_ms_per_1k"] = latency

    joblib.dump(model, model_file)
    out = test[["sensor_id", "timestamp", "flow", "true_event_effect"]].copy()
    out[pred_col] = preds
    out.to_parquet(pred_file, index=False)
    return metrics
