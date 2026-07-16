"""Shared train/evaluate/save routine for the traffic model runs (A and A+).

Runs A and A+ differ only in their feature set and output names. Sharing the mechanics here
is what keeps the A vs A+ comparison fair.
"""

from __future__ import annotations

from pathlib import Path
import time

import joblib

from ..config import target_column
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
    df = load_dataset(cfg)
    for col in require_cols or []:
        if col not in df.columns:
            raise RuntimeError(f"required column {col!r} missing — did an earlier stage run?")

    train, test = df[df["is_train"]], df[~df["is_train"]]
    tgt = target_column(cfg)

    model = build_model(cfg)
    train_started = time.perf_counter()
    model.fit(train[feats], train["target"])
    training_seconds = time.perf_counter() - train_started

    # Predictions come out in the (possibly per-sensor-normalized) target space. Invert them to
    # real units so the metrics are comparable no matter how normalization is configured.
    preds_norm, latency = timed_predict(model, test[feats])
    preds = preds_norm * test["flow_scale"].to_numpy()
    metrics = regression_metrics(test[tgt], preds)
    metrics["timing"] = {**latency, "training_seconds": training_seconds}

    joblib.dump(model, model_file)
    # Keep only what compare and the figures need. in_event_window flags the event-affected subset
    # for both real and synthetic data, true_event_effect is the synthetic ground truth, and raw
    # flow is kept for the flow-based figures when it isn't already the target.
    keep = ["sensor_id", "timestamp", tgt, "in_event_window"]
    if "flow" in test.columns and "flow" not in keep:
        keep.append("flow")
    if "true_event_effect" in test.columns:
        keep.append("true_event_effect")
    out = test[keep].copy()
    out[pred_col] = preds
    out.to_parquet(pred_file, index=False)
    return metrics
