"""Model B, the event-impact model.

Trains on event-exposure features to predict the event impact score, then writes that
predicted score back into the cached dataset as `event_impact_score` so run A+ can use it.
On synthetic data the target is the ground-truth `true_event_effect`; on real data we fall
back to a proxy built from the baseline residual, which needs run A first.
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.model_selection import KFold

from ..config import CFG, MODELS_DIR, target_column
from ..features.build_features import EVENT_FEATURE_COLUMNS, traffic_feature_columns
from ..models.event_impact_model import build_event_model
from ..models.metrics import regression_metrics
from ..models.traffic_model import build_model
from .prepare import DATASET_FILE, load_dataset

MODEL_FILE = MODELS_DIR / "model_B_event.joblib"


def _proxy_target(df, cfg: dict) -> np.ndarray:
    """Real-data target: the baseline model's positive residual as a fraction of its prediction,
    computed as clip(actual/predicted - 1, 0, inf). This is the traffic a history-only model
    could not explain, which on event days is mostly the event's doing.

    The baseline predictions are cross-fitted so the proxy is not inflated by in-sample
    overfitting. Each train row is predicted by a baseline fit on the other folds, and test rows
    are predicted by a baseline fit on all train rows. This is framed for flow and occupancy,
    where events raise the target; for speed the residual sign inverts.
    """
    feats = traffic_feature_columns(cfg)
    tgt = target_column(cfg)
    scale = df["flow_scale"].to_numpy()
    actual = df[tgt].to_numpy()
    is_train = df["is_train"].to_numpy()
    train_idx = np.where(is_train)[0]

    x_all = df[feats].to_numpy()
    y_train = df["target"].to_numpy()[train_idx]
    x_train = x_all[train_idx]

    pred = np.empty(len(df))
    # Out-of-fold predictions for the train rows.
    n_splits = int(cfg["model"].get("cross_fit_folds", 5))
    n_splits = max(2, min(n_splits, len(train_idx)))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg["seed"])
    for tr, va in kf.split(train_idx):
        m = build_model(cfg)
        m.fit(x_train[tr], y_train[tr])
        pred[train_idx[va]] = m.predict(x_train[va])
    test_idx = np.where(~is_train)[0]
    if len(test_idx):
        m = build_model(cfg)
        m.fit(x_train, y_train)
        pred[test_idx] = m.predict(x_all[test_idx])

    pred = np.maximum(pred * scale, 1.0)
    return np.clip(actual / pred - 1.0, 0.0, None)


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    df = load_dataset()

    if "true_event_effect" in df.columns:  # synthetic data has ground truth
        y = df["true_event_effect"].to_numpy()
        target_kind = "true_event_effect"
    else:
        y = _proxy_target(df, cfg)
        target_kind = "proxy_residual"

    is_train = df["is_train"].to_numpy()
    model = build_event_model(cfg)
    model.fit(df[EVENT_FEATURE_COLUMNS][is_train], y[is_train])

    # Score every row and cache it, since this becomes the extra feature for run A+.
    df["event_impact_score"] = model.predict(df[EVENT_FEATURE_COLUMNS])
    df.to_parquet(DATASET_FILE, index=False)

    metrics = regression_metrics(y[~is_train], model.predict(df[EVENT_FEATURE_COLUMNS][~is_train]))
    joblib.dump(model, MODEL_FILE)

    print(f"[B/event]  target={target_kind}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.3f}")
    print(f"[B/event]  wrote event_impact_score into dataset, saved {MODEL_FILE.name}")
    return metrics


if __name__ == "__main__":
    main()
