"""Run primary A+raw, the event-aware traffic model.

Same model and traffic features as run A, plus the predeclared raw planned-event features.
"""

from __future__ import annotations

from ..config import CFG, MODELS_DIR, PROCESSED_DIR
from ..features.build_features import EVENT_FEATURE_COLUMNS, traffic_feature_columns
from ._common import train_eval_traffic

MODEL_FILE = MODELS_DIR / "model_A_plus_event_aware.joblib"
PRED_FILE = PROCESSED_DIR / "pred_A_plus_event_aware.parquet"


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    feats = traffic_feature_columns(cfg) + EVENT_FEATURE_COLUMNS
    metrics = train_eval_traffic(
        cfg,
        feats,
        MODEL_FILE,
        PRED_FILE,
        "event_aware_pred",
        require_cols=EVENT_FEATURE_COLUMNS,
    )

    print(
        f"[A+raw] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
        f"WAPE={metrics['WAPE']:.2f}%  R2={metrics['R2']:.3f}"
    )
    print(f"[A+raw] saved {MODEL_FILE.name} and {PRED_FILE.name}")
    return metrics


if __name__ == "__main__":
    main()
