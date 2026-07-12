"""Run A+ — event-aware traffic model.

Identical model type to run A, identical traffic features, PLUS the single extra feature
`event_impact_score` produced by Model B. Requires train_event_model to have run first.
"""

from __future__ import annotations

from ..config import CFG, MODELS_DIR, PROCESSED_DIR
from ..features.build_features import traffic_feature_columns
from ._common import train_eval_traffic

MODEL_FILE = MODELS_DIR / "model_A_plus_event_aware.joblib"
PRED_FILE = PROCESSED_DIR / "pred_A_plus_event_aware.parquet"


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    feats = traffic_feature_columns(cfg) + ["event_impact_score"]
    metrics = train_eval_traffic(
        cfg,
        feats,
        MODEL_FILE,
        PRED_FILE,
        "event_aware_pred",
        require_cols=["event_impact_score"],
    )

    print(
        f"[A+/event] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
        f"MAPE={metrics['MAPE']:.2f}%  R2={metrics['R2']:.3f}"
    )
    print(f"[A+/event] saved {MODEL_FILE.name} and {PRED_FILE.name}")
    return metrics


if __name__ == "__main__":
    main()
