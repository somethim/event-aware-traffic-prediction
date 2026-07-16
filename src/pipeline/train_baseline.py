"""Run A — baseline traffic model. Uses ONLY traffic features (no event info)."""

from __future__ import annotations

from ..config import CFG, MODELS_DIR, PROCESSED_DIR
from ..features.build_features import traffic_feature_columns
from ._common import train_eval_traffic

MODEL_FILE = MODELS_DIR / "model_A_baseline.joblib"
PRED_FILE = PROCESSED_DIR / "pred_A_baseline.parquet"


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    feats = traffic_feature_columns(cfg)
    metrics = train_eval_traffic(cfg, feats, MODEL_FILE, PRED_FILE, "baseline_pred")

    print(
        f"[A/baseline] MAE={metrics['MAE']:.2f}  RMSE={metrics['RMSE']:.2f}  "
        f"WAPE={metrics['WAPE']:.2f}%  R2={metrics['R2']:.3f}"
    )
    print(f"[A/baseline] saved {MODEL_FILE.name} and {PRED_FILE.name}")
    return metrics


if __name__ == "__main__":
    main()
