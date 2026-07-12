"""Model B — event-impact model.

Trains on event-exposure features to predict the event impact score, then writes that
predicted score back into the cached dataset as `event_impact_score` so run A+ can use it.

Target selection:
  * synthetic data -> `true_event_effect` (ground truth).
  * real data      -> proxy from the baseline residual (needs run A first).
"""

from __future__ import annotations

import joblib

from ..config import CFG, MODELS_DIR
from ..features.build_features import EVENT_FEATURE_COLUMNS
from ..models.event_impact_model import build_event_model
from ..models.metrics import regression_metrics
from .prepare import DATASET_FILE, load_dataset

MODEL_FILE = MODELS_DIR / "model_B_event.joblib"


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    df = load_dataset()

    if "true_event_effect" in df.columns:
        target = "true_event_effect"
    else:
        raise RuntimeError(
            "No ground-truth event effect (real-data mode). Derive a proxy target with "
            "models.event_impact_model.derive_proxy_target using run A's residuals, then "
            "train Model B on it. See the module docstring."
        )

    train, test = df[df["is_train"]], df[~df["is_train"]]
    model = build_event_model(cfg)
    model.fit(train[EVENT_FEATURE_COLUMNS], train[target])

    # Predict impact for ALL rows and store it — this becomes the extra A+ feature.
    df["event_impact_score"] = model.predict(df[EVENT_FEATURE_COLUMNS])
    df.to_parquet(DATASET_FILE, index=False)

    metrics = regression_metrics(test[target], model.predict(test[EVENT_FEATURE_COLUMNS]))
    joblib.dump(model, MODEL_FILE)

    print(f"[B/event]  target={target}  MAE={metrics['MAE']:.4f}  R2={metrics['R2']:.3f}")
    print(f"[B/event]  wrote event_impact_score into dataset, saved {MODEL_FILE.name}")
    return metrics


if __name__ == "__main__":
    main()
