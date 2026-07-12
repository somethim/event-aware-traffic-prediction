"""Model B — the event-impact model.

Given the event-exposure features for a (sensor, timestamp), predict a scalar "event
impact score": how much extra traffic the nearby planned events are expected to cause,
expressed as a fractional flow uplift (same units as `true_event_effect`).

That single predicted score is the extra column that turns run A into run A+.

Target
------
- Synthetic data: we train directly on the known `true_event_effect` (clean supervision).
- Real data: you won't have this. The documented approach is to use a PROXY target —
  train the baseline traffic model first, then use its residual (actual - predicted) on
  event days as Model B's target. See `derive_proxy_target()`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from ..config import CFG


def build_event_model(cfg: dict | None = None):
    cfg = cfg or CFG
    # Random Forest here too: robust, gives feature importances over the event features.
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=cfg["seed"],
    )


def derive_proxy_target(flow_with_pred: pd.DataFrame) -> np.ndarray:
    """Proxy target for REAL data: fractional residual of the baseline model.

    Expects columns `flow` (actual) and `baseline_pred`. Returns clipped fractional uplift
    (actual/pred - 1), floored at 0 since events add traffic. Use this in place of
    `true_event_effect` when no ground-truth event effect exists.
    """
    pred = np.maximum(flow_with_pred["baseline_pred"].to_numpy(), 1.0)
    resid = flow_with_pred["flow"].to_numpy() / pred - 1.0
    return np.clip(resid, 0.0, None)
