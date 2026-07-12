"""Model B, the event-impact model.

From the event-exposure features for a (sensor, timestamp) it predicts a scalar event
impact score: the extra traffic nearby planned events are expected to cause, as a
fractional flow uplift in the same units as `true_event_effect`. That predicted score is
the one extra column that turns run A into run A+.

On synthetic data it trains directly on the known `true_event_effect`. On real data there
is no ground truth, so it trains on a proxy target built from the baseline model's positive
residual (see `train_event_model._proxy_target`).
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from ..config import CFG


def build_event_model(cfg: dict | None = None):
    cfg = cfg or CFG
    # A Random Forest again, since it is robust and gives feature importances over the events.
    return RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=cfg["seed"],
    )
