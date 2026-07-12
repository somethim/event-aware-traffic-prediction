"""The traffic-prediction model (used for BOTH run A and run A+).

`build_model()` returns a fresh estimator of the configured type. Both the baseline and the
event-aware runs call this, guaranteeing they use the identical model type/hyperparameters —
so the only thing that differs between A and A+ is the feature set.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from ..config import CFG


def build_model(cfg: dict | None = None):
    cfg = cfg or CFG
    mcfg = cfg["model"]
    mtype = mcfg["type"]
    seed = cfg["seed"]

    if mtype == "random_forest":
        p = mcfg["random_forest"]
        return RandomForestRegressor(
            n_estimators=p["n_estimators"],
            max_depth=p["max_depth"],
            min_samples_leaf=p["min_samples_leaf"],
            n_jobs=p["n_jobs"],
            random_state=seed,
        )
    if mtype == "gradient_boosting":
        p = mcfg["gradient_boosting"]
        return GradientBoostingRegressor(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            random_state=seed,
        )
    if mtype == "xgboost":
        # Lazy import so xgboost is only required when this model is actually selected.
        from xgboost import XGBRegressor

        from ..device import resolve_device

        p = mcfg["xgboost"]
        device = resolve_device(mcfg.get("device", "auto"))
        print(f"[model] xgboost on device={device}")
        return XGBRegressor(
            n_estimators=p["n_estimators"],
            learning_rate=p["learning_rate"],
            max_depth=p["max_depth"],
            subsample=p.get("subsample", 0.9),
            colsample_bytree=p.get("colsample_bytree", 0.9),
            tree_method="hist",  # GPU path uses the same histogram algorithm
            device=device,  # "cuda" or "cpu"
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(
        f"Unknown model.type: {mtype!r} " "(use 'random_forest', 'gradient_boosting', or 'xgboost')"
    )
