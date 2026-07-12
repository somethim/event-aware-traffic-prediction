"""The traffic-prediction model used for both run A and run A+.

Both runs build their estimator here so they share the same model type and
hyperparameters. That way the feature set is the only thing that differs between A and A+.
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
        # Imported lazily so xgboost is only a dependency when this model is selected.
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
            tree_method="hist",
            device=device,
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(
        f"Unknown model.type: {mtype!r} " "(use 'random_forest', 'gradient_boosting', or 'xgboost')"
    )
