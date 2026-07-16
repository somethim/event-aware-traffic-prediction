"""Predeclared non-ML and linear reference predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


def reference_predictions(
    train: pd.DataFrame, test: pd.DataFrame, target: str
) -> dict[str, np.ndarray]:
    """Return persistence/calendar references using information available before each row."""
    sensor_mean = train.groupby(
        ["sensor_id", train["timestamp"].dt.dayofweek, train["timestamp"].dt.time]
    )[target].mean()
    fallback = train.groupby("sensor_id")[target].mean()
    hist = []
    timestamps = pd.DatetimeIndex(test["timestamp"])
    for sensor_id, timestamp in zip(test["sensor_id"].to_numpy(), timestamps):
        key = (sensor_id, timestamp.dayofweek, timestamp.time())
        hist.append(sensor_mean.get(key, fallback.get(sensor_id, train[target].mean())))
    out = {"historical_sensor_weekday_time_mean": np.asarray(hist, dtype=float)}
    for label, lag in (
        ("previous_interval", 1),
        ("same_time_yesterday", 96),
        ("same_time_last_week", 672),
    ):
        col = f"lag_{lag}"
        if col in test:
            out[label] = test[col].to_numpy(dtype=float)
    return out


def ridge_predictions(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> np.ndarray:
    model = Ridge(alpha=1.0)
    model.fit(train[features], train["target"])
    return model.predict(test[features]) * test["flow_scale"].to_numpy()
