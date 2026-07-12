"""The two configurable experiment knobs: how the data is split, and whether flow is
normalized per sensor.

Split modes (`model.split.mode`): "time" holds out the last `time_size` of the timeline for
within-city forecasting, which never trains on the future and is the clean setting for the
event hypothesis. "city" trains on `train_city` and tests on `test_city` (e.g. LA->SF) to ask
whether the benefit generalizes across cities.

Normalization (`model.normalize`): "per_sensor" divides every flow-valued column by that
sensor's mean flow so the model learns relative patterns. Trees split on absolute thresholds
and don't extrapolate, so this is what makes LA->SF transfer fair when the two cities have
different flow magnitudes. The scale is learned on train rows only, and predictions are
converted back to real units before scoring.
"""

from __future__ import annotations

import pandas as pd

from ..config import target_column
from ..utils import time_split_mask

WARMUP_FRAC = 0.2  # early slice used to estimate a test-only sensor's scale in the city split


def flow_unit_columns(cfg: dict) -> list[str]:
    """Feature columns measured in flow units (so they scale with per-sensor normalization)."""
    fcfg = cfg["features"]
    return (
        [f"lag_{lag}" for lag in fcfg["lags"]]
        + [f"roll_mean_{w}" for w in fcfg["rolling_windows"]]
        + [f"roll_std_{w}" for w in fcfg["rolling_windows"]]
    )


def assign_split(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add an `is_train` column per `model.split.mode`. For city mode, also drops rows from
    cities other than train/test."""
    split = cfg["model"]["split"]
    mode = split["mode"]
    df = df.copy()

    if mode == "time":
        is_train, _ = time_split_mask(df["timestamp"], split["time_size"])
        df["is_train"] = is_train
        return df

    if mode == "city":
        if "city" not in df.columns:
            raise ValueError("split.mode='city' needs a 'city' column (multi-city dataset).")
        train_city, test_city = split["train_city"], split["test_city"]
        df = df[df["city"].isin([train_city, test_city])].copy()
        df["is_train"] = df["city"] == train_city
        return df.reset_index(drop=True)

    raise ValueError(f"unknown split.mode: {mode!r} (use 'time' or 'city')")


def _warmup_scale(df: pd.DataFrame, tgt: str, frac: float = WARMUP_FRAC) -> pd.Series:
    """Per-sensor mean of `tgt` over each sensor's earliest `frac` of timestamps.

    For a test-only sensor in the city split this approximates a typical volume known before the
    forecast window. Using it as the normalization scale avoids leaking test-period information,
    which the sensor's full-period mean would do.
    """
    d = df.sort_values(["sensor_id", "timestamp"])
    rank = d.groupby("sensor_id").cumcount()
    kthr = (d.groupby("sensor_id")["sensor_id"].transform("size") * frac).astype(int).clip(lower=1)
    return d[rank < kthr].groupby("sensor_id")[tgt].mean()


def normalize_per_sensor(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add `flow_scale` and `target` columns, and scale flow-unit features when enabled.

    `target` is what the model is trained to predict: the normalized metric, or the raw metric
    when normalization is off. `flow_scale` maps it back to real units. The raw target column is
    left untouched so metrics are always computed in real units.
    """
    df = df.copy()
    tgt = target_column(cfg)
    if cfg["model"].get("normalize", "none") == "per_sensor":
        # The scale is learned on train rows. Test-only sensors in city mode instead fall back to
        # a warm-up-slice mean, their typical volume known before deployment, since their full
        # test-period mean would leak test-period information into the normalization.
        ids = df["sensor_id"].unique()
        train_mean = df[df["is_train"]].groupby("sensor_id")[tgt].mean().reindex(ids)
        scale = train_mean.fillna(_warmup_scale(df, tgt).reindex(ids)).clip(lower=1e-6)
        df["flow_scale"] = df["sensor_id"].map(scale).astype(float)
        for col in flow_unit_columns(cfg):
            df[col] = df[col] / df["flow_scale"]
    else:
        df["flow_scale"] = 1.0

    df["target"] = df[tgt] / df["flow_scale"]
    return df
