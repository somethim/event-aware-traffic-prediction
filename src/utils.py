"""Small shared helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def haversine_km(
    lat1: float | FloatArray,
    lon1: float | FloatArray,
    lat2: float | FloatArray,
    lon2: float | FloatArray,
) -> FloatArray:
    """Great-circle distance in km. Accepts scalars or numpy arrays (broadcasts)."""
    r = 6371.0
    rlat1, rlon1, rlat2, rlon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = np.sin(dlat / 2) ** 2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2) ** 2
    return np.asarray(r * 2 * np.arcsin(np.sqrt(a)), dtype=float)


def hours_since(index: pd.DatetimeIndex, origin: pd.Timestamp) -> FloatArray:
    """Hours elapsed from `origin` for each timestamp, as a plain float array.

    Concentrates the (poorly-typed) pandas Timedelta arithmetic in one place so callers can
    work in clean numpy float space.
    """
    delta = (index - origin) / pd.Timedelta(hours=1)
    return np.asarray(delta, dtype=float)


def time_split_mask(timestamps: npt.ArrayLike, test_size: float) -> tuple[BoolArray, BoolArray]:
    """Boolean train/test masks with a time-based split (test = last `test_size` of the
    timeline). Time-based (not random) so we never train on the future — essential for a
    forecasting evaluation."""
    ts = np.asarray(timestamps)
    order = np.sort(np.unique(ts))
    cutoff = order[int(len(order) * (1 - test_size))]
    is_train: BoolArray = ts < cutoff
    return is_train, ~is_train
