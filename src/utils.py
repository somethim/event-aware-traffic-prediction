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

    Keeps the awkwardly-typed pandas Timedelta arithmetic in one place so callers can work in
    plain numpy floats.
    """
    delta = (index - origin) / pd.Timedelta(hours=1)
    return np.asarray(delta, dtype=float)


def to_local_naive(times: pd.Series, tz: str) -> pd.Series:
    """Convert event timestamps to `tz` local wall-clock, then drop the tz to get naive-local.

    PeMS flow timestamps are Pacific-local wall-clock with no timezone attached, while event
    feeds return UTC or tz-aware times. Matching the two naively puts events about 7-8 hours
    off, so this maps event times into the same local wall-clock PeMS uses. tz-aware input is
    converted, and tz-naive input is assumed to already be UTC.
    """
    ts = pd.to_datetime(times, utc=True)
    return ts.dt.tz_convert(tz).dt.tz_localize(None)


def steps_per(freq: str, period: str = "1D") -> int:
    """How many `freq` timesteps fit in `period` (e.g. steps_per('15min','1D') == 96).

    Lets feature windows be expressed relative to the data resolution instead of hard-coding them.
    """
    return int(pd.Timedelta(period) / pd.Timedelta(freq))


def time_split_mask(timestamps: npt.ArrayLike, test_size: float) -> tuple[BoolArray, BoolArray]:
    """Boolean train/test masks with a time-based split, where the test set is the last
    `test_size` fraction of the timeline. The split is time-based rather than random so we
    never train on the future, which matters for an honest forecasting evaluation."""
    ts = np.asarray(timestamps)
    order = np.sort(np.unique(ts))
    cutoff = order[int(len(order) * (1 - test_size))]
    is_train: BoolArray = ts < cutoff
    return is_train, ~is_train
