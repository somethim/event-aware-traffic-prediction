"""Feature engineering.

Two families of features, kept separate on purpose:

  traffic_features()  -> temporal + historical (lag/rolling) features. These are what the
                         BASELINE traffic model (A) is allowed to see. No event info.

  event_features()    -> per (sensor, timestamp) "event exposure": how close/large/imminent
                         the nearby planned events are. These feed Model B, and Model B's
                         prediction is the single extra column that A+ gets over A.

Keeping them separate is what makes the A vs A+ comparison clean: A = traffic_features,
A+ = traffic_features + event_impact_score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG
from ..utils import haversine_km, hours_since


# --- Traffic (baseline) features ----------------------------------------------------
def traffic_features(flow: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add temporal + lag + rolling features to the long flow table.

    Lags/rollings are computed WITHIN each sensor (groupby) so no sensor leaks into another,
    and only use past values. Rows with undefined lags (start of each series) are dropped.
    """
    cfg = cfg or CFG
    fcfg = cfg["features"]
    df = flow.sort_values(["sensor_id", "timestamp"]).copy()

    ts = df["timestamp"].dt
    df["hour"] = ts.hour
    df["dayofweek"] = ts.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = ts.month
    # Cyclical encodings so the model sees 23:00 and 00:00 as adjacent.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    g = df.groupby("sensor_id")["flow"]
    for lag in fcfg["lags"]:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in fcfg["rolling_windows"]:
        # transform keeps the rolling window WITHIN each sensor (never crosses boundaries);
        # shift(1) first so the window never includes the current (target) timestep.
        # win=w binds the current window into each lambda (avoids late-binding closures).
        df[f"roll_mean_{w}"] = g.transform(lambda s, win=w: s.shift(1).rolling(win).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s, win=w: s.shift(1).rolling(win).std())

    lag_cols = [f"lag_{l}" for l in fcfg["lags"]]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)
    return df


def traffic_feature_columns(cfg: dict | None = None) -> list[str]:
    cfg = cfg or CFG
    fcfg = cfg["features"]
    cols = [
        "hour",
        "dayofweek",
        "is_weekend",
        "month",
        "hour_sin",
        "hour_cos",
        "dow_sin",
        "dow_cos",
    ]
    cols += [f"lag_{l}" for l in fcfg["lags"]]
    cols += [f"roll_mean_{w}" for w in fcfg["rolling_windows"]]
    cols += [f"roll_std_{w}" for w in fcfg["rolling_windows"]]
    return cols


# --- Event exposure features --------------------------------------------------------
EVENT_FEATURE_COLUMNS = [
    "n_active_events",  # events overlapping this timestep within radius
    "nearest_event_km",  # distance to nearest relevant event (radius if none)
    "sum_attendance_exposed",  # attendance-weighted, distance-decayed exposure
    "max_attendance_exposed",
    "hours_to_next_event",  # signed-ish: hours until the nearest upcoming event start
    "in_event_window",  # 1 if inside [start-1h, end+1h] of any nearby event
]


def event_features(
    flow: pd.DataFrame, events: pd.DataFrame, sensors: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Compute per (sensor, timestamp) event-exposure features.

    Returns a frame keyed by (sensor_id, timestamp) with EVENT_FEATURE_COLUMNS. This is
    deliberately built ONLY from event metadata (location/time/attendance) + sensor
    location — exactly what a real event scraper provides — so it is honest about what
    information the event signal actually carries.
    """
    cfg = cfg or CFG
    radius = cfg["features"]["event_radius_km"]

    # Everything to numpy up front -> the loops stay in clean float space.
    sensor_ids = sensors["sensor_id"].to_numpy()
    sensor_lat = sensors["lat"].to_numpy(dtype=float)
    sensor_lon = sensors["lon"].to_numpy(dtype=float)

    timestamps = np.sort(flow["timestamp"].unique())
    t = pd.DatetimeIndex(pd.to_datetime(timestamps))
    t_h = hours_since(t, t[0])
    n_t = len(t_h)

    ev_lat = events["lat"].to_numpy(dtype=float)
    ev_lon = events["lon"].to_numpy(dtype=float)
    ev_dur = events["duration_h"].to_numpy(dtype=float)
    ev_att = events["expected_attendance"].to_numpy(dtype=float)
    ev_start_h = hours_since(pd.DatetimeIndex(events["start_time"]), t[0])
    ev_end_h = ev_start_h + ev_dur

    frames = []
    for i in range(len(sensor_ids)):
        dist = haversine_km(sensor_lat[i], sensor_lon[i], ev_lat, ev_lon)
        rel = dist <= radius

        n_active = np.zeros(n_t)
        sum_att = np.zeros(n_t)
        max_att = np.zeros(n_t)
        in_win = np.zeros(n_t)
        nearest_km = np.full(n_t, radius)
        hours_to_next = np.full(n_t, 999.0)

        for j in np.where(rel)[0]:
            s, e = ev_start_h[j], ev_end_h[j]
            d = dist[j]
            att = ev_att[j]
            decay = max(0.0, 1.0 - d / radius)

            window = (t_h >= s - 1.0) & (t_h <= e + 1.0)
            n_active += window
            in_win = np.maximum(in_win, window.astype(float))
            sum_att += window * att * decay
            max_att = np.maximum(max_att, window * att * decay)
            nearest_km = np.where(window & (d < nearest_km), d, nearest_km)

            upcoming = s - t_h  # hours until this event starts (negative once passed)
            eligible = upcoming >= -ev_dur[j] - 1.0
            cand = np.where(eligible, np.abs(upcoming), 999.0)
            hours_to_next = np.minimum(hours_to_next, cand)

        frames.append(
            pd.DataFrame(
                {
                    "sensor_id": sensor_ids[i],
                    "timestamp": t,
                    "n_active_events": n_active,
                    "nearest_event_km": nearest_km,
                    "sum_attendance_exposed": sum_att,
                    "max_attendance_exposed": max_att,
                    "hours_to_next_event": np.clip(hours_to_next, 0, 72),
                    "in_event_window": in_win,
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    return out
