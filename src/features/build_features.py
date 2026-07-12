"""Feature engineering.

Two families of features are kept separate on purpose. traffic_features() builds the
temporal and historical lag/rolling features that the baseline model (A) is allowed to see,
with no event information. event_features() builds the per (sensor, timestamp) event
exposure that feeds Model B, whose prediction becomes the one extra column A+ gets over A.

Keeping them apart is what makes the comparison clean: A uses traffic_features, A+ uses
traffic_features plus the event_impact_score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, target_column
from ..utils import haversine_km, hours_since


# --- Traffic (baseline) features ----------------------------------------------------
def traffic_features(flow: pd.DataFrame, cfg: dict | None = None) -> pd.DataFrame:
    """Add temporal, lag, and rolling features to the long flow table.

    Lags and rollings are computed on the configured target metric (data.target) within each
    sensor so no sensor leaks into another, and they only use past values. Rows with an
    undefined lag (start of each series) or a missing target are dropped.
    """
    cfg = cfg or CFG
    fcfg = cfg["features"]
    tgt = target_column(cfg)
    df = flow.sort_values(["sensor_id", "timestamp"]).copy()

    ts = df["timestamp"].dt
    df["hour"] = ts.hour
    df["dayofweek"] = ts.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = ts.month
    # Cyclical encodings so the model treats 23:00 and 00:00 as adjacent.
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)

    g = df.groupby("sensor_id")[tgt]
    for lag in fcfg["lags"]:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in fcfg["rolling_windows"]:
        # transform keeps each rolling window within one sensor, and shift(1) first so the
        # window never includes the current target timestep. win=w is bound per lambda to
        # avoid the late-binding closure trap.
        df[f"roll_mean_{w}"] = g.transform(lambda s, win=w: s.shift(1).rolling(win).mean())
        df[f"roll_std_{w}"] = g.transform(lambda s, win=w: s.shift(1).rolling(win).std())

    lag_cols = [f"lag_{l}" for l in fcfg["lags"]]
    roll_cols = [f"roll_{stat}_{w}" for w in fcfg["rolling_windows"] for stat in ("mean", "std")]
    # Rolling columns must be checked too: a NaN gap in the series (real data has them from the
    # %Observed gating) makes a rolling feature NaN even when every lag is defined, and
    # sklearn's GradientBoostingRegressor rejects NaN inputs outright.
    df = df.dropna(subset=lag_cols + roll_cols + [tgt]).reset_index(drop=True)
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
    "n_active_events",
    "nearest_event_km",  # distance to nearest relevant event, or the radius if none
    "sum_attendance_exposed",  # attendance weighted and distance decayed
    "max_attendance_exposed",
    "hours_to_next_event",
    "in_event_window",  # 1 while inside [start-lead, end+post] of any nearby event
    # Rises as a nearby event approaches, before it even starts, which lets the model predict
    # leaving-early congestion earlier than the lag features can.
    "pre_event_pressure",
]


def event_features(
    flow: pd.DataFrame, events: pd.DataFrame, sensors: pd.DataFrame, cfg: dict | None = None
) -> pd.DataFrame:
    """Compute per (sensor, timestamp) event-exposure features.

    Returns a frame keyed by (sensor_id, timestamp) with EVENT_FEATURE_COLUMNS. It is built
    only from event metadata (location, time, attendance) and sensor location, which is
    exactly what a real event scraper provides, so it stays honest about what information the
    event signal actually carries.
    """
    cfg = cfg or CFG
    radius = cfg["features"]["event_radius_km"]
    lead = cfg["features"].get("event_lead_hours", 3.0)  # how early the approach window opens
    post = 1.0  # departure buffer after the event ends

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
        pre_pressure = np.zeros(n_t)
        nearest_km = np.full(n_t, radius)
        hours_to_next = np.full(n_t, 999.0)

        for j in np.where(rel)[0]:
            s, e = ev_start_h[j], ev_end_h[j]
            d = dist[j]
            att = ev_att[j]
            decay = max(0.0, 1.0 - d / radius)

            # Window opens `lead` hours before the start for the approach period and closes
            # `post` hours after the end for the departure period.
            window = (t_h >= s - lead) & (t_h <= e + post)
            n_active += window
            in_win = np.maximum(in_win, window.astype(float))
            sum_att += window * att * decay
            max_att = np.maximum(max_att, window * att * decay)
            nearest_km = np.where(window & (d < nearest_km), d, nearest_km)

            # Anticipatory pressure lives only in the [start-lead, start] approach window and
            # ramps from 0 at `lead` hours out to 1 at the start, scaled by attendance and
            # proximity. It is the "people are already heading there" cue.
            approach = (t_h >= s - lead) & (t_h < s)
            imminence = np.clip(1.0 - (s - t_h) / lead, 0.0, 1.0)
            pre_pressure = np.maximum(pre_pressure, approach * imminence * att * decay)

            upcoming = s - t_h  # hours until this event starts, negative once it has passed
            eligible = upcoming >= -ev_dur[j] - post
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
                    "pre_event_pressure": pre_pressure,
                }
            )
        )

    out = pd.concat(frames, ignore_index=True)
    return out
