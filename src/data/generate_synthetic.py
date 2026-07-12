"""Generate a synthetic PeMS-style dataset so the whole pipeline runs end-to-end.

We simulate traffic FLOW (vehicles/hour) for a set of sensors over a timeline, with:
  * a base level per sensor,
  * a daily double-peak (morning + evening rush),
  * a weekly pattern (weekends lighter),
  * random noise,
  * and — crucially — a KNOWN, ground-truth uplift caused by planned events.

The event uplift is what makes this a meaningful testbed: a model that ignores events
literally cannot explain the event-driven spikes, while the event-aware model can. The
per-row ground-truth uplift is saved (`true_event_effect`) so Model B has a learnable
target and so the comparison can zoom in on event-affected rows.

Outputs (parquet, in data/synthetic/):
  sensors.parquet : sensor_id, lat, lon, base_flow
  events.parquet  : the "scraped" event metadata (this is all a real scraper would give)
  flow.parquet    : sensor_id, timestamp, flow, true_event_effect
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, EVENTS_FILE, FLOW_FILE, SENSORS_FILE
from ..utils import haversine_km, hours_since

EVENT_CATEGORIES = ["concert", "sports", "festival", "conference"]
# Rough per-category multiplier on how "traffic-heavy" a given attendance is.
CATEGORY_INTENSITY = {"concert": 1.0, "sports": 1.2, "festival": 0.9, "conference": 0.6}


def _sensors(rng, cfg) -> pd.DataFrame:
    n = cfg["n_sensors"]
    area = cfg["area"]
    return pd.DataFrame(
        {
            "sensor_id": np.arange(n),
            "lat": rng.uniform(area["lat_min"], area["lat_max"], n),
            "lon": rng.uniform(area["lon_min"], area["lon_max"], n),
            # Base flow (veh/h) varies by sensor: some are arterials, some quiet.
            "base_flow": rng.uniform(200, 1200, n).round(0),
        }
    )


def _events(rng, cfg, timeline) -> pd.DataFrame:
    n = cfg["n_events"]
    area = cfg["area"]
    start_idx = rng.integers(0, len(timeline), n)
    starts = timeline[start_idx]
    durations_h = rng.choice([2, 3, 4, 5], n)  # event length in hours
    cats = rng.choice(EVENT_CATEGORIES, n)
    return pd.DataFrame(
        {
            "event_id": np.arange(n),
            "lat": rng.uniform(area["lat_min"], area["lat_max"], n),
            "lon": rng.uniform(area["lon_min"], area["lon_max"], n),
            "start_time": starts,
            "duration_h": durations_h,
            "category": cats,
            # Expected popularity / attendance — the key "how big" signal from scraping.
            "expected_attendance": rng.integers(500, 60000, n),
        }
    )


def _daily_weekly_profile(ts: pd.DatetimeIndex) -> np.ndarray:
    """Multiplicative base profile (dimensionless) with rush-hour peaks + weekend dip."""
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()  # 0=Mon
    # Two Gaussian rush peaks (08:00 and 17:30).
    morning = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening = np.exp(-0.5 * ((hour - 17.5) / 1.8) ** 2)
    daily = 0.35 + 0.9 * morning + 1.0 * evening  # floor so nights aren't zero
    weekly = np.where(dow >= 5, 0.7, 1.0)  # weekends lighter
    return daily * weekly


def _event_effect(
    events: pd.DataFrame, sensors: pd.DataFrame, timeline: pd.DatetimeIndex, radius_km: float
) -> np.ndarray:
    """Ground-truth fractional flow uplift, shape (n_sensors, n_timesteps).

    Each event raises flow on nearby sensors, peaking around its start (arrival) and its
    end (departure), scaled by attendance, category, and decaying with distance.
    """
    n_s, n_t = len(sensors), len(timeline)
    effect = np.zeros((n_s, n_t))
    origin = timeline[0]
    t_hours = hours_since(timeline, origin)  # float array (n_t,)

    s_lat = sensors["lat"].to_numpy(dtype=float)
    s_lon = sensors["lon"].to_numpy(dtype=float)

    # Pull every event column into numpy up front so the loop is pure float arithmetic.
    ev_lat = events["lat"].to_numpy(dtype=float)
    ev_lon = events["lon"].to_numpy(dtype=float)
    ev_dur = events["duration_h"].to_numpy(dtype=float)
    ev_att = events["expected_attendance"].to_numpy(dtype=float)
    ev_cat = events["category"].to_numpy()
    ev_start_h = hours_since(pd.DatetimeIndex(events["start_time"]), origin)

    for j in range(len(events)):
        dist = haversine_km(s_lat, s_lon, ev_lat[j], ev_lon[j])  # (n_s,)
        near = dist <= radius_km
        if not np.any(near):
            continue
        # Distance decay (1 at the venue -> 0 at the radius edge).
        dist_factor = np.clip(1.0 - dist / radius_km, 0.0, 1.0)

        start_h = ev_start_h[j]
        end_h = start_h + ev_dur[j]
        # Arrival bump before start, departure bump after end.
        arrival = np.exp(-0.5 * ((t_hours - (start_h - 1)) / 1.0) ** 2)
        departure = np.exp(-0.5 * ((t_hours - (end_h + 0.5)) / 1.0) ** 2)
        temporal = arrival + departure  # (n_t,)

        intensity = CATEGORY_INTENSITY[ev_cat[j]]
        # Attendance -> magnitude (saturating). 60k people ~ up to ~1.5x uplift at venue.
        magnitude = intensity * 1.6 * (ev_att[j] / 60000.0)

        effect += magnitude * np.outer(dist_factor * near, temporal)

    return effect


def generate(cfg: dict | None = None, seed: int | None = None) -> None:
    cfg = cfg or CFG
    seed = CFG["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    dcfg = cfg["data"]

    timeline = pd.date_range(
        dcfg["start_date"], periods=dcfg["n_days"] * _steps_per_day(dcfg["freq"]), freq=dcfg["freq"]
    )

    sensors = _sensors(rng, dcfg)
    events = _events(rng, dcfg, timeline)
    effect = _event_effect(events, sensors, timeline, cfg["features"]["event_radius_km"])

    profile = _daily_weekly_profile(timeline)  # (n_t,)
    base = sensors["base_flow"].to_numpy()[:, None]  # (n_s, 1)
    noise = rng.normal(1.0, 0.08, size=(len(sensors), len(timeline)))

    flow = base * profile[None, :] * (1.0 + effect) * noise
    flow = np.clip(flow, 0, None).round(1)

    # Long format (one row per sensor x timestamp) — matches how PeMS is usually loaded.
    long = pd.DataFrame(
        {
            "sensor_id": np.repeat(sensors["sensor_id"].to_numpy(), len(timeline)),
            "timestamp": np.tile(timeline.to_numpy(), len(sensors)),
            "flow": flow.reshape(-1),
            "true_event_effect": effect.reshape(-1),
        }
    )

    sensors.to_parquet(SENSORS_FILE, index=False)
    events.to_parquet(EVENTS_FILE, index=False)
    long.to_parquet(FLOW_FILE, index=False)

    n_affected = int((long["true_event_effect"] > 0.05).sum())
    print(f"[generate] sensors={len(sensors)} events={len(events)} rows={len(long):,}")
    print(
        f"[generate] event-affected rows (effect>0.05): {n_affected:,} "
        f"({100 * n_affected / len(long):.1f}%)"
    )
    print(f"[generate] wrote:\n  {SENSORS_FILE}\n  {EVENTS_FILE}\n  {FLOW_FILE}")


def _steps_per_day(freq: str) -> int:
    # pd.Timedelta parses fixed offset aliases like "1h" / "15min" directly.
    delta = pd.Timedelta(freq)
    return int(pd.Timedelta(days=1) / delta)


if __name__ == "__main__":
    generate()
