"""Generate a synthetic PeMS-style dataset so the whole pipeline runs end-to-end.

We simulate traffic flow (vehicles/hour) for a set of sensors over a timeline from a per-sensor
base level, a daily double-peak (morning and evening rush), a lighter weekly weekend pattern,
random noise, and a known ground-truth uplift caused by planned events.

That event uplift is what makes this a meaningful testbed: a model that ignores events cannot
explain the event-driven spikes, while an event-aware model can. We save the per-row uplift as
`true_event_effect` so Model B has a learnable target and so the comparison can zoom in on
event-affected rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CFG, EVENTS_FILE, FLOW_FILE, SENSORS_FILE
from ..utils import haversine_km, hours_since

EVENT_CATEGORIES = ["concert", "sports", "festival", "conference"]
# Rough per-category multiplier on how traffic-heavy a given attendance is.
CATEGORY_INTENSITY = {"concert": 1.0, "sports": 1.2, "festival": 0.9, "conference": 0.6}


def _sensors(rng, city: dict, id_offset: int) -> pd.DataFrame:
    n = city["n_sensors"]
    area = city["area"]
    lo, hi = city["base_flow"]
    return pd.DataFrame(
        {
            "sensor_id": np.arange(n) + id_offset,  # id_offset keeps ids unique across cities
            "city": city["name"],
            "lat": rng.uniform(area["lat_min"], area["lat_max"], n),
            "lon": rng.uniform(area["lon_min"], area["lon_max"], n),
            # Base flow range differs per city (LA is busier than SF).
            "base_flow": rng.uniform(lo, hi, n).round(0),
        }
    )


def _events(rng, city: dict, timeline, id_offset: int) -> pd.DataFrame:
    n = city["n_events"]
    area = city["area"]
    start_idx = rng.integers(0, len(timeline), n)
    starts = timeline[start_idx]
    durations_h = rng.choice([2, 3, 4, 5], n)
    cats = rng.choice(EVENT_CATEGORIES, n)
    return pd.DataFrame(
        {
            "event_id": np.arange(n) + id_offset,
            "city": city["name"],
            "lat": rng.uniform(area["lat_min"], area["lat_max"], n),
            "lon": rng.uniform(area["lon_min"], area["lon_max"], n),
            "start_time": starts,
            "duration_h": durations_h,
            "category": cats,
            # Expected attendance is the main "how big" signal a real scraper would give.
            "expected_attendance": rng.integers(500, 60000, n),
        }
    )


def _daily_weekly_profile(ts: pd.DatetimeIndex) -> np.ndarray:
    """Multiplicative base profile (dimensionless) with rush-hour peaks + weekend dip."""
    hour = ts.hour.to_numpy()
    dow = ts.dayofweek.to_numpy()  # 0 is Monday
    # Two Gaussian rush peaks at 08:00 and 17:30.
    morning = np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
    evening = np.exp(-0.5 * ((hour - 17.5) / 1.8) ** 2)
    daily = 0.35 + 0.9 * morning + 1.0 * evening  # the 0.35 floor keeps nights above zero
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
    t_hours = hours_since(timeline, origin)  # (n_t,)

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
        # Linear distance decay: 1 at the venue down to 0 at the radius edge.
        dist_factor = np.clip(1.0 - dist / radius_km, 0.0, 1.0)

        start_h = ev_start_h[j]
        end_h = start_h + ev_dur[j]
        # Traffic bumps as people arrive before the start and leave after the end.
        arrival = np.exp(-0.5 * ((t_hours - (start_h - 1)) / 1.0) ** 2)
        departure = np.exp(-0.5 * ((t_hours - (end_h + 0.5)) / 1.0) ** 2)
        temporal = arrival + departure  # (n_t,)

        intensity = CATEGORY_INTENSITY[ev_cat[j]]
        # Attendance drives a saturating magnitude: 60k people gives roughly 1.5x at the venue.
        magnitude = intensity * 1.6 * (ev_att[j] / 60000.0)

        effect += magnitude * np.outer(dist_factor * near, temporal)

    return effect


def _city_flow(rng, city: dict, timeline, radius_km: float, sid_offset: int, eid_offset: int):
    """Generate (sensors, events, long-flow) for one city."""
    sensors = _sensors(rng, city, sid_offset)
    events = _events(rng, city, timeline, eid_offset)
    effect = _event_effect(events, sensors, timeline, radius_km)

    profile = _daily_weekly_profile(timeline)
    base = sensors["base_flow"].to_numpy()[:, None]  # (n_s, 1) to broadcast over time
    noise = rng.normal(1.0, 0.08, size=(len(sensors), len(timeline)))
    flow = np.clip(base * profile[None, :] * (1.0 + effect) * noise, 0, None).round(1)

    long = pd.DataFrame(
        {
            "sensor_id": np.repeat(sensors["sensor_id"].to_numpy(), len(timeline)),
            "city": city["name"],
            "timestamp": np.tile(timeline.to_numpy(), len(sensors)),
            "flow": flow.reshape(-1),
            "true_event_effect": effect.reshape(-1),
        }
    )
    return sensors, events, long


def generate(cfg: dict | None = None, seed: int | None = None) -> None:
    cfg = cfg or CFG
    seed = CFG["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    dcfg = cfg["data"]
    radius = cfg["features"]["event_radius_km"]

    timeline = pd.date_range(
        dcfg["start_date"], periods=dcfg["n_days"] * _steps_per_day(dcfg["freq"]), freq=dcfg["freq"]
    )

    sensor_frames, event_frames, flow_frames = [], [], []
    sid_offset = eid_offset = 0
    for city in dcfg["cities"]:
        s, e, f = _city_flow(rng, city, timeline, radius, sid_offset, eid_offset)
        sensor_frames.append(s)
        event_frames.append(e)
        flow_frames.append(f)
        sid_offset += len(s)
        eid_offset += len(e)

    sensors = pd.concat(sensor_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    long = pd.concat(flow_frames, ignore_index=True)

    sensors.to_parquet(SENSORS_FILE, index=False)
    events.to_parquet(EVENTS_FILE, index=False)
    long.to_parquet(FLOW_FILE, index=False)

    names = ", ".join(c["name"] for c in dcfg["cities"])
    n_affected = int((long["true_event_effect"] > 0.05).sum())
    print(
        f"[generate] cities=[{names}] sensors={len(sensors)} events={len(events)} "
        f"rows={len(long):,}"
    )
    print(
        f"[generate] event-affected rows (effect>0.05): {n_affected:,} "
        f"({100 * n_affected / len(long):.1f}%)"
    )
    print(f"[generate] wrote:\n  {SENSORS_FILE}\n  {EVENTS_FILE}\n  {FLOW_FILE}")


def _steps_per_day(freq: str) -> int:
    # pd.Timedelta parses fixed offset aliases like "1h" or "15min" directly.
    delta = pd.Timedelta(freq)
    return int(pd.Timedelta(days=1) / delta)


if __name__ == "__main__":
    generate()
