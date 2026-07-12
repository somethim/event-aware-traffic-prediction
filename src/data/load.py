"""Load traffic, event, and sensor tables from the parquet files.

The synthetic generator and the real-data build both write these same files, so the pipeline
is source-agnostic.

Schemas
-------
flow    : sensor_id (int), timestamp (datetime64), flow (float), city (str)
          [+ true_event_effect (float) only in synthetic data]
events  : event_id, city, lat, lon, start_time (datetime64), duration_h, category,
          expected_attendance (int)
sensors : sensor_id (int), city (str), lat (float), lon (float)
"""

from __future__ import annotations

import pandas as pd

from ..config import EVENTS_FILE, FLOW_FILE, SENSORS_FILE


def load_flow() -> pd.DataFrame:
    df = pd.read_parquet(FLOW_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)


def load_events() -> pd.DataFrame:
    df = pd.read_parquet(EVENTS_FILE)
    df["start_time"] = pd.to_datetime(df["start_time"])
    return df


def load_sensors() -> pd.DataFrame:
    return pd.read_parquet(SENSORS_FILE)
