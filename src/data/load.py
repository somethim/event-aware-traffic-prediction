"""Load traffic / event / sensor tables.

Right now these read the synthetic parquet files. To move to REAL data for the thesis,
implement the `load_real_*` functions to return DataFrames with the SAME columns and the
rest of the pipeline keeps working unchanged.

Expected schemas
----------------
flow    : sensor_id (int), timestamp (datetime64), flow (float)
          [+ true_event_effect (float) only in synthetic data]
events  : event_id, lat, lon, start_time (datetime64), duration_h, category,
          expected_attendance (int)
sensors : sensor_id (int), lat (float), lon (float)
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


# --- Real-data hooks (implement for the thesis's real-world evaluation) -------------
def load_real_flow_pems(path: str) -> pd.DataFrame:
    """TODO: parse PEMS-BAY / PeMS flow into the schema above.

    PEMS-BAY commonly ships as an HDF5 (`pems-bay.h5`) of shape (time, sensors). Melt it to
    long format: columns sensor_id, timestamp, flow. Then return.
    """
    raise NotImplementedError("Plug in PEMS-BAY / PeMS parsing here.")


def load_real_events(path: str) -> pd.DataFrame:
    """TODO: load events produced by src/data/scrape_events.py into the events schema."""
    raise NotImplementedError("Plug in scraped events here.")
