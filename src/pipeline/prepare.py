"""Build the modelling dataset once and cache it.

Produces data/processed/dataset.parquet with, per (sensor_id, timestamp):
  * traffic (baseline) features,
  * event-exposure features,
  * flow (target for A / A+),
  * true_event_effect (target for Model B on synthetic data),
  * is_train (time-based split flag).

Run A, B and A+ all read this single cached frame so they operate on identical rows/splits.
"""

from __future__ import annotations

import pandas as pd

from ..config import CFG, PROCESSED_DIR
from ..data.load import load_events, load_flow, load_sensors
from ..features.build_features import event_features, traffic_features
from ..utils import time_split_mask

DATASET_FILE = PROCESSED_DIR / "dataset.parquet"


def prepare(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or CFG
    flow = load_flow()
    events = load_events()
    sensors = load_sensors()

    df = traffic_features(flow, cfg)  # adds temporal + lag/rolling
    ev = event_features(flow, events, sensors, cfg)  # per (sensor,timestamp) exposure
    df = df.merge(ev, on=["sensor_id", "timestamp"], how="left")

    is_train, _ = time_split_mask(df["timestamp"], cfg["model"]["test_size"])
    df["is_train"] = is_train

    df.to_parquet(DATASET_FILE, index=False)
    print(
        f"[prepare] dataset rows={len(df):,}  train={int(is_train.sum()):,}  "
        f"test={int((~is_train).sum()):,}"
    )
    print(f"[prepare] wrote {DATASET_FILE}")
    return df


def load_dataset() -> pd.DataFrame:
    if not DATASET_FILE.exists():
        return prepare()
    df = pd.read_parquet(DATASET_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    prepare()
