"""Build the modelling dataset once and cache it to data/processed/dataset.parquet.

Runs A, B and A+ all read this single cached frame, which is what keeps them on identical
rows and splits.
"""

from __future__ import annotations

import pandas as pd

from ..config import CFG, PROCESSED_DIR
from ..data.load import load_events, load_flow, load_sensors
from ..features.build_features import event_features, traffic_features
from ..provenance import validate_manifest, write_manifest
from .preprocess import assign_split, normalize_per_sensor

DATASET_FILE = PROCESSED_DIR / "dataset.parquet"
DATASET_MANIFEST_FILE = PROCESSED_DIR / "dataset_manifest.json"


def build_feature_frame(cfg: dict | None = None) -> pd.DataFrame:
    """Traffic and event-exposure features per (sensor, timestamp), before split and normalize.

    Kept separate so the cross-validation evaluator (pipeline.evaluate) can build features once
    and then reuse them across many train/test splits without recomputing.
    """
    cfg = cfg or CFG
    flow = load_flow()
    events = load_events()
    sensors = load_sensors()

    df = traffic_features(flow, cfg)
    ev = event_features(flow, events, sensors, cfg)
    return df.merge(ev, on=["sensor_id", "timestamp"], how="left")


def prepare(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or CFG
    df = build_feature_frame(cfg)

    df = assign_split(df, cfg)
    df = normalize_per_sensor(df, cfg)

    df.to_parquet(DATASET_FILE, index=False)
    write_manifest(DATASET_MANIFEST_FILE, cfg)
    split = cfg["model"]["split"]
    detail = (
        f"time (test={split['time_size']})"
        if split["mode"] == "time"
        else f"city ({split['train_city']}->{split['test_city']})"
    )
    print(
        f"[prepare] rows={len(df):,}  train={int(df['is_train'].sum()):,}  "
        f"test={int((~df['is_train']).sum()):,}  split={detail}  "
        f"normalize={cfg['model'].get('normalize', 'none')}"
    )
    print(f"[prepare] wrote {DATASET_FILE}")
    return df


def load_dataset(cfg: dict | None = None) -> pd.DataFrame:
    if not DATASET_FILE.exists():
        return prepare(cfg)
    if cfg is not None:
        if not DATASET_MANIFEST_FILE.exists():
            raise RuntimeError("stale cache rejected: dataset manifest is missing")
        import json

        validate_manifest(json.loads(DATASET_MANIFEST_FILE.read_text()), cfg)
    df = pd.read_parquet(DATASET_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


if __name__ == "__main__":
    prepare()
