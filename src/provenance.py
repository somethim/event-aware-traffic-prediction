"""Dataset/run provenance and stale-cache guards."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .config import (
    CFG,
    CONFIG_PATH,
    EVENTS_FILE,
    FLOW_FILE,
    RESULTS_DIR,
    SENSORS_FILE,
    config_checksum,
)

INPUT_FILES = (FLOW_FILE, EVENTS_FILE, SENSORS_FILE)


def file_checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def input_checksums() -> dict[str, str]:
    missing = [str(p) for p in INPUT_FILES if not p.exists()]
    if missing:
        raise RuntimeError(f"cannot create manifest; missing inputs: {missing}")
    return {p.name: file_checksum(p) for p in INPUT_FILES}


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or CFG
    flow, events, sensors = (pd.read_parquet(p) for p in INPUT_FILES)
    target = cfg.get("data", {}).get("target", "flow")
    healthy = sensors.groupby("city")["sensor_id"].nunique().astype(int).to_dict()
    deps = {}
    for name in ("numpy", "pandas", "scikit-learn", "xgboost", "pyarrow"):
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "schema_version": 1,
        "source_profile": cfg["data"]["source"],
        "config_path": str(CONFIG_PATH),
        "config_checksum": config_checksum(cfg),
        "input_checksums": input_checksums(),
        "date_range": [
            str(pd.to_datetime(flow["timestamp"]).min()),
            str(pd.to_datetime(flow["timestamp"]).max()),
        ],
        "counts": {
            "rows": int(len(flow)),
            "events": int(len(events)),
            "sensors": int(sensors["sensor_id"].nunique()),
            "healthy_sensors_by_city": healthy,
        },
        "target_availability": {
            c: int(flow[c].notna().sum()) for c in ("flow", "speed", "occupancy") if c in flow
        },
        "seed": int(cfg["seed"]),
        "git_commit": _git_commit(),
        "dependencies": deps,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(path: Path, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    manifest = build_manifest(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp.replace(path)
    return manifest


def validate_manifest(manifest: dict[str, Any], cfg: dict[str, Any] | None = None) -> None:
    cfg = cfg or CFG
    if manifest.get("config_checksum") != config_checksum(cfg):
        raise RuntimeError("stale cache rejected: configuration checksum differs from manifest")
    if manifest.get("input_checksums") != input_checksums():
        raise RuntimeError("stale cache rejected: input checksum differs from manifest")


def write_run_manifest(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    return write_manifest(RESULTS_DIR.parent / "run_manifest.json", cfg)
