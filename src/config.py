"""Central config + path helpers. Everything reads knobs from config/config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/config.py -> src -> repo root).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config as a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve(*parts: str) -> Path:
    """Resolve a path relative to the repo root and ensure its parent exists."""
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# Convenience: paths used across modules ---------------------------------------
CFG = load_config()

SYNTHETIC_DIR = ROOT / CFG["data"]["raw_dir"]
PROCESSED_DIR = ROOT / CFG["data"]["processed_dir"]
MODELS_DIR = ROOT / CFG["paths"]["models_dir"]
RESULTS_DIR = ROOT / CFG["paths"]["results_dir"]

for _d in (SYNTHETIC_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Canonical file names
FLOW_FILE = SYNTHETIC_DIR / "flow.parquet"  # sensor x time flow (long format)
EVENTS_FILE = SYNTHETIC_DIR / "events.parquet"  # one row per planned event
SENSORS_FILE = SYNTHETIC_DIR / "sensors.parquet"  # sensor id + location
