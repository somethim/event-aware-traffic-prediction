"""Central config and path helpers. Everything reads knobs from config/config.yaml.

API keys come from the environment via `get_env`, loaded from a gitignored `.env` at import
time so secrets are never hard-coded.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Repo root is two levels up from this file (src/config.py -> src -> repo root).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "config.yaml"

# Load .env if present. Real environment variables take precedence over the file.
load_dotenv(ROOT / ".env")


def get_env(name: str, required: bool = False, default: str | None = None) -> str | None:
    """Read an API key / secret from the environment (populated from .env)."""
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(
            f"Environment variable {name!r} is not set. Copy .env.example to .env and fill it in."
        )
    return val


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the YAML config as a plain dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


# Prediction target ------------------------------------------------------------
# data.target selects what the traffic model predicts so one pipeline can forecast flow
# (default), speed, or occupancy. That string is also the column name in the flow table.
TARGET_UNITS = {
    "flow": "veh/interval",
    "speed": "mph",
    "occupancy": "occupancy",
}


def target_column(cfg: dict[str, Any] | None = None) -> str:
    """Name of the column being predicted (data.target; defaults to 'flow')."""
    cfg = cfg or CFG
    return cfg.get("data", {}).get("target", "flow")


def target_units(cfg: dict[str, Any] | None = None) -> str:
    """Human-readable unit label for the current target (for figures/metrics)."""
    return TARGET_UNITS.get(target_column(cfg), "units")


def resolve(*parts: str) -> Path:
    """Resolve a path relative to the repo root and ensure its parent exists."""
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# Paths used across modules ----------------------------------------------------
CFG = load_config()

# All data/output directories hang off this root. Tests set EATP_DATA_ROOT to a temp dir so
# running the suite never reads or overwrites real data under the repo. Because every path
# constant below (and the copies other modules import) derives from it, the override reaches
# the whole pipeline from this one place.
DATA_ROOT = Path(os.environ.get("EATP_DATA_ROOT") or ROOT)

INPUTS_DIR = DATA_ROOT / CFG["data"]["inputs_dir"]
PROCESSED_DIR = DATA_ROOT / CFG["data"]["processed_dir"]
MODELS_DIR = DATA_ROOT / CFG["paths"]["models_dir"]
RESULTS_DIR = DATA_ROOT / CFG["paths"]["results_dir"]
FIGURES_DIR = DATA_ROOT / CFG["paths"]["figures_dir"]

for _d in (INPUTS_DIR, PROCESSED_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

FLOW_FILE = INPUTS_DIR / "flow.parquet"  # sensor by time flow, long format
EVENTS_FILE = INPUTS_DIR / "events.parquet"  # one row per planned event
SENSORS_FILE = INPUTS_DIR / "sensors.parquet"  # sensor id and location
