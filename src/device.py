"""GPU/CPU device detection for GPU-capable models (XGBoost).

sklearn's RandomForest and GradientBoosting are CPU-only, so this only matters when
`model.type = xgboost`. XGBoost 2.x takes a single `device` string of "cuda" or "cpu",
resolved here from the config preference and auto-detecting a usable NVIDIA GPU.
"""

from __future__ import annotations

import shutil
import subprocess


def cuda_available() -> bool:
    """True if an NVIDIA GPU looks usable (``nvidia-smi`` is present and exits cleanly)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return False
    try:
        subprocess.run([exe], capture_output=True, timeout=5, check=True)
        return True
    except (subprocess.SubprocessError, OSError):
        return False


def resolve_device(preference: str = "auto") -> str:
    """Map a config preference to an XGBoost device string.

    "cpu" stays on CPU, "cuda" or "gpu" forces CUDA, and "auto" picks CUDA when a GPU is
    detected and CPU otherwise.
    """
    pref = preference.lower()
    if pref == "cpu":
        return "cpu"
    if pref in ("cuda", "gpu"):
        return "cuda"
    return "cuda" if cuda_available() else "cpu"
