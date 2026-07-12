"""Force a rebuild of the real dataset (PeMS flow from Google Drive plus setlist.fm events).

This is a thin wrapper around src.data.build_real.build_real, which holds the actual logic so
the runners can reuse it via ensure_dataset. Afterward the other scripts operate on real data.

    uv run python -m scripts.build_real_dataset
"""

from __future__ import annotations

from src.config import CFG
from src.data.build_real import build_real


def main() -> None:
    build_real(CFG)


if __name__ == "__main__":
    main()
