"""Generate all thesis figures into media/figures/.

python -m scripts.run_visuals
"""

from __future__ import annotations

from src.pipeline.visualize import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
