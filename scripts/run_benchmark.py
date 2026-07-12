"""Run every model in config's benchmark.models through A and A+, then compare them all.

python -m scripts.run_benchmark
"""

from __future__ import annotations

from src.pipeline.benchmark import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
