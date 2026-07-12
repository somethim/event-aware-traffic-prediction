"""Statistical-rigor run: rolling-origin CV across multiple seeds, plus a significance test
and bootstrap CI on the A -> A+ MAE gap. Reads the `evaluation` block in config/config.yaml
and runs on whichever dataset is currently built. Writes media/results/stats.json.

    uv run python -m scripts.run_stats
"""

from __future__ import annotations

from src.config import CFG
from src.data.build_real import ensure_dataset
from src.pipeline import evaluate


def main() -> None:
    ensure_dataset(CFG)
    evaluate.evaluate(CFG)


if __name__ == "__main__":
    main()
