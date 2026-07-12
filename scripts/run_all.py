"""Run the whole thing end to end: build the dataset if missing, the full experiment
matrix, the default headline pipeline, and all figures.

This takes several minutes. gradient_boosting is the slow part, so trim GRID in
src/pipeline/experiments.py if you need it faster. Which dataset is used comes from
data.source in config (real PeMS data by default, synthetic for the controlled testbed).

    uv run python -m scripts.run_all
"""

from __future__ import annotations

from src.config import CFG
from src.data.build_real import ensure_dataset
from src.pipeline import (
    compare,
    experiments,
    prepare,
    train_baseline,
    train_event_aware,
    train_event_model,
    visualize,
)


def main() -> None:
    ensure_dataset(CFG)

    print("\n########## 1/3  EXPERIMENT MATRIX (all models × all conditions) ##########")
    experiments.run()

    # The experiment matrix leaves the dataset in whatever state its last cell used, so
    # re-prepare from the default config before computing the headline metrics and figures.
    print("\n########## 2/3  DEFAULT headline pipeline (A / B / A+ / compare) ##########")
    prepare.prepare(CFG)
    train_baseline.main(CFG)
    train_event_model.main(CFG)
    train_event_aware.main(CFG)
    compare.main(CFG)

    print("\n########## 3/3  FIGURES ##########")
    visualize.run()

    print(
        "\nDONE.\n"
        "  matrix   -> docs/thesis-notes.md (Experiment matrix section)\n"
        "  metrics  -> media/results/ (experiments.json, metrics.json)\n"
        "  figures  -> media/figures/"
    )


if __name__ == "__main__":
    main()
