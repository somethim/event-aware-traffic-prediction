"""Run the full experiment matrix (split x normalize x model) and compile the comparison
table into docs/thesis-notes.md.

    python -m scripts.run_experiments
"""

from __future__ import annotations

from src.pipeline.experiments import run


def main() -> None:
    run()


if __name__ == "__main__":
    main()
