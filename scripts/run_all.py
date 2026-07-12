"""One-shot runner: generate data -> prepare -> train A, B, A+ -> compare.

python -m scripts.run_all
"""

from __future__ import annotations

from src.config import FLOW_FILE
from src.data.generate_synthetic import generate
from src.pipeline import compare, prepare, train_baseline, train_event_aware, train_event_model


def main() -> None:
    if not FLOW_FILE.exists():
        print("== 1/6 generate synthetic data ==")
        generate()
    else:
        print(f"== 1/6 data already present ({FLOW_FILE.name}), skipping generation ==")

    print("\n== 2/6 prepare features ==")
    prepare.prepare()

    print("\n== 3/6 train A (baseline) ==")
    train_baseline.main()

    print("\n== 4/6 train B (event impact) ==")
    train_event_model.main()

    print("\n== 5/6 train A+ (event-aware) ==")
    train_event_aware.main()

    print("\n== 6/6 compare ==")
    compare.main()


if __name__ == "__main__":
    main()
