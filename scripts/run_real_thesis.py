"""Focused claim-bearing LA/SF study (three targets, RF/XGBoost, raw events/placebo)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/real.yaml")
    parser.add_argument(
        "--skip-rebuild",
        action="store_true",
        help="reuse inputs only when their manifest is already trusted",
    )
    args = parser.parse_args()
    os.environ["EATP_CONFIG"] = str(Path(args.config).resolve())
    run_id = "real-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.environ["EATP_RUN_ID"] = run_id

    from src.config import CFG, RESULTS_DIR, ROOT, config_checksum
    from src.data.build_real import build_real, ensure_dataset
    from src.pipeline.evaluate import evaluate
    from src.provenance import input_checksums, write_run_manifest

    if args.skip_rebuild:
        ensure_dataset(CFG)
    else:
        build_real(CFG)  # real.yaml predeclares refresh_events: true
    runs = []
    for target in ("flow", "speed", "occupancy"):
        for model in ("random_forest", "xgboost"):
            cfg = json.loads(json.dumps(CFG))
            cfg["data"]["target"] = target
            cfg["model"]["type"] = model
            result = evaluate(cfg)
            result.update(
                {
                    "run_id": run_id,
                    "source_profile": "real",
                    "target": target,
                    "horizon_minutes": 15,
                    "model": model,
                    "primary_comparison": "A_vs_A+raw",
                }
            )
            runs.append(result)

    payload = {
        "run_id": run_id,
        "config_checksum": config_checksum(CFG),
        "dataset_checksums": input_checksums(),
        "runs": runs,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / "fold_results.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(RESULTS_DIR / "fold_results.json")
    write_run_manifest(CFG)
    pointer = ROOT / "runs" / "latest-verified"
    pointer.parent.mkdir(exist_ok=True)
    temp = pointer.with_name(".latest-verified.tmp")
    temp.unlink(missing_ok=True)
    temp.symlink_to(run_id)
    temp.replace(pointer)
    print(f"[verified] completed {run_id}; all reported cells are in fold_results.json")


if __name__ == "__main__":
    main()
