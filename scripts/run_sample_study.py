"""Minutes-scale synthetic study with RF/XGBoost, rolling folds, seeds, and placebo."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/sample.yaml")
    args = parser.parse_args()
    os.environ["EATP_CONFIG"] = str(Path(args.config).resolve())
    run_id = "sample-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.environ["EATP_RUN_ID"] = run_id
    started = time.perf_counter()

    from src.config import CFG, RESULTS_DIR, ROOT, config_checksum
    from src.data.generate_synthetic import generate
    from src.pipeline.evaluate import evaluate
    from src.provenance import input_checksums, write_run_manifest

    generate(CFG)
    studies = []
    for model_type in ("random_forest", "xgboost"):
        cfg = copy.deepcopy(CFG)
        cfg["model"]["type"] = model_type
        print(f"\n########## {model_type.upper()} ##########")
        result = evaluate(cfg)
        result.update(
            {
                "run_id": run_id,
                "source_profile": "sample",
                "horizon_minutes": 15,
                "primary_comparison": "A_vs_A+raw",
            }
        )
        studies.append(result)

    elapsed = time.perf_counter() - started
    payload = {
        "run_id": run_id,
        "config_checksum": config_checksum(CFG),
        "dataset_checksums": input_checksums(),
        "elapsed_seconds": elapsed,
        "model_fits": len(CFG["evaluation"]["seeds"]) * CFG["evaluation"]["cv"]["n_folds"] * 6 * 2,
        "note": "Each cell fits A, three raw-feature ablations, A+raw, and +7-day placebo.",
        "studies": studies,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / "sample_results.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(RESULTS_DIR / "sample_results.json")
    write_run_manifest(CFG)

    pointer = ROOT / "runs" / "latest-verified"
    pointer.parent.mkdir(exist_ok=True)
    temp = pointer.with_name(".latest-verified.tmp")
    temp.unlink(missing_ok=True)
    temp.symlink_to(run_id)
    temp.replace(pointer)
    print(
        f"\n[verified] {run_id}: {payload['model_fits']} fits in {elapsed:.1f}s "
        f"-> {RESULTS_DIR / 'sample_results.json'}"
    )


if __name__ == "__main__":
    main()
