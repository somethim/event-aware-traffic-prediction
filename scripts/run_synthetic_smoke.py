"""Under-10-minute mechanism smoke test on the compact synthetic profile."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def _bootstrap(config: str) -> str:
    os.environ["EATP_CONFIG"] = str(Path(config).resolve())
    run_id = "synthetic-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.environ["EATP_RUN_ID"] = run_id
    return run_id


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/synthetic.yaml")
    args = p.parse_args()
    run_id = _bootstrap(args.config)
    started = time.perf_counter()

    from src.config import CFG, RESULTS_DIR, ROOT, config_checksum, target_column
    from src.data.generate_synthetic import generate
    from src.features.build_features import EVENT_FEATURE_COLUMNS, traffic_feature_columns
    from src.models.metrics import regression_metrics
    from src.models.references import reference_predictions, ridge_predictions
    from src.pipeline import compare, prepare, train_baseline, train_event_aware
    from src.provenance import input_checksums, write_run_manifest

    generate(CFG)
    df = prepare.prepare(CFG)
    train, test = df[df["is_train"]], df[~df["is_train"]]
    feats = traffic_feature_columns(CFG)
    refs = {
        name: regression_metrics(test[target_column(CFG)], pred)
        for name, pred in reference_predictions(train, test, target_column(CFG)).items()
    }
    refs["ridge"] = regression_metrics(
        test[target_column(CFG)], ridge_predictions(train, test, feats)
    )
    train_baseline.main(CFG)
    train_event_aware.main(CFG)
    result = compare.main(CFG)
    event = result["event_affected_only"]
    if event["event_aware_A_plus"]["MAE"] >= event["baseline_A"]["MAE"]:
        raise RuntimeError("synthetic acceptance failed: A+raw did not improve event-affected MAE")
    if set(feats) & set(EVENT_FEATURE_COLUMNS):
        raise AssertionError("baseline feature leakage")
    elapsed = time.perf_counter() - started
    if elapsed >= 600:
        raise RuntimeError(f"synthetic acceptance failed: elapsed {elapsed:.1f}s >= 600s")
    manifest = write_run_manifest(CFG)
    summary = {
        "run_id": run_id,
        "dataset_checksum": input_checksums(),
        "config_checksum": config_checksum(CFG),
        "source_profile": "synthetic",
        "target": target_column(CFG),
        "horizon_minutes": 15,
        "model": "random_forest",
        "treatment": "A_vs_A+raw",
        "fold": 0,
        "seed": CFG["seed"],
        "metrics": result,
        "references": refs,
        "timing": {"elapsed_seconds": elapsed},
        "sample_counts": {"test": result["n_test_rows"], "event": result["n_event_affected_rows"]},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = RESULTS_DIR / "smoke_results.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(RESULTS_DIR / "smoke_results.json")
    pointer = ROOT / "runs" / "latest-verified"
    pointer.parent.mkdir(exist_ok=True)
    temp_pointer = pointer.with_name(".latest-verified.tmp")
    temp_pointer.unlink(missing_ok=True)
    temp_pointer.symlink_to(Path(run_id))
    temp_pointer.replace(pointer)
    print(
        f"[verified] {run_id} completed in {elapsed:.1f}s; manifest schema={manifest['schema_version']}"
    )


if __name__ == "__main__":
    main()
