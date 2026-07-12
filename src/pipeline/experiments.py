"""Experiment matrix. Runs every combination of the key config knobs (see GRID below),
records the A-vs-A+ result for each, and compiles a comparison table into docs/thesis-notes.md.

Data prep and Model B depend only on (split, normalize), so they run once per cell and only
runs A and A+ repeat per model.

    uv run python -m scripts.run_experiments

Writes media/results/experiments.json and the table in docs/thesis-notes.md (between the
EXPERIMENTS markers).
"""

from __future__ import annotations

import copy
import json

from ..config import CFG, RESULTS_DIR, ROOT
from ..data.build_real import ensure_dataset
from . import compare, prepare, train_baseline, train_event_aware, train_event_model

# Trim or extend as needed. gradient_boosting is the slow one.
GRID = {
    "split": ["time", "city"],
    "normalize": ["none", "per_sensor"],
    "model": ["random_forest", "gradient_boosting", "xgboost"],
}

MARK_START = "<!-- EXPERIMENTS:START -->"
MARK_END = "<!-- EXPERIMENTS:END -->"
NOTES = ROOT / "docs" / "thesis-notes.md"


def _cfg_for(split: str, normalize: str, model: str) -> dict:
    cfg = copy.deepcopy(CFG)
    cfg["model"]["split"]["mode"] = split
    cfg["model"]["normalize"] = normalize
    cfg["model"]["type"] = model
    return cfg


def run(cfg_base: dict | None = None) -> list[dict]:
    base = cfg_base or CFG
    ensure_dataset(base)

    rows: list[dict] = []
    for split in GRID["split"]:
        for normalize in GRID["normalize"]:
            # Prep and Model B run once per (split, normalize) cell.
            cell_cfg = _cfg_for(split, normalize, GRID["model"][0])
            prepare.prepare(cell_cfg)
            train_event_model.main(cell_cfg)

            for model in GRID["model"]:
                cfg = _cfg_for(split, normalize, model)
                train_baseline.main(cfg)
                m_ap = train_event_aware.main(cfg)
                cmp = compare.compute_results(cfg)
                rows.append(
                    {
                        "split": split,
                        "normalize": normalize,
                        "model": model,
                        "mae_A": cmp["overall"]["baseline_A"]["MAE"],
                        "mae_Aplus": cmp["overall"]["event_aware_A_plus"]["MAE"],
                        "impr_overall": cmp["improvement_pct_overall"].get("MAE", 0.0),
                        "mae_A_event": cmp["event_affected_only"]["baseline_A"]["MAE"],
                        "mae_Aplus_event": cmp["event_affected_only"]["event_aware_A_plus"]["MAE"],
                        "impr_event": cmp["improvement_pct_event_affected"].get("MAE", 0.0),
                        "latency_ms_per_1k": m_ap["inference_ms_per_1k"],
                    }
                )
                print(
                    f"[exp] split={split:5} norm={normalize:10} model={model:17} "
                    f"event-Δ={rows[-1]['impr_event']:+.2f}%"
                )

    (RESULTS_DIR / "experiments.json").write_text(json.dumps(rows, indent=2))
    _write_table(rows)
    return rows


def _table_md(rows: list[dict]) -> str:
    header = (
        "| split | normalize | model | MAE A | MAE A+ | Δ% overall | "
        "MAE A (event) | MAE A+ (event) | **Δ% event** | infer ms/1k |\n"
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    lines = []
    best = max(rows, key=lambda row: row["impr_event"])
    for r in rows:
        star = " ⭐" if r is best else ""
        lines.append(
            f"| {r['split']} | {r['normalize']} | {r['model']}{star} | "
            f"{r['mae_A']:.2f} | {r['mae_Aplus']:.2f} | {r['impr_overall']:+.2f}% | "
            f"{r['mae_A_event']:.2f} | {r['mae_Aplus_event']:.2f} | "
            f"**{r['impr_event']:+.2f}%** | {r['latency_ms_per_1k']:.2f} |"
        )

    def best_for(split: str) -> dict:
        return max(
            (row for row in rows if row["split"] == split), key=lambda row: row["impr_event"]
        )

    summary = [
        "",
        "**Best configuration per condition** (largest event-affected MAE reduction):",
        "",
        f"- *Within-city (time split):* `{best_for('time')['model']}` + "
        f"normalize=`{best_for('time')['normalize']}` "
        f"→ {best_for('time')['impr_event']:+.2f}% on event rows.",
        f"- *Cross-city transfer (LA→SF):* `{best_for('city')['model']}` + "
        f"normalize=`{best_for('city')['normalize']}` "
        f"→ {best_for('city')['impr_event']:+.2f}% on event rows.",
        "",
        "Δ% = MAE reduction from baseline A to event-aware A+ (higher is better). "
        "The event feature helps most on event-affected rows; ⭐ = best cell overall.",
    ]
    return header + "\n".join(lines) + "\n" + "\n".join(summary) + "\n"


def _write_table(rows: list[dict]) -> None:
    table = _table_md(rows)
    block = f"{MARK_START}\n\n{table}\n{MARK_END}"
    text = NOTES.read_text()
    if MARK_START in text and MARK_END in text:
        pre = text.split(MARK_START)[0]
        posttext = text.split(MARK_END)[1]
        text = pre + block + posttext
    else:
        text += f"\n\n## Experiment matrix (auto-generated)\n\n{block}\n"
    NOTES.write_text(text)
    print(f"[exp] wrote {len(rows)} rows to {RESULTS_DIR/'experiments.json'} and updated {NOTES}")


if __name__ == "__main__":
    run()
