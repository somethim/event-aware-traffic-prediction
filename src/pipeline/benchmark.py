"""Multi-model benchmark. Runs A and A+ for every model in config's benchmark.models and
compares them in one table and chart, recording accuracy and inference latency for each.

Data prep and Model B run once up front. The event-impact score does not depend on the
traffic model type, so only runs A and A+ repeat per model.

    uv run python -m scripts.run_benchmark      # or: python -m src.pipeline.benchmark

Outputs: results/benchmark.json and results/benchmark.png
"""

from __future__ import annotations

import copy
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import CFG, FIGURES_DIR, RESULTS_DIR
from ..data.build_real import ensure_dataset
from . import compare, prepare, train_baseline, train_event_aware, train_event_model

DEFAULT_MODELS = ["random_forest", "gradient_boosting", "xgboost"]


def run(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    models = cfg.get("benchmark", {}).get("models", DEFAULT_MODELS)

    ensure_dataset(cfg)
    print("== preparing features ==")
    prepare.prepare(cfg)
    print("== training Model B (event impact, shared) ==")
    train_event_model.main(cfg)

    results: dict = {}
    for mtype in models:
        print(f"\n===== model: {mtype} =====")
        c = copy.deepcopy(cfg)
        c["model"]["type"] = mtype
        m_a = train_baseline.main(c)
        m_ap = train_event_aware.main(c)
        comparison = compare.compute_results(c)
        results[mtype] = {
            "comparison": comparison,
            "latency_A_ms_per_1k": m_a["inference_ms_per_1k"],
            "latency_A_plus_ms_per_1k": m_ap["inference_ms_per_1k"],
        }

    (RESULTS_DIR / "benchmark.json").write_text(json.dumps(results, indent=2))
    _print_table(results)
    _plot(results)
    print(f"\nwrote {RESULTS_DIR / 'benchmark.json'} and {RESULTS_DIR / 'benchmark.png'}")
    return results


def _row(model: str, r: dict) -> tuple:
    c = r["comparison"]
    a_over = c["overall"]["baseline_A"]["MAE"]
    ap_over = c["overall"]["event_aware_A_plus"]["MAE"]
    a_evt = c["event_affected_only"]["baseline_A"]["MAE"]
    ap_evt = c["event_affected_only"]["event_aware_A_plus"]["MAE"]
    imp_over = c["improvement_pct_overall"].get("MAE", 0.0)
    imp_evt = c["improvement_pct_event_affected"].get("MAE", 0.0)
    return model, a_over, ap_over, imp_over, a_evt, ap_evt, imp_evt, r["latency_A_ms_per_1k"]


def _print_table(results: dict) -> None:
    header = (
        "model",
        "A MAE",
        "A+ MAE",
        "Δ% all",
        "A MAE(ev)",
        "A+ MAE(ev)",
        "Δ% event",
        "infer ms/1k",
    )
    print("\n=== Model benchmark (A = baseline, A+ = event-aware) ===")
    print(
        f"{header[0]:<18}{header[1]:>9}{header[2]:>9}{header[3]:>9}"
        f"{header[4]:>11}{header[5]:>12}{header[6]:>10}{header[7]:>13}"
    )
    for model, r in results.items():
        _, a_o, ap_o, i_o, a_e, ap_e, i_e, lat = _row(model, r)
        print(
            f"{model:<18}{a_o:>9.2f}{ap_o:>9.2f}{i_o:>8.2f}%"
            f"{a_e:>11.2f}{ap_e:>12.2f}{i_e:>9.2f}%{lat:>13.2f}"
        )


def _plot(results: dict) -> None:
    models = list(results.keys())
    a_evt = [results[m]["comparison"]["event_affected_only"]["baseline_A"]["MAE"] for m in models]
    ap_evt = [
        results[m]["comparison"]["event_affected_only"]["event_aware_A_plus"]["MAE"] for m in models
    ]
    x = range(len(models))
    fig, ax = plt.subplots(figsize=(1.8 * len(models) + 4, 4.6))
    ax.bar([i - 0.19 for i in x], a_evt, width=0.38, color="#8899aa", label="A (baseline)")
    ax.bar([i + 0.19 for i in x], ap_evt, width=0.38, color="#2f7d4f", label="A+ (event-aware)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models)
    ax.set_ylabel("MAE on event-affected rows (veh/h)")
    ax.set_title("Event-affected accuracy: baseline vs event-aware, per model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "benchmark.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run()
