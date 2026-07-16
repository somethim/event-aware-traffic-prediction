"""Compare run A (baseline) against run A+ (event-aware), the core thesis result.

Reports metrics for both runs overall and on the event-affected subset, where the hypothesis
predicts the biggest gain. Writes results/metrics.json and two figures.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # headless, no display needed
import matplotlib.pyplot as plt
import pandas as pd

from ..config import CFG, FIGURES_DIR, RESULTS_DIR, target_column, target_units
from ..models.metrics import regression_metrics
from .train_baseline import PRED_FILE as PRED_A
from .train_event_aware import PRED_FILE as PRED_A_PLUS

EVENT_EFFECT_THRESHOLD = 0.05  # a row is "event-affected" if true uplift exceeds this


def load_merged() -> pd.DataFrame:
    a = pd.read_parquet(PRED_A)
    b = pd.read_parquet(PRED_A_PLUS)[["sensor_id", "timestamp", "event_aware_pred"]]
    return a.merge(b, on=["sensor_id", "timestamp"], how="inner")


def _block(df: pd.DataFrame, y: str) -> dict:
    return {
        "baseline_A": regression_metrics(df[y], df["baseline_pred"]),
        "event_aware_A_plus": regression_metrics(df[y], df["event_aware_pred"]),
    }


def _improvement(a: dict, b: dict) -> dict:
    return {k: round(100 * (a[k] - b[k]) / a[k], 2) for k in ("MAE", "RMSE", "WAPE") if a[k] != 0}


def _event_affected(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where an event is influencing traffic. Uses the synthetic ground truth when present,
    otherwise falls back to being inside an event window, which also works for real data."""
    if "true_event_effect" in df.columns:
        return df[df["true_event_effect"] > EVENT_EFFECT_THRESHOLD]
    return df[df["in_event_window"] == 1]


def compute_results(cfg: dict) -> dict:
    """A-vs-A+ metrics for the current run's prediction files (no plotting/side effects).

    Reused by the multi-model benchmark, which calls this once per model type.
    """
    y = target_column(cfg)
    df = load_merged()
    affected = _event_affected(df)
    overall = _block(df, y)
    event_only = _block(affected, y)
    return {
        "model_type": cfg["model"]["type"],
        "target": y,
        "n_test_rows": int(len(df)),
        "n_event_affected_rows": int(len(affected)),
        "overall": overall,
        "event_affected_only": event_only,
        "improvement_pct_overall": _improvement(
            overall["baseline_A"], overall["event_aware_A_plus"]
        ),
        "improvement_pct_event_affected": _improvement(
            event_only["baseline_A"], event_only["event_aware_A_plus"]
        ),
    }


def main(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    results = compute_results(cfg)
    overall = results["overall"]
    event_only = results["event_affected_only"]

    out = RESULTS_DIR / "metrics.json"
    out.write_text(json.dumps(results, indent=2))

    _plot_bars(overall, event_only, target_units(cfg))
    _plot_event_window(load_merged(), target_column(cfg), target_units(cfg))

    print("\n=== A vs A+ ===")
    print(
        f"model: {results['model_type']}   test rows: {results['n_test_rows']:,}   "
        f"event-affected: {results['n_event_affected_rows']:,}"
    )
    print(
        f"OVERALL         MAE  A={overall['baseline_A']['MAE']:.2f}  "
        f"A+={overall['event_aware_A_plus']['MAE']:.2f}  "
        f"-> {results['improvement_pct_overall'].get('MAE', 0)}% better"
    )
    print(
        f"EVENT-AFFECTED  MAE  A={event_only['baseline_A']['MAE']:.2f}  "
        f"A+={event_only['event_aware_A_plus']['MAE']:.2f}  "
        f"-> {results['improvement_pct_event_affected'].get('MAE', 0)}% better"
    )
    print(f"\nwrote {out} and figures in {RESULTS_DIR}/")
    return results


def _plot_bars(overall: dict, event_only: dict, units: str = "veh/interval") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, block, title in (
        (axes[0], overall, "Overall test set"),
        (axes[1], event_only, "Event-affected rows only"),
    ):
        for i, metric in enumerate(("MAE", "RMSE")):
            ax.bar(
                i - 0.19,
                block["baseline_A"][metric],
                width=0.38,
                color="#8899aa",
                label="A (baseline)" if i == 0 else None,
            )
            ax.bar(
                i + 0.19,
                block["event_aware_A_plus"][metric],
                width=0.38,
                color="#2f7d4f",
                label="A+ (event-aware)" if i == 0 else None,
            )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["MAE", "RMSE"])
        ax.set_title(title)
        ax.set_ylabel(f"error ({units})")
        ax.legend()
    fig.suptitle("Baseline vs Event-Aware traffic prediction")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comparison_bars.png", dpi=130)
    plt.close(fig)


def _plot_event_window(df: pd.DataFrame, tgt: str = "flow", units: str = "veh/interval") -> None:
    """Time series for the single sensor with the largest event effect in the test set."""
    if "true_event_effect" not in df.columns or df["true_event_effect"].max() <= 0:
        return
    sid = int(df["sensor_id"].to_numpy()[df["true_event_effect"].to_numpy().argmax()])
    s = df[df["sensor_id"] == sid].sort_values("timestamp").reset_index(drop=True)
    peak_pos = int(s["true_event_effect"].to_numpy().argmax())
    peak = pd.Timestamp(s["timestamp"].to_numpy()[peak_pos])
    win = s[
        (s["timestamp"] >= peak - pd.Timedelta(hours=24))
        & (s["timestamp"] <= peak + pd.Timedelta(hours=24))
    ]

    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(win["timestamp"], win[tgt], color="black", lw=2, label=f"actual {tgt}")
    ax.plot(win["timestamp"], win["baseline_pred"], "--", color="#8899aa", label="A (baseline)")
    ax.plot(
        win["timestamp"], win["event_aware_pred"], "--", color="#2f7d4f", label="A+ (event-aware)"
    )
    ax.axvline(peak, color="#c04040", alpha=0.4, lw=1)  # type: ignore[arg-type]
    ax.set_title(f"Sensor {sid} around its biggest event (test set)")
    ax.set_ylabel(f"{tgt} ({units})")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "event_window_sensor.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
