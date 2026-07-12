"""Thesis figures. Each `plot_*` function is a small, self-contained matplotlib snippet
that writes one PNG into media/figures/, ready to drop into the thesis appendix as-is.

    uv run python -m scripts.run_visuals     # or: python -m src.pipeline.visualize
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..config import CFG, FIGURES_DIR, target_column, target_units
from ..data.build_real import ensure_dataset
from ..data.load import load_events, load_flow, load_sensors
from ..features.build_features import traffic_feature_columns
from . import compare, prepare, train_baseline, train_event_aware, train_event_model

BASELINE_COLOR = "#8899aa"
EVENT_COLOR = "#2f7d4f"


def _ensure_pipeline(cfg: dict) -> None:
    """Build any pipeline artifacts the figures need, reusing whatever is already computed."""
    from .prepare import DATASET_FILE

    ensure_dataset(cfg)
    if not DATASET_FILE.exists():
        prepare.prepare(cfg)
    if "event_impact_score" not in prepare.load_dataset().columns:
        train_event_model.main(cfg)
    if not train_baseline.PRED_FILE.exists():
        train_baseline.main(cfg)
    if not train_event_aware.PRED_FILE.exists():
        train_event_aware.main(cfg)


def plot_flow_profile(flow: pd.DataFrame) -> None:
    """Average flow by hour of day, split weekday vs weekend."""
    df = flow.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["weekend"] = df["timestamp"].dt.dayofweek >= 5
    prof = df.groupby(["weekend", "hour"])["flow"].mean().unstack(0)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(prof.index, prof[False], color="#2f6d9e", lw=2, label="weekday")
    ax.plot(prof.index, prof[True], color="#c0803a", lw=2, label="weekend")
    ax.set_xlabel("hour of day")
    ax.set_ylabel("mean flow (veh/interval)")
    ax.set_title("Average daily traffic profile")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    _save(fig, "flow_profile.png")


def plot_sensor_event_map(sensors: pd.DataFrame, events: pd.DataFrame) -> None:
    """Spatial layout of sensors and events, with events sized by expected attendance."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        sensors["lon"], sensors["lat"], marker="^", color="#2f6d9e", s=60, label="sensors", zorder=3
    )
    sizes = 10 + events["expected_attendance"] / 60000 * 400
    ax.scatter(
        events["lon"],
        events["lat"],
        s=sizes,
        color="#c0403a",
        alpha=0.35,
        label="events (size ∝ attendance)",
        zorder=2,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Sensor and event locations")
    ax.legend(loc="upper right")
    _save(fig, "sensor_event_map.png")


def plot_event_effect_hist(flow: pd.DataFrame) -> None:
    """Distribution of the (ground-truth) fractional flow uplift on event-affected rows."""
    eff = flow.loc[flow["true_event_effect"] > 0.01, "true_event_effect"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(eff, bins=40, color=EVENT_COLOR, alpha=0.85)
    ax.set_xlabel("event effect (fractional flow uplift)")
    ax.set_ylabel("number of (sensor, time) rows")
    ax.set_title("How strong are event-driven traffic increases?")
    _save(fig, "event_effect_hist.png")


def plot_feature_importance(cfg: dict, top_n: int = 15) -> None:
    """Feature importances of the event-aware model, highlighting the event score."""
    import joblib

    model = joblib.load(train_event_aware.MODEL_FILE)
    feats = traffic_feature_columns(cfg) + ["event_impact_score"]
    imp = pd.Series(model.feature_importances_, index=feats).nlargest(top_n)[::-1]
    colors = [EVENT_COLOR if f == "event_impact_score" else BASELINE_COLOR for f in imp.index]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.barh(list(imp.index), np.asarray(imp.values, dtype=float), color=colors)
    ax.set_xlabel("importance")
    ax.set_title("Event-aware model: feature importances\n(event score highlighted)")
    _save(fig, "feature_importance.png")


def plot_pred_vs_actual(merged: pd.DataFrame, tgt: str = "flow", sample: int = 5000) -> None:
    """Predicted vs actual target for A and A+. Points near the diagonal are accurate."""
    rng = np.random.default_rng(CFG["seed"])
    idx = rng.choice(len(merged), size=min(sample, len(merged)), replace=False)
    s = merged.iloc[idx]
    lim = max(
        float(np.nanmax(s[tgt].to_numpy())), float(np.nanmax(s["event_aware_pred"].to_numpy()))
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, col, title, color in (
        (axes[0], "baseline_pred", "A (baseline)", BASELINE_COLOR),
        (axes[1], "event_aware_pred", "A+ (event-aware)", EVENT_COLOR),
    ):
        ax.scatter(s[tgt], s[col], s=5, alpha=0.25, color=color)
        ax.plot([0, lim], [0, lim], "k--", lw=1)
        ax.set_xlabel(f"actual {tgt}")
        ax.set_title(title)
    axes[0].set_ylabel(f"predicted {tgt}")
    fig.suptitle(f"Predicted vs actual {tgt}")
    _save(fig, "pred_vs_actual.png")


def plot_error_vs_event_effect(merged: pd.DataFrame, tgt: str = "flow") -> None:
    """Key figure. Mean absolute error binned by event size. A+ should pull ahead of A as the
    event effect grows, which is the thesis hypothesis made visible."""
    df = merged.copy()
    df["err_A"] = (df[tgt] - df["baseline_pred"]).abs()
    df["err_Aplus"] = (df[tgt] - df["event_aware_pred"]).abs()
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.4, np.inf]
    labels = ["~0", "0.01–.05", ".05–.1", ".1–.2", ".2–.4", ">0.4"]
    df["bin"] = pd.cut(df["true_event_effect"], bins=bins, labels=labels, right=False)
    grp = df.groupby("bin", observed=True)[["err_A", "err_Aplus"]].mean()

    x = np.arange(len(grp))
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.19, grp["err_A"], width=0.38, color=BASELINE_COLOR, label="A (baseline)")
    ax.bar(x + 0.19, grp["err_Aplus"], width=0.38, color=EVENT_COLOR, label="A+ (event-aware)")
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index)
    ax.set_xlabel("event effect bin (fractional uplift)")
    ax.set_ylabel(f"mean absolute error ({target_units()})")
    ax.set_title("Prediction error vs event size — where event data helps")
    ax.legend()
    _save(fig, "error_vs_event_effect.png")


def plot_xgb_training_curve(cfg: dict) -> None:
    """Train an XGBoost model while recording train and test error each boosting round, to
    visualize the sequential error correction described in the thesis notes."""
    from xgboost import XGBRegressor

    from ..device import resolve_device

    df = prepare.load_dataset()
    feats = traffic_feature_columns(cfg) + ["event_impact_score"]
    train, test = df[df["is_train"]], df[~df["is_train"]]
    device = resolve_device(cfg["model"].get("device", "auto"))

    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        tree_method="hist",
        device=device,
        eval_metric="rmse",
    )
    model.fit(
        train[feats],
        train["target"],
        eval_set=[(train[feats], train["target"]), (test[feats], test["target"])],
        verbose=False,
    )
    hist = model.evals_result()
    rounds = range(len(hist["validation_0"]["rmse"]))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(rounds, hist["validation_0"]["rmse"], color=BASELINE_COLOR, label="train RMSE")
    ax.plot(rounds, hist["validation_1"]["rmse"], color=EVENT_COLOR, label="test RMSE")
    ax.set_xlabel("boosting round (tree number)")
    ax.set_ylabel("RMSE (veh/interval)")
    ax.set_title("XGBoost learning curve — error falls as trees are added")
    ax.legend()
    _save(fig, "xgb_training_curve.png")


def _save(fig, name: str) -> None:
    fig.tight_layout()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"[viz] wrote {path}")


def run(cfg: dict | None = None) -> None:
    cfg = cfg or CFG
    _ensure_pipeline(cfg)

    flow = load_flow()
    merged = compare.load_merged()
    tgt = target_column(cfg)

    plot_flow_profile(flow)
    plot_sensor_event_map(load_sensors(), load_events())
    plot_feature_importance(cfg)
    plot_pred_vs_actual(merged, tgt)
    plot_xgb_training_curve(cfg)
    # The ground-truth-effect figures only work on the synthetic testbed; real data has no label.
    if "true_event_effect" in flow.columns:
        plot_event_effect_hist(flow)
    if "true_event_effect" in merged.columns:
        plot_error_vs_event_effect(merged, tgt)
    else:
        print("[viz] skipping event-effect figures (no ground-truth effect on real data)")
    print(f"\n[viz] all figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    run()
