"""Statistical rigor for the A -> A+ comparison.

A single split with one seed can't tell whether the event feature really helps: the gap could
be split luck or estimator randomness. Driven by the `evaluation` config block, this module
combines rolling-origin CV (several forward test blocks that never train on the future),
multiple seeds to average out estimator randomness, and a paired significance test with a
bootstrap CI on the per-(fold, seed) MAE gap, so the improvement comes with a p-value and a
confidence interval rather than a single point estimate.

The event-affected subset is the primary lens, since that is where the hypothesis predicts the
gain.

    uv run python -m scripts.run_stats
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd
from scipy import stats

from ..config import CFG, RESULTS_DIR, target_column
from ..data.load import load_events, load_flow, load_sensors
from ..features.build_features import (
    EVENT_FEATURE_COLUMNS,
    event_features,
    shift_events_placebo,
    traffic_feature_columns,
)
from ..models.traffic_model import build_model
from .preprocess import normalize_per_sensor
from .prepare import build_feature_frame

EVENT_EFFECT_THRESHOLD = 0.05
SAMPLE_TREATMENTS = {
    "A": [],
    "A+window": ["in_event_window"],
    "A+spatiotemporal": ["nearest_event_km", "hours_to_next_event"],
    "A+attendance": ["sum_attendance_exposed", "max_attendance_exposed"],
    "A+raw": EVENT_FEATURE_COLUMNS,
    "A+placebo": [f"placebo__{c}" for c in EVENT_FEATURE_COLUMNS],
}


def rolling_origin_folds(n_times: int, n_folds: int, test_frac: float) -> list[tuple[int, int]]:
    """Return (train_end, test_end) index cut points into the sorted-unique timestamp array.

    Expanding window: fold i trains on all timestamps before `train_end` and tests on the next
    contiguous block [train_end, test_end). Blocks march forward and never overlap.
    """
    test_len = max(1, int(n_times * test_frac))
    first_train = n_times - n_folds * test_len
    if first_train < test_len:  # not enough history, so use fewer but still valid folds
        n_folds = max(1, (n_times // test_len) - 1)
        first_train = n_times - n_folds * test_len
    return [(first_train + i * test_len, first_train + (i + 1) * test_len) for i in range(n_folds)]


def _event_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows an event is influencing (synthetic ground truth if present, else event window)."""
    if "true_event_effect" in df.columns:
        return (df["true_event_effect"] > EVENT_EFFECT_THRESHOLD).to_numpy()
    return (df["in_event_window"] == 1).to_numpy()


def _run_fold_seed(fold: pd.DataFrame, cfg: dict, feats: list[str], tgt: str) -> dict:
    """Fit every treatment with identical rows, split, model family, seed, and parameters."""
    train, test = fold[fold["is_train"]], fold[~fold["is_train"]]
    scale = test["flow_scale"].to_numpy()
    y_true = test[tgt].to_numpy()
    errors = {}
    for treatment, extra in SAMPLE_TREATMENTS.items():
        columns = feats + extra
        model = build_model(cfg)
        model.fit(train[columns], train["target"])
        errors[treatment] = np.abs(y_true - model.predict(test[columns]) * scale)
    ev = _event_mask(test)
    return {
        "errors": errors,
        "event": ev,
        "city": test["city"].to_numpy(),
        "day": test["timestamp"].dt.strftime("%Y-%m-%d").to_numpy(),
    }


def _paired_stats(gap: np.ndarray, alpha: float, n_boot: int, seed: int) -> dict:
    """Paired significance test + bootstrap CI on a vector of per-(fold, seed) MAE gaps (A-A+)."""
    gap = np.asarray(gap, dtype=float)
    mean_gap = float(gap.mean())
    # Paired t-test against 0, with Wilcoxon as a distribution-free backup that needs at least
    # one non-zero difference.
    t_p = float(stats.ttest_1samp(gap, 0.0).pvalue) if len(gap) > 1 else float("nan")
    try:
        w_p = (
            float(stats.wilcoxon(gap).pvalue) if np.any(gap != 0) and len(gap) > 1 else float("nan")
        )
    except ValueError:
        w_p = float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([rng.choice(gap, size=len(gap), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean_mae_gap": mean_gap,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ttest_p": t_p,
        "wilcoxon_p": w_p,
        "interval_supports_improvement": bool(len(gap) >= 10 and lo > 0),
        "inference_unit": "paired city-day blocks",
        "n_cells": int(len(gap)),
    }


def _stratified_day_stats(day: pd.DataFrame, alpha: float, n_boot: int, seed: int) -> dict:
    """Paired city-day bootstrap, resampling days independently within each city."""
    gap = day["mae_a"] - day["mae_ap"]
    rng = np.random.default_rng(seed)
    groups = [g.to_numpy() for _, g in gap.groupby(day["city"], sort=True)]
    boots = np.array(
        [
            np.concatenate([rng.choice(g, len(g), replace=True) for g in groups]).mean()
            for _ in range(n_boot)
        ]
    )
    result = _paired_stats(gap.to_numpy(), alpha, max(1, min(n_boot, 100)), seed)
    result["ci_low"], result["ci_high"] = map(
        float, np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    )
    result["interval_supports_improvement"] = bool(len(gap) >= 10 and result["ci_low"] > 0)
    result["n_days"] = int(day["day"].nunique())
    result["n_city_days"] = int(len(day))
    return result


def evaluate(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    ecfg = cfg.get("evaluation", {})
    seeds = ecfg.get("seeds", [cfg["seed"]])
    n_folds = int(ecfg.get("cv", {}).get("n_folds", 4))
    cv_cfg = ecfg.get("cv", {})
    test_frac = float(cv_cfg.get("test_frac", 0.15))
    alpha = float(ecfg.get("alpha", 0.05))
    n_boot = int(ecfg.get("n_bootstrap", 2000))

    tgt = target_column(cfg)
    feats = traffic_feature_columns(cfg)
    base = build_feature_frame(cfg).sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    placebo = event_features(
        load_flow(), shift_events_placebo(load_events()), load_sensors(), cfg
    ).rename(columns={c: f"placebo__{c}" for c in EVENT_FEATURE_COLUMNS})
    base = base.merge(
        placebo[["sensor_id", "timestamp"] + [f"placebo__{c}" for c in EVENT_FEATURE_COLUMNS]],
        on=["sensor_id", "timestamp"],
        how="left",
    )
    times = pd.DatetimeIndex(base["timestamp"].unique()).sort_values()
    if "test_days" in cv_cfg:
        test_len = int(pd.Timedelta(days=float(cv_cfg["test_days"])) / (times[1] - times[0]))
        test_frac = test_len / len(times)
    folds = rolling_origin_folds(len(times), n_folds, test_frac)

    cells: list[dict] = []
    row_err_a_evt: list[np.ndarray] = []
    row_err_ap_evt: list[np.ndarray] = []
    day_frames: list[pd.DataFrame] = []
    for fi, (tr_end, te_end) in enumerate(folds):
        train_end, test_end = times[tr_end], times[te_end - 1]
        fold = base[base["timestamp"] <= test_end].copy()
        fold["is_train"] = fold["timestamp"] < train_end
        fold = normalize_per_sensor(fold, cfg)
        for seed in seeds:
            c = copy.deepcopy(cfg)
            c["seed"] = seed
            r = _run_fold_seed(fold, c, feats, tgt)
            evt = r["event"]
            errors = r["errors"]
            cell = {
                "fold": fi,
                "seed": seed,
                "n_test": int(len(errors["A"])),
                "n_event": int(evt.sum()),
                "mae_a_all": float(errors["A"].mean()),
                "mae_ap_all": float(errors["A+raw"].mean()),
                "mae_placebo_all": float(errors["A+placebo"].mean()),
                "mae_a_event": float(errors["A"][evt].mean()) if evt.any() else float("nan"),
                "mae_ap_event": (float(errors["A+raw"][evt].mean()) if evt.any() else float("nan")),
                "mae_placebo_event": (
                    float(errors["A+placebo"][evt].mean()) if evt.any() else float("nan")
                ),
            }
            for treatment in SAMPLE_TREATMENTS:
                key = treatment.lower().replace("+", "_plus_").replace("-", "_")
                cell[f"mae_{key}_all"] = float(errors[treatment].mean())
                cell[f"mae_{key}_event"] = (
                    float(errors[treatment][evt].mean()) if evt.any() else float("nan")
                )
            cells.append(cell)
            day_frames.append(
                pd.DataFrame(
                    {
                        "fold": fi,
                        "seed": seed,
                        "city": r["city"],
                        "day": r["day"],
                        "event": evt,
                        "err_a": errors["A"],
                        "err_ap": errors["A+raw"],
                    }
                )
            )
            if evt.any():
                row_err_a_evt.append(errors["A"][evt])
                row_err_ap_evt.append(errors["A+raw"][evt])
            print(
                f"[stats] fold {fi} seed {seed}: "
                f"event MAE A={cell['mae_a_event']:.3f} A+raw={cell['mae_ap_event']:.3f}"
            )

    cell_df = pd.DataFrame(cells)
    # Seeds measure stability; inference uses one seed-averaged observation per fold.
    fold_df = cell_df.groupby("fold", as_index=False).mean(numeric_only=True)
    row_days = pd.concat(day_frames, ignore_index=True)
    # Average stochastic seeds first. The inferential observations are paired city-days.
    day_seed = row_days.groupby(["fold", "seed", "city", "day"], as_index=False).agg(
        mae_a=("err_a", "mean"), mae_ap=("err_ap", "mean")
    )
    day_overall = day_seed.groupby(["fold", "city", "day"], as_index=False)[
        ["mae_a", "mae_ap"]
    ].mean()
    event_rows = row_days[row_days["event"]]
    event_day_seed = event_rows.groupby(["fold", "seed", "city", "day"], as_index=False).agg(
        mae_a=("err_a", "mean"), mae_ap=("err_ap", "mean")
    )
    day_event = event_day_seed.groupby(["fold", "city", "day"], as_index=False)[
        ["mae_a", "mae_ap"]
    ].mean()

    # Secondary check: a pooled row-level paired test on event rows. The n is large, but the rows
    # aren't independent, so treat it as supporting evidence only.
    pooled_p = float("nan")
    if row_err_a_evt:
        ea = np.concatenate(row_err_a_evt)
        eap = np.concatenate(row_err_ap_evt)
        if np.any(ea != eap):
            pooled_p = float(stats.wilcoxon(ea, eap).pvalue)

    results = {
        "target": tgt,
        "model_type": cfg["model"]["type"],
        "seeds": list(seeds),
        "n_folds": len(folds),
        "alpha": alpha,
        "event_affected": _stratified_day_stats(day_event, alpha, n_boot, cfg["seed"]),
        "overall": _stratified_day_stats(day_overall, alpha, n_boot, cfg["seed"]),
        "ablations": {
            treatment: {
                "mean_mae_all": float(
                    fold_df[
                        f"mae_{treatment.lower().replace('+', '_plus_').replace('-', '_')}_all"
                    ].mean()
                ),
                "mean_mae_event": float(
                    fold_df[
                        f"mae_{treatment.lower().replace('+', '_plus_').replace('-', '_')}_event"
                    ].mean()
                ),
            }
            for treatment in SAMPLE_TREATMENTS
        },
        "placebo": {
            "mean_real_minus_placebo_mae": float(
                (fold_df["mae_ap_all"] - fold_df["mae_placebo_all"]).mean()
            ),
            "interpretation": "negative favors true event dates",
        },
        "pooled_row_wilcoxon_p_event": pooled_p,
        "cells": cells,
    }

    out = RESULTS_DIR / f"stats_{cfg['model']['type']}_{tgt}.json"
    out.write_text(json.dumps(results, indent=2))
    _print_summary(results)
    print(f"\n[stats] wrote {out}")
    return results


def _print_summary(r: dict) -> None:
    ev, ov = r["event_affected"], r["overall"]
    print("\n=== A -> A+raw paired city-day evidence ===")
    print(
        f"target={r['target']}  model={r['model_type']}  folds={r['n_folds']}  seeds={r['seeds']}"
    )
    for name, blk in (("EVENT-AFFECTED", ev), ("OVERALL", ov)):
        verdict = (
            "interval supports improvement"
            if blk["interval_supports_improvement"]
            else "interval does not support a directional claim"
        )
        print(
            f"{name:<15} mean MAE gap={blk['mean_mae_gap']:+.3f} "
            f"[{blk['ci_low']:+.3f}, {blk['ci_high']:+.3f}]  "
            f"city-days={blk['n_city_days']} days={blk['n_days']}  -> {verdict}"
        )
    placebo = r["placebo"]
    print(
        f"PLACEBO         A+raw minus +7-day-placebo MAE="
        f"{placebo['mean_real_minus_placebo_mae']:+.3f} "
        f"({placebo['interpretation']})"
    )
    print(
        "ABLATIONS       "
        + ", ".join(
            f"{name}={values['mean_mae_event']:.2f}" for name, values in r["ablations"].items()
        )
        + "  [event MAE]"
    )


if __name__ == "__main__":
    evaluate()
