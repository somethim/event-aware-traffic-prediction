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
from ..features.build_features import traffic_feature_columns
from ..models.traffic_model import build_model
from .preprocess import normalize_per_sensor
from .prepare import build_feature_frame
from .train_event_model import fit_event_score

EVENT_EFFECT_THRESHOLD = 0.05


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


def _fit_event_score(fold: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Model B on this fold: out-of-fold scores for train rows, train-fit for the test block."""
    return np.asarray(fit_event_score(fold, cfg)[0], dtype=float)


def _run_fold_seed(fold: pd.DataFrame, cfg: dict, feats: list[str], tgt: str) -> dict:
    """Train A and A+ on this fold with cfg's seed; return per-row abs errors on the test block."""
    train, test = fold[fold["is_train"]], fold[~fold["is_train"]]
    scale = test["flow_scale"].to_numpy()
    y_true = test[tgt].to_numpy()

    model_a = build_model(cfg)
    model_a.fit(train[feats], train["target"])
    err_a = np.abs(y_true - model_a.predict(test[feats]) * scale)

    feats_plus = feats + ["event_impact_score"]
    model_ap = build_model(cfg)
    model_ap.fit(train[feats_plus], train["target"])
    err_ap = np.abs(y_true - model_ap.predict(test[feats_plus]) * scale)

    ev = _event_mask(test)
    return {"err_a": err_a, "err_ap": err_ap, "event": ev}


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
        # Significant only when the t-test clears alpha and the bootstrap CI excludes 0.
        "significant": bool(t_p == t_p and t_p < alpha and lo > 0),
        "n_cells": int(len(gap)),
    }


def evaluate(cfg: dict | None = None) -> dict:
    cfg = cfg or CFG
    ecfg = cfg.get("evaluation", {})
    seeds = ecfg.get("seeds", [cfg["seed"]])
    n_folds = int(ecfg.get("cv", {}).get("n_folds", 4))
    test_frac = float(ecfg.get("cv", {}).get("test_frac", 0.15))
    alpha = float(ecfg.get("alpha", 0.05))
    n_boot = int(ecfg.get("n_bootstrap", 2000))

    tgt = target_column(cfg)
    feats = traffic_feature_columns(cfg)
    base = build_feature_frame(cfg).sort_values(["sensor_id", "timestamp"]).reset_index(drop=True)
    times = pd.DatetimeIndex(base["timestamp"].unique()).sort_values()
    folds = rolling_origin_folds(len(times), n_folds, test_frac)

    cells: list[dict] = []
    row_err_a_evt: list[np.ndarray] = []
    row_err_ap_evt: list[np.ndarray] = []
    for fi, (tr_end, te_end) in enumerate(folds):
        train_end, test_end = times[tr_end], times[te_end - 1]
        fold = base[base["timestamp"] <= test_end].copy()
        fold["is_train"] = fold["timestamp"] < train_end
        fold = normalize_per_sensor(fold, cfg)
        fold["event_impact_score"] = _fit_event_score(fold, cfg)

        for seed in seeds:
            c = copy.deepcopy(cfg)
            c["seed"] = seed
            r = _run_fold_seed(fold, c, feats, tgt)
            evt = r["event"]
            cell = {
                "fold": fi,
                "seed": seed,
                "n_test": int(len(r["err_a"])),
                "n_event": int(evt.sum()),
                "mae_a_all": float(r["err_a"].mean()),
                "mae_ap_all": float(r["err_ap"].mean()),
                "mae_a_event": float(r["err_a"][evt].mean()) if evt.any() else float("nan"),
                "mae_ap_event": float(r["err_ap"][evt].mean()) if evt.any() else float("nan"),
            }
            cells.append(cell)
            if evt.any():
                row_err_a_evt.append(r["err_a"][evt])
                row_err_ap_evt.append(r["err_ap"][evt])
            print(
                f"[stats] fold {fi} seed {seed}: "
                f"event MAE A={cell['mae_a_event']:.3f} A+={cell['mae_ap_event']:.3f}"
            )

    cell_df = pd.DataFrame(cells)
    gap_event = (cell_df["mae_a_event"] - cell_df["mae_ap_event"]).dropna().to_numpy()
    gap_overall = (cell_df["mae_a_all"] - cell_df["mae_ap_all"]).to_numpy()

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
        "event_affected": _paired_stats(gap_event, alpha, n_boot, cfg["seed"]),
        "overall": _paired_stats(gap_overall, alpha, n_boot, cfg["seed"]),
        "pooled_row_wilcoxon_p_event": pooled_p,
        "cells": cells,
    }

    out = RESULTS_DIR / "stats.json"
    out.write_text(json.dumps(results, indent=2))
    _print_summary(results)
    print(f"\n[stats] wrote {out}")
    return results


def _print_summary(r: dict) -> None:
    ev, ov = r["event_affected"], r["overall"]
    print("\n=== A -> A+ significance (rolling-origin CV x seeds) ===")
    print(
        f"target={r['target']}  model={r['model_type']}  folds={r['n_folds']}  seeds={r['seeds']}"
    )
    for name, blk in (("EVENT-AFFECTED", ev), ("OVERALL", ov)):
        verdict = "SIGNIFICANT" if blk["significant"] else "not significant"
        print(
            f"{name:<15} mean MAE gap={blk['mean_mae_gap']:+.3f} "
            f"[{blk['ci_low']:+.3f}, {blk['ci_high']:+.3f}]  "
            f"t-p={blk['ttest_p']:.4f}  wilcoxon-p={blk['wilcoxon_p']:.4f}  -> {verdict}"
        )


if __name__ == "__main__":
    evaluate()
