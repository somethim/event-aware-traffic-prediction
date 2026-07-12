"""Smoke tests: run the full pipeline on a tiny config and assert the core hypothesis
plumbing works (A+ has the extra feature, and it should not be worse than A overall).
"""

from __future__ import annotations

import copy

from src.config import CFG
from src.data.generate_synthetic import generate
from src.features.build_features import EVENT_FEATURE_COLUMNS, traffic_feature_columns
from src.pipeline import compare, prepare, train_baseline, train_event_aware, train_event_model


def _tiny_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["data"].update(n_sensors=6, n_days=20, n_events=25)
    cfg["model"]["random_forest"].update(n_estimators=40)
    return cfg


def test_end_to_end():
    cfg = _tiny_cfg()
    generate(cfg)
    df = prepare.prepare(cfg)

    # event-exposure features are present and non-trivial
    for c in EVENT_FEATURE_COLUMNS:
        assert c in df.columns
    assert df["sum_attendance_exposed"].sum() > 0

    train_baseline.main(cfg)
    train_event_model.main(cfg)
    df2 = prepare.load_dataset()
    assert "event_impact_score" in df2.columns  # Model B wrote its score back

    train_event_aware.main(cfg)
    results = compare.main(cfg)

    a = results["overall"]["baseline_A"]["MAE"]
    ap = results["overall"]["event_aware_A_plus"]["MAE"]
    # Adding a genuinely informative feature should not hurt overall accuracy.
    assert ap <= a * 1.05


def test_feature_columns_disjoint_from_event_score():
    # The baseline must not accidentally include the event score.
    assert "event_impact_score" not in traffic_feature_columns(CFG)
