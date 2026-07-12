"""Smoke tests that run the full pipeline on a tiny config, checking that A+ carries the
extra event feature and that it is not worse than the plain baseline A overall.
"""

from __future__ import annotations

import copy

from src.config import CFG
from src.data.generate_synthetic import generate
from src.features.build_features import EVENT_FEATURE_COLUMNS, traffic_feature_columns
from src.pipeline import compare, prepare, train_baseline, train_event_aware, train_event_model


def _tiny_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["data"]["n_days"] = 20
    cfg["data"]["cities"] = [
        {
            "name": "LA",
            "n_sensors": 5,
            "n_events": 20,
            "base_flow": [300, 1400],
            "area": {"lat_min": 34.0, "lat_max": 34.2, "lon_min": -118.35, "lon_max": -118.10},
        },
        {
            "name": "SF",
            "n_sensors": 4,
            "n_events": 15,
            "base_flow": [150, 900],
            "area": {"lat_min": 37.7, "lat_max": 37.85, "lon_min": -122.48, "lon_max": -122.38},
        },
    ]
    cfg["model"]["random_forest"].update(n_estimators=40)
    return cfg


def _run_pipeline(cfg) -> dict:
    generate(cfg)
    prepare.prepare(cfg)
    train_baseline.main(cfg)
    train_event_model.main(cfg)
    train_event_aware.main(cfg)
    return compare.main(cfg)


def test_end_to_end_time_split():
    cfg = _tiny_cfg()  # default: split.mode = time, normalize = none
    generate(cfg)
    df = prepare.prepare(cfg)

    # The event-exposure features should be present and actually carry signal.
    for c in EVENT_FEATURE_COLUMNS:
        assert c in df.columns
    assert df["sum_attendance_exposed"].sum() > 0
    assert {"target", "flow_scale"}.issubset(df.columns)

    train_baseline.main(cfg)
    train_event_model.main(cfg)
    assert "event_impact_score" in prepare.load_dataset().columns  # Model B wrote its score

    train_event_aware.main(cfg)
    results = compare.main(cfg)
    a = results["overall"]["baseline_A"]["MAE"]
    ap = results["overall"]["event_aware_A_plus"]["MAE"]
    # An informative feature should not make the overall error worse.
    assert ap <= a * 1.05


def test_city_split_with_per_sensor_norm():
    # Cross-city transfer (train on LA, test on SF) with per-sensor normalization.
    cfg = _tiny_cfg()
    cfg["model"]["split"] = {
        "mode": "city",
        "time_size": 0.2,
        "train_city": "LA",
        "test_city": "SF",
    }
    cfg["model"]["normalize"] = "per_sensor"

    generate(cfg)
    df = prepare.prepare(cfg)
    # Test rows should be exactly the SF sensors, and normalization gives varied scales.
    assert (df.loc[~df["is_train"], "city"] == "SF").all()
    assert (df.loc[df["is_train"], "city"] == "LA").all()
    assert df["flow_scale"].nunique() > 1

    results = _run_pipeline(cfg)
    assert results["n_test_rows"] > 0


def test_feature_columns_disjoint_from_event_score():
    # The baseline must not accidentally leak Model B's event score into its features.
    assert "event_impact_score" not in traffic_feature_columns(CFG)
