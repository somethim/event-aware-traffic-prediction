from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.config import CFG
from src.features.build_features import EVENT_FEATURE_COLUMNS, event_features, shift_events_placebo
from src.models.metrics import wape
from src.pipeline.train_event_model import forward_expanding_splits
from src.provenance import validate_manifest


def test_forward_cross_fit_never_uses_same_or_later_timestamp():
    times = np.repeat(pd.date_range("2026-01-01", periods=20, freq="15min").to_numpy(), 2)
    for train, valid in forward_expanding_splits(times, 4):
        assert times[train].max() < times[valid].min()


def test_placebo_shifts_exactly_seven_days_and_preserves_metadata():
    events = pd.DataFrame(
        {
            "event_id": ["a"],
            "start_time": [pd.Timestamp("2026-05-11 20:00")],
            "lat": [1.0],
            "expected_attendance": [10],
        }
    )
    shifted = shift_events_placebo(events)
    delta = shifted["start_time"].to_numpy()[0] - events["start_time"].to_numpy()[0]
    assert delta == np.timedelta64(7, "D")
    pd.testing.assert_frame_equal(
        shifted.drop(columns="start_time"), events.drop(columns="start_time")
    )


def test_wape_handles_zero_flow_and_all_zero_denominator():
    assert wape([0, 10], [5, 8]) == 70.0
    assert np.isnan(wape([0, 0], [1, 2]))


def test_manifest_rejects_config_and_input_mismatch(monkeypatch):
    cfg = copy.deepcopy(CFG)
    monkeypatch.setattr("src.provenance.input_checksums", lambda: {"flow.parquet": "now"})
    from src.config import config_checksum

    valid = {"config_checksum": config_checksum(cfg), "input_checksums": {"flow.parquet": "now"}}
    validate_manifest(valid, cfg)
    changed = copy.deepcopy(cfg)
    changed["seed"] += 1
    try:
        validate_manifest(valid, changed)
    except RuntimeError as exc:
        assert "configuration checksum" in str(exc)
    else:
        raise AssertionError("configuration mismatch was accepted")
    valid["input_checksums"] = {"flow.parquet": "stale"}
    try:
        validate_manifest(valid, cfg)
    except RuntimeError as exc:
        assert "input checksum" in str(exc)
    else:
        raise AssertionError("input mismatch was accepted")


def test_event_attribution_none_one_and_overlap():
    cfg = copy.deepcopy(CFG)
    cfg["features"].update(event_radius_km=10.0, event_lead_hours=2.0)
    timestamps = pd.date_range("2026-05-11 17:00", periods=25, freq="15min")
    flow = pd.DataFrame({"sensor_id": 1, "timestamp": timestamps})
    sensors = pd.DataFrame({"sensor_id": [1], "lat": [34.0], "lon": [-118.0]})
    events = pd.DataFrame(
        {
            "event_id": ["small", "large"],
            "lat": [34.0, 34.0],
            "lon": [-118.0, -118.0],
            "start_time": [pd.Timestamp("2026-05-11 20:00")] * 2,
            "duration_h": [2.0, 2.0],
            "expected_attendance": [1000, 5000],
        }
    )
    out = event_features(flow, events, sensors, cfg)
    assert (
        out.loc[out["timestamp"] == pd.Timestamp("2026-05-11 17:00"), "event_phase"].iloc[0]
        == "none"
    )
    at_start = out.loc[out["timestamp"] == pd.Timestamp("2026-05-11 20:00")].iloc[0]
    assert at_start["dominant_event_id"] == "large"
    assert at_start["event_phase"] == "during"
    assert all(c in out for c in EVENT_FEATURE_COLUMNS)
