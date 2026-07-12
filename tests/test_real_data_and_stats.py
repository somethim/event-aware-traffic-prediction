"""Offline tests for the real-data and statistical-rigor additions: setlist.fm parsing,
event-time timezone normalization, the configurable prediction target, Model B's
cross-fitted proxy target, and the rolling-origin CV significance evaluator.

None of these hit the network; the API layer is exercised only through its pure parser.
"""

from __future__ import annotations

import copy
import tarfile

import numpy as np
import pandas as pd

from src.config import CFG, target_column
from src.data import setlistfm

from src.data.build_real import _drive_file_id, _extract_archive, _flatten_extracted
from src.data.generate_synthetic import generate
from src.features.build_features import traffic_feature_columns, traffic_features
from src.pipeline import evaluate, prepare
from src.pipeline.preprocess import normalize_per_sensor

from src.pipeline.train_event_model import _proxy_target
from src.utils import time_block_folds, to_local_naive


def _setlist(
    i: int,
    venue_name: str,
    city_name: str = "Inglewood",
    state_code: str = "CA",
    lat: float = 33.96,
    lon: float = -118.35,
) -> dict:
    """One setlist.fm search result with the city block the API always attaches to a venue."""
    return {
        "id": f"e{i}",
        "eventDate": "11-05-2026",
        "venue": {
            "name": venue_name,
            "city": {
                "name": city_name,
                "stateCode": state_code,
                "coords": {"lat": lat, "long": lon},
            },
        },
    }


def _payload(*setlists: dict) -> dict:
    return {"total": len(setlists), "setlist": list(setlists)}


def test_setlistfm_parse_keeps_known_venue_with_capacity():
    df = setlistfm.parse_setlists(
        _payload(_setlist(0, "SoFi Stadium"), _setlist(1, "Some Tiny Unlisted Bar"))
    )
    # Only the known reference venue should survive; the unlisted bar is dropped.
    assert len(df) == 1
    row = df.iloc[0]
    assert row["expected_attendance"] == 70000  # SoFi capacity, used as a popularity signal
    assert row["category"] == "concert"
    assert row["city"] == "LA"
    # A date-only event is assumed to start at 20:00 local wall-clock, tz-naive like PeMS.
    assert row["start_time"] == pd.Timestamp("2026-05-11 20:00:00")
    assert row["start_time"].tzinfo is None
    # Coordinates come from the reference table, not from coarse city-level coords.
    assert abs(row["lat"] - 33.9535) < 1e-6


def test_setlistfm_norm_and_reference_cover_both_cities():
    assert setlistfm._norm("The Wiltern") == "wiltern"
    assert setlistfm._norm("Crypto.com Arena") == "cryptocom arena"
    cities = {v.city for v in setlistfm.VENUE_REFERENCE.values()}
    assert cities == {"LA", "SF"}
    assert {v.city for v in setlistfm.venues_for(["SF"])} == {"SF"}


def test_drive_file_id_handles_link_forms():
    assert _drive_file_id("1AbC_dEfg-hi") == "1AbC_dEfg-hi"
    assert _drive_file_id("https://drive.google.com/file/d/1AbC_dEfg-hi/view") == "1AbC_dEfg-hi"
    assert (
        _drive_file_id("https://drive.google.com/uc?id=1AbC_dEfg-hi&export=download")
        == "1AbC_dEfg-hi"
    )


def test_tar_xz_archive_extracts_and_flattens(tmp_path):
    # Simulate an archive that nested the district dirs under a top-level "pems/" folder.
    src = tmp_path / "src" / "pems"
    for dist in ("4", "7"):
        (src / dist).mkdir(parents=True)
        (src / dist / f"d0{dist}_text_station_5min_2026_05_11.txt.gz").write_text("x")
    archive = tmp_path / "pems.tar.xz"
    with tarfile.open(archive, "w:xz") as tf:
        tf.add(src, arcname="pems")

    dest = tmp_path / "raw" / "pems"
    dest.mkdir(parents=True)
    _extract_archive(archive, dest)
    _flatten_extracted(dest, ["4", "7"])

    for dist in ("4", "7"):
        assert any((dest / dist).glob("*station_5min_*.txt.gz"))


def test_setlistfm_custom_start_hour():
    chase = _setlist(0, "Chase Center", "San Francisco", "CA", 37.7749, -122.4194)
    df = setlistfm.parse_setlists(_payload(chase), default_start_hour=19)
    assert df.iloc[0]["start_time"] == pd.Timestamp("2026-05-11 19:00:00")


def test_setlistfm_rejects_same_name_venue_in_other_city():
    # setlist.fm's venueName search matches nationwide: a Philadelphia Fillmore concert must
    # not be stamped with the San Francisco Fillmore's coordinates and capacity.
    ref = setlistfm.VENUE_REFERENCE[setlistfm._norm("The Fillmore")]
    df = setlistfm.parse_setlists(
        _payload(
            _setlist(0, "The Fillmore", "San Francisco", "CA", 37.7749, -122.4194),
            _setlist(1, "The Fillmore", "Philadelphia", "PA", 39.9526, -75.1652),
        ),
        venue=ref,
    )
    assert len(df) == 1
    assert df.iloc[0]["city"] == "SF"


def test_setlistfm_city_state_fallback_when_coords_missing():
    ref = setlistfm.VENUE_REFERENCE[setlistfm._norm("The Fillmore")]
    sl = _setlist(0, "The Fillmore", "San Francisco", "CA")
    del sl["venue"]["city"]["coords"]  # no coords: fall back to the state code
    assert len(setlistfm.parse_setlists(_payload(sl), venue=ref)) == 1
    sl["venue"]["city"]["stateCode"] = "PA"
    assert len(setlistfm.parse_setlists(_payload(sl), venue=ref)) == 0


def test_setlistfm_dedupes_multi_artist_bills():
    # setlist.fm returns one setlist per artist, so a headliner + opener at the same venue and
    # date must collapse to a single event (otherwise attendance exposure double-counts).
    df = setlistfm.parse_setlists(
        _payload(_setlist(0, "SoFi Stadium"), _setlist(1, "SoFi Stadium"))
    )
    assert len(df) == 2  # the parser keeps both artists' setlists ...
    assert len(setlistfm.dedupe_events(df)) == 1  # ... and the dedupe collapses the bill


def test_time_block_folds_are_chronological_blocks():
    times = pd.date_range("2026-05-11", periods=10, freq="1h").to_numpy()
    folds = time_block_folds(times, 3)
    positions = np.concatenate(folds)
    assert sorted(positions.tolist()) == list(range(10))
    # With chronologically sorted input, blocks are contiguous runs marching forward in time.
    flat = [int(p) for blk in folds for p in blk]
    assert flat == sorted(flat)


def test_traffic_features_drop_rows_with_nan_rolling_gap():
    # A NaN gap (real PeMS data has them from %Observed gating) makes rolling features NaN on
    # rows whose lags are all defined; those rows must be dropped or GradientBoosting crashes.
    cfg = copy.deepcopy(CFG)
    cfg["features"]["lags"] = [1, 2]
    cfg["features"]["rolling_windows"] = [3]
    ts = pd.date_range("2026-05-11", periods=30, freq="15min")
    vals = np.arange(30.0) + 100
    vals[10] = np.nan
    flow = pd.DataFrame({"sensor_id": 1, "city": "LA", "timestamp": ts, "flow": vals})
    feat = traffic_features(flow, cfg)
    assert len(feat) > 0
    assert not feat[traffic_feature_columns(cfg)].isna().to_numpy().any()


def test_to_local_naive_shifts_utc_to_pacific():
    # 03:00 UTC on 2026-05-11 is 20:00 PDT on 2026-05-10, since May is UTC-7.
    out = to_local_naive(pd.Series(["2026-05-11T03:00:00Z"]), "America/Los_Angeles")
    assert out.iloc[0] == pd.Timestamp("2026-05-10 20:00:00")
    assert out.dt.tz is None


def test_target_column_threads_through_features_and_normalize():
    cfg = copy.deepcopy(CFG)
    cfg["data"]["target"] = "speed"
    cfg["features"]["lags"] = [1, 2]
    cfg["features"]["rolling_windows"] = [2]
    assert target_column(cfg) == "speed"

    ts = pd.date_range("2026-05-11", periods=10, freq="15min")
    flow = pd.DataFrame(
        {
            "sensor_id": 1,
            "city": "LA",
            "timestamp": ts,
            "flow": np.arange(10.0) + 100,
            "speed": np.arange(10.0) + 60,
        }
    )
    feat = traffic_features(flow, cfg)
    # Lags are computed on the target metric (speed here), not on flow.
    assert feat["lag_1"].iloc[0] == flow["speed"].iloc[1]  # sorted; first surviving row
    feat["is_train"] = True
    normed = normalize_per_sensor(feat, cfg)
    # With normalization off, the target is just the raw speed.
    assert np.allclose(normed["target"], normed["speed"])


def _prepared_tiny():
    cfg = copy.deepcopy(CFG)
    cfg["data"]["n_days"] = 12
    cfg["data"]["cities"] = [
        {
            "name": "LA",
            "n_sensors": 4,
            "n_events": 12,
            "base_flow": [300, 1400],
            "area": {"lat_min": 34.0, "lat_max": 34.2, "lon_min": -118.35, "lon_max": -118.10},
        }
    ]
    cfg["features"]["lags"] = [1, 2, 3, 96]
    cfg["features"]["rolling_windows"] = [4]
    cfg["model"]["random_forest"].update(n_estimators=25)
    cfg["model"]["cross_fit_folds"] = 3
    generate(cfg)
    df = prepare.prepare(cfg)
    return cfg, df


def test_proxy_target_cross_fit_is_nonnegative_and_finite():
    cfg, df = _prepared_tiny()
    # On real data there is no ground-truth effect, so compute the proxy target directly.
    y = _proxy_target(df, cfg)
    assert y.shape[0] == len(df)
    assert np.all(y >= 0.0)
    assert np.all(np.isfinite(y))
    assert y.max() > 0.0  # some rows carry unexplained, event-driven traffic


def test_rolling_origin_folds_are_forward_and_disjoint():
    folds = evaluate.rolling_origin_folds(1000, 4, 0.15)
    assert len(folds) == 4
    prev_end = 0
    for tr_end, te_end in folds:
        assert tr_end < te_end <= 1000
        assert te_end > prev_end  # each fold ends later than the last
        prev_end = te_end


def test_evaluate_runs_and_reports_significance_fields():
    cfg, _ = _prepared_tiny()
    cfg["evaluation"] = {
        "seeds": [42, 43],
        "cv": {"n_folds": 2, "test_frac": 0.2},
        "alpha": 0.05,
        "n_bootstrap": 100,
    }
    res = evaluate.evaluate(cfg)
    for block in ("event_affected", "overall"):
        assert "mean_mae_gap" in res[block]
        assert "ttest_p" in res[block]
        assert "ci_low" in res[block] and "ci_high" in res[block]
    assert len(res["cells"]) == 2 * 2  # 2 folds x 2 seeds
