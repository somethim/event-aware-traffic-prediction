"""Build the real dataset into the parquet files the pipeline reads.

It combines Caltrans PeMS flow (auto-fetched from Google Drive, since the ~2.7 GB is not kept
in the repo) with setlist.fm historical concerts. Concerts are fetched for the PeMS date
window, and their start times are on PeMS's Pacific-local clock. We keep only the sensors near
an event (within event_radius_km plus a buffer), load their 5-min flow resampled to the config
frequency, and write flow.parquet, sensors.parquet, and events.parquet.

`ensure_dataset` is what the runners call. It builds the configured dataset only when the flow
parquet is missing.

    uv run python -m scripts.build_real_dataset      # force a real rebuild
"""

from __future__ import annotations

import re
import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

from ..config import CFG, EVENTS_FILE, FLOW_FILE, INPUTS_DIR, ROOT, SENSORS_FILE
from ..data import setlistfm
from ..data.pems import (
    build_flow_and_sensors_wanted,
    date_range,
    find_meta_file,
    load_station_meta,
)
from ..utils import haversine_km


def _drive_file_id(url_or_id: str) -> str:
    """Pull the file id out of a Drive share link (/file/d/<id>/... or ?id=<id>), or pass a
    bare id straight through."""
    m = re.search(r"/d/([\w-]+)", url_or_id) or re.search(r"[?&]id=([\w-]+)", url_or_id)
    return m.group(1) if m else url_or_id


def _extract_archive(path: Path, dest: Path) -> None:
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            zf.extractall(dest)
    elif tarfile.is_tarfile(path):  # tarfile auto-detects gz/bz2/xz, so .tar.xz works here
        with tarfile.open(path) as tf:
            tf.extractall(dest, filter="data")
    else:
        raise SystemExit(f"[fetch] {path.name} is neither a zip nor a tar archive")


def _flatten_extracted(dest: Path, dir_names: list[str]) -> None:
    """Move each district folder up to `dest` if the archive nested it under a top-level dir.

    Covers the common case where the archive was made from the parent folder (so it contains
    pems/4 and pems/7 rather than 4 and 7 at the root), which would otherwise land in the wrong
    place. Only moves a folder that actually holds 5-min files.
    """
    for name in dir_names:
        target = dest / name
        if target.exists() and any(target.glob("*station_5min_*.txt.gz")):
            continue
        for cand in dest.rglob(name):
            if cand.is_dir() and cand != target and any(cand.glob("*station_5min_*.txt.gz")):
                shutil.move(str(cand), str(target))
                break


def fetch_pems_from_drive(cfg: dict | None = None) -> None:
    """Download the raw PeMS 5-min data from Google Drive if it isn't already present.

    Prefers a single archive (gdrive_archive_url) because Google rate-limits the per-file folder
    download once it has served many files. Falls back to the folder URL if no archive is set.
    Skips entirely when both district directories already hold 5-min files.
    """
    cfg = cfg or CFG
    real = cfg["data"]["real"]
    dirs = [ROOT / d for d in real["pems_dirs"].values()]
    if all(d.exists() and any(d.glob("*station_5min_*.txt.gz")) for d in dirs):
        return

    import gdown  # imported here so the download dependency stays optional

    dest = ROOT / real["pems_root"]
    dest.mkdir(parents=True, exist_ok=True)

    archive_url = real.get("gdrive_archive_url")
    if archive_url:
        print("[fetch] downloading PeMS archive from Google Drive (single file)")
        tmp = dest / "_pems_archive.download"
        path = gdown.download(id=_drive_file_id(archive_url), output=str(tmp), quiet=False)
        if not isinstance(path, str):
            raise SystemExit("[fetch] archive download failed (check the link / sharing settings)")
        print(f"[fetch] extracting into {dest}")
        archive = Path(path)
        _extract_archive(archive, dest)
        _flatten_extracted(archive.parent, [Path(d).name for d in real["pems_dirs"].values()])
        archive.unlink(missing_ok=True)
    else:
        print(
            f"[fetch] downloading PeMS folder from Google Drive -> {dest} (large). "
            "Folder downloads get rate-limited by Google; if this fails, publish a single "
            "zip/tar of the {4,7} dirs and set data.real.gdrive_archive_url instead."
        )
        gdown.download_folder(
            url=real["gdrive_folder_url"], output=str(dest), quiet=False, use_cookies=False
        )

    missing = [d for d in dirs if not (d.exists() and any(d.glob("*station_5min_*.txt.gz")))]
    if missing:
        raise SystemExit(
            f"[fetch] download incomplete, no 5-min files in {missing}. Google likely throttled "
            "the folder download. Either retry later, set data.real.gdrive_archive_url to a "
            "single archive, or download the folder from the browser into data/raw/pems/."
        )


def _fetch_or_load_events(window: tuple[str, str], cfg: dict) -> pd.DataFrame:
    """Fetch setlist.fm concerts for the window, caching the raw result.

    Historical concerts don't change, so we save the fetched events to a per-window parquet and
    reuse it on later builds instead of hitting the API again. Set data.real.refresh_events: true
    (or delete the cache file) to force a re-fetch.
    """
    cache = INPUTS_DIR / f"events_setlistfm_{window[0]}_{window[1]}.parquet"
    if cache.exists() and not cfg["data"]["real"].get("refresh_events", False):
        events = pd.read_parquet(cache)
        events["start_time"] = pd.to_datetime(events["start_time"])
        print(f"[build] loaded {len(events)} cached setlist.fm events from {cache.name}")
        return events

    print(f"[build] fetching setlist.fm concerts {window[0]}..{window[1]}")
    events = setlistfm.fetch_events(window, cfg)
    events.to_parquet(cache, index=False)
    print(f"[build] cached {len(events)} setlist.fm events -> {cache.name}")
    return events


def _sensors_near_events(meta: pd.DataFrame, events: pd.DataFrame, keep_km: float) -> set[int]:
    """Sensor IDs within `keep_km` of any event."""
    if events.empty:
        return set()
    ev_lat = events["lat"].to_numpy(dtype=float)
    ev_lon = events["lon"].to_numpy(dtype=float)
    s_lat = meta["lat"].to_numpy(dtype=float)
    s_lon = meta["lon"].to_numpy(dtype=float)
    s_id = meta["sensor_id"].to_numpy()
    keep = set()
    for i in range(len(s_id)):
        if haversine_km(s_lat[i], s_lon[i], ev_lat, ev_lon).min() <= keep_km:
            keep.add(int(s_id[i]))
    return keep


def build_real(cfg: dict | None = None) -> None:
    cfg = cfg or CFG
    real = cfg["data"]["real"]
    freq = cfg["data"]["freq"]
    min_observed = float(real.get("min_observed", 0.5))
    keep_km = cfg["features"]["event_radius_km"] + real["station_buffer_km"]

    fetch_pems_from_drive(cfg)

    # Use the date range shared by both districts.
    ranges = {c: date_range(ROOT / d) for c, d in real["pems_dirs"].items()}
    start = max(r[0] for r in ranges.values())
    end = min(r[1] for r in ranges.values())
    print(f"[build] date window {start} .. {end}")

    events = _fetch_or_load_events((start, end), cfg)
    print(f"[build] {len(events)} events total")

    flow_parts: list[pd.DataFrame] = []
    sensor_parts: list[pd.DataFrame] = []
    for city, rel_dir in real["pems_dirs"].items():
        directory = ROOT / rel_dir
        meta = load_station_meta(find_meta_file(directory))
        city_events = events.loc[events["city"] == city] if not events.empty else events
        wanted = _sensors_near_events(meta, city_events, keep_km)
        print(f"[build] {city}: {len(wanted)} sensors near {len(city_events)} events")
        if not wanted:
            print(f"[build] WARNING {city}: no events/sensors — skipping.")
            continue
        flow, sensors = build_flow_and_sensors_wanted(
            directory, city, meta, wanted, freq, min_observed=min_observed
        )
        flow_parts.append(flow)
        sensor_parts.append(sensors)

    if not flow_parts:
        raise SystemExit("[build] no data built — no events returned in the PeMS date window.")

    flow_all = pd.concat(flow_parts, ignore_index=True)
    sensors_all = pd.concat(sensor_parts, ignore_index=True)
    # Drop events in cities where no sensor survived, so every event touches the network.
    events = events[events["city"].isin(sensors_all["city"].unique())]

    flow_all.to_parquet(FLOW_FILE, index=False)
    sensors_all.to_parquet(SENSORS_FILE, index=False)
    events.to_parquet(EVENTS_FILE, index=False)
    print(
        f"[build] wrote REAL dataset -> {FLOW_FILE.parent}/\n"
        f"  flow: {len(flow_all):,} rows, {sensors_all['sensor_id'].nunique()} sensors  "
        f"| events: {len(events)}"
    )


def ensure_dataset(cfg: dict | None = None) -> None:
    """Build the configured dataset if the flow parquet is missing (used by the runners).

    Real is the default and auto-fetches PeMS from Drive. Synthetic is opt-in for the
    controlled-experiment testbed and tests.
    """
    cfg = cfg or CFG
    if FLOW_FILE.exists():
        return
    if cfg["data"].get("source", "real") == "synthetic":
        from .generate_synthetic import generate

        generate(cfg)
    else:
        build_real(cfg)


if __name__ == "__main__":
    build_real()
