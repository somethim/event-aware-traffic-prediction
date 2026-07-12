"""Load raw Caltrans PeMS Station 5-Minute files into this project's flow/sensor schema.

The 5-min files (``dNN_text_station_5min_YYYY_MM_DD.txt.gz``) are headerless CSVs with 52
positional columns, of which we use only a handful. We read (0-indexed) Total Flow (9),
Avg Occupancy (10), and Avg Speed (11) so the model can target flow, speed, or occupancy,
plus %Observed (8) to gate imputed data. Lane-type "ML" is mainline; we drop ramps.

PeMS backfills gaps and reports how much of each reading is real via %Observed. A resampled
bin with no underlying samples, or an average %Observed below `min_observed`, is left as NaN
rather than a fake 0, so it is dropped downstream instead of masquerading as real traffic. We
keep the regular time grid so positional lag features stay aligned.

Station lat/lon are not in the 5-min files. They live in the Station Metadata file
(``dNN_text_meta_*.txt``, tab-separated, with a header), which must be downloaded per district
from the PeMS Clearinghouse (Type = "Station Metadata"). Without it we cannot compute
event-distance features.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pandas as pd


def find_meta_file(directory: str | Path) -> Path:
    """Locate the Station Metadata (*meta*.txt) file dropped alongside the 5-min files."""
    hits = sorted(Path(directory).glob("*meta*.txt"))
    if not hits:
        raise FileNotFoundError(
            f"no Station Metadata (*meta*.txt) in {directory} — download it from the PeMS "
            "Clearinghouse (Type = 'Station Metadata')."
        )
    return hits[0]


def date_range(directory: str | Path) -> tuple[str, str]:
    """Min/max date (YYYY-MM-DD) parsed from the 5-min filenames in a district folder."""
    dates = sorted(
        m.group(0).replace("_", "-")
        for f in Path(directory).glob("*station_5min_*.txt.gz")
        if (m := re.search(r"\d{4}_\d{2}_\d{2}", f.name))
    )
    return dates[0], dates[-1]


# Station-level positional columns of the Station 5-Minute file (the rest are per-lane).
# 8=%Observed, 9=Total Flow, 10=Avg Occupancy, 11=Avg Speed.
_5MIN_USECOLS = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]
_5MIN_NAMES = [
    "timestamp",
    "sensor_id",
    "district",
    "freeway",
    "direction",
    "lane_type",
    "pct_observed",
    "flow",
    "occupancy",
    "speed",
]


def load_station_meta(path: str | Path) -> pd.DataFrame:
    """Load a PeMS Station Metadata file -> sensor_id, lat, lon (+ freeway, direction)."""
    meta = pd.read_csv(path, sep="\t")
    # Column names vary slightly by district and year, so normalize the ones we need.
    rename = {
        "ID": "sensor_id",
        "Latitude": "lat",
        "Longitude": "lon",
        "Fwy": "freeway",
        "Dir": "direction",
    }
    meta = meta.rename(columns=rename)
    keep = [c for c in ["sensor_id", "lat", "lon", "freeway", "direction"] if c in meta.columns]
    return meta[keep].dropna(subset=["lat", "lon"])


def load_5min_dir(
    directory: str | Path,
    city: str,
    freq: str = "15min",
    lane_type: str = "ML",
    stations: set[int] | None = None,
    min_observed: float = 0.5,
) -> pd.DataFrame:
    """Read every 5-min file in `directory`, keep mainline data, resample per sensor to `freq`.

    Flow is summed across the sub-intervals (vehicles per `freq`), while speed and occupancy are
    averaged. A bin with no underlying 5-min samples, or an average %Observed below
    `min_observed`, is set to NaN so imputed data does not masquerade as measured traffic. Pass
    `stations` to keep only those sensor IDs. Filtering happens per file so memory stays bounded
    even though a district-day file has around 1.4M rows.
    """
    files = sorted(glob.glob(str(Path(directory) / "*station_5min_*.txt.gz")))
    if not files:
        raise FileNotFoundError(f"no PeMS 5-min files in {directory}")

    frames: list[pd.DataFrame] = []
    for f in files:
        df = pd.read_csv(
            f,
            header=None,
            usecols=_5MIN_USECOLS,
            names=_5MIN_NAMES,
            compression="gzip",
            dtype={"lane_type": "string"},
        )
        df = df[(df["lane_type"] == lane_type) & df["flow"].notna()]
        if stations is not None:
            df = df[df["sensor_id"].isin(stations)]
        frames.append(df[["timestamp", "sensor_id", "flow", "speed", "occupancy", "pct_observed"]])
    raw = pd.concat(frames, ignore_index=True)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="%m/%d/%Y %H:%M:%S")

    # Resample each sensor to the target frequency. `n_obs` counts real samples in the bin and
    # `observed` averages %Observed, which lets us null out imputed or empty bins below.
    agg = (
        raw.set_index("timestamp")
        .groupby("sensor_id")
        .resample(freq)
        .agg(
            flow=("flow", "sum"),
            speed=("speed", "mean"),
            occupancy=("occupancy", "mean"),
            observed=("pct_observed", "mean"),
            n_obs=("flow", "count"),
        )
        .reset_index()
    )
    # Empty bins sum to a fake zero flow and low-%Observed bins are mostly imputed, so NaN both.
    untrusted = (agg["n_obs"] == 0) | (agg["observed"].fillna(0) / 100.0 < min_observed)
    agg.loc[untrusted, ["flow", "speed", "occupancy"]] = pd.NA
    agg["city"] = city
    return agg[["sensor_id", "city", "timestamp", "flow", "speed", "occupancy"]]


def build_flow_and_sensors_wanted(
    directory: str | Path,
    city: str,
    meta: pd.DataFrame,
    wanted: set[int],
    freq: str = "15min",
    min_observed: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load flow for a specific set of stations and pair it with their coordinates.

    Returns (flow, sensors), keeping only stations present in both the flow data and the
    metadata.
    """
    flow = load_5min_dir(directory, city, freq=freq, stations=wanted, min_observed=min_observed)
    sensors = meta[meta["sensor_id"].isin(flow["sensor_id"].unique())].copy()
    sensors["city"] = city
    flow = flow[flow["sensor_id"].isin(sensors["sensor_id"])]
    return (
        flow[["sensor_id", "city", "timestamp", "flow", "speed", "occupancy"]],
        sensors[["sensor_id", "city", "lat", "lon"]],
    )


def build_flow_and_sensors(
    dirs_by_city: dict[str, str | Path],
    meta_by_city: dict[str, str | Path],
    freq: str = "15min",
    max_stations_per_city: int | None = None,
    min_observed: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine multiple districts (cities) into the project's flow + sensors tables.

    `max_stations_per_city` optionally subsamples the busiest N stations per city to keep the
    dataset tractable (PeMS districts have thousands of detectors).
    """
    flow_parts, sensor_parts = [], []
    for city, directory in dirs_by_city.items():
        flow = load_5min_dir(directory, city, freq=freq, min_observed=min_observed)
        meta = load_station_meta(meta_by_city[city])
        meta["city"] = city

        if max_stations_per_city is not None:
            busiest = flow.groupby("sensor_id")["flow"].mean().nlargest(max_stations_per_city).index
            flow = flow[flow["sensor_id"].isin(busiest)]

        # Keep only sensors that have both flow and known coordinates.
        sensors = meta[meta["sensor_id"].isin(flow["sensor_id"].unique())].copy()
        flow = flow[flow["sensor_id"].isin(sensors["sensor_id"])]

        flow_parts.append(flow)
        sensor_parts.append(sensors[["sensor_id", "city", "lat", "lon"]])
        print(f"[pems] {city}: {sensors['sensor_id'].nunique()} sensors, {len(flow):,} rows")

    return (
        pd.concat(flow_parts, ignore_index=True),
        pd.concat(sensor_parts, ignore_index=True),
    )
