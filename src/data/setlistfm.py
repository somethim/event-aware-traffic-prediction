"""Fetch historical concerts from the setlist.fm API into this project's events schema.

We use setlist.fm rather than Ticketmaster because Ticketmaster's Discovery API only lists
upcoming events, and our PeMS window is already in the past. setlist.fm is a free historical
concert database, which is what we need to label past traffic.

setlist.fm has a few gaps that we fill from VENUE_REFERENCE. It gives a date but no start time,
so we assume a fixed local start hour (config default_start_hour). It gives only city-level
coordinates, so we substitute the venue's precise coords. It has no capacity or attendance
figure, so we use the venue's announced capacity as the popularity signal; for the thesis,
capacity rather than actual turnout is what matters. Because coords and capacity come from
VENUE_REFERENCE, only concerts at a known reference venue are usable, and setlists at unlisted
venues are dropped. Extend VENUE_REFERENCE to widen coverage.

Since setlist.fm is date-only, the assembled start time is already local wall-clock (naive),
which lines up with PeMS's Pacific-local-naive timestamps without any conversion.

Two data-quality traps in the API's shape are handled here. The venueName search matches
venues nationwide and several reference names are chains or duplicates (The Fillmore, Fox
Theater, Greek Theatre ...), so every setlist is validated against its own city block before
it may inherit the reference venue's coords and capacity. And setlist.fm returns one setlist
per performing artist, so multi-artist bills are collapsed to one event per venue and day.
"""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

from ..config import get_env
from ..utils import haversine_km

if TYPE_CHECKING:
    import requests

SEARCH_URL = "https://api.setlist.fm/rest/1.0/search/setlists"
ITEMS_PER_PAGE = 20  # fixed by the API
CATEGORY = "concert"
USER_AGENT = "event-aware-traffic-prediction (thesis research)"
# How far a setlist's own city coords may sit from the reference venue and still count as the
# same place. City coords are metro-level, so this only needs to separate metros, not blocks.
MAX_CITY_VENUE_KM = 40.0


class Venue(NamedTuple):
    """A reference venue with precise location, announced capacity, and project city."""

    display: str  # name to query setlist.fm with
    lat: float
    lon: float
    capacity: int
    city: str  # "LA" or "SF"


def _norm(name: str) -> str:
    """Normalize a venue name for matching: lowercase, drop punctuation and a leading 'the'."""
    s = re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()
    s = re.sub(r"\s+", " ", s)
    return s[4:] if s.startswith("the ") else s


# Major LA and SF-Bay venues with precise coords and announced capacity. Keyed by normalized
# name so setlist.fm's venueName results match robustly. Extend as needed.
_VENUES = [
    # --- Los Angeles (District 7) ---
    Venue("SoFi Stadium", 33.9535, -118.3392, 70000, "LA"),
    Venue("Rose Bowl", 34.1613, -118.1676, 88000, "LA"),
    Venue("Dodger Stadium", 34.0739, -118.2400, 56000, "LA"),
    Venue("Crypto.com Arena", 34.0430, -118.2673, 20000, "LA"),
    Venue("BMO Stadium", 34.0128, -118.2848, 22000, "LA"),
    Venue("Kia Forum", 33.9583, -118.3417, 17500, "LA"),
    Venue("Hollywood Bowl", 34.1122, -118.3390, 17500, "LA"),
    Venue("Greek Theatre", 34.1197, -118.2903, 5900, "LA"),
    Venue("YouTube Theater", 33.9560, -118.3392, 6000, "LA"),
    Venue("Peacock Theater", 34.0444, -118.2673, 7100, "LA"),
    Venue("The Novo", 34.0452, -118.2660, 2300, "LA"),
    Venue("The Wiltern", 34.0617, -118.3089, 1850, "LA"),
    # --- SF Bay Area (District 4) ---
    Venue("Levi's Stadium", 37.4030, -121.9700, 68500, "SF"),
    Venue("Oracle Park", 37.7786, -122.3893, 41000, "SF"),
    Venue("Oakland Coliseum", 37.7516, -122.2005, 63000, "SF"),
    Venue("Stanford Stadium", 37.4347, -122.1611, 50000, "SF"),
    Venue("Shoreline Amphitheatre", 37.4267, -122.0806, 22500, "SF"),
    Venue("Chase Center", 37.7680, -122.3877, 18000, "SF"),
    Venue("SAP Center", 37.3327, -121.9010, 17500, "SF"),
    Venue("Bill Graham Civic Auditorium", 37.7786, -122.4183, 8500, "SF"),
    Venue("Frost Amphitheater", 37.4265, -122.1610, 6900, "SF"),
    Venue("Fox Theater", 37.8080, -122.2687, 2800, "SF"),
    Venue("The Masonic", 37.7917, -122.4126, 3300, "SF"),
    Venue("The Fillmore", 37.7840, -122.4331, 1150, "SF"),
]

VENUE_REFERENCE: dict[str, Venue] = {_norm(v.display): v for v in _VENUES}


def venues_for(cities: list[str] | None = None) -> list[Venue]:
    """Reference venues, optionally restricted to the given project cities."""
    if cities is None:
        return list(VENUE_REFERENCE.values())
    return [v for v in VENUE_REFERENCE.values() if v.city in cities]


def parse_setlists(
    payload: dict,
    default_start_hour: int = 20,
    default_duration_h: float = 3.0,
    venue: Venue | None = None,
) -> pd.DataFrame:
    """Map one setlist.fm response page into rows of the events schema.

    Keeps only setlists whose venue matches a reference venue (so location + capacity are
    known). If `venue` is given (per-venue query), every returned setlist is matched against it;
    otherwise the setlist's own venue name is looked up in VENUE_REFERENCE. Either way the
    setlist must also pass `_city_matches`, since a name match alone would let a same-named
    venue elsewhere in the country inherit our venue's coords and capacity.
    """
    rows = []
    for sl in payload.get("setlist", []):
        date = sl.get("eventDate")  # dd-MM-yyyy
        if not date:
            continue
        ref = venue if venue is not None else VENUE_REFERENCE.get(_norm(_venue_name(sl)))
        if ref is None:  # unlisted venue, so no capacity or coords to use
            continue
        if not _city_matches(sl, ref):  # same-named venue somewhere else in the country
            continue
        day = pd.Timestamp(pd.to_datetime(date, format="%d-%m-%Y"))
        start = day + pd.Timedelta(hours=default_start_hour)  # local wall-clock, naive
        rows.append(
            {
                "event_id": sl.get("id"),
                "city": ref.city,
                "lat": ref.lat,
                "lon": ref.lon,
                "start_time": start,
                "duration_h": default_duration_h,
                "category": CATEGORY,
                "expected_attendance": ref.capacity,
            }
        )
    return pd.DataFrame(rows)


def _venue_name(setlist: dict) -> str:
    return (setlist.get("venue") or {}).get("name", "") or ""


def _city_matches(setlist: dict, ref: Venue, max_km: float = MAX_CITY_VENUE_KM) -> bool:
    """True when the setlist's own venue city is plausibly the reference venue's location.

    Uses the city coordinates setlist.fm attaches to every venue, falling back to the state
    code when coords are missing. No usable city info means the match cannot be proven, so
    the setlist is dropped rather than risking cross-country contamination.
    """
    city = (setlist.get("venue") or {}).get("city") or {}
    coords = city.get("coords") or {}
    lat, lon = coords.get("lat"), coords.get("long")
    if lat is not None and lon is not None:
        return float(haversine_km(float(lat), float(lon), ref.lat, ref.lon)) <= max_km
    return city.get("stateCode") == "CA"


def dedupe_events(events: pd.DataFrame) -> pd.DataFrame:
    """Collapse multi-artist bills into one event per (venue, day).

    setlist.fm returns one setlist per performing artist, so a headliner-plus-openers show
    would otherwise appear several times and multiply the attendance exposure downstream.
    """
    if events.empty:
        return events
    return events.drop_duplicates(subset=["lat", "lon", "start_time"]).reset_index(drop=True)


def _get(headers: dict, params: dict, sleep_s: float) -> requests.Response:
    """One request to the search endpoint, paced to respect the 2 requests/second cap and
    retried with backoff when the API returns 429. Always waits `sleep_s` afterwards so the
    next call (including the first page of the next venue) stays under the limit."""
    import requests  # imported here so the network dependency stays optional

    resp = None
    for attempt in range(6):
        resp = requests.get(SEARCH_URL, headers=headers, params=params, timeout=30)
        if resp.status_code != 429:
            break
        wait = float(resp.headers.get("Retry-After") or 2**attempt)
        print(f"[setlistfm] rate-limited, waiting {wait:.0f}s")
        time.sleep(min(max(wait, 1.0), 30.0))
    time.sleep(sleep_s)
    assert resp is not None  # range(6) always runs, so the loop assigned resp
    return resp


def fetch_events(
    window: tuple[str, str],
    cfg: dict,
    api_key: str | None = None,
    max_pages_per_venue: int = 40,
    sleep_s: float = 0.6,
) -> pd.DataFrame:
    """Query setlist.fm for concerts at every reference venue within a date window.

    `window` is ("YYYY-MM-DD", "YYYY-MM-DD"). Iterates reference venues for the configured
    cities, pulls their setlists year-by-year, and filters to the window. Returns the events
    table with local-naive start times and capacity as expected_attendance.
    """
    scfg = cfg["data"]["real"].get("setlistfm", {})
    start_hour = int(scfg.get("default_start_hour", 20))
    duration_h = float(scfg.get("default_duration_h", 3.0))
    cities = scfg.get("cities")
    key = api_key or get_env("SETLISTFM_API_KEY", required=True)
    assert key is not None  # narrows the type for mypy; get_env already raises if missing
    headers = {"x-api-key": key, "Accept": "application/json", "User-Agent": USER_AGENT}

    start = pd.Timestamp(window[0])
    end = pd.Timestamp(window[1])
    years = list(range(start.year, end.year + 1))

    frames = []
    for v in venues_for(cities):
        for year in years:
            for page in range(1, max_pages_per_venue + 1):
                params = {"venueName": v.display, "year": str(year), "p": str(page)}
                resp = _get(headers, params, sleep_s)
                if resp.status_code == 404:
                    break  # no setlists for this venue and year
                if resp.status_code in (401, 403):
                    raise SystemExit(
                        "[setlistfm] setlist.fm rejected the request "
                        f"(HTTP {resp.status_code}). Its API runs on AWS API Gateway, which "
                        "returns this when SETLISTFM_API_KEY is missing, wrong, or not yet "
                        "activated. Check the key at https://www.setlist.fm/settings/apps and "
                        "confirm your API access has been granted."
                    )
                resp.raise_for_status()
                payload = resp.json()
                frames.append(parse_setlists(payload, start_hour, duration_h, venue=v))
                total = int(payload.get("total", 0))
                if page * ITEMS_PER_PAGE >= total:
                    break
            else:
                print(
                    f"[setlistfm] WARNING {v.display} {year}: truncated at "
                    f"{max_pages_per_venue * ITEMS_PER_PAGE} of {total} setlists — raise "
                    "max_pages_per_venue if the missing pages could fall inside the window"
                )

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        # Keep only events inside the PeMS window, then collapse multi-artist bills.
        mask = (out["start_time"] >= start) & (out["start_time"] <= end + pd.Timedelta(days=1))
        out = dedupe_events(out[mask].drop_duplicates(subset="event_id"))
    print(f"[setlistfm] fetched {len(out)} concerts across {len(venues_for(cities))} venues")
    return out
