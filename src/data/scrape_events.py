"""Event web-scraper — STUB for the thesis's real-data path.

The abstract commits to collecting event information "through web scraping techniques".
This module is where that lives. It is intentionally not wired into the default run (the
pipeline uses synthetic events), but the structure below shows exactly what to fill in and
what schema to return so `src/data/load.load_real_events` can consume it.

Two realistic options for a metro like LA / the Bay Area:
  1. Ticketmaster Discovery API (free key) — cleanest, returns lat/lon + datetime + name.
  2. Scrape a venue / city events calendar with requests + BeautifulSoup.

Return a DataFrame with columns:
    event_id, lat, lon, start_time, duration_h, category, expected_attendance
"""

from __future__ import annotations

import pandas as pd

# import requests
# from bs4 import BeautifulSoup


def scrape_ticketmaster(
    api_key: str,
    lat: float,
    lon: float,
    radius_km: int = 30,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """TODO: call https://app.ticketmaster.com/discovery/v2/events.json

    Map each returned event to the schema. `expected_attendance` can be approximated from
    the venue capacity (also in the API) when a true attendance figure is unavailable.
    """
    raise NotImplementedError(
        "Get a free Ticketmaster Discovery API key and map the JSON to the events schema."
    )


def scrape_calendar_html(url: str) -> pd.DataFrame:
    """TODO: requests.get(url) -> BeautifulSoup -> parse event cards into the schema."""
    raise NotImplementedError("Implement HTML scraping for your chosen local events page.")


if __name__ == "__main__":
    print(__doc__)
