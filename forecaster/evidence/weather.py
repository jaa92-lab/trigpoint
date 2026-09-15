"""Weather model forecasts from Open-Meteo, for weather questions.

Open-Meteo is free and needs no key. The place comes from the question: capitalized
names after a preposition, tried longest first until the geocoder recognizes one.
The evidence names the place it found, so an analyst can see a wrong match.
Forecasts as they looked on a past day cannot be fetched, so pastcasting skips
this source and says so, as it does for prediction markets.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from typing import Any

import httpx

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.http import get_json
from forecaster.triage import Triage, Window

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
SOURCE_URL = "https://open-meteo.com/"
MAX_FORECAST_DAYS = 16
DEFAULT_DAYS = 7
MAX_PLACE_WORDS = 5
MAX_GEOCODE_ATTEMPTS = 6
DAILY_VARIABLES = (
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "precipitation_probability_max",
    "wind_gusts_10m_max",
)

_PLACE = re.compile(r"\b(?:in|at|for|over|near|of)\s+([A-Z][\w'.-]*(?:(?:\s+|,\s*)[A-Z][\w'.-]*)*)")
_STORM_NAME = re.compile(r"\b(storm|hurricane|tropical|cyclone|typhoon|depression)\b", re.IGNORECASE)
_FAHRENHEIT = re.compile(r"°\s?F\b|\bfahrenheit\b|\bdegrees\s+F\b", re.IGNORECASE)
_INCHES = re.compile(r"\binch(?:es)?\b", re.IGNORECASE)
_NOT_PLACES = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july", "august",
        "september", "october", "november", "december", "jan", "feb", "mar", "apr",
        "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "monday", "tuesday",
        "wednesday", "thursday", "friday", "saturday", "sunday", "the", "utc", "gmt",
        "celsius", "fahrenheit",
    }
)


def place_candidates(*texts: str | None) -> list[tuple[str, str | None]]:
    """Possible place names with an optional region qualifier, most specific first.

    "at Phoenix Sky Harbor International Airport" yields the full name, then
    shorter prefixes down to "Phoenix". "in Denver, Colorado" yields ("Denver",
    "Colorado"). Storm names and dates are skipped.
    """
    found: list[tuple[str, str | None]] = []
    for text in texts:
        for match in _PLACE.finditer(text or ""):
            parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
            qualifier = parts[1] if len(parts) > 1 else None
            words: list[str] = []
            for word in parts[0].split()[:MAX_PLACE_WORDS]:
                if word.endswith("'s"):
                    words.append(word[:-2])
                    break
                words.append(word)
            if not words or words[0].lower().strip(".") in _NOT_PLACES or _STORM_NAME.search(parts[0]):
                continue
            for count in range(len(words), 0, -1):
                name = " ".join(words[:count]).strip(".")
                if name.lower() in _NOT_PLACES or (name, qualifier) in found:
                    continue
                found.append((name, qualifier))
    return found


def forecast_dates(window: Window | None, today: date) -> tuple[date, date] | None:
    """The days to request, or None when the question's dates are outside the forecast range."""
    if window is None:
        return today, today + timedelta(days=DEFAULT_DAYS - 1)
    last = today + timedelta(days=MAX_FORECAST_DAYS - 1)
    if window.end.date() < today or window.start.date() > last:
        return None
    return max(window.start.date(), today), min(window.end.date(), last)


def _matches_region(result: dict[str, Any], qualifier: str) -> bool:
    wanted = qualifier.lower()
    fields = (result.get("admin1"), result.get("country"), result.get("country_code"))
    return any(str(value).lower().startswith(wanted) for value in fields if value)


async def geocode(client: httpx.AsyncClient, name: str, qualifier: str | None) -> dict[str, Any] | None:
    data = await get_json(
        client, GEOCODE_URL, params={"name": name, "count": 10, "language": "en", "format": "json"}, attempts=2
    )
    results = (data or {}).get("results") or []
    if not results:
        return None
    if qualifier:
        for result in results:
            if _matches_region(result, qualifier):
                return result
    return results[0]


def _with_unit(value: float, unit: str) -> str:
    joiner = "" if unit in ("°C", "°F", "%") else " "
    return f"{value:g}{joiner}{unit}"


def render_forecast(data: dict[str, Any]) -> str:
    daily = data.get("daily") or {}
    units = data.get("daily_units") or {}

    def value(key: str, index: int) -> str | None:
        values = daily.get(key) or []
        if index >= len(values) or values[index] is None:
            return None
        return _with_unit(values[index], units.get(key, ""))

    lines = [f"Daily values in local time ({data.get('timezone') or 'time zone not given'}):"]
    for index, day in enumerate(daily.get("time") or []):
        bits = []
        if (high := value("temperature_2m_max", index)) is not None:
            bits.append(f"high {high}")
        if (low := value("temperature_2m_min", index)) is not None:
            bits.append(f"low {low}")
        if (rain := value("precipitation_sum", index)) is not None:
            chance = value("precipitation_probability_max", index)
            bits.append(f"precipitation {rain}" + (f" ({chance} chance)" if chance else ""))
        if (gust := value("wind_gusts_10m_max", index)) is not None:
            bits.append(f"gusts {gust}")
        lines.append(f"- {day}: {', '.join(bits) if bits else 'no data'}")
    lines.append("Model forecasts are sharp for a few days and much less reliable beyond a week.")
    return "\n".join(lines)


async def fetch_weather(
    client: httpx.AsyncClient,
    question_text: str,
    resolution_criteria: str | None,
    triage: Triage,
    clock: Clock,
) -> tuple[list[EvidenceItem], list[str]]:
    if not clock.is_live:
        return [], ["Weather forecasts were not consulted: past forecasts are not available for pastcasting."]

    candidates = place_candidates(question_text, resolution_criteria)[:MAX_GEOCODE_ATTEMPTS]
    if not candidates:
        return [], ["No place name was found in the question, so no weather forecast was fetched."]
    place = None
    for name, qualifier in candidates:
        try:
            place = await geocode(client, name, qualifier)
        except Exception as error:  # one failed lookup should not stop the next candidate
            logger.info("Geocoding %r failed: %r", name, error)
            continue
        if place is not None:
            break
    if place is None:
        return [], [f"The weather geocoder did not recognize {candidates[0][0]!r}, so no forecast was fetched."]

    today = clock.now().date()
    dates = forecast_dates(triage.window, today)
    if dates is None:
        return [], [
            f"The question's dates fall outside Open-Meteo's forecast range (today to {MAX_FORECAST_DAYS} days ahead), "
            "so no weather forecast was fetched."
        ]
    notes: list[str] = []
    if triage.window is not None and triage.window.end.date() > dates[1]:
        notes.append(f"The weather forecast covers only the first {MAX_FORECAST_DAYS} days of the question's window.")

    wording = f"{question_text} {resolution_criteria or ''}"
    params: dict[str, Any] = {
        "latitude": place["latitude"],
        "longitude": place["longitude"],
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": "auto",
        "start_date": dates[0].isoformat(),
        "end_date": dates[1].isoformat(),
    }
    if _FAHRENHEIT.search(wording):
        params["temperature_unit"] = "fahrenheit"
        params["wind_speed_unit"] = "mph"
    if _INCHES.search(wording):
        params["precipitation_unit"] = "inch"
    data = await get_json(client, FORECAST_URL, params=params, attempts=2)

    label = ", ".join(str(part) for part in (place.get("name"), place.get("admin1"), place.get("country")) if part)
    item = EvidenceItem(
        source="weather",
        title=f"Weather model forecast for {label}",
        text=render_forecast(data),
        url=SOURCE_URL,
        published_at=clock.now(),
    )
    return [item], notes
