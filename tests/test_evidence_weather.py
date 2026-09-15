from datetime import date, datetime, timezone

import httpx

from forecaster.clock import Clock
from forecaster.evidence.weather import fetch_weather, forecast_dates, place_candidates
from forecaster.triage import Window, triage

NOW = datetime(2026, 9, 15, 12, 0, tzinfo=timezone.utc)
AIRPORT_TITLE = "What will be the highest temperature recorded at Phoenix Sky Harbor International Airport on September 18, 2026?"
PHOENIX = {
    "name": "Phoenix",
    "latitude": 33.44838,
    "longitude": -112.07404,
    "country": "United States",
    "country_code": "US",
    "admin1": "Arizona",
    "timezone": "America/Phoenix",
}
FORECAST = {
    "timezone": "America/Phoenix",
    "daily_units": {
        "time": "iso8601",
        "temperature_2m_max": "°C",
        "temperature_2m_min": "°C",
        "precipitation_sum": "mm",
        "precipitation_probability_max": "%",
        "wind_gusts_10m_max": "km/h",
    },
    "daily": {
        "time": ["2026-09-18"],
        "temperature_2m_max": [37.8],
        "temperature_2m_min": [31.1],
        "precipitation_sum": [0.0],
        "precipitation_probability_max": [1],
        "wind_gusts_10m_max": [12.6],
    },
}


class LiveAt:
    """A live clock stopped at a fixed moment, so date arithmetic is testable."""

    is_live = True

    def __init__(self, moment):
        self.moment = moment

    def now(self):
        return self.moment


def weather_triage(title: str):
    return triage(question_text=title, question_type="numeric", now=NOW)


def open_meteo(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            if request.url.params["name"] == "Phoenix":
                return httpx.Response(200, json={"results": [PHOENIX]})
            return httpx.Response(200, json={"generationtime_ms": 0.1})
        if request.url.host == "api.open-meteo.com":
            return httpx.Response(200, json=FORECAST)
        return httpx.Response(404)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_place_candidates_try_the_full_name_then_shorter_prefixes():
    names = [name for name, _ in place_candidates(AIRPORT_TITLE)]
    assert names[0] == "Phoenix Sky Harbor International Airport"
    assert names[-1] == "Phoenix"


def test_place_candidates_keep_a_region_and_skip_dates_storms_and_possessives():
    assert place_candidates("Will it rain in Phoenix, Arizona on September 18?")[0] == ("Phoenix", "Arizona")
    assert place_candidates("What will the high be in September for Denver?") == [("Denver", None)]
    assert place_candidates("What will the PSI be in any region of Singapore on September 18, 2026?") == [("Singapore", None)]
    assert place_candidates("Will the NHC issue advisories for Tropical Storm Gabrielle by September 20?") == []
    assert place_candidates("How much rain will fall in New York City's Central Park?")[0] == ("New York City", None)


def test_forecast_dates_follow_the_question_window_within_range():
    today = NOW.date()
    window = Window(datetime(2026, 9, 18, tzinfo=timezone.utc), datetime(2026, 9, 18, 23, 59, tzinfo=timezone.utc), False)
    assert forecast_dates(window, today) == (date(2026, 9, 18), date(2026, 9, 18))
    assert forecast_dates(None, today) == (today, date(2026, 9, 21))
    far = Window(datetime(2026, 11, 1, tzinfo=timezone.utc), datetime(2026, 11, 2, tzinfo=timezone.utc), False)
    assert forecast_dates(far, today) is None
    long = Window(NOW, datetime(2026, 10, 30, tzinfo=timezone.utc), True)
    assert forecast_dates(long, today) == (today, date(2026, 9, 30))


async def test_live_fetch_geocodes_the_place_and_renders_the_days():
    seen = []
    async with open_meteo(seen) as client:
        items, notes = await fetch_weather(client, AIRPORT_TITLE, "", weather_triage(AIRPORT_TITLE), LiveAt(NOW))
    assert notes == []
    [item] = items
    assert item.source == "weather"
    assert item.title == "Weather model forecast for Phoenix, Arizona, United States"
    assert "- 2026-09-18: high 37.8°C, low 31.1°C, precipitation 0 mm (1% chance), gusts 12.6 km/h" in item.text
    assert item.url == "https://open-meteo.com/"
    forecast_request = seen[-1]
    assert forecast_request.url.params["start_date"] == "2026-09-18"
    assert forecast_request.url.params["end_date"] == "2026-09-18"
    assert "temperature_unit" not in forecast_request.url.params


async def test_fahrenheit_questions_request_fahrenheit():
    title = "Will the high temperature in Phoenix exceed 100°F on September 18, 2026?"
    seen = []
    async with open_meteo(seen) as client:
        await fetch_weather(client, title, "", weather_triage(title), LiveAt(NOW))
    assert seen[-1].url.params["temperature_unit"] == "fahrenheit"


async def test_pastcasting_skips_weather_without_any_request():
    seen = []
    async with open_meteo(seen) as client:
        items, notes = await fetch_weather(client, AIRPORT_TITLE, "", weather_triage(AIRPORT_TITLE), Clock.pinned(NOW))
    assert items == [] and seen == []
    assert "not available for pastcasting" in notes[0]


async def test_unknown_or_missing_places_become_notes():
    seen = []
    async with open_meteo(seen) as client:
        title = "Will it snow at Nowhereville on September 18, 2026?"
        items, notes = await fetch_weather(client, title, "", weather_triage(title), LiveAt(NOW))
        assert items == [] and "did not recognize 'Nowhereville'" in notes[0]
        title = "Will a new named storm form in the Atlantic basin by September 20, 2026?"
        items, notes = await fetch_weather(client, title, "", weather_triage(title), LiveAt(NOW))
        assert items == [] and "No place name" in notes[0]
