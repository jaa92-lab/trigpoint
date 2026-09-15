from datetime import datetime, timezone

import httpx

from forecaster.clock import Clock
from forecaster.evidence.sources import fetch_sources, html_to_text, wayback_capture_url

AS_OF = datetime(2026, 9, 10, 15, 30, tzinfo=timezone.utc)


def test_html_to_text_drops_scripts_and_navigation():
    html = (
        "<html><head><title>Outlook</title><script>var x=1;</script></head><body>"
        "<nav>Menu</nav><h1>Tropical Weather Outlook</h1>"
        "<p>No new storms expected&nbsp;during the next 48 hours.</p></body></html>"
    )
    title, text = html_to_text(html)
    assert title == "Outlook"
    assert text.splitlines() == [
        "Tropical Weather Outlook",
        "No new storms expected during the next 48 hours.",
    ]


async def test_wayback_uses_the_latest_capture_before_the_pinned_moment():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["to"] = request.url.params["to"]
        return httpx.Response(
            200,
            json=[
                ["timestamp", "original"],
                ["20260901000000", "https://nhc.noaa.gov/"],
                ["20260910120000", "https://nhc.noaa.gov/"],
            ],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        url = await wayback_capture_url(client, "https://nhc.noaa.gov/", AS_OF)
    assert seen["to"] == "20260910153000"
    assert url == "https://web.archive.org/web/20260910120000id_/https://nhc.noaa.gov/"


async def test_pastcast_skips_pages_without_an_archived_copy():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_sources(client, ("https://example.gov/report",), Clock.pinned(AS_OF))
    assert items == []
    assert "No archived copy" in notes[0]


async def test_live_fetch_reads_and_labels_the_page():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text="<title>Report</title><p>42 cases</p>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_sources(client, ("https://example.gov/report",), Clock.live())
    assert notes == []
    assert items[0].title == "Report (fetched live)"
    assert items[0].text == "42 cases"
    assert items[0].url == "https://example.gov/report"


async def test_broken_pages_become_notes_not_crashes():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_sources(client, ("https://example.gov/missing",), Clock.live())
    assert items == []
    assert "Could not read" in notes[0]
