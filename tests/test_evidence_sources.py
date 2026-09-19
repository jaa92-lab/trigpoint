from datetime import datetime, timezone

import httpx

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceBundle, EvidenceItem
from forecaster.evidence.sources import excerpt, fetch_sources, html_to_text, pdf_to_text, wayback_capture_url

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


def html(text: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html"}, text=text)


def minimal_pdf(text: str) -> bytes:
    """A one-page PDF holding a line of text, built by hand so the test needs no PDF writer."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def test_long_pages_keep_the_passages_that_mention_the_question():
    filler = [f"Unrelated paragraph {i} about agency staffing updates." for i in range(400)]
    text = "\n".join(["Florida Arbovirus Surveillance"] + filler[:200] + ["Hillsborough County: 115 locally acquired dengue cases"] + filler[200:])
    focus = "How many cumulative locally acquired 2026 dengue cases will Hillsborough County, Florida show?"
    short = excerpt(text, focus, max_chars=1000)
    assert short.startswith("Florida Arbovirus Surveillance")
    assert "Hillsborough County: 115 locally acquired dengue cases" in short
    assert "[...]" in short
    assert len(short) < 1300
    assert excerpt("A short page.", focus) == "A short page."


def test_pdf_text_is_extracted():
    assert "42 confirmed cases" in pdf_to_text(minimal_pdf("42 confirmed cases"))


async def test_live_pdf_sources_are_read():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/pdf"}, content=minimal_pdf("Week 27: 3.1 cases per sentinel"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_sources(client, ("https://example.go.jp/idwr/report.pdf",), Clock.live())
    assert notes == []
    assert "3.1 cases per sentinel" in items[0].text
    assert items[0].title == "report.pdf (fetched live)"


async def test_refused_live_pages_fall_back_to_the_archive():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.gov":
            return httpx.Response(403)
        if request.url.path.startswith("/cdx/"):
            return httpx.Response(200, json=[["timestamp", "original"], ["20260910120000", "https://example.gov/report"]])
        return html("<title>Report</title><p>42 cases</p>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_sources(client, ("https://example.gov/report",), Clock.live())
    assert notes == []
    assert items[0].title == "Report (archived copy from 2026-09-10; the live page returned HTTP 403)"
    assert items[0].text == "42 cases"


async def test_metaculus_links_are_skipped_without_a_request():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_sources(client, ("https://www.metaculus.com/questions/43339/",), Clock.live())
    assert items == [] and seen == []
    assert "Metaculus page" in notes[0]


async def test_official_counts_can_add_an_earlier_copy_for_the_trend():
    seen_to = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/cdx/"):
            to = request.url.params["to"]
            seen_to.append(to)
            return httpx.Response(200, json=[["timestamp", "original"], [to, "https://example.gov/report"]])
        if "/web/20260903" in request.url.path:
            return html("<title>Report</title><p>30 cases</p>")
        return html("<title>Report</title><p>42 cases</p>")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_sources(client, ("https://example.gov/report",), Clock.pinned(AS_OF), compare_days=7)
    assert seen_to == ["20260910153000", "20260903153000"]
    assert [i.title for i in items] == [
        "Report (archived copy from 2026-09-10)",
        "Report (earlier archived copy from 2026-09-03, for comparison)",
    ]
    assert items[1].text == "30 cases"
    assert notes == []


def test_source_pages_get_more_room_than_news_items():
    page = EvidenceItem("sources", "Report", "x" * 3000, "https://example.gov/r")
    story = EvidenceItem("news", "Story", "x" * 3000, "https://news.example/s")
    bundle = EvidenceBundle(items=[page, story])
    assert "[item truncated]" not in bundle.render(("sources",))
    assert "[item truncated]" in bundle.render(("news",))
