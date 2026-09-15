"""Resolution-source pages: the URLs a question names as how it will be resolved.

Live runs fetch the page directly. Pastcast runs fetch the Internet Archive's
latest capture taken at or before the pinned moment, and skip the page if there
is none, so no later version of a page can leak into an old forecast.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from html.parser import HTMLParser

import httpx

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.http import get_json

logger = logging.getLogger(__name__)

WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
ARCHIVE_TIMEOUT_SECONDS = 60.0
MAX_PAGE_CHARS = 6000
_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "pre", "td",
}


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
        elif self._skip_depth == 0:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.parts)
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in raw.splitlines()]
    return parser.title.strip(), "\n".join(line for line in lines if line)


def _limit(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n[Page truncated: {len(text) - max_chars} more characters not shown.]"


async def wayback_capture_url(client: httpx.AsyncClient, url: str, as_of: datetime) -> str | None:
    stamp = as_of.strftime("%Y%m%d%H%M%S")
    rows = await get_json(
        client,
        WAYBACK_CDX,
        params={
            "url": url,
            "to": stamp,
            "output": "json",
            "limit": "-1",
            "filter": "statuscode:200",
            "fl": "timestamp,original",
        },
        attempts=2,
        backoff_seconds=5.0,
        timeout=ARCHIVE_TIMEOUT_SECONDS,
    )
    if not rows or len(rows) < 2:
        return None
    timestamp, original = rows[-1][0], rows[-1][1]
    if timestamp > stamp:
        return None
    return f"https://web.archive.org/web/{timestamp}id_/{original}"


async def fetch_page(client: httpx.AsyncClient, url: str, timeout: float | None = None) -> tuple[str, str]:
    request_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
    response = await client.get(url, timeout=request_timeout)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" in content_type:
        return url, "[This source is a PDF, which the bot does not read.]"
    if "html" in content_type or "xml" in content_type or not content_type:
        title, text = html_to_text(response.text)
        return title or url, text
    return url, response.text


async def fetch_sources(
    client: httpx.AsyncClient,
    urls: tuple[str, ...],
    clock: Clock,
    max_urls: int = 3,
) -> tuple[list[EvidenceItem], list[str]]:
    items: list[EvidenceItem] = []
    notes: list[str] = []
    for url in urls[:max_urls]:
        try:
            if clock.is_live:
                title, text = await fetch_page(client, url)
                label = "fetched live"
            else:
                capture = await wayback_capture_url(client, url, clock.now())
                if capture is None:
                    notes.append(f"No archived copy of {url} exists from before the forecast date.")
                    continue
                title, text = await fetch_page(client, capture, timeout=ARCHIVE_TIMEOUT_SECONDS)
                stamp = capture.split("/web/", 1)[1][:8]
                label = f"archived copy from {stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
            if not text.strip():
                notes.append(f"{url} returned no readable text.")
                continue
            items.append(
                EvidenceItem(
                    source="sources",
                    title=f"{title} ({label})",
                    text=_limit(text, MAX_PAGE_CHARS),
                    url=url,
                )
            )
        except Exception as error:  # a broken source page must never sink the forecast
            notes.append(f"Could not read {url}: {error!r}")
    return items, notes
