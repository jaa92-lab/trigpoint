"""Resolution-source pages: the URLs a question names as how it will be resolved.

Live runs fetch the page directly, and fall back to the Internet Archive's
latest copy when the site refuses the bot. Pastcast runs fetch the latest
capture taken at or before the pinned moment, and skip the page if there is
none, so no later version of a page can leak into an old forecast.

Long pages and PDFs are cut to the passages that mention the question's terms,
because the figure a question resolves on is rarely at the top of a report.
Official-count questions can also get an earlier copy of the first page, so the
analyst sees the trend and not just the latest value.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from datetime import datetime, timedelta
from html.parser import HTMLParser
from urllib.parse import urlparse

import httpx

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.http import get_json

logger = logging.getLogger(__name__)

WAYBACK_CDX = "https://web.archive.org/cdx/search/cdx"
ARCHIVE_TIMEOUT_SECONDS = 60.0
FALLBACK_TIMEOUT_SECONDS = 40.0
COMPARISON_TIMEOUT_SECONDS = 45.0
EXCERPT_CHARS = 3200
HEADER_LINES = 3
CONTEXT_LINES = 2
MAX_PDF_PAGES = 20
# Links to other Metaculus questions: they refuse the bot, and they are not resolution data.
SKIP_DOMAINS = ("metaculus.com",)
_SKIP_TAGS = {"script", "style", "noscript", "svg", "nav", "footer", "header", "form", "iframe"}
_BLOCK_TAGS = {
    "p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
    "section", "article", "pre", "td",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{3,}|\d[\d,.]*\d")
_STOPWORDS = frozenset(
    {
        "will", "what", "when", "where", "which", "while", "with", "without", "that", "this", "these", "those",
        "there", "their", "they", "them", "then", "than", "have", "been", "being", "from", "into", "onto",
        "about", "above", "below", "after", "before", "between", "during", "under", "over", "more", "less",
        "least", "most", "many", "much", "only", "also", "such", "each", "other", "some", "same", "does",
        "would", "should", "could", "resolve", "resolves", "resolved", "resolution", "question", "questions",
        "according", "source", "sources", "official", "reported", "report", "total", "number", "value",
        "yes", "no", "any", "all", "the", "and", "for", "not", "per", "day", "days", "week", "weeks",
    }
)


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


def _clean_lines(raw: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> tuple[str, str]:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return parser.title.strip(), _clean_lines("".join(parser.parts))


def pdf_to_text(data: bytes, max_pages: int = MAX_PDF_PAGES) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # one unreadable page should not lose the rest
            continue
    return _clean_lines("\n".join(parts))


def _normalize(text: str) -> str:
    return text.lower().replace(",", "")


def focus_terms(text: str) -> set[str]:
    """The distinctive words and numbers of a question, for finding the passages that matter."""
    terms = set()
    for token in _WORD.findall(text or ""):
        term = _normalize(token).rstrip(".")
        if term.endswith("'s"):
            term = term[:-2]
        if len(term) >= 2 and term not in _STOPWORDS:
            terms.add(term)
    return terms


def excerpt(text: str, focus: str, max_chars: int = EXCERPT_CHARS) -> str:
    """Cut a long page to its opening lines plus the passages that mention the question's terms, in page order."""
    if len(text) <= max_chars:
        return text
    terms = focus_terms(focus)
    lines = text.splitlines()
    scores = [sum(1 for term in terms if term in _normalize(line)) for line in lines]
    keep = set(range(min(HEADER_LINES, len(lines))))
    used = sum(len(lines[i]) + 1 for i in keep)
    for index in sorted(range(len(lines)), key=lambda i: (-scores[i], i)):
        if scores[index] == 0:
            break
        window = [
            j for j in range(max(0, index - CONTEXT_LINES), min(len(lines), index + CONTEXT_LINES + 1)) if j not in keep
        ]
        cost = sum(len(lines[j]) + 1 for j in window)
        if used + cost <= max_chars:
            keep.update(window)
            used += cost
    if len(keep) <= HEADER_LINES:
        return text[:max_chars] + f"\n[Page truncated: {len(text) - max_chars} more characters not shown.]"
    parts: list[str] = []
    previous = -1
    for j in sorted(keep):
        if j != previous + 1:
            parts.append("[...]")
        parts.append(lines[j])
        previous = j
    if previous != len(lines) - 1:
        parts.append("[...]")
    return (
        "\n".join(parts)
        + f"\n[Showing the opening lines and the passages that mention the question's terms; "
        f"{len(text) - used} other characters not shown.]"
    )


def _stamp(capture_url: str) -> str:
    stamp = capture_url.split("/web/", 1)[1][:8]
    return f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"


def _brief(error: Exception) -> str:
    if isinstance(error, httpx.HTTPStatusError):
        return f"HTTP {error.response.status_code}"
    return type(error).__name__


def _on_domain(url: str, domain: str) -> bool:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host == domain or host.endswith("." + domain)


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
    path = urlparse(url).path.lower()
    if "pdf" in content_type or path.endswith(".pdf"):
        return path.rsplit("/", 1)[-1] or url, pdf_to_text(response.content)
    if "html" in content_type or "xml" in content_type or not content_type:
        title, text = html_to_text(response.text)
        return title or url, text
    return url, response.text


async def _archived_copy(client: httpx.AsyncClient, url: str, as_of: datetime) -> tuple[str, str, str] | None:
    capture = await wayback_capture_url(client, url, as_of)
    if capture is None:
        return None
    title, text = await fetch_page(client, capture, timeout=ARCHIVE_TIMEOUT_SECONDS)
    return title, text, _stamp(capture)


async def _read_live(client: httpx.AsyncClient, url: str, now: datetime) -> tuple[str, str, str]:
    try:
        title, text = await fetch_page(client, url)
        return title, text, "fetched live"
    except Exception as live_error:
        try:
            archived = await asyncio.wait_for(_archived_copy(client, url, now), FALLBACK_TIMEOUT_SECONDS)
        except Exception:
            archived = None
        if archived is None:
            raise live_error
        title, text, stamp = archived
        return title, text, f"archived copy from {stamp}; the live page returned {_brief(live_error)}"


async def _earlier_copy(client: httpx.AsyncClient, url: str, as_of: datetime, focus: str) -> EvidenceItem | None:
    archived = await _archived_copy(client, url, as_of)
    if archived is None or not archived[1].strip():
        return None
    title, text, stamp = archived
    return EvidenceItem(
        source="sources",
        title=f"{title} (earlier archived copy from {stamp}, for comparison)",
        text=excerpt(text, focus),
        url=url,
    )


async def fetch_sources(
    client: httpx.AsyncClient,
    urls: tuple[str, ...],
    clock: Clock,
    max_urls: int = 3,
    focus: str = "",
    compare_days: int | None = None,
) -> tuple[list[EvidenceItem], list[str]]:
    items: list[EvidenceItem] = []
    notes: list[str] = []
    readable = []
    for url in urls:
        if any(_on_domain(url, domain) for domain in SKIP_DOMAINS):
            notes.append(f"Skipped {url}: a Metaculus page, not a resolution source.")
        else:
            readable.append(url)

    fetched: list[str] = []
    for url in readable[:max_urls]:
        try:
            if clock.is_live:
                title, text, label = await _read_live(client, url, clock.now())
            else:
                archived = await _archived_copy(client, url, clock.now())
                if archived is None:
                    notes.append(f"No archived copy of {url} exists from before the forecast date.")
                    continue
                title, text, stamp = archived
                label = f"archived copy from {stamp}"
            if not text.strip():
                notes.append(f"{url} returned no readable text.")
                continue
            items.append(EvidenceItem(source="sources", title=f"{title} ({label})", text=excerpt(text, focus), url=url))
            fetched.append(url)
        except Exception as error:  # a broken source page must never sink the forecast
            notes.append(f"Could not read {url}: {error!r}")

    if compare_days and fetched:
        earlier = clock.now() - timedelta(days=compare_days)
        try:
            comparison = await asyncio.wait_for(
                _earlier_copy(client, fetched[0], earlier, focus), COMPARISON_TIMEOUT_SECONDS
            )
        except Exception as error:
            comparison = None
            logger.info("Earlier copy of %s failed: %r", fetched[0], error)
        if comparison is None:
            notes.append(f"No archived copy of {fetched[0]} from {compare_days} or more days earlier, so no trend comparison.")
        else:
            items.append(comparison)
    return items, notes
