"""Prediction-market prices for related questions, from Manifold and Polymarket.

Live runs only. Neither site offers a clean way to read what a market showed at
a past moment, so pastcasting skips this source and says so in the evidence.

Both search endpoints do poorly with long queries, so the bot sends two short
ones (proper nouns first) and keeps only markets that share at least two
meaningful words with the question.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone

import httpx

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.http import get_json

logger = logging.getLogger(__name__)

MANIFOLD_SEARCH = "https://api.manifold.markets/v0/search-markets"
POLYMARKET_SEARCH = "https://gamma-api.polymarket.com/public-search"
MIN_RELEVANCE = 0.4

_STOPWORDS = frozenset(
    """a an the and or of in on at by for to from with will be is are was were have has had
    than more less before after between during as its it this that which what who whom whose
    new any least most how many much does do did not no yes if when where there their they
    january february march april may june july august september october november december
    jan feb mar apr jun jul aug sep sept oct nov dec""".split()
)
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'.&-]*")


def keywords(title: str, max_terms: int = 6) -> list[str]:
    words = []
    for raw in _WORD.findall(title):
        word = raw.strip(".'&-")
        if len(word) < 2 or word.lower() in _STOPWORDS:
            continue
        words.append(word)
    return list(dict.fromkeys(words))[:max_terms]


def search_queries(title: str) -> list[str]:
    words = keywords(title, max_terms=12)
    proper = [w for w in words if w[0].isupper()]
    queries = []
    if proper:
        queries.append(" ".join(proper[:3]))
    queries.append(" ".join(words[:3]))
    return list(dict.fromkeys(q for q in queries if q))


def _normalized_set(text: str) -> set[str]:
    return {w.lower().rstrip("s") for w in keywords(text, max_terms=50)}


def relevance(title_a: str, title_b: str) -> float:
    """Share of the shorter title's words found in the other; zero below two shared words."""
    a, b = _normalized_set(title_a), _normalized_set(title_b)
    shared = len(a & b)
    if shared < 2:
        return 0.0
    return shared / min(len(a), len(b))


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except ValueError:
            return []
    return []


def manifold_items(payload: list[dict]) -> list[EvidenceItem]:
    items = []
    for market in payload or []:
        probability = market.get("probability")
        if market.get("outcomeType") != "BINARY" or probability is None or market.get("isResolved"):
            continue
        close = market.get("closeTime")
        close_text = (
            f", closes {datetime.fromtimestamp(close / 1000, tz=timezone.utc):%Y-%m-%d}" if close else ""
        )
        volume = market.get("volume") or 0
        items.append(
            EvidenceItem(
                source="markets",
                title=market.get("question") or "Manifold market",
                text=(
                    f"Manifold (play money) shows {float(probability):.0%} Yes. "
                    f"Trading volume M${float(volume):,.0f}{close_text}."
                ),
                url=market.get("url"),
            )
        )
    return items


def polymarket_items(payload: dict) -> list[EvidenceItem]:
    items = []
    for event in (payload or {}).get("events") or []:
        for market in event.get("markets") or []:
            if market.get("closed"):
                continue
            outcomes = _as_list(market.get("outcomes"))
            prices = _as_list(market.get("outcomePrices"))
            if not outcomes or len(outcomes) != len(prices):
                continue
            try:
                pairs = ", ".join(f"{o} {float(p):.0%}" for o, p in zip(outcomes, prices))
            except (TypeError, ValueError):
                continue
            volume = market.get("volumeNum") or market.get("volume") or 0
            try:
                volume_text = f"${float(volume):,.0f}"
            except (TypeError, ValueError):
                volume_text = "unknown"
            slug = event.get("slug")
            items.append(
                EvidenceItem(
                    source="markets",
                    title=market.get("question") or event.get("title") or "Polymarket market",
                    text=f"Polymarket (real money) prices: {pairs}. Trading volume {volume_text}.",
                    url=f"https://polymarket.com/event/{slug}" if slug else None,
                )
            )
    return items


async def fetch_markets(
    client: httpx.AsyncClient, title: str, clock: Clock, max_items: int = 6
) -> tuple[list[EvidenceItem], list[str]]:
    if not clock.is_live:
        return [], [
            "Prediction markets were not consulted: their past prices are not available for pastcasting."
        ]
    queries = search_queries(title)
    if not queries:
        return [], []
    calls = []
    for query in queries:
        calls.append(("Manifold", get_json(client, MANIFOLD_SEARCH, params={"term": query, "limit": 10}), manifold_items))
        calls.append(("Polymarket", get_json(client, POLYMARKET_SEARCH, params={"q": query, "limit_per_type": 5}), polymarket_items))
    results = await asyncio.gather(*(call[1] for call in calls), return_exceptions=True)

    notes: list[str] = []
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    for (name, _, parser), result in zip(calls, results):
        if isinstance(result, BaseException):
            notes.append(f"{name} search failed: {result}")
            continue
        for item in parser(result):
            key = item.url or item.title
            if key in seen:
                continue
            seen.add(key)
            items.append(item)

    scored = sorted(((relevance(title, item.title), item) for item in items), key=lambda pair: pair[0], reverse=True)
    kept = [item for score, item in scored if score >= MIN_RELEVANCE]
    return kept[:max_items], list(dict.fromkeys(notes))
