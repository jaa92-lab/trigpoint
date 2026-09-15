"""News evidence from AskNews.

Live runs combine the last 48 hours with a broader recent-news search. Pastcast
runs query the archive with an end timestamp at the pinned moment, drop any
article dated after it, and cache results so replays spend no further calls.
The free tier allows one call every ten seconds, so calls are serialized.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Iterable
from datetime import timedelta

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.cache import DiskCache

logger = logging.getLogger(__name__)

PASTCAST_LOOKBACK_DAYS = 30


def articles_to_items(responses: Iterable[object], clock: Clock) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    seen: set[str] = set()
    cutoff = clock.now()
    for response in responses:
        for article in getattr(response, "as_dicts", None) or []:
            url = str(getattr(article, "article_url", "") or "")
            if url and url in seen:
                continue
            published = getattr(article, "pub_date", None)
            if not clock.is_live and published is not None and published > cutoff:
                continue
            if url:
                seen.add(url)
            title = getattr(article, "eng_title", None) or getattr(article, "title", None) or "Untitled"
            source = getattr(article, "source_id", None) or getattr(article, "domain_url", None)
            summary = (getattr(article, "summary", None) or "").strip()
            points = getattr(article, "key_points", None) or []
            text = "\n".join([summary] + [f"- {point}" for point in points]).strip()
            items.append(
                EvidenceItem(
                    source="news",
                    title=f"{title} ({source})" if source else title,
                    text=text,
                    url=url or None,
                    published_at=published,
                )
            )
    return items


class NewsSearcher:
    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        api_key: str | None = None,
        cache: DiskCache | None = None,
        pause_seconds: float = 10.0,
        max_articles: int = 8,
    ) -> None:
        self.client_id = client_id or os.getenv("ASKNEWS_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("ASKNEWS_SECRET")
        self.api_key = api_key or os.getenv("ASKNEWS_API_KEY")
        self.cache = cache or DiskCache(None)
        self.pause_seconds = pause_seconds
        self.max_articles = max_articles
        self._lock = asyncio.Lock()

    @property
    def available(self) -> bool:
        return bool((self.client_id and self.client_secret) or self.api_key)

    async def search(self, query: str, clock: Clock) -> list[EvidenceItem]:
        key = f"{query}|{'live' if clock.is_live else clock.now().isoformat()}"
        if not clock.is_live:
            cached = self.cache.get("news", key)
            if cached is not None:
                return [EvidenceItem.from_json(item) for item in cached]
        items = await self._search_uncached(query, clock)
        if not clock.is_live:
            self.cache.set("news", key, [item.to_json() for item in items])
        return items

    async def _search_uncached(self, query: str, clock: Clock) -> list[EvidenceItem]:
        from asknews_sdk import AsyncAskNewsSDK

        async with self._lock:
            async with AsyncAskNewsSDK(
                client_id=self.client_id,
                client_secret=self.client_secret,
                api_key=self.api_key,
                scopes={"news"},
            ) as ask:
                if clock.is_live:
                    latest = await ask.news.search_news(
                        query=query,
                        n_articles=self.max_articles,
                        return_type="dicts",
                        strategy="latest news",
                    )
                    await asyncio.sleep(self.pause_seconds)
                    background = await ask.news.search_news(
                        query=query,
                        n_articles=self.max_articles,
                        return_type="dicts",
                        strategy="news knowledge",
                    )
                    responses = [latest, background]
                else:
                    end = clock.now()
                    start = end - timedelta(days=PASTCAST_LOOKBACK_DAYS)
                    archive = await ask.news.search_news(
                        query=query,
                        n_articles=self.max_articles * 2,
                        return_type="dicts",
                        historical=True,
                        start_timestamp=int(start.timestamp()),
                        end_timestamp=int(end.timestamp()),
                        method="both",
                    )
                    responses = [archive]
            # Keep holding the lock so the next caller waits out the rate limit.
            await asyncio.sleep(self.pause_seconds)
        return articles_to_items(responses, clock)
