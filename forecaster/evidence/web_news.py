"""News evidence from web search through OpenRouter.

AskNews's free credit ran out on 2026-10-19 and does not refill. OpenRouter's web
search plugin runs a search, hands the results to a small model, and returns the
pages it cited. It is paid from the same OpenRouter credit as the analysts, so it
needs no other account or key.

Web search cannot be pinned to a past moment, so it only runs live: a pastcast
that used it would read news from after the question closed.
"""

from __future__ import annotations

import logging
import os
from datetime import timezone
from typing import Any

import httpx

from forecaster.clock import Clock
from forecaster.config import PARSER_MODEL
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.cache import DiskCache
from forecaster.evidence.http import USER_AGENT

logger = logging.getLogger(__name__)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"

DIGEST_PROMPT = """Today is {today}. Search the web for the latest news on: {query}

Write a short digest of what the coverage says, newest first. Give the date of each development and name the outlet.
Separate what has happened from what is scheduled or expected. Say plainly if nothing recent turned up."""


def response_to_items(data: dict[str, Any], query: str) -> list[EvidenceItem]:
    """The model's dated digest first, then each page it cited."""
    message = ((data.get("choices") or [{}])[0]).get("message") or {}
    items: list[EvidenceItem] = []
    digest = (message.get("content") or "").strip()
    if digest:
        items.append(EvidenceItem(source="news", title=f"Web search digest: {query}", text=digest))
    seen: set[str] = set()
    for annotation in message.get("annotations") or []:
        citation = annotation.get("url_citation") or {}
        url = citation.get("url")
        if annotation.get("type") != "url_citation" or not url or url in seen:
            continue
        seen.add(url)
        items.append(
            EvidenceItem(
                source="news",
                title=citation.get("title") or url,
                text=(citation.get("content") or "").strip(),
                url=url,
            )
        )
    return items


class WebNewsSearcher:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = PARSER_MODEL,
        max_results: int = 5,
        narrow_max_results: int = 3,
        timeout: float = 90.0,
        cache: DiskCache | None = None,
    ) -> None:
        self.api_key = (api_key or os.getenv("OPENROUTER_API_KEY") or "").strip()
        self.model = model.removeprefix("openrouter/")
        self.max_results = max_results
        self.narrow_max_results = narrow_max_results
        self.timeout = timeout
        self.cache = cache or DiskCache(None)

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, clock: Clock, broad: bool = True) -> list[EvidenceItem]:
        if not clock.is_live:
            return []
        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": DIGEST_PROMPT.format(today=f"{clock.now().astimezone(timezone.utc):%Y-%m-%d}", query=query),
                }
            ],
            "plugins": [{"id": "web", "max_results": self.max_results if broad else self.narrow_max_results}],
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "User-Agent": USER_AGENT}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(ENDPOINT, json=body, headers=headers)
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter web search returned HTTP {response.status_code}: {response.text[:200]}")
        return response_to_items(response.json(), query)


if __name__ == "__main__":
    # A live check with the real key: uv run python -m forecaster.evidence.web_news "query"
    import asyncio
    import sys

    found = asyncio.run(WebNewsSearcher().search(" ".join(sys.argv[1:]) or "world news today", Clock.live()))
    for item in found:
        print(f"## {item.title}\n{item.url or ''}\n{item.text[:600]}\n")
    print(f"{len(found)} items")
