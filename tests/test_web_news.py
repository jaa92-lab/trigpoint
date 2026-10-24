import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from forecaster.bot import default_news_searcher
from forecaster.clock import Clock
from forecaster.config import BotConfig
from forecaster.evidence import web_news
from forecaster.evidence.news import NewsSearcher
from forecaster.evidence.web_news import WebNewsSearcher, response_to_items

REPLY = {
    "choices": [
        {
            "message": {
                "content": "Oct 23 (Reuters): Texas filed its notice of appeal.",
                "annotations": [
                    {"type": "url_citation", "url_citation": {"url": "https://x/a", "title": "Texas appeals", "content": "Texas filed..."}},
                    {"type": "url_citation", "url_citation": {"url": "https://x/a", "title": "Duplicate"}},
                    {"type": "url_citation", "url_citation": {"url": "https://x/b", "title": "", "content": ""}},
                ],
            }
        }
    ]
}


def test_reply_becomes_a_digest_then_each_cited_page_once():
    items = response_to_items(REPLY, "Texas prison appeal")
    assert [i.title for i in items] == ["Web search digest: Texas prison appeal", "Texas appeals", "https://x/b"]
    assert items[0].text.startswith("Oct 23")
    assert all(i.source == "news" for i in items)


def test_an_empty_reply_gives_no_items():
    assert response_to_items({"choices": [{"message": {"content": ""}}]}, "q") == []
    assert response_to_items({}, "q") == []


def test_availability_and_pastcasts(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert not WebNewsSearcher().available
    searcher = WebNewsSearcher(api_key=" k ")
    assert searcher.available
    # Web search cannot be pinned to the past, so a pastcast gets nothing rather than a leak.
    pinned = Clock.pinned(datetime(2026, 9, 7, tzinfo=timezone.utc))
    assert asyncio.run(searcher.search("anything", pinned)) == []


def test_search_sends_the_web_plugin_and_parses_the_reply(monkeypatch):
    sent = {}

    def handler(request):
        sent["auth"] = request.headers["Authorization"]
        sent["body"] = json.loads(request.content)
        return httpx.Response(200, json=REPLY)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(web_news.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(handler)))
    searcher = WebNewsSearcher(api_key="k", model="openrouter/openai/some-model")
    items = asyncio.run(searcher.search("Texas prison appeal", Clock.live(), broad=False))
    assert sent["auth"] == "Bearer k"
    assert sent["body"]["model"] == "openai/some-model"
    assert sent["body"]["plugins"] == [{"id": "web", "max_results": searcher.narrow_max_results}]
    assert len(items) == 3


def test_a_failed_search_raises(monkeypatch):
    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        web_news.httpx, "AsyncClient", lambda **kw: real_client(transport=httpx.MockTransport(lambda r: httpx.Response(402, text="no credit")))
    )
    with pytest.raises(RuntimeError, match="HTTP 402"):
        asyncio.run(WebNewsSearcher(api_key="k").search("q", Clock.live()))


def test_the_config_picks_the_news_source():
    assert isinstance(default_news_searcher(BotConfig()), WebNewsSearcher)
    assert isinstance(default_news_searcher(BotConfig(news_source="asknews")), NewsSearcher)
