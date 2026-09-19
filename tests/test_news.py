from datetime import datetime, timezone
from types import SimpleNamespace

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.cache import DiskCache
from forecaster.evidence.news import NewsSearcher, articles_to_items, merge_news, parse_queries

AS_OF = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)


def article(title, url, published, points=None):
    return SimpleNamespace(
        eng_title=title,
        title=title,
        article_url=url,
        pub_date=published,
        summary=f"Summary of {title}",
        key_points=points,
        source_id="Reuters",
    )


def test_articles_become_deduplicated_items_with_key_points():
    first = SimpleNamespace(as_dicts=[article("A", "https://x/a", AS_OF, ["point one"]), article("B", "https://x/b", AS_OF)])
    second = SimpleNamespace(as_dicts=[article("A", "https://x/a", AS_OF)])
    items = articles_to_items([first, second], Clock.live())
    assert [i.url for i in items] == ["https://x/a", "https://x/b"]
    assert items[0].title == "A (Reuters)"
    assert "- point one" in items[0].text


def test_pastcast_drops_articles_published_after_the_pinned_moment():
    later = datetime(2026, 9, 8, tzinfo=timezone.utc)
    response = SimpleNamespace(as_dicts=[article("Old", "https://x/old", AS_OF), article("Leak", "https://x/leak", later)])
    items = articles_to_items([response], Clock.pinned(AS_OF))
    assert [i.title for i in items] == ["Old (Reuters)"]


def test_availability_depends_on_credentials(monkeypatch):
    for name in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert not NewsSearcher().available
    assert NewsSearcher(api_key="k").available
    assert NewsSearcher(client_id="i", client_secret="s").available


async def test_pastcast_searches_are_served_from_the_cache(tmp_path):
    cache = DiskCache(tmp_path)
    cache.set("news", f"storm|{AS_OF.isoformat()}", [EvidenceItem("news", "Cached", "text").to_json()])
    searcher = NewsSearcher(api_key="k", cache=cache)
    items = await searcher.search("storm", Clock.pinned(AS_OF))
    assert [i.title for i in items] == ["Cached"]


class FakeAsk:
    calls = []

    def __init__(self, **kwargs) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> bool:
        return False

    @property
    def news(self):
        return self

    async def search_news(self, **kwargs):
        FakeAsk.calls.append(kwargs)
        return SimpleNamespace(as_dicts=[article(kwargs["query"], f"https://x/{len(FakeAsk.calls)}", AS_OF)])


async def test_narrow_live_searches_make_one_latest_news_call(monkeypatch):
    import asknews_sdk

    FakeAsk.calls = []
    monkeypatch.setattr(asknews_sdk, "AsyncAskNewsSDK", FakeAsk)
    searcher = NewsSearcher(api_key="k", pause_seconds=0)
    await searcher.search("broad question", Clock.live())
    await searcher.search("narrow query", Clock.live(), broad=False)
    assert [call["strategy"] for call in FakeAsk.calls] == ["latest news", "news knowledge", "latest news"]


def test_generated_queries_are_cleaned_and_capped():
    title = "Will Trump have formally nominated a permanent Secretary of the Army by September 18, 2026?"
    text = f'1. Army Secretary nominee Senate\n- "Driscoll nomination hearing"\n{title}\na third query'
    assert parse_queries(text, title, 2) == ["Army Secretary nominee Senate", "Driscoll nomination hearing"]
    assert parse_queries("", title, 2) == []


def test_merged_news_is_deduplicated_and_newest_first():
    old = EvidenceItem("news", "Old", "t", "https://x/old", datetime(2026, 9, 1, tzinfo=timezone.utc))
    new = EvidenceItem("news", "New", "t", "https://x/new", datetime(2026, 9, 6, tzinfo=timezone.utc))
    undated = EvidenceItem("news", "Undated", "t", "https://x/u", None)
    assert [i.title for i in merge_news([[old, undated], [new, old]], limit=5)] == ["New", "Old", "Undated"]
    assert len(merge_news([[old, new, undated]], limit=2)) == 2
