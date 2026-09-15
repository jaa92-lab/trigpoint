from datetime import datetime, timezone
from types import SimpleNamespace

from forecaster.clock import Clock
from forecaster.evidence.base import EvidenceItem
from forecaster.evidence.cache import DiskCache
from forecaster.evidence.news import NewsSearcher, articles_to_items

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
