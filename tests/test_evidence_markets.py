from datetime import datetime, timezone

import httpx

from forecaster.clock import Clock
from forecaster.evidence.markets import (
    fetch_markets,
    keywords,
    manifold_items,
    polymarket_items,
    relevance,
)

ARMY = "Will Trump have formally nominated a permanent Secretary of the Army by September 18, 2026?"


def test_keywords_drop_stopwords_dates_and_numbers():
    assert keywords(ARMY) == ["Trump", "formally", "nominated", "permanent", "Secretary", "Army"]


def test_relevance_prefers_matching_titles():
    question = "Will NASA announce a launch date for SpaceX Crew-13?"
    related = "SpaceX Crew-13 launch date announced by NASA?"
    unrelated = "Will Bitcoin hit $150k?"
    assert relevance(question, related) > relevance(question, unrelated)


def test_manifold_parsing_keeps_open_binary_markets():
    payload = [
        {
            "outcomeType": "BINARY",
            "probability": 0.23,
            "question": "Army secretary nominated by October?",
            "url": "https://manifold.markets/x",
            "volume": 1500,
            "closeTime": 1790000000000,
        },
        {"outcomeType": "MULTIPLE_CHOICE", "question": "Who?", "url": "https://manifold.markets/y"},
        {"outcomeType": "BINARY", "probability": 0.9, "isResolved": True, "question": "Old", "url": "https://manifold.markets/z"},
    ]
    items = manifold_items(payload)
    assert [i.title for i in items] == ["Army secretary nominated by October?"]
    assert "23% Yes" in items[0].text


def test_polymarket_parsing_handles_json_encoded_lists():
    payload = {
        "events": [
            {
                "slug": "zcash-price",
                "title": "Zcash price",
                "markets": [
                    {
                        "question": "Zcash above $1,100 on September 18?",
                        "outcomes": '["Yes", "No"]',
                        "outcomePrices": '["0.61", "0.39"]',
                        "volumeNum": 25000,
                    },
                    {"question": "Closed one", "closed": True, "outcomes": '["Yes", "No"]', "outcomePrices": '["1", "0"]'},
                    {"question": "Broken", "outcomes": '["Yes"]', "outcomePrices": "[]"},
                ],
            }
        ]
    }
    items = polymarket_items(payload)
    assert len(items) == 1
    assert "Yes 61%" in items[0].text
    assert items[0].url == "https://polymarket.com/event/zcash-price"


async def test_markets_are_skipped_when_pastcasting():
    async with httpx.AsyncClient() as client:
        items, notes = await fetch_markets(client, "Will X happen?", Clock.pinned(datetime(2026, 9, 1, tzinfo=timezone.utc)))
    assert items == []
    assert "pastcasting" in notes[0]


async def test_live_market_search_keeps_only_relevant_markets():
    def handler(request: httpx.Request) -> httpx.Response:
        if "manifold" in request.url.host:
            return httpx.Response(
                200,
                json=[
                    {"outcomeType": "BINARY", "probability": 0.3, "question": "Will Trump nominate an Army Secretary?", "url": "https://manifold.markets/a"},
                    {"outcomeType": "BINARY", "probability": 0.5, "question": "Will it rain in Paris?", "url": "https://manifold.markets/b"},
                ],
            )
        return httpx.Response(200, json={"events": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        items, notes = await fetch_markets(client, ARMY, Clock.live())
    assert [i.url for i in items] == ["https://manifold.markets/a"]
    assert notes == []
