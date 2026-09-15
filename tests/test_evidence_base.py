from datetime import datetime, timezone

from forecaster.evidence.base import DataPrior, EvidenceBundle, EvidenceItem, PriceSnapshot
from forecaster.evidence.cache import DiskCache

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)


def test_render_includes_only_the_requested_slices():
    bundle = EvidenceBundle()
    bundle.add(
        EvidenceItem("news", "Launch slips", "NASA says no date yet.", "https://nasa.gov/a", AS_OF),
        EvidenceItem("markets", "Crew-13 by September?", "Polymarket prices: Yes 20%", "https://polymarket.com/event/x"),
    )
    text = bundle.render(("news",))
    assert "Launch slips" in text
    assert "Polymarket" not in text
    assert "(2026-09-14)" in text
    assert "<https://nasa.gov/a>" in text


def test_empty_slices_say_so():
    assert "Nothing was found." in EvidenceBundle().render(("sources",))
    assert EvidenceBundle().render(()) == "No evidence is provided for this analysis."


def test_prices_and_prior_render_only_when_requested():
    bundle = EvidenceBundle(
        prices=[PriceSnapshot("ZEC", "CoinGecko", AS_OF, 1131.28, 0.045, 0.012, -0.03, "USD")],
        prior=DataPrior(probability=0.62, explanation="Model says 62%."),
    )
    plain = bundle.render(("news",))
    assert "Price data" not in plain
    assert "Model says" not in plain
    full = bundle.render(("news",), include_prices=True, include_prior=True)
    assert "last price 1,131.28 USD" in full
    assert "24-hour change +1.2%" in full
    assert "typical daily move 4.5%" in full
    assert "Model says 62%." in full


def test_truncation_is_announced():
    bundle = EvidenceBundle()
    for i in range(30):
        bundle.add(EvidenceItem("news", f"Story {i}", "x" * 1400))
    assert bundle.render(("news",), max_chars=5000).endswith("more characters were not shown.]")
    long_item = EvidenceBundle(items=[EvidenceItem("news", "Long", "y" * 5000)])
    assert "[item truncated]" in long_item.render(("news",))


def test_urls_and_reference_price():
    bundle = EvidenceBundle(items=[EvidenceItem("sources", "NHC", "quiet", "https://nhc.noaa.gov")])
    assert bundle.urls("sources") == ["https://nhc.noaa.gov"]
    assert bundle.reference_price() is None


def test_evidence_items_round_trip_through_json():
    item = EvidenceItem("news", "T", "body", "https://x.test", AS_OF)
    assert EvidenceItem.from_json(item.to_json()) == item


def test_disk_cache_round_trip(tmp_path):
    cache = DiskCache(tmp_path)
    assert cache.get("news", "k") is None
    cache.set("news", "k", {"a": [1, 2]})
    assert cache.get("news", "k") == {"a": [1, 2]}
    assert DiskCache(None).get("news", "k") is None
