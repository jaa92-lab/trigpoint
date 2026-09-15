from datetime import datetime, timezone

import pytest

from forecaster.evidence.base import PriceSnapshot
from forecaster.evidence.prices import (
    coingecko_points,
    derive_price_prior,
    price_question_shape,
    series_stats,
    yahoo_points,
)
from forecaster.quant import prob_above_at
from forecaster.triage import triage

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
T = AS_OF.timestamp()
DAY = 86400.0
ZEC_TITLE = "Will Zcash (ZEC) trade above $1,100 on September 17 or 18, 2026?"


def snapshot(price: float = 1000.0, sigma: float | None = 0.04) -> PriceSnapshot:
    return PriceSnapshot("ZEC", "CoinGecko", AS_OF, price, sigma)


def test_yahoo_payload_parsing_skips_missing_closes():
    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"currency": "HKD"},
                    "timestamp": [1, 2, 3],
                    "indicators": {"quote": [{"close": [2.9, None, 3.0]}]},
                }
            ],
            "error": None,
        }
    }
    points, currency = yahoo_points(payload)
    assert points == [(1.0, 2.9), (3.0, 3.0)]
    assert currency == "HKD"


def test_yahoo_error_payload_raises():
    with pytest.raises(ValueError):
        yahoo_points({"chart": {"result": None, "error": {"code": "Not Found"}}})


def test_coingecko_parsing_converts_milliseconds():
    assert coingecko_points({"prices": [[1789174800000, 1155.25]]}) == [(1789174800.0, 1155.25)]


def test_series_stats_ignore_points_after_the_pinned_moment():
    points = [(T - 8 * DAY, 100.0), (T - DAY - 60, 110.0), (T - 60, 121.0), (T + 3600, 999.0)]
    last, change_1d, change_7d = series_stats(points, AS_OF)
    assert last == 121.0
    assert change_1d == pytest.approx(0.1)
    assert change_7d == pytest.approx(0.21)
    assert series_stats([(T + 10, 5.0)], AS_OF) is None


@pytest.mark.parametrize(
    ("title", "direction", "style"),
    [
        (ZEC_TITLE, "above", "touch"),
        ("Will Bitcoin close above $120k on Friday?", "above", "close"),
        ("Will Tesla (TSLA) fall below $200 before October?", "below", "touch"),
        ("Will Zcash (ZEC) be a top-10 cryptocurrency by market cap on September 18, 2026?", None, "close"),
    ],
)
def test_price_question_shape(title, direction, style):
    assert price_question_shape(title) == (direction, style)


def test_binary_touch_prior_uses_the_single_threshold():
    t = triage(question_text=ZEC_TITLE, question_type="binary", now=AS_OF)
    prior = derive_price_prior(snapshot(), t, ZEC_TITLE, "binary", horizon_days=4.0)
    assert prior is not None
    assert 0.0 < prior.probability < 1.0
    assert "at any point" in prior.explanation


def test_touch_prior_is_not_certain_before_the_window_opens():
    t = triage(question_text=ZEC_TITLE, question_type="binary", now=AS_OF)
    prior = derive_price_prior(snapshot(1163.0, 0.064), t, ZEC_TITLE, "binary", horizon_days=None)
    assert 0.6 < prior.probability < 0.999
    assert "Sep 17" in prior.explanation


def test_close_style_below_prior_is_the_complement_of_above():
    title = "Will Bitcoin close below $90k on Friday?"
    t = triage(question_text=title, question_type="binary", now=AS_OF)
    below = derive_price_prior(snapshot(100_000.0, 0.03), t, title, "binary", 3.0).probability
    assert below == pytest.approx(1 - prob_above_at(100_000.0, 90_000.0, 0.03, 3.0, 1.25))


def test_numeric_prior_gives_quantiles():
    title = "What will Boyaa Interactive (0434.HK) close at on September 18, 2026?"
    t = triage(question_text=title, question_type="numeric", now=AS_OF)
    prior = derive_price_prior(snapshot(2.95, 0.03), t, title, "numeric", 4.0)
    assert prior.quantiles[0.1] < prior.quantiles[0.9]
    assert prior.probability is None


def test_no_prior_without_volatility_threshold_or_direction():
    title = "Will Zcash (ZEC) be a top-10 cryptocurrency by market cap on September 18, 2026?"
    t = triage(question_text=title, question_type="binary", now=AS_OF)
    assert derive_price_prior(snapshot(), t, title, "binary", 4.0) is None
    assert derive_price_prior(snapshot(sigma=None), t, title, "binary", 4.0) is None
    assert derive_price_prior(None, t, title, "binary", 4.0) is None
