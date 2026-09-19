from datetime import datetime, timedelta, timezone

import pytest

from forecaster.triage import extract_thresholds, extract_window, triage

NOW = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)

# Titles copied from the September 7-25, 2026 MiniBench round.
REAL_MINIBENCH_TITLES = [
    ("Will Maykel 'Osorbo' Castillo Pérez have left Cuba by September 19, 2026?", "binary", "event"),
    ("Will Zcash (ZEC) be a top-10 cryptocurrency by market cap on September 18, 2026?", "binary", "price_threshold"),
    (
        "Will the House Energy & Commerce Committee begin a markup on the Trahan–Obernolte "
        "comprehensive AI framework (or a bill derived from the Great American AI Act) on "
        "September 17 or 18, 2026?",
        "binary",
        "event",
    ),
    ("What will Boyaa Interactive (0434.HK) close at on September 18, 2026?", "numeric", "price_level"),
    ("As of September 18, 2026, will NASA have announced a specific-day launch date for SpaceX Crew-13?", "binary", "event"),
    ("Will a new named storm form in the Atlantic basin between September 17 and September 18, 2026?", "binary", "weather"),
    (
        "How many cumulative locally acquired 2026 dengue cases will Hillsborough County, Florida "
        "show in the Florida Arbovirus Surveillance report published on or around September 18, 2026?",
        "numeric",
        "official_count",
    ),
    ("Will Trump have formally nominated a permanent Secretary of the Army by September 18, 2026?", "binary", "event"),
    ("How many combined points will be scored in the Detroit Lions at Buffalo Bills game on September 17, 2026?", "numeric", "sports"),
    ("Will Zcash (ZEC) trade above $1,100 on September 17 or 18, 2026?", "binary", "price_threshold"),
    # From earlier MiniBench rounds, summer 2026
    ("What will be the high temperature at NYC Central Park on August 20, 2026?", "numeric", "weather"),
    ("Will Central Park record measurable precipitation on August 20, 2026?", "binary", "weather"),
    # Written for the plain word "rain", which once fell through to generic
    ("How much rain will fall in New York City's Central Park between September 20 and September 22, 2026?", "numeric", "weather"),
]


@pytest.mark.parametrize(("title", "question_type", "expected_kind"), REAL_MINIBENCH_TITLES)
def test_real_minibench_titles_route_to_the_right_kind(title, question_type, expected_kind):
    assert triage(question_text=title, question_type=question_type, now=NOW).kind == expected_kind


def test_crypto_ticker_and_threshold_extraction():
    t = triage(
        question_text="Will Zcash (ZEC) trade above $1,100 on September 17 or 18, 2026?",
        question_type="binary",
        now=NOW,
    )
    assert t.tickers == ("ZEC",)
    assert t.asset_class == "crypto"
    assert t.crypto_names == ("zcash",)
    assert t.thresholds == (1100.0,)
    assert t.is_market_data
    assert t.evidence_plan[0] == "prices"


def test_equity_ticker_with_exchange_suffix():
    t = triage(
        question_text="What will Boyaa Interactive (0434.HK) close at on September 18, 2026?",
        question_type="numeric",
        now=NOW,
    )
    assert t.tickers == ("0434.HK",)
    assert t.asset_class == "equity"


def test_urls_and_horizon_use_every_field():
    t = triage(
        question_text="Will a new named storm form in the Atlantic basin between September 17 and September 18, 2026?",
        question_type="binary",
        now=NOW,
        resolution_criteria="Resolves using https://www.nhc.noaa.gov/gtwo.php.",
        resolve_time=NOW + timedelta(days=4),
    )
    assert t.urls == ("https://www.nhc.noaa.gov/gtwo.php",)
    assert t.horizon_days == pytest.approx(4.0)


def test_basis_points_are_not_sports():
    t = triage(
        question_text="Will the Fed cut rates by 50 basis points at its September meeting?",
        question_type="binary",
        now=NOW,
    )
    assert t.kind != "sports"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Will Bitcoin close above $120k on Friday?", (120000.0,)),
        ("Will revenue exceed $2.5 billion in Q3?", (2.5e9,)),
        ("Will at least 40 cases be reported?", (40.0,)),
        ("Will the storm form by September 18, 2026?", ()),
    ],
)
def test_extract_thresholds(text, expected):
    assert extract_thresholds(text) == expected


def test_window_for_either_of_two_days():
    window = triage(
        question_text="Will Zcash (ZEC) trade above $1,100 on September 17 or 18, 2026?",
        question_type="binary",
        now=NOW,
    ).window
    assert window.start == datetime(2026, 9, 17, tzinfo=timezone.utc)
    assert window.end == datetime(2026, 9, 18, 23, 59, 59, tzinfo=timezone.utc)
    assert not window.is_deadline


def test_window_between_two_full_dates():
    window = extract_window(
        "Will a new named storm form in the Atlantic basin between September 17 and September 18, 2026?", NOW
    )
    assert (window.start.day, window.end.day, window.is_deadline) == (17, 18, False)


@pytest.mark.parametrize(
    "title",
    [
        "Will Trump have formally nominated a permanent Secretary of the Army by September 18, 2026?",
        "As of September 18, 2026, will NASA have announced a specific-day launch date for SpaceX Crew-13?",
    ],
)
def test_deadline_windows_start_now(title):
    window = extract_window(title, NOW)
    assert window.is_deadline
    assert window.start == NOW
    assert window.end.day == 18


def test_titles_without_dates_have_no_window():
    assert extract_window("Will Bitcoin close above $120k on Friday?", NOW) is None


def test_missing_year_uses_the_nearest_future_year():
    window = extract_window("Will it snow on January 3?", datetime(2026, 11, 20, tzinfo=timezone.utc))
    assert window.end.year == 2027


# Price questions from summer 2026 MiniBench rounds that name the market in words.
MARKETS_IN_WORDS = [
    ("What will Oklo Inc. (NYSE: OKLO) common stock close at on July 9, 2026?", "numeric", "price_level", "OKLO", "equity"),
    ("What will the S&P 500 closing level be on Thursday, August 20, 2026?", "numeric", "price_level", "^GSPC", "market"),
    ("Will the S&P 500 close at a new all-time record high on August 20, 2026?", "binary", "price_threshold", "^GSPC", "market"),
    ("What will the KOSPI Composite Index closing level be on July 10, 2026?", "numeric", "price_level", "^KS11", "market"),
    ("What will the front-month Brent crude oil futures settlement price be on July 24, 2026?", "numeric", "price_level", "BZ=F", "market"),
    ("What will the USD/KRW exchange rate be at the close on August 20, 2026?", "numeric", "price_level", "KRW=X", "market"),
    ("What will the ECB EUR/USD reference rate be on July 23, 2026?", "numeric", "price_level", "EURUSD=X", "market"),
    ("What will the US 10-year Treasury par yield be at the close on Thursday, August 20, 2026?", "numeric", "price_level", "^TNX", "market"),
    ("What will XRP's closing price be on August 20, 2026?", "numeric", "price_level", "XRP", "crypto"),
]


@pytest.mark.parametrize(("title", "question_type", "kind", "symbol", "asset_class"), MARKETS_IN_WORDS)
def test_markets_named_in_words_route_to_price_data(title, question_type, kind, symbol, asset_class):
    t = triage(question_text=title, question_type=question_type, now=NOW)
    assert (t.kind, t.asset_class) == (kind, asset_class)
    assert symbol in t.tickers


@pytest.mark.parametrize(
    "title",
    [
        "How many gold medals will Russia win through the July 9 finals session at the 2026 European Junior Swimming Championships?",
        "Will the DOT approve the merger before September 30, 2026?",
        "Will Dax Shepard announce a new podcast before October 1, 2026?",
        "What will the French 10-year OAT–German Bund yield spread be at close on July 24, 2026?",
    ],
)
def test_ordinary_words_do_not_become_market_symbols(title):
    assert triage(question_text=title, question_type="binary", now=NOW).tickers == ()


def test_exchange_prefixed_tickers_become_yahoo_symbols():
    def tickers(title):
        return triage(question_text=title, question_type="numeric", now=NOW).tickers

    assert tickers("What will Tencent (HKEX: 700) close at on October 2, 2026?") == ("0700.HK",)
    assert tickers("What will Samsung Electronics (KRX: 005930) close at on October 2, 2026?") == ("005930.KS",)
    assert tickers("What will Shopify (TSX: SHOP) close at on October 2, 2026?") == ("SHOP.TO",)
    assert tickers("What will IonQ (Nasdaq: IONQ) close at on October 2, 2026?") == ("IONQ",)
