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
