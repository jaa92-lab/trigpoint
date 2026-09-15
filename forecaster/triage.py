"""Question triage: what kind of question is this, and what evidence does it need?

Deterministic and network-free on purpose. It runs first on every question, so it
has to be cheap, predictable, and testable against real tournament titles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

KNOWN_CRYPTO_SYMBOLS = frozenset(
    {
        "BTC", "ETH", "SOL", "XRP", "BNB", "ADA", "DOGE", "TRX", "TON", "AVAX",
        "DOT", "LINK", "LTC", "BCH", "XLM", "XMR", "ZEC", "SHIB", "SUI", "HYPE",
        "UNI", "NEAR", "APT", "ICP", "ETC", "PEPE", "ATOM", "HBAR",
    }
)
KNOWN_CRYPTO_NAMES = (
    "bitcoin", "ethereum", "solana", "ripple", "cardano", "dogecoin", "toncoin",
    "avalanche", "polkadot", "chainlink", "litecoin", "stellar", "monero", "zcash",
    "shiba inu", "hyperliquid", "uniswap", "aptos", "hedera",
)

_UNIT = r"(thousand|million|billion|trillion|[kmbt])?"
_TICKER_IN_PARENS = re.compile(r"\(([A-Z0-9]{1,6}(?:\.[A-Z]{1,3})?)\)")
_URL = re.compile(r"https?://[^\s)\]>\x22\x27<]+")
_MONEY = re.compile(r"\$\s?(\d[\d,]*(?:\.\d+)?)\s*" + _UNIT + r"\b", re.IGNORECASE)
_THRESHOLD_NUMBER = re.compile(
    r"\b(?:above|below|over|under|exceeds?|exceeding|at\s+least|at\s+most|greater\s+than|"
    r"less\s+than|higher\s+than|lower\s+than|more\s+than|fewer\s+than)\s+\$?\s?"
    r"(\d[\d,]*(?:\.\d+)?)\s*" + _UNIT + r"\b",
    re.IGNORECASE,
)
_CRYPTO_WORDS = re.compile(
    r"\b(crypto|cryptocurrency|cryptocurrencies|token|coin|stablecoin)\b", re.IGNORECASE
)
_PRICE_WORDS = re.compile(
    r"\b(price|prices|trade|trades|traded|trading|close|closes|closed|closing|"
    r"market\s+cap|market\s+capitalization|share\s+price|stock|shares|index|"
    r"exchange\s+rate|yield|valuation)\b",
    re.IGNORECASE,
)
_RANK_WORDS = re.compile(r"\btop[-\s]?\d+\b|\brank(?:ed|ing)?\b", re.IGNORECASE)
_WEATHER_WORDS = re.compile(
    r"\b(storm|storms|hurricane|hurricanes|tropical|cyclone|typhoon|rain|rainfall|precipitation|"
    r"temperature|temperatures|heat\s?wave|snow|snowfall|tornado|tornadoes|earthquake|wildfire|"
    r"flood|flooding)\b",
    re.IGNORECASE,
)
_SPORTS_WORDS = re.compile(
    r"\b(game|match|points|goals|scored|touchdowns?|innings|quarterback|championship|"
    r"league|nfl|nba|mlb|nhl|mls|premier\s+league|world\s+cup|grand\s+prix|ufc|playoffs?|"
    r"super\s+bowl|world\s+series|grand\s+slam)\b",
    re.IGNORECASE,
)
_NON_SPORTS_POINTS = re.compile(r"\b(basis|percentage)\s+points?\b", re.IGNORECASE)
_COUNT_WORDS = re.compile(
    r"\b(how\s+many|number\s+of|cases|deaths|cumulative|count|reported|according\s+to|report)\b",
    re.IGNORECASE,
)
_EVENT_WORDS = re.compile(
    r"\b(announc\w*|nominat\w*|sign|signs|signed|launch\w*|releas\w*|confirm\w*|vote|votes|"
    r"voted|pass|passes|passed|markup|resign\w*|appoint\w*|approv\w*|indict\w*|arrest\w*|"
    r"leave|left|visit\w*|meet|meets|met|declar\w*|ban|bans|banned|begin|begins|began)\b",
    re.IGNORECASE,
)

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}
_MONTH = (
    r"(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)\.?"
)
_DATE_RANGE = re.compile(
    rf"\b{_MONTH}\s+(\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:\s*(?:,\s*)?(?:or|and|to|through|-|–)\s*(?:{_MONTH}\s+)?(\d{{1,2}})(?:st|nd|rd|th)?)?"
    rf"(?:,?\s*(\d{{4}}))?",
    re.IGNORECASE,
)
_DEADLINE_PREFIX = re.compile(r"\b(by|before|no\s+later\s+than|until|as\s+of)\s*$", re.IGNORECASE)

_UNIT_MULTIPLIER = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "million": 1e6,
    "b": 1e9, "billion": 1e9,
    "t": 1e12, "trillion": 1e12,
}

EVIDENCE_PLAN: dict[str, tuple[str, ...]] = {
    "price_threshold": ("prices", "markets", "news", "sources"),
    "price_level": ("prices", "news", "sources"),
    "sports": ("markets", "news", "sources"),
    "weather": ("weather", "sources", "news"),
    "official_count": ("sources", "news"),
    "event": ("news", "markets", "sources"),
    "generic": ("news", "markets", "sources"),
}


@dataclass(frozen=True)
class Window:
    """The period a question is about, in UTC.

    For "by <date>" and "as of <date>" questions the window starts now, because
    anything that happens before the date counts.
    """

    start: datetime
    end: datetime
    is_deadline: bool


@dataclass(frozen=True)
class Triage:
    kind: str
    question_type: str
    asset_class: str | None
    tickers: tuple[str, ...]
    crypto_names: tuple[str, ...]
    thresholds: tuple[float, ...]
    urls: tuple[str, ...]
    horizon_days: float | None
    window: Window | None = None

    @property
    def evidence_plan(self) -> tuple[str, ...]:
        return EVIDENCE_PLAN[self.kind]

    @property
    def is_market_data(self) -> bool:
        return self.kind in ("price_threshold", "price_level")


def triage(
    *,
    question_text: str,
    question_type: str,
    now: datetime,
    resolution_criteria: str | None = "",
    fine_print: str | None = "",
    background_info: str | None = "",
    resolve_time: datetime | None = None,
) -> Triage:
    title = question_text or ""
    body = " ".join(p for p in (resolution_criteria, fine_print, background_info) if p)

    tickers = tuple(dict.fromkeys(_TICKER_IN_PARENS.findall(title)))
    lowered = title.lower()
    crypto_names = tuple(
        name for name in KNOWN_CRYPTO_NAMES if re.search(rf"\b{re.escape(name)}\b", lowered)
    )
    crypto_hit = (
        bool(crypto_names)
        or any(t in KNOWN_CRYPTO_SYMBOLS for t in tickers)
        or bool(_CRYPTO_WORDS.search(title))
    )
    has_asset = bool(tickers) or crypto_hit

    kind = _classify(title, question_type, has_asset)
    if kind == "generic" and body:
        kind = _classify(f"{title} {body}", question_type, has_asset)

    asset_class = "crypto" if crypto_hit else ("equity" if tickers else None)
    urls = tuple(dict.fromkeys(u.rstrip(".,;:") for u in _URL.findall(f"{title} {body}")))
    horizon = None
    if resolve_time is not None:
        horizon = (resolve_time - now).total_seconds() / 86400.0

    return Triage(
        kind=kind,
        question_type=question_type,
        asset_class=asset_class,
        tickers=tickers,
        crypto_names=crypto_names,
        thresholds=extract_thresholds(title),
        urls=urls,
        horizon_days=horizon,
        window=extract_window(title, now),
    )


def _classify(text: str, question_type: str, has_asset: bool) -> str:
    numeric = question_type in ("numeric", "discrete")
    if has_asset and (_PRICE_WORDS.search(text) or _RANK_WORDS.search(text)):
        return "price_level" if numeric else "price_threshold"
    if _WEATHER_WORDS.search(text):
        return "weather"
    if _SPORTS_WORDS.search(text) and not _NON_SPORTS_POINTS.search(text):
        return "sports"
    if numeric and _COUNT_WORDS.search(text):
        return "official_count"
    if _EVENT_WORDS.search(text):
        return "event"
    return "generic"


def _to_number(digits: str, unit: str | None) -> float:
    value = float(digits.replace(",", ""))
    if unit:
        value *= _UNIT_MULTIPLIER[unit.lower()]
    return value


def extract_thresholds(text: str) -> tuple[float, ...]:
    found: list[float] = []
    for match in _MONEY.finditer(text):
        found.append(_to_number(match.group(1), match.group(2)))
    for match in _THRESHOLD_NUMBER.finditer(text):
        found.append(_to_number(match.group(1), match.group(2)))
    return tuple(dict.fromkeys(found))


def _infer_year(month: int, day: int, now: datetime) -> int:
    try:
        candidate = date(now.year, month, day)
    except ValueError:
        return now.year
    if candidate < now.date() - timedelta(days=150):
        return now.year + 1
    return now.year


def extract_window(text: str, now: datetime) -> Window | None:
    dates: list[date] = []
    deadline = False
    for match in _DATE_RANGE.finditer(text):
        month1 = _MONTHS[match.group(1).lower()]
        day1 = int(match.group(2))
        month2 = _MONTHS[match.group(3).lower()] if match.group(3) else month1
        day2 = int(match.group(4)) if match.group(4) else None
        year = int(match.group(5)) if match.group(5) else _infer_year(month1, day1, now)
        try:
            dates.append(date(year, month1, day1))
            if day2 is not None:
                dates.append(date(year, month2, day2))
        except ValueError:
            continue
        if _DEADLINE_PREFIX.search(text[: match.start()]):
            deadline = True
    if not dates:
        return None
    first, last = min(dates), max(dates)
    end = datetime(last.year, last.month, last.day, 23, 59, 59, tzinfo=timezone.utc)
    start = now if deadline else datetime(first.year, first.month, first.day, tzinfo=timezone.utc)
    return Window(start=min(start, end), end=end, is_deadline=deadline)
