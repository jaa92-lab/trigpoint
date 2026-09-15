"""Price feeds and the statistical prior for asset-price questions.

Equities come from Yahoo's chart endpoint and crypto from CoinGecko. Neither
needs a key. Both accept an end timestamp, so pastcasting sees only prices that
existed at the pinned moment, and every point after it is dropped again locally.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from datetime import datetime, timezone

import httpx

from forecaster import quant
from forecaster.aggregate import median_of
from forecaster.clock import Clock
from forecaster.evidence.base import DataPrior, PriceSnapshot
from forecaster.evidence.http import get_json
from forecaster.triage import Triage

logger = logging.getLogger(__name__)

YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
COINGECKO = "https://api.coingecko.com/api/v3"
PERCENTILE_LEVELS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)
DAY = 86400.0

_ABOVE = re.compile(
    r"\b(above|over|exceeds?|exceeding|higher\s+than|more\s+than|at\s+least|greater\s+than|reach\w*|hits?)\b",
    re.IGNORECASE,
)
_BELOW = re.compile(
    r"\b(below|under|lower\s+than|less\s+than|at\s+most|falls?|drops?|dips?)\b", re.IGNORECASE
)
_CLOSE_STYLE = re.compile(r"\b(close|closes|closed|closing|settle\w*)\b", re.IGNORECASE)
_TOUCH_STYLE = re.compile(
    r"\b(trade|trades|traded|trading|reach\w*|hits?|touch\w*|falls?|drops?|dips?|rises?|"
    r"surpass\w*|at\s+any\s+(?:point|time))\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------ parsing


def yahoo_points(payload: dict) -> tuple[list[tuple[float, float]], str | None]:
    chart = payload.get("chart") or {}
    results = chart.get("result") or []
    if not results:
        raise ValueError(f"Yahoo returned no data: {chart.get('error')}")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators") or {}).get("quote") or [{}]
    closes = quotes[0].get("close") or []
    currency = (result.get("meta") or {}).get("currency")
    points = [(float(t), float(c)) for t, c in zip(timestamps, closes) if c is not None]
    return points, currency


def coingecko_points(payload: dict) -> list[tuple[float, float]]:
    return [(ms / 1000.0, float(price)) for ms, price in payload.get("prices") or []]


def series_stats(
    points: Sequence[tuple[float, float]], as_of: datetime
) -> tuple[float, float | None, float | None] | None:
    """Last price and 1-day / 7-day changes, using only points at or before as_of."""
    cutoff = as_of.timestamp()
    usable = sorted((t, p) for t, p in points if p > 0 and t <= cutoff)
    if not usable:
        return None
    last_t, last_p = usable[-1]

    def price_before(target: float) -> float | None:
        earlier = [p for t, p in usable if t <= target]
        return earlier[-1] if earlier else None

    day_ago = price_before(last_t - DAY)
    week_ago = price_before(last_t - 7 * DAY)
    change_1d = last_p / day_ago - 1.0 if day_ago else None
    change_7d = last_p / week_ago - 1.0 if week_ago else None
    return last_p, change_1d, change_7d


def price_question_shape(title: str) -> tuple[str | None, str]:
    """Direction ('above'/'below'/None) and style ('touch' or 'close') of a threshold question."""
    direction = "above" if _ABOVE.search(title) else ("below" if _BELOW.search(title) else None)
    style = "touch" if _TOUCH_STYLE.search(title) and not _CLOSE_STYLE.search(title) else "close"
    return direction, style


def derive_price_prior(
    snapshot: PriceSnapshot | None,
    triage_result: Triage,
    title: str,
    question_type: str,
    horizon_days: float | None,
    tail_factor: float = 1.25,
) -> DataPrior | None:
    if snapshot is None or snapshot.sigma_per_day is None:
        return None
    window = triage_result.window
    if window is not None:
        start_days = max((window.start - snapshot.as_of).total_seconds() / DAY, 0.0)
        days = max((window.end - snapshot.as_of).total_seconds() / DAY, 0.0)
        span = f"between {window.start:%b %d %H:%M} and {window.end:%b %d %H:%M} UTC"
    elif horizon_days is not None:
        start_days = 0.0
        days = max(horizon_days, 0.0)
        span = f"within the next {days:.1f} days"
    else:
        return None
    sigma = snapshot.sigma_per_day
    caveat = "It ignores news, scheduled events, and the exact wording of the resolution criteria."

    if question_type in ("numeric", "discrete"):
        q = quant.lognormal_quantiles(snapshot.last_price, sigma, days, PERCENTILE_LEVELS, tail_factor)
        return DataPrior(
            quantiles=q,
            explanation=(
                f"A driftless lognormal model using {snapshot.symbol}'s recent volatility "
                f"({sigma:.1%} per day, widened by {tail_factor}x for fat tails) puts the value "
                f"{days:.1f} days out at a median of {median_of(q):,.4g}, with a 10th percentile of "
                f"{q[0.1]:,.4g} and a 90th percentile of {q[0.9]:,.4g}. {caveat}"
            ),
        )

    if question_type != "binary" or len(triage_result.thresholds) != 1:
        return None
    direction, style = price_question_shape(title)
    if direction is None:
        return None
    strike = triage_result.thresholds[0]
    spot = snapshot.last_price
    if style == "touch":
        p = quant.prob_touch_in_window(spot, strike, sigma, start_days, days, direction, tail_factor)
        phrase = f"trades {direction} {strike:,g} at any point {span}"
    else:
        p_above = quant.prob_above_at(spot, strike, sigma, days, tail_factor)
        p = p_above if direction == "above" else 1.0 - p_above
        phrase = f"is {direction} {strike:,g} at the end of the period {span}"
    return DataPrior(
        probability=p,
        explanation=(
            f"A driftless lognormal model using {snapshot.symbol}'s recent volatility "
            f"({sigma:.1%} per day, widened by {tail_factor}x) and a last price of {spot:,.6g} "
            f"gives {p:.0%} that it {phrase}. {caveat}"
        ),
    )


# ----------------------------------------------------------- fetching


async def _coingecko_lookup(
    client: httpx.AsyncClient, triage_result: Triage
) -> tuple[str, str, int | None] | None:
    wanted_symbols = {t.upper() for t in triage_result.tickers}
    for query in list(triage_result.crypto_names) + list(triage_result.tickers):
        data = await get_json(client, f"{COINGECKO}/search", params={"query": query})
        coins = data.get("coins") or []
        matches = [
            c
            for c in coins
            if (c.get("symbol") or "").upper() in wanted_symbols
            or (c.get("name") or "").lower() in triage_result.crypto_names
        ]
        pool = matches or coins[:1]
        if pool:
            pool.sort(key=lambda c: c.get("market_cap_rank") or 10**9)
            best = pool[0]
            return best["id"], (best.get("symbol") or query).upper(), best.get("market_cap_rank")
    return None


async def _fetch_crypto(
    client: httpx.AsyncClient, triage_result: Triage, clock: Clock
) -> list[PriceSnapshot]:
    found = await _coingecko_lookup(client, triage_result)
    if found is None:
        return []
    coin_id, symbol, rank = found
    as_of = clock.now()
    end = int(as_of.timestamp())
    payload = await get_json(
        client,
        f"{COINGECKO}/coins/{coin_id}/market_chart/range",
        params={"vs_currency": "usd", "from": end - 30 * int(DAY), "to": end},
    )
    points = coingecko_points(payload)
    stats = series_stats(points, as_of)
    if stats is None:
        return []
    last, change_1d, change_7d = stats
    note = None
    if rank and clock.is_live:
        note = f"CoinGecko currently ranks it #{rank} by market capitalization."
    return [
        PriceSnapshot(
            symbol=symbol,
            provider="CoinGecko",
            as_of=as_of,
            last_price=last,
            sigma_per_day=quant.realized_sigma_per_day([(t, p) for t, p in points if t <= end]),
            change_1d=change_1d,
            change_7d=change_7d,
            currency="USD",
            note=note,
        )
    ]


async def _fetch_equity(client: httpx.AsyncClient, ticker: str, clock: Clock) -> PriceSnapshot | None:
    as_of = clock.now()
    end = int(as_of.timestamp())
    url = YAHOO_CHART.format(symbol=ticker)
    daily_payload = await get_json(
        client, url, params={"period1": end - 120 * int(DAY), "period2": end, "interval": "1d"}
    )
    hourly_payload = await get_json(
        client, url, params={"period1": end - 7 * int(DAY), "period2": end, "interval": "1h"}
    )
    daily, currency = yahoo_points(daily_payload)
    hourly, _ = yahoo_points(hourly_payload)
    stats = series_stats(daily + hourly, as_of)
    if stats is None:
        return None
    last, change_1d, change_7d = stats
    return PriceSnapshot(
        symbol=ticker,
        provider="Yahoo Finance",
        as_of=as_of,
        last_price=last,
        sigma_per_day=quant.realized_sigma_per_day([(t, p) for t, p in daily if t <= end]),
        change_1d=change_1d,
        change_7d=change_7d,
        currency=currency,
    )


async def fetch_prices(
    client: httpx.AsyncClient, triage_result: Triage, clock: Clock
) -> list[PriceSnapshot]:
    if triage_result.asset_class == "crypto":
        return await _fetch_crypto(client, triage_result, clock)
    snapshots = []
    for ticker in triage_result.tickers[:2]:
        snapshot = await _fetch_equity(client, ticker, clock)
        if snapshot is not None:
            snapshots.append(snapshot)
    return snapshots


def utc_from_unix(seconds: float) -> datetime:
    return datetime.fromtimestamp(seconds, tz=timezone.utc)
