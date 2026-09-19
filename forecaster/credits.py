"""The OpenRouter credit balance, and which tournaments a run can afford.

Metaculus funds this bot incrementally: about $100 up front, and more only after
above-average MiniBench results. So MiniBench always runs first and is protected.
When the balance runs low, the seasonal tournament pauses, and below a floor the
bot stops instead of failing question after question.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

KEY_STATUS_URL = "https://openrouter.ai/api/v1/key"


@dataclass(frozen=True)
class KeyStatus:
    limit: float | None
    remaining: float | None
    usage: float | None


def _number(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


async def fetch_key_status(client: httpx.AsyncClient, api_key: str) -> KeyStatus:
    response = await client.get(KEY_STATUS_URL, headers={"Authorization": f"Bearer {api_key}"})
    response.raise_for_status()
    data = response.json().get("data") or {}
    return KeyStatus(
        limit=_number(data.get("limit")),
        remaining=_number(data.get("limit_remaining")),
        usage=_number(data.get("usage")),
    )


def spent_between(before: KeyStatus, after: KeyStatus) -> float | None:
    """Spend between two readings. The balance comes first: Metaculus's keys report usage as 0.

    Charges can post a little after a run ends, so this slightly undercounts.
    """
    if before.remaining is not None and after.remaining is not None:
        return before.remaining - after.remaining
    if before.usage is not None and after.usage is not None:
        return after.usage - before.usage
    return None


@dataclass(frozen=True)
class RunPlan:
    tournaments: tuple[str | int, ...]
    reason: str


def plan_run(
    remaining: float | None,
    minibench: str | int,
    seasonal: str | int,
    floor: float,
    minibench_reserve: float,
) -> RunPlan:
    if remaining is None:
        return RunPlan((minibench, seasonal), "The credit balance is unknown, so both tournaments run.")
    if remaining < floor:
        return RunPlan((), f"Only ${remaining:.2f} of credit is left, below the ${floor:.2f} floor, so nothing runs.")
    if remaining < minibench_reserve:
        return RunPlan(
            (minibench,),
            f"${remaining:.2f} of credit is left, below the ${minibench_reserve:.2f} kept for MiniBench, "
            "so the seasonal tournament is paused.",
        )
    return RunPlan((minibench, seasonal), f"${remaining:.2f} of credit is left, so both tournaments run.")
