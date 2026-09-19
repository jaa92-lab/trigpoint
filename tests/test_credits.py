import httpx
import pytest

from forecaster.credits import KeyStatus, fetch_key_status, plan_run, spent_between

SEASONAL = 33121


def key_client(payload: dict, seen: dict) -> httpx.AsyncClient:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json=payload)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_key_status_reads_the_balance_with_a_bearer_key():
    seen = {}
    payload = {"data": {"label": "test", "limit": 100, "limit_remaining": 90.5, "usage": 9.5}}
    async with key_client(payload, seen) as client:
        status = await fetch_key_status(client, "test-key")
    assert status == KeyStatus(limit=100.0, remaining=90.5, usage=9.5)
    assert seen == {"url": "https://openrouter.ai/api/v1/key", "auth": "Bearer test-key"}


async def test_keys_without_a_limit_have_no_remaining_balance():
    async with key_client({"data": {"limit": None, "limit_remaining": None, "usage": 3}}, {}) as client:
        status = await fetch_key_status(client, "test-key")
    assert status.remaining is None
    assert status.usage == 3.0


def test_minibench_runs_first_and_is_protected_when_credit_runs_low():
    assert plan_run(80.0, "minibench", SEASONAL, 3.0, 25.0).tournaments == ("minibench", SEASONAL)
    low = plan_run(20.0, "minibench", SEASONAL, 3.0, 25.0)
    assert low.tournaments == ("minibench",)
    assert "seasonal tournament is paused" in low.reason
    empty = plan_run(2.0, "minibench", SEASONAL, 3.0, 25.0)
    assert empty.tournaments == ()
    assert "nothing runs" in empty.reason
    assert plan_run(None, "minibench", SEASONAL, 3.0, 25.0).tournaments == ("minibench", SEASONAL)


def test_spend_prefers_the_balance_because_usage_can_read_zero():
    assert spent_between(KeyStatus(100, 100, 0), KeyStatus(100, 98.85, 0)) == pytest.approx(1.15)
    assert spent_between(KeyStatus(None, None, 10), KeyStatus(None, None, 12.5)) == 2.5
    assert spent_between(KeyStatus(None, None, None), KeyStatus(None, None, None)) is None
