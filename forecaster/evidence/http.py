"""Shared HTTP helpers: one client configuration, retries on rate limits."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (compatible; forecaster-bot/0.1)"


def make_client(timeout: float = 20.0) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    attempts: int = 3,
    backoff_seconds: float = 2.0,
    timeout: float | None = None,
) -> Any:
    """GET and decode JSON, retrying on 429, 5xx, timeouts, and connection errors."""
    request_timeout = timeout if timeout is not None else httpx.USE_CLIENT_DEFAULT
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = await client.get(url, params=params, timeout=request_timeout)
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"HTTP {response.status_code}")
            else:
                response.raise_for_status()
                return response.json()
        except (httpx.TransportError, ValueError) as error:
            last_error = error
        if attempt < attempts - 1:
            await asyncio.sleep(backoff_seconds * (2**attempt))
    raise RuntimeError(f"GET {url} failed after {attempts} attempts: {last_error!r}")
