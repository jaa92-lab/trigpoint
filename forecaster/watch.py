"""Keep checking for new questions inside one run.

GitHub's scheduler is unreliable: on 2026-09-21 a cron set to every 20 minutes
actually fired every 2.5 hours, and the bot forecast 7 of 27 MiniBench
questions. Questions are open about 3 hours, so a run now stays alive and polls
instead of exiting after one pass, and the workflow starts its successor.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

# A pass returns its exit code and whether watching should continue.
Pass = Callable[[], Awaitable[tuple[int, bool]]]


async def watch_loop(
    run_pass: Pass,
    watch_seconds: float,
    interval_seconds: float,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> tuple[int, bool]:
    """Run passes until the window is too short for another. Returns the last pass's result.

    A pass that raises is logged and retried at the next interval: a network blip
    must not end the watch. The final pass may start just before the deadline and
    run well past it, so give the surrounding job plenty of headroom.
    """
    deadline = monotonic() + watch_seconds
    code, keep_going = 0, True
    while True:
        try:
            code, keep_going = await run_pass()
        except Exception:
            logger.exception("Forecast pass failed; still watching")
            code, keep_going = 1, True
        if not keep_going:
            logger.info("Stopping the watch early.")
            return code, False
        if monotonic() + interval_seconds >= deadline:
            logger.info("Watch window finished.")
            return code, True
        await sleep(interval_seconds)
