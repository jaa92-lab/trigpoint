from forecaster.watch import watch_loop


class FakeClock:
    """A clock that only moves when the loop sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def test_passes_repeat_until_the_window_is_too_short():
    clock = FakeClock()
    passes = []

    async def run_pass():
        passes.append(clock.now)
        return 0, True

    code, watched = await watch_loop(run_pass, 100.0, 30.0, clock.monotonic, clock.sleep)
    # A pass starts whenever the window still has time; the one at 90s is the last.
    assert passes == [0.0, 30.0, 60.0, 90.0]
    assert clock.sleeps == [30.0, 30.0, 30.0]
    assert (code, watched) == (0, True)


async def test_a_pass_can_stop_the_watch_early():
    clock = FakeClock()
    calls = []

    async def run_pass():
        calls.append(1)
        return 3, False

    assert await watch_loop(run_pass, 600.0, 30.0, clock.monotonic, clock.sleep) == (3, False)
    assert len(calls) == 1
    assert clock.sleeps == []


async def test_a_failing_pass_is_logged_and_the_watch_continues():
    clock = FakeClock()
    attempts = []

    async def run_pass():
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise RuntimeError("network blip")
        return 0, True

    code, watched = await watch_loop(run_pass, 100.0, 40.0, clock.monotonic, clock.sleep)
    assert attempts == [0.0, 40.0, 80.0]  # the first raised, the rest ran
    assert (code, watched) == (0, True)


async def test_no_watching_means_a_single_pass():
    clock = FakeClock()
    passes = []

    async def run_pass():
        passes.append(1)
        return 0, True

    assert await watch_loop(run_pass, 0.0, 30.0, clock.monotonic, clock.sleep) == (0, True)
    assert len(passes) == 1
