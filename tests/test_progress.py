from lightsaber_fx.progress import EtaTracker, format_duration


class FakeClock:
    """Deterministic stand-in for time.monotonic."""

    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_first_event_of_a_stage_has_no_eta():
    clock = FakeClock()
    tracker = EtaTracker(clock=clock)

    elapsed, eta = tracker.update("track", 0.0)

    assert elapsed == 0.0
    assert eta is None


def test_eta_extrapolates_from_progress_within_a_stage():
    clock = FakeClock()
    tracker = EtaTracker(clock=clock)
    tracker.update("track", 0.0)

    clock.advance(10.0)
    elapsed, eta = tracker.update("track", 25.0)

    # 25% took 10s, so the remaining 75% should be about 30s.
    assert elapsed == 10.0
    assert eta == 30.0


def test_eta_is_zero_at_completion():
    clock = FakeClock()
    tracker = EtaTracker(clock=clock)
    tracker.update("track", 0.0)
    clock.advance(8.0)

    _, eta = tracker.update("track", 100.0)

    assert eta == 0.0


def test_stage_change_restarts_the_eta_estimate():
    clock = FakeClock()
    tracker = EtaTracker(clock=clock)
    tracker.update("track", 0.0)
    clock.advance(120.0)
    tracker.update("track", 100.0)

    # A new, much faster stage must not inherit tracking's slow rate.
    clock.advance(1.0)
    elapsed, eta = tracker.update("glow", 50.0)

    assert elapsed == 121.0
    assert eta == 1.0


def test_elapsed_accumulates_across_stages():
    clock = FakeClock()
    tracker = EtaTracker(clock=clock)
    tracker.update("extract", 100.0)
    clock.advance(5.0)
    tracker.update("track", 50.0)
    clock.advance(5.0)

    elapsed, _ = tracker.update("mux", 100.0)

    assert elapsed == 10.0


def test_zero_percent_after_time_has_passed_still_has_no_eta():
    clock = FakeClock()
    tracker = EtaTracker(clock=clock)
    tracker.update("track", 0.0)
    clock.advance(30.0)

    _, eta = tracker.update("track", 0.0)

    assert eta is None


def test_format_duration_seconds():
    assert format_duration(0) == "0s"
    assert format_duration(42.4) == "42s"
    assert format_duration(59.9) == "59s"


def test_format_duration_minutes():
    assert format_duration(60) == "1m 00s"
    assert format_duration(130) == "2m 10s"


def test_format_duration_hours():
    assert format_duration(3600) == "1h 00m"
    assert format_duration(3780) == "1h 03m"


def test_format_duration_handles_none_and_negative():
    assert format_duration(None) == "?"
    assert format_duration(-5) == "0s"
