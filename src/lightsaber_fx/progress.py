import time


def format_duration(seconds):
    """Render a duration compactly: "42s", "2m 10s", "1h 03m". None -> "?"."""
    if seconds is None:
        return "?"
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60:02d}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60:02d}m"


class EtaTracker:
    """Turns the pipeline's (stage, pct) progress into elapsed time and an ETA.

    The estimate is deliberately per-stage rather than whole-job: tracking takes
    minutes while audio and mux take seconds, so extrapolating from overall
    progress would report badly wrong numbers early on.
    """

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._job_start = None
        self._stage = None
        self._stage_start = None
        self._last_event = None

    def update(self, stage, pct):
        """Record progress, returning (elapsed_total_seconds, eta_seconds_or_None)."""
        now = self._clock()
        if self._job_start is None:
            self._job_start = now
        if stage != self._stage:
            self._stage = stage
            # The stage really began somewhere between the previous event and
            # now, so the previous event's timestamp beats "now" as its start:
            # it lets the first event of a stage carry an ETA instead of a blank.
            self._stage_start = now if self._last_event is None else self._last_event
        self._last_event = now

        elapsed_total = now - self._job_start
        stage_elapsed = now - self._stage_start

        eta = None
        if pct > 0 and stage_elapsed > 0:
            eta = stage_elapsed * (100.0 - pct) / pct
        return elapsed_total, eta
