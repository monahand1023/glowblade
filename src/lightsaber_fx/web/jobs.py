import threading
import traceback
from dataclasses import dataclass, field


@dataclass
class JobState:
    job_id: str
    status: str = "running"
    events: list = field(default_factory=list)
    result_path: str | None = None
    error_message: str | None = None


class JobManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._current = None
        self._thread = None

    def is_busy(self):
        with self._lock:
            return self._current is not None and self._current.status == "running"

    def start(self, job_id, pipeline_fn):
        with self._lock:
            if self._current is not None and self._current.status == "running":
                raise RuntimeError("A job is already running")
            state = JobState(job_id=job_id)
            self._current = state

        def progress_cb(stage, pct, message):
            with self._lock:
                state.events.append((stage, pct, message))

        def target():
            try:
                result = pipeline_fn(progress_cb)
                with self._lock:
                    state.result_path = result
                    state.status = "done"
            except Exception as exc:
                traceback.print_exc()
                with self._lock:
                    state.status = "error"
                    state.error_message = str(exc)

        self._thread = threading.Thread(target=target, daemon=True)
        self._thread.start()
        return state

    def get(self, job_id):
        with self._lock:
            if self._current and self._current.job_id == job_id:
                return self._current
            return None

    def wait(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)
