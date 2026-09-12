import threading

import pytest

from lightsaber_fx.web.jobs import JobManager


def test_start_runs_pipeline_and_records_done_state():
    manager = JobManager()

    def pipeline_fn(progress_cb):
        progress_cb("extract", 50, "halfway")
        progress_cb("extract", 100, "done")
        return "/tmp/final.mp4"

    manager.start("job-1", pipeline_fn)
    manager.wait(timeout=2)

    state = manager.get("job-1")
    assert state.status == "done"
    assert state.result_path == "/tmp/final.mp4"
    assert ("extract", 50, "halfway") in state.events


def test_start_records_error_state_on_exception():
    manager = JobManager()

    def pipeline_fn(progress_cb):
        raise ValueError("tracking failed")

    manager.start("job-1", pipeline_fn)
    manager.wait(timeout=2)

    state = manager.get("job-1")
    assert state.status == "error"
    assert "tracking failed" in state.error_message


def test_start_raises_when_a_job_is_already_running():
    manager = JobManager()
    release = threading.Event()

    def slow_pipeline_fn(progress_cb):
        release.wait(timeout=2)
        return "/tmp/final.mp4"

    manager.start("job-1", slow_pipeline_fn)

    with pytest.raises(RuntimeError):
        manager.start("job-2", lambda progress_cb: "/tmp/other.mp4")

    release.set()
    manager.wait(timeout=2)


def test_get_returns_none_for_unknown_job_id():
    manager = JobManager()
    assert manager.get("does-not-exist") is None
