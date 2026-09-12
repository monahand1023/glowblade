import json
import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import lightsaber_fx.paths as paths_module
import lightsaber_fx.web.server as server_module


@pytest.fixture(autouse=True)
def fresh_job_manager(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "manager", server_module.JobManager())
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path))
    # submit_points now pre-flight-checks that SAM2 setup has been run (A7);
    # plant a stand-in checkpoint so existing tests still reach the pipeline.
    checkpoint = tmp_path / "checkpoints" / "sam2.1_hiera_small.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake checkpoint")
    yield


@pytest.fixture
def client():
    return TestClient(server_module.app)


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]


def test_upload_returns_job_id_and_frame0(client, tiny_video_bytes):
    resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})

    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["width"] == 64
    assert data["height"] == 48

    frame0_resp = client.get(data["frame0_url"])
    assert frame0_resp.status_code == 200
    assert frame0_resp.headers["content-type"] == "image/jpeg"


def test_points_then_events_then_result(client, tiny_video_bytes, monkeypatch, tmp_path):
    def fake_run_pipeline(*, output_path, progress_cb, **kwargs):
        for stage in ("extract", "track", "glow", "audio", "mux"):
            progress_cb(stage, 100, "done")
        with open(output_path, "wb") as f:
            f.write(b"fake final video bytes")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline", fake_run_pipeline)

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    points_resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"points": [[10, 10, 1]], "color": "red", "intensity": 0.35},
    )
    assert points_resp.status_code == 200

    server_module.manager.wait(timeout=2)

    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        body = b"".join(stream.iter_bytes())
    assert b'"stage": "done"' in body or b'"stage":"done"' in body

    # Every progress event carries timing so the UI can show elapsed/ETA.
    progress_events = [
        json.loads(line[len("data: "):])
        for line in body.decode().splitlines()
        if line.startswith("data: ") and '"pct"' in line
    ]
    assert progress_events, "expected at least one progress event"
    for event in progress_events:
        assert "elapsed" in event and event["elapsed"] >= 0
        assert "eta" in event  # may be null when there is nothing to extrapolate from

    result_resp = client.get(f"/api/jobs/{job_id}/result")
    assert result_resp.status_code == 200
    assert result_resp.content == b"fake final video bytes"


def test_points_rejected_without_include_point(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(f"/api/jobs/{job_id}/points", json={"points": [[10, 10, 0]]})

    assert resp.status_code == 400


@pytest.mark.parametrize("intensity", [-1.0, 1.5, 200.0])
def test_points_rejected_with_out_of_range_intensity(client, tiny_video_bytes, intensity):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"points": [[10, 10, 1]], "intensity": intensity},
    )

    assert resp.status_code == 400


def test_points_rejected_when_sam2_checkpoint_missing(client, tiny_video_bytes, tmp_path):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    (tmp_path / "checkpoints" / "sam2.1_hiera_small.pt").unlink()

    resp = client.post(f"/api/jobs/{job_id}/points", json={"points": [[10, 10, 1]]})

    assert resp.status_code == 400
    assert "lightsaber-fx setup" in resp.json()["detail"]


def test_second_upload_returns_409_while_a_job_is_running(client, tiny_video_bytes, monkeypatch):
    release = threading.Event()

    def slow_run_pipeline(*, output_path, progress_cb, **kwargs):
        release.wait(timeout=2)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline", slow_run_pipeline)

    up1 = client.post("/api/upload", files={"file": ("a.mp4", tiny_video_bytes, "video/mp4")})
    job1 = up1.json()["job_id"]
    client.post(f"/api/jobs/{job1}/points", json={"points": [[1, 1, 1]]})

    up2 = client.post("/api/upload", files={"file": ("b.mp4", tiny_video_bytes, "video/mp4")})

    assert up2.status_code == 409
    release.set()
    server_module.manager.wait(timeout=2)


def test_index_html_references_expected_elements(client):
    resp = client.get("/")
    html = resp.text
    for element_id in ("dropzone", "picker-canvas", "submit-button", "progress-fill", "result-player"):
        assert f'id="{element_id}"' in html


@pytest.mark.parametrize(
    "job_id",
    ["..", "../../etc/passwd", "a" * 65, "", "..."],
)
def test_validate_job_id_rejects_traversal_style_ids(job_id):
    with pytest.raises(HTTPException) as exc_info:
        server_module._validate_job_id(job_id)
    assert exc_info.value.status_code == 404


@pytest.mark.parametrize("job_id", ["abcd1234", "a" * 64, "job-id_1"])
def test_validate_job_id_accepts_normal_ids(job_id):
    server_module._validate_job_id(job_id)  # must not raise


def test_traversal_style_job_id_cannot_escape_jobs_dir(client, tmp_path):
    # get_jobs_dir() is tmp_path / "jobs" (see the fresh_job_manager fixture).
    # A job_id of ".." would, without validation, make
    # `get_jobs_dir() / job_id / "frame0.jpg"` resolve to tmp_path/"frame0.jpg" —
    # one level above the jobs directory. Plant a file there and confirm the
    # traversal id is rejected before that file is ever served.
    #
    # A *literal* ".." segment (e.g. "/api/jobs/../frame0") never reaches this
    # app: httpx/TestClient normalizes ".." out of the URL client-side before
    # dispatch, collapsing the request to "/api/frame0" and hitting
    # Starlette's router-level 404 ("Not Found") -- `get_frame0` and
    # `_validate_job_id` are never called. Percent-encoding the dots
    # (%2E%2E) survives that client-side normalization intact and is decoded
    # back to a literal ".." job_id by Starlette's router when it extracts
    # the path parameter, so this is the request shape that actually drives
    # a hostile job_id into the route and its `_validate_job_id` guard.
    # Verified empirically (see fixwave2-report.md): with the guard
    # neutered, this same request returns 200 with the planted bytes.
    outside_file = tmp_path / "frame0.jpg"
    outside_file.write_bytes(b"should never be reachable via job_id traversal")

    resp = client.get("/api/jobs/%2E%2E/frame0")

    assert resp.status_code == 404
    # Confirms this 404 came from `_validate_job_id` (detail="Job not found"),
    # not Starlette's generic router-miss 404 (detail="Not Found").
    assert resp.json()["detail"] == "Job not found"
    assert resp.content != b"should never be reachable via job_id traversal"


def test_second_points_submission_returns_409_while_job_is_running(client, tiny_video_bytes, monkeypatch):
    release = threading.Event()

    def slow_run_pipeline(*, output_path, progress_cb, **kwargs):
        release.wait(timeout=2)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline", slow_run_pipeline)

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    first = client.post(f"/api/jobs/{job_id}/points", json={"points": [[1, 1, 1]]})
    assert first.status_code == 200

    # The job is now running (blocked on `release`). A second points submission
    # for the same job_id passes the upload-level `is_busy()` guard entirely
    # (it never touches /api/upload) and must instead be rejected by
    # `manager.start()` raising RuntimeError, which `submit_points` translates
    # into a 409.
    second = client.post(f"/api/jobs/{job_id}/points", json={"points": [[2, 2, 1]]})
    assert second.status_code == 409

    release.set()
    server_module.manager.wait(timeout=2)
