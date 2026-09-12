import threading

import pytest
from fastapi.testclient import TestClient

import lightsaber_fx.paths as paths_module
import lightsaber_fx.web.server as server_module


@pytest.fixture(autouse=True)
def fresh_job_manager(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "manager", server_module.JobManager())
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path))
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

    result_resp = client.get(f"/api/jobs/{job_id}/result")
    assert result_resp.status_code == 200
    assert result_resp.content == b"fake final video bytes"


def test_points_rejected_without_include_point(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(f"/api/jobs/{job_id}/points", json={"points": [[10, 10, 0]]})

    assert resp.status_code == 400


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
