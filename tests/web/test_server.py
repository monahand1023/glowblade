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
    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        for stage in ("extract", "track", "glow", "audio", "mux"):
            progress_cb(stage, 100, "done")
        with open(output_path, "wb") as f:
            f.write(b"fake final video bytes")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", fake_run_pipeline_multi)

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    points_resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 1]], "color": "red", "intensity": 0.35}]},
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

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 0]]}]},
    )

    assert resp.status_code == 400


@pytest.mark.parametrize("intensity", [-1.0, 1.5, 200.0])
def test_points_rejected_with_out_of_range_intensity(client, tiny_video_bytes, intensity):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 1]], "intensity": intensity}]},
    )

    assert resp.status_code == 400


def test_points_rejected_when_sam2_checkpoint_missing(client, tiny_video_bytes, tmp_path):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    (tmp_path / "checkpoints" / "sam2.1_hiera_small.pt").unlink()

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 1]]}]},
    )

    assert resp.status_code == 400
    assert "lightsaber-fx setup" in resp.json()["detail"]


def test_second_upload_returns_409_while_a_job_is_running(client, tiny_video_bytes, monkeypatch):
    release = threading.Event()

    def slow_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        release.wait(timeout=2)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", slow_run_pipeline_multi)

    up1 = client.post("/api/upload", files={"file": ("a.mp4", tiny_video_bytes, "video/mp4")})
    job1 = up1.json()["job_id"]
    client.post(f"/api/jobs/{job1}/points", json={"sabers": [{"points": [[1, 1, 1]]}]})

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

    def slow_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        release.wait(timeout=2)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", slow_run_pipeline_multi)

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    first = client.post(f"/api/jobs/{job_id}/points", json={"sabers": [{"points": [[1, 1, 1]]}]})
    assert first.status_code == 200

    # The job is now running (blocked on `release`). A second points submission
    # for the same job_id passes the upload-level `is_busy()` guard entirely
    # (it never touches /api/upload) and must instead be rejected by
    # `manager.start()` raising RuntimeError, which `submit_points` translates
    # into a 409.
    second = client.post(f"/api/jobs/{job_id}/points", json={"sabers": [{"points": [[2, 2, 1]]}]})
    assert second.status_code == 409

    release.set()
    server_module.manager.wait(timeout=2)


# ---------------------------------------------------------------------------
# rerender: change color/intensity/voice/blade_extend on an existing job
# without re-uploading, re-tracking, or inventing a second job-manager path.
# ---------------------------------------------------------------------------


def _fake_run_pipeline_multi_writing(content: bytes):
    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        for stage in ("extract", "track", "glow", "audio", "mux"):
            progress_cb(stage, 100, "done")
        with open(output_path, "wb") as f:
            f.write(content)
        return output_path
    return fake_run_pipeline_multi


def test_rerender_endpoint_accepts_multiple_sabers(client, tiny_video_bytes, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"first"))

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]
    points_resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [
            {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[20, 20, 1]], "color": "blue", "intensity": 0.35, "voice": "neutral"},
        ]},
    )
    assert points_resp.status_code == 200
    server_module.manager.wait(timeout=2)

    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": [0, 1]})())

    captured = {}

    def fake_rerender_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"rerendered")
        return output_path

    monkeypatch.setattr(server_module, "rerender_pipeline_multi", fake_rerender_pipeline_multi)

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [
            {"color": "green", "intensity": 0.6, "voice": "jedi"},
            {"color": "red", "intensity": 0.2, "voice": "sith"},
        ]},
    )

    assert resp.status_code == 200
    server_module.manager.wait(timeout=2)
    assert len(captured["sabers"]) == 2
    assert captured["sabers"][0]["voice"] == "jedi"


def test_rerender_endpoint_rejects_a_saber_count_mismatch(client, tiny_video_bytes, monkeypatch):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"x"))
    client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[1, 1, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )
    server_module.manager.wait(timeout=2)

    # This job only has one tracked object -- posting 2 sabers to /rerender
    # must trip the object_ids-count-mismatch check itself, not the
    # pre-existing rerenderability guard (require_rerenderable is stubbed
    # out here so it can't coincidentally 400 for the wrong reason, same
    # pattern as test_rerender_endpoint_accepts_multiple_sabers above).
    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": [0]})())

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [
            {"color": "red", "intensity": 0.35, "voice": "neutral"},
            {"color": "blue", "intensity": 0.35, "voice": "neutral"},
        ]},
    )

    assert resp.status_code == 400
    assert "tracked object" in resp.json()["detail"]


def test_rerender_endpoint_starts_job_and_produces_new_result(client, tiny_video_bytes, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"first render bytes"))

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    points_resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )
    assert points_resp.status_code == 200
    server_module.manager.wait(timeout=2)

    first_result = client.get(f"/api/jobs/{job_id}/result")
    assert first_result.content == b"first render bytes"

    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": [0]})())

    def fake_rerender_pipeline_multi(*, output_path, progress_cb, **kwargs):
        for stage in ("extract", "glow", "audio", "mux"):
            progress_cb(stage, 100, "done")
        with open(output_path, "wb") as f:
            f.write(b"rerendered bytes")
        return output_path

    monkeypatch.setattr(server_module, "rerender_pipeline_multi", fake_rerender_pipeline_multi)

    rerender_resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"color": "blue", "intensity": 0.6, "voice": "neutral"}]},
    )
    assert rerender_resp.status_code == 200
    server_module.manager.wait(timeout=2)

    result_resp = client.get(f"/api/jobs/{job_id}/result")
    assert result_resp.status_code == 200
    assert result_resp.content == b"rerendered bytes"


def test_rerender_endpoint_explains_that_a_legacy_job_predates_multi_saber(
    client, tiny_video_bytes, monkeypatch
):
    # A job with no recorded object_ids isn't a saber-count mismatch at all --
    # it predates multi-saber support (or came from the CLI). Reporting it as
    # "This job has 1 tracked object(s)" sent the user off to change the saber
    # count, which can never fix it.
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": None})())

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"color": "blue", "intensity": 0.35, "voice": "neutral"}]},
    )

    assert resp.status_code == 400
    detail = resp.json()["detail"]
    assert "no recorded object_ids" in detail
    assert "Re-upload it through the web app" in detail
    assert "tracked object(s)" not in detail  # not the count-mismatch message


def test_rerender_endpoint_404_for_unknown_job(client):
    resp = client.post("/api/jobs/doesnotexist/rerender", json={"sabers": []})
    assert resp.status_code == 404


def test_rerender_endpoint_400_when_job_has_no_masks_yet(client, tiny_video_bytes):
    # Upload only -- /points was never called, so there's no masks/,
    # motion/, or video_meta.txt for this job yet.
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"color": "blue", "intensity": 0.35, "voice": "neutral"}]},
    )

    assert resp.status_code == 400
    assert "masks" in resp.json()["detail"]


@pytest.mark.parametrize("intensity", [-1.0, 1.5, 200.0])
def test_rerender_endpoint_rejects_out_of_range_intensity(client, tiny_video_bytes, intensity):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"intensity": intensity}]},
    )

    assert resp.status_code == 400


def test_rerender_endpoint_rejects_invalid_voice(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"voice": "yoda"}]},
    )

    assert resp.status_code == 400


def test_rerender_endpoint_409_when_a_job_is_already_running(client, tiny_video_bytes, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"first"))

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]
    client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )
    server_module.manager.wait(timeout=2)

    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": [0]})())

    release = threading.Event()

    def slow_rerender_pipeline_multi(*, output_path, progress_cb, **kwargs):
        release.wait(timeout=2)
        with open(output_path, "wb") as f:
            f.write(b"slow")
        return output_path

    monkeypatch.setattr(server_module, "rerender_pipeline_multi", slow_rerender_pipeline_multi)

    first = client.post(f"/api/jobs/{job_id}/rerender", json={"sabers": [{"color": "blue", "intensity": 0.35, "voice": "neutral"}]})
    assert first.status_code == 200

    second = client.post(f"/api/jobs/{job_id}/rerender", json={"sabers": [{"color": "green", "intensity": 0.35, "voice": "neutral"}]})
    assert second.status_code == 409

    release.set()
    server_module.manager.wait(timeout=2)


# --------------------------------------------------------------------------
# Automatic detection
# --------------------------------------------------------------------------

def _fake_proposal(frame_index=3):  # inside the 5-frame fixture clip
    import numpy as np

    from lightsaber_fx.pipeline.detect import BladeProposal, MotionSeed

    mask = np.zeros((48, 64), dtype=bool)
    mask[22:26, 8:56] = True
    points = [[12, 24], [32, 24], [52, 24]]
    return BladeProposal(
        frame_index=frame_index, points=points, labels=[1, 1, 1], mask=mask,
        elongation=8.4,
        seed=MotionSeed(frame_index=frame_index, point=[52, 24], speed=7.1, area=180),
    )


def _upload(client, tiny_video_bytes):
    resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    assert resp.status_code == 200
    return resp.json()["job_id"]


def test_detect_returns_a_proposal_with_its_frame_and_points(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([_fake_proposal(3)], "motion"))
    job_id = _upload(client, tiny_video_bytes)

    resp = client.post(f"/api/jobs/{job_id}/detect")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["frame_index"] == 3
    assert data["source"] == "motion"
    assert len(data["proposals"]) == 1
    proposal = data["proposals"][0]
    assert proposal["elongation"] == 8.4
    # Points come back in the same [x, y, label] shape /points takes, all
    # includes -- detection never proposes carving anything out.
    assert proposal["points"] == [[12, 24, 1], [32, 24, 1], [52, 24, 1]]


def test_detect_returns_multiple_proposals_from_the_vision_path(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(
        server_module, "_detect_proposals",
        lambda *a, **k: ([_fake_proposal(3), _fake_proposal(3)], "vlm"),
    )
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    assert data["found"] is True
    assert data["source"] == "vlm"
    assert len(data["proposals"]) == 2


def test_detect_serves_the_frame_the_points_refer_to_and_a_mask_overlay(
    client, tiny_video_bytes, monkeypatch
):
    # The frame matters as much as the points: a proposal from mid-swing is
    # meaningless drawn over frame 0, because the object has moved.
    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([_fake_proposal(3)], "motion"))
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    frame_resp = client.get(data["frame_url"])
    assert frame_resp.status_code == 200
    assert frame_resp.headers["content-type"] == "image/jpeg"

    mask_resp = client.get(data["proposals"][0]["mask_url"])
    assert mask_resp.status_code == 200
    assert mask_resp.headers["content-type"] == "image/png"


def test_detect_mask_overlay_is_transparent_outside_the_mask(
    client, tiny_video_bytes, monkeypatch
):
    # The page composites this over the frame, so anything outside the mask
    # must be fully transparent or it paints over the footage.
    import cv2
    import numpy as np

    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([_fake_proposal()], "motion"))
    job_id = _upload(client, tiny_video_bytes)
    client.post(f"/api/jobs/{job_id}/detect")

    overlay = cv2.imread(
        str(paths_module.get_jobs_dir() / job_id / "detect_mask_0.png"), cv2.IMREAD_UNCHANGED
    )
    assert overlay.shape[2] == 4, "overlay has no alpha channel"
    assert overlay[24, 32, 3] > 0, "masked pixels are transparent"
    assert overlay[5, 5, 3] == 0, "unmasked pixels are not transparent"
    assert np.count_nonzero(overlay[:, :, 3]) == 4 * 48


def test_detect_reports_not_found_without_erroring(client, tiny_video_bytes, monkeypatch):
    # "I couldn't find it" is a normal outcome, not a failure: the page falls
    # back to asking the user to click, which is what it did before detection
    # existed. Returning an error status would surface a scary message for
    # something entirely expected.
    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([], "motion"))
    job_id = _upload(client, tiny_video_bytes)

    resp = client.post(f"/api/jobs/{job_id}/detect")

    assert resp.status_code == 200
    assert resp.json() == {"found": False}


def test_detect_falls_back_to_motion_when_vlm_raises(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(
        server_module, "detect_blades_vlm",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api key")),
    )
    monkeypatch.setattr(server_module, "detect_blade", lambda *a, **k: _fake_proposal(3))
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    assert data["found"] is True
    assert data["source"] == "motion"
    assert len(data["proposals"]) == 1


def test_detect_falls_back_to_motion_when_vlm_finds_nothing(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(server_module, "detect_blades_vlm", lambda *a, **k: [])
    monkeypatch.setattr(server_module, "detect_blade", lambda *a, **k: _fake_proposal(3))
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    assert data["found"] is True
    assert data["source"] == "motion"


def test_detect_404_for_unknown_job(client):
    assert client.post("/api/jobs/deadbeef/detect").status_code == 404


def test_detect_400_when_sam2_checkpoint_missing(client, tiny_video_bytes, tmp_path):
    job_id = _upload(client, tiny_video_bytes)
    (paths_module.get_checkpoint_path()).unlink()

    resp = client.post(f"/api/jobs/{job_id}/detect")

    assert resp.status_code == 400
    assert "setup" in resp.json()["detail"]


def test_detect_routes_reject_traversal_style_job_ids(client):
    for path in ("detect", "detect-frame", "detect-mask"):
        method = client.post if path == "detect" else client.get
        resp = method(f"/api/jobs/%2E%2E/{path}")
        assert resp.status_code == 404, f"{path} accepted a traversal-style id"


def test_points_threads_each_sabers_prompt_frame_through(client, tiny_video_bytes, monkeypatch):
    # /detect reports the frame an object was easiest to find -- usually
    # mid-swing, not frame 0 -- and the page sends it back with the points.
    # Dropping it here silently applies those points to frame 0, against a
    # frame the object has already left.
    captured = {}

    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", fake_run_pipeline_multi)
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 20, 1]], "prompt_frame": 17}]},
    )
    assert resp.status_code == 200
    server_module.manager.wait(timeout=10)

    assert captured["sabers"][0]["prompt_frame"] == 17


def test_points_defaults_prompt_frame_to_zero_when_the_client_omits_it(
    client, tiny_video_bytes, monkeypatch
):
    captured = {}

    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", fake_run_pipeline_multi)
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 20, 1]]}, {"points": [[30, 40, 1]]}]},
    )
    assert resp.status_code == 200
    server_module.manager.wait(timeout=10)

    assert [s["prompt_frame"] for s in captured["sabers"]] == [0, 0]


def test_points_rejects_sabers_with_different_prompt_frames(client, tiny_video_bytes):
    # Mixing conditioning frames inside one SAM2 session breaks its memory
    # attention -- on MPS with a Metal assertion that kills the process, which
    # the JobManager cannot turn into an error event. Caught here so it is a
    # 400 on the POST rather than a dead server seconds later.
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [
            {"points": [[10, 20, 1]], "prompt_frame": 0},
            {"points": [[30, 40, 1]], "prompt_frame": 12},
        ]},
    )

    assert resp.status_code == 400
    assert "same prompt_frame" in resp.json()["detail"]


def test_points_accepts_sabers_sharing_one_non_zero_prompt_frame(client, tiny_video_bytes, monkeypatch):
    # Only the mix is refused -- a shared mid-clip frame is exactly what
    # automatic detection produces and must still go through.
    captured = {}

    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", fake_run_pipeline_multi)
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [
            {"points": [[10, 20, 1]], "prompt_frame": 12},
            {"points": [[30, 40, 1]], "prompt_frame": 12},
        ]},
    )

    assert resp.status_code == 200
    server_module.manager.wait(timeout=10)
    assert [s["prompt_frame"] for s in captured["sabers"]] == [12, 12]


def test_points_rejects_a_negative_prompt_frame(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 20, 1]], "prompt_frame": -1}]},
    )

    assert resp.status_code == 400
    assert "prompt_frame" in resp.json()["detail"]


def test_points_accepts_multiple_sabers_and_starts_a_multi_object_job(client, tiny_video_bytes, monkeypatch):
    captured = {}

    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"multi render bytes")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", fake_run_pipeline_multi)

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={
            "sabers": [
                {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"},
                {"points": [[20, 20, 1]], "color": "blue", "intensity": 0.5, "voice": "sith"},
            ],
        },
    )

    assert resp.status_code == 200
    server_module.manager.wait(timeout=2)
    assert len(captured["sabers"]) == 2
    assert captured["sabers"][0]["color"] == "red"
    assert captured["sabers"][1]["voice"] == "sith"


def test_points_rejects_zero_sabers(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(f"/api/jobs/{job_id}/points", json={"sabers": []})

    assert resp.status_code == 400


def test_points_rejects_more_than_four_sabers(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[1, 1, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}] * 5},
    )

    assert resp.status_code == 400


def test_points_rejects_a_saber_with_no_include_point(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[1, 1, 0]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )

    assert resp.status_code == 400


def test_detect_reports_not_found_when_its_frame_cannot_be_extracted(
    client, tiny_video_bytes, monkeypatch
):
    # Detection names a frame index from the same file, so this should not
    # happen -- but a container whose reported frame count disagrees with its
    # actual frames is real. Falling back to "found nothing" leaves the user
    # clicking the object, which is the pre-detection behaviour; a 500 would
    # break a page that has a working fallback.
    monkeypatch.setattr(
        server_module, "detect_blade", lambda *a, **k: _fake_proposal(9999)
    )
    job_id = _upload(client, tiny_video_bytes)

    resp = client.post(f"/api/jobs/{job_id}/detect")

    assert resp.status_code == 200
    assert resp.json() == {"found": False}


# --------------------------------------------------------------------------
# Live preview: the glow stage's own in-progress PNG sequence, so the page
# can show what the render currently looks like instead of a bare percentage.
# --------------------------------------------------------------------------


def test_preview_404s_before_the_glow_stage_has_written_any_frame(
    client, tiny_video_bytes
):
    # Right after upload there is no glow_frames/ dir at all yet -- the render
    # hasn't even started tracking, let alone reached the glow stage.
    job_id = _upload(client, tiny_video_bytes)

    resp = client.get(f"/api/jobs/{job_id}/preview")

    assert resp.status_code == 404


def test_preview_returns_the_most_recently_written_glow_frame(
    client, tiny_video_bytes
):
    job_id = _upload(client, tiny_video_bytes)
    glow_dir = paths_module.get_jobs_dir() / job_id / "glow_frames"
    glow_dir.mkdir(parents=True)
    (glow_dir / "00000.png").write_bytes(b"oldest frame")
    (glow_dir / "00001.png").write_bytes(b"newest frame")

    resp = client.get(f"/api/jobs/{job_id}/preview")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == b"newest frame"


def test_preview_rejects_traversal_style_job_ids(client):
    resp = client.get("/api/jobs/%2E%2E/preview")
    assert resp.status_code == 404
    assert resp.json()["detail"] == "Job not found"


def test_parse_saber_specs_defaults_source_to_manual():
    from lightsaber_fx.web.server import _parse_saber_specs

    parsed = _parse_saber_specs({"sabers": [
        {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"},
    ]})

    assert parsed[0]["source"] == "manual"


def test_parse_saber_specs_passes_through_a_given_source():
    from lightsaber_fx.web.server import _parse_saber_specs

    parsed = _parse_saber_specs({"sabers": [
        {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral", "source": "vlm"},
    ]})

    assert parsed[0]["source"] == "vlm"
