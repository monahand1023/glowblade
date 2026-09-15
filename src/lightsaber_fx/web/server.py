import json
import re
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import paths
from ..device import select_device
from ..pipeline.detect import detect_blade
from ..pipeline.frames import extract_first_frame, extract_frame_at
from ..pipeline.job_meta import JobNotRerenderableError, require_rerenderable
from ..pipeline.runner import rerender_pipeline, rerender_pipeline_multi, run_pipeline, run_pipeline_multi
from .jobs import JobManager

STATIC_DIR = Path(__file__).parent / "static"
JOB_ID_RE = re.compile(r"[0-9a-zA-Z_-]{1,64}")
VALID_VOICES = ("neutral", "jedi", "sith")

app = FastAPI()
manager = JobManager()


def _validate_job_id(job_id: str) -> None:
    if not JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(status_code=404, detail="Job not found")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if manager.is_busy():
        raise HTTPException(status_code=409, detail="A render is already in progress")

    job_id = uuid.uuid4().hex[:8]
    job_dir = paths.new_job_dir(job_id)
    input_path = job_dir / "input.mp4"
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    frame0_path = job_dir / "frame0.jpg"
    extract_first_frame(str(input_path), str(frame0_path))

    img = cv2.imread(str(frame0_path))
    height, width = img.shape[:2]

    return {
        "job_id": job_id,
        "frame0_url": f"/api/jobs/{job_id}/frame0",
        "width": width,
        "height": height,
    }


@app.get("/api/jobs/{job_id}/frame0")
def get_frame0(job_id: str):
    _validate_job_id(job_id)
    frame0_path = paths.get_jobs_dir() / job_id / "frame0.jpg"
    if not frame0_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(frame0_path, media_type="image/jpeg")


DETECT_TINT = (0, 255, 0)  # BGR; matches the CLI picker's overlay.


@app.post("/api/jobs/{job_id}/detect")
def detect(job_id: str):
    """Look for the swung object and return a proposal to confirm.

    Deliberately a *sync* route: detection runs optical flow and one SAM2
    image pass, several seconds of blocking CPU work, and FastAPI runs sync
    routes in a threadpool rather than on the event loop. Declaring this
    `async def` would stall every other request, including the progress
    stream, for the duration.

    Writes two files into the job dir rather than returning pixels inline:
    the clean frame the points refer to, and an RGBA tint of the mask for
    the page to composite over it. The frame matters -- a proposal's points
    are meaningless against frame 0, since the object has moved by then.
    """
    _validate_job_id(job_id)
    job_dir = paths.get_jobs_dir() / job_id
    input_path = job_dir / "input.mp4"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    if not paths.get_checkpoint_path().exists():
        raise HTTPException(
            status_code=400,
            detail="SAM2 is not installed yet — run `lightsaber-fx setup` first.",
        )

    proposal = detect_blade(
        str(input_path), str(paths.get_checkpoint_path()),
        "configs/sam2.1/sam2.1_hiera_s.yaml", select_device(),
    )
    if proposal is None:
        return {"found": False}

    try:
        extract_frame_at(
            str(input_path), proposal.frame_index, str(job_dir / "detect_frame.jpg")
        )
    except ValueError:
        # Detection read this same file, so a frame it named should always be
        # seekable -- but a container whose index disagrees with its actual
        # frames is a real thing, and "found nothing" leaves the user clicking
        # the object as they would have anyway. A 500 here would instead break
        # a page that has a perfectly good fallback.
        return {"found": False}

    height, width = proposal.mask.shape[:2]
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[proposal.mask, :3] = DETECT_TINT
    overlay[proposal.mask, 3] = 110
    cv2.imwrite(str(job_dir / "detect_mask.png"), overlay)

    return {
        "found": True,
        "frame_index": proposal.frame_index,
        "elongation": round(proposal.elongation, 1),
        "points": [[x, y, 1] for x, y in proposal.points],
        "frame_url": f"/api/jobs/{job_id}/detect-frame",
        "mask_url": f"/api/jobs/{job_id}/detect-mask",
        "width": width,
        "height": height,
    }


@app.get("/api/jobs/{job_id}/detect-frame")
def get_detect_frame(job_id: str):
    _validate_job_id(job_id)
    path = paths.get_jobs_dir() / job_id / "detect_frame.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/jobs/{job_id}/detect-mask")
def get_detect_mask(job_id: str):
    _validate_job_id(job_id)
    path = paths.get_jobs_dir() / job_id / "detect_mask.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(path, media_type="image/png")


def _parse_render_params(body: dict):
    """Validate and extract color/intensity/blade_extend/voice from a
    request body. Shared by `/points` (the first render) and `/rerender`
    (later renders of the same job) so the two endpoints can't quietly
    drift apart on what counts as a valid --intensity or --voice."""
    color = body.get("color", "red")
    intensity = float(body.get("intensity", 0.35))
    if not 0.0 <= intensity <= 1.0:
        raise HTTPException(status_code=400, detail="intensity must be between 0.0 and 1.0")
    blade_extend = bool(body.get("blade_extend", True))
    voice = body.get("voice", "neutral")
    if voice not in VALID_VOICES:
        raise HTTPException(
            status_code=400, detail=f"voice must be one of {', '.join(VALID_VOICES)}"
        )
    return color, intensity, blade_extend, voice


def _parse_saber_specs(body: dict):
    """Validate and extract the list of per-saber specs from a `/points`
    request body: 1-4 entries, each needing at least one include point and
    a valid color/intensity/voice, using the exact same per-field rules
    `_parse_render_params` already enforces for the single-object endpoints."""
    sabers = body.get("sabers", [])
    if not 1 <= len(sabers) <= 4:
        raise HTTPException(status_code=400, detail="sabers must have between 1 and 4 entries")

    parsed = []
    for i, saber in enumerate(sabers):
        points_and_labels = saber.get("points", [])
        if not any(p[2] == 1 for p in points_and_labels):
            raise HTTPException(status_code=400, detail=f"saber {i}: at least one include point is required")
        color, intensity, _, voice = _parse_render_params(saber)
        parsed.append({
            "points": [[p[0], p[1]] for p in points_and_labels],
            "labels": [p[2] for p in points_and_labels],
            "color": color,
            "intensity": intensity,
            "voice": voice,
        })
    return parsed


@app.post("/api/jobs/{job_id}/points")
async def submit_points(job_id: str, body: dict):
    _validate_job_id(job_id)
    if not paths.get_checkpoint_path().exists():
        raise HTTPException(
            status_code=400,
            detail="SAM2 is not installed yet — run `lightsaber-fx setup` first.",
        )
    job_dir = paths.get_jobs_dir() / job_id
    input_path = job_dir / "input.mp4"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    sabers = _parse_saber_specs(body)
    blade_extend = bool(body.get("blade_extend", True))

    output_path = job_dir / "final.mp4"
    device = select_device()

    def pipeline_fn(progress_cb):
        return run_pipeline_multi(
            input_video=str(input_path),
            sabers=sabers,
            output_path=str(output_path),
            job_dir=str(job_dir),
            checkpoint_path=str(paths.get_checkpoint_path()),
            device=device,
            blade_extend=blade_extend,
            progress_cb=progress_cb,
        )

    try:
        manager.start(job_id, pipeline_fn)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"status": "started"}


@app.post("/api/jobs/{job_id}/rerender")
async def rerender_job(job_id: str, body: dict):
    """Re-render an existing job with a new color/intensity/voice/
    blade_extend, reusing its cached masks instead of re-tracking. Goes
    through the same `JobManager` (one job at a time) as `/points` --
    rerender_pipeline() itself never calls track_object, so this can't
    contend with anything except another render of some job."""
    _validate_job_id(job_id)
    job_dir = paths.get_jobs_dir() / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    color, intensity, blade_extend, voice = _parse_render_params(body)

    try:
        require_rerenderable(str(job_dir))
    except JobNotRerenderableError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    output_path = job_dir / "final.mp4"

    def pipeline_fn(progress_cb):
        return rerender_pipeline(
            job_dir=str(job_dir),
            output_path=str(output_path),
            color=color,
            intensity=intensity,
            blade_extend=blade_extend,
            voice=voice,
            progress_cb=progress_cb,
        )

    try:
        manager.start(job_id, pipeline_fn)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"status": "started"}


@app.get("/api/jobs/{job_id}/events")
def stream_events(job_id: str):
    _validate_job_id(job_id)

    def event_gen():
        last_sent = 0
        while True:
            state = manager.get(job_id)
            if state is None:
                yield f"data: {json.dumps({'stage': 'error', 'message': 'job not found'})}\n\n"
                return
            for event in state.events[last_sent:]:
                yield "data: " + json.dumps({
                    "stage": event.stage,
                    "pct": event.pct,
                    "message": event.message,
                    "elapsed": event.elapsed,
                    "eta": event.eta,
                }) + "\n\n"
            last_sent = len(state.events)
            if state.status == "done":
                yield f"data: {json.dumps({'stage': 'done', 'result_url': f'/api/jobs/{job_id}/result'})}\n\n"
                return
            if state.status == "error":
                yield f"data: {json.dumps({'stage': 'error', 'message': state.error_message})}\n\n"
                return
            time.sleep(0.2)

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str):
    _validate_job_id(job_id)
    result_path = paths.get_jobs_dir() / job_id / "final.mp4"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not ready")
    return FileResponse(result_path, media_type="video/mp4")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
