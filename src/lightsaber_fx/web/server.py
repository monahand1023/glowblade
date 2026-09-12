import json
import re
import shutil
import time
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .. import paths
from ..device import select_device
from ..pipeline.frames import extract_first_frame
from ..pipeline.runner import run_pipeline
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

    points_and_labels = body.get("points", [])
    if not any(p[2] == 1 for p in points_and_labels):
        raise HTTPException(status_code=400, detail="At least one include point is required")

    points = [[p[0], p[1]] for p in points_and_labels]
    labels = [p[2] for p in points_and_labels]
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

    output_path = job_dir / "final.mp4"
    device = select_device()

    def pipeline_fn(progress_cb):
        return run_pipeline(
            input_video=str(input_path),
            points=points,
            labels=labels,
            output_path=str(output_path),
            job_dir=str(job_dir),
            checkpoint_path=str(paths.get_checkpoint_path()),
            device=device,
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
