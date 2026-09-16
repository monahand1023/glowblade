import json
import logging
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
from ..pipeline.detect import _build_image_predictor, detect_blade
from ..pipeline.frames import extract_first_frame, extract_frame_at
from ..pipeline.job_meta import JobNotRerenderableError, require_rerenderable
from ..pipeline.runner import rerender_pipeline_multi, run_pipeline_multi
from ..pipeline.vision_detect import detect_blades_vlm
from .jobs import JobManager

STATIC_DIR = Path(__file__).parent / "static"
JOB_ID_RE = re.compile(r"[0-9a-zA-Z_-]{1,64}")
VALID_VOICES = ("neutral", "jedi", "sith")

app = FastAPI()
manager = JobManager()


@app.middleware("http")
async def no_cache_static(request, call_next):
    """Static files (index.html/app.js/style.css) ship with no explicit
    cache headers, so browsers apply heuristic caching and can silently
    keep serving an old version after a reload -- confirmed directly: a
    CSS change was invisible after a normal reload because Chrome served
    the previous style.css from its disk cache without revalidating.
    This is a local dev tool under active iteration, so correctness of a
    reload matters far more than the bandwidth saved by caching it."""
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-store"
    return response


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


def _mask_overlay_png(mask, tint=DETECT_TINT, alpha=110):
    """A transparent RGBA tint of `mask`, in the same style `/detect` has
    always used -- shared so the manual-selection preview looks like the
    same kind of thing as an automatic proposal, not a different feature."""
    height, width = mask.shape[:2]
    overlay = np.zeros((height, width, 4), dtype=np.uint8)
    overlay[mask, :3] = tint
    overlay[mask, 3] = alpha
    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("failed to encode mask overlay")
    return buf.tobytes()


def _detect_proposals(input_path, device):
    """Try vision-assisted multi-object detection first, falling back to
    today's single-object motion-based detect_blade on any failure --
    missing API key, network error, bad response, or zero validated
    candidates. Returns `(list[BladeProposal], source)` where `source` is
    `"vlm"` or `"motion"`.

    Partial success is still success: if the VLM finds 2 of 3 actual
    objects, those 2 are returned as-is -- this never tops up a VLM result
    with a motion-detected one, since they could disagree about which
    frame to use and every saber in one render must share a prompt_frame.
    """
    try:
        proposals = detect_blades_vlm(
            input_path, str(paths.get_checkpoint_path()),
            "configs/sam2.1/sam2.1_hiera_s.yaml", device,
        )
        if proposals:
            return proposals, "vlm"
    except Exception:
        logging.getLogger(__name__).warning(
            "vision-assisted detection failed, falling back to motion detection", exc_info=True,
        )

    proposal = detect_blade(
        input_path, str(paths.get_checkpoint_path()),
        "configs/sam2.1/sam2.1_hiera_s.yaml", device,
    )
    return ([proposal] if proposal else []), "motion"


@app.post("/api/jobs/{job_id}/detect")
def detect(job_id: str):
    """Look for the swung object(s) and return proposals to confirm.

    Deliberately a *sync* route: detection runs (vision or motion) plus one
    or more SAM2 passes, several seconds of blocking CPU/network work, and
    FastAPI runs sync routes in a threadpool rather than on the event loop.
    Declaring this `async def` would stall every other request, including
    the progress stream, for the duration.

    Writes one clean frame plus one RGBA mask overlay per proposal into the
    job dir rather than returning pixels inline. The frame matters -- a
    proposal's points are meaningless against frame 0, since the object has
    moved by then.
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

    proposals, source = _detect_proposals(str(input_path), select_device())
    if not proposals:
        return {"found": False}

    frame_index = proposals[0].frame_index  # all proposals share one frame -- see Global Constraints
    try:
        extract_frame_at(
            str(input_path), frame_index, str(job_dir / "detect_frame.jpg")
        )
    except ValueError:
        # Detection read this same file, so a frame it named should always be
        # seekable -- but a container whose index disagrees with its actual
        # frames is a real thing, and "found nothing" leaves the user clicking
        # the object as they would have anyway. A 500 here would instead break
        # a page that has a perfectly good fallback.
        return {"found": False}

    height, width = proposals[0].mask.shape[:2]
    result_proposals = []
    for i, proposal in enumerate(proposals):
        (job_dir / f"detect_mask_{i}.png").write_bytes(_mask_overlay_png(proposal.mask))
        result_proposals.append({
            "elongation": round(proposal.elongation, 1),
            "points": [[x, y, 1] for x, y in proposal.points],
            "mask_url": f"/api/jobs/{job_id}/detect-mask/{i}",
        })

    return {
        "found": True,
        "frame_index": frame_index,
        "frame_url": f"/api/jobs/{job_id}/detect-frame",
        "proposals": result_proposals,
        "source": source,
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


@app.get("/api/jobs/{job_id}/detect-mask/{index}")
def get_detect_mask(job_id: str, index: int):
    _validate_job_id(job_id)
    if not 0 <= index < 4:
        raise HTTPException(status_code=404, detail="Job not found")
    path = paths.get_jobs_dir() / job_id / f"detect_mask_{index}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(path, media_type="image/png")


@app.post("/api/jobs/{job_id}/preview_mask")
def preview_mask(job_id: str, body: dict):
    """Segment the points the user has placed so far on a single frame and
    return the mask as an overlay, so an obviously-wrong selection (SAM2
    latching onto the background instead of the thin object clicked near)
    is visible before committing to a full track-and-render.

    Deliberately cheap compared to `/points`: one SAM2 *image* pass on one
    frame, not video propagation across the whole clip -- this exists to be
    called after every click/drag while the user is still choosing.
    Same sync-route reasoning as `/detect`: this blocks on real CPU/GPU
    work, so it must not be `async def`.
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

    points_and_labels = body.get("points", [])
    if not any(p[2] == 1 for p in points_and_labels):
        raise HTTPException(status_code=400, detail="at least one include point is required")
    prompt_frame = int(body.get("prompt_frame", 0))
    if prompt_frame < 0:
        raise HTTPException(status_code=400, detail="prompt_frame must not be negative")

    frame_path = job_dir / "preview_mask_frame.jpg"
    try:
        extract_frame_at(str(input_path), prompt_frame, str(frame_path))
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Could not read frame {prompt_frame}")

    frame = cv2.imread(str(frame_path))
    predictor = _build_image_predictor(
        str(paths.get_checkpoint_path()), "configs/sam2.1/sam2.1_hiera_s.yaml", select_device(),
    )
    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[p[0], p[1]] for p in points_and_labels], dtype=np.float32),
        point_labels=np.array([p[2] for p in points_and_labels], dtype=np.int32),
        multimask_output=False,
    )
    mask = np.asarray(masks)[0].astype(bool)

    (job_dir / "preview_mask.png").write_bytes(_mask_overlay_png(mask))
    height, width = mask.shape[:2]
    return {
        "mask_url": f"/api/jobs/{job_id}/preview-mask-image",
        "score": round(float(np.asarray(scores)[0]), 3),
        "area_frac": round(float(mask.sum()) / (height * width), 4),
    }


@app.get("/api/jobs/{job_id}/preview-mask-image")
def get_preview_mask_image(job_id: str):
    _validate_job_id(job_id)
    path = paths.get_jobs_dir() / job_id / "preview_mask.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No preview available")
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
    `_parse_render_params` already enforces for the single-object endpoints.

    Each entry may also carry a `prompt_frame` -- the frame its points were
    placed on, which automatic detection usually reports as mid-swing
    rather than frame 0. It is validated (non-negative) the same way the
    old single-object endpoint validated it, and passed through to the
    tracker, which prompts there and propagates both ways."""
    sabers = body.get("sabers", [])
    if not 1 <= len(sabers) <= 4:
        raise HTTPException(status_code=400, detail="sabers must have between 1 and 4 entries")

    parsed = []
    for i, saber in enumerate(sabers):
        points_and_labels = saber.get("points", [])
        if not any(p[2] == 1 for p in points_and_labels):
            raise HTTPException(status_code=400, detail=f"saber {i}: at least one include point is required")
        color, intensity, _, voice = _parse_render_params(saber)
        prompt_frame = int(saber.get("prompt_frame", 0))
        if prompt_frame < 0:
            raise HTTPException(status_code=400, detail=f"saber {i}: prompt_frame must not be negative")
        parsed.append({
            "points": [[p[0], p[1]] for p in points_and_labels],
            "labels": [p[2] for p in points_and_labels],
            "color": color,
            "intensity": intensity,
            "voice": voice,
            "prompt_frame": prompt_frame,
        })

    # Caught here as well as in track_objects so a bad request fails as a
    # 400 on the POST, rather than being accepted and then killing the
    # background render seconds later.
    prompt_frames = {s["prompt_frame"] for s in parsed}
    if len(prompt_frames) > 1:
        raise HTTPException(
            status_code=400,
            detail="All sabers must share the same prompt_frame in this release "
                   f"(mixing frames isn't supported yet) -- got {sorted(prompt_frames)}",
        )
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


def _parse_saber_style_specs(body: dict):
    """Like `_parse_saber_specs`, for `/rerender` -- no points/labels here,
    tracking is never re-run, just color/intensity/voice per saber."""
    sabers = body.get("sabers", [])
    if not 1 <= len(sabers) <= 4:
        raise HTTPException(status_code=400, detail="sabers must have between 1 and 4 entries")
    parsed = []
    for saber in sabers:
        color, intensity, _, voice = _parse_render_params(saber)
        parsed.append({"color": color, "intensity": intensity, "voice": voice})
    return parsed


@app.post("/api/jobs/{job_id}/rerender")
async def rerender_job(job_id: str, body: dict):
    """Re-render an existing job with new per-saber color/intensity/voice
    (plus blade_extend), reusing its cached masks instead of re-tracking.
    Goes through the same `JobManager` (one job at a time) as `/points` --
    rerender_pipeline_multi() itself never calls track_objects, so this
    can't contend with anything except another render of some job."""
    _validate_job_id(job_id)
    job_dir = paths.get_jobs_dir() / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    sabers = _parse_saber_style_specs(body)
    blade_extend = bool(body.get("blade_extend", True))

    try:
        info = require_rerenderable(str(job_dir))
    except JobNotRerenderableError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # Two genuinely different failures, so two different messages: a job
    # with no recorded object_ids predates multi-saber support entirely
    # (reporting it as "1 tracked object" sends the user off to change the
    # saber count, which can never fix it), while a count mismatch is a
    # real, fixable mismatch against a real multi-saber job.
    if info.object_ids is None:
        raise HTTPException(
            status_code=400,
            detail="This job has no recorded object_ids -- it isn't a multi-saber job "
                   "(it may predate multi-saber support, or was created by the CLI). "
                   "Re-upload it through the web app to re-render with multiple sabers.",
        )
    if len(sabers) != len(info.object_ids):
        raise HTTPException(
            status_code=400,
            detail=f"This job has {len(info.object_ids)} tracked object(s), but {len(sabers)} saber(s) were given",
        )

    output_path = job_dir / "final.mp4"

    def pipeline_fn(progress_cb):
        return rerender_pipeline_multi(
            job_dir=str(job_dir),
            output_path=str(output_path),
            sabers=sabers,
            blade_extend=blade_extend,
            progress_cb=progress_cb,
        )

    try:
        manager.start(job_id, pipeline_fn)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"status": "started"}


@app.get("/api/jobs/{job_id}/preview")
def get_preview(job_id: str):
    """The most recently written frame of the glow stage's in-progress PNG
    sequence, so the page can show what the render currently looks like
    instead of a bare percentage.

    `glow_frames/` only exists while the glow stage is running -- it is
    absent before that stage starts and removed once it finishes -- so both
    "not created yet" and "already cleaned up" are the same ordinary 404,
    not an error.
    """
    _validate_job_id(job_id)
    glow_dir = paths.get_jobs_dir() / job_id / "glow_frames"
    try:
        frames = sorted(glow_dir.glob("*.png"))
    except OSError:
        frames = []
    if not frames:
        raise HTTPException(status_code=404, detail="No preview available yet")
    return FileResponse(frames[-1], media_type="image/png")


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
