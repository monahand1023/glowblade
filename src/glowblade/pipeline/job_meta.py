"""Job-level metadata: recording a render's source clip path so a later
`rerender` can re-extract frames from it, and deciding whether an existing
job directory has everything `rerender` needs.

This is deliberately separate from `blade.py`'s per-frame mask/motion I/O:
it is whole-job bookkeeping (one small JSON file per job), not a per-frame
artifact format. Frames are re-extracted from the source clip rather than
kept -- they are cheap to recompute and expensive to store, the exact
opposite of masks (docs/design-notes.md, "Two storage trade-offs") -- which
is what makes `source_video`, recorded here, the one thing a job can't do
without.
"""

import json
import os
import time
from typing import NamedTuple, Optional

from .blade import mask_frame_indices

JOB_META_FILENAME = "job_meta.json"


class JobNotRerenderableError(Exception):
    """Raised when a job directory is asked to re-render but is missing
    something it needs. `str(exc)` is a clear, user-facing explanation of
    exactly what's missing -- a moved/deleted source clip, or a job that
    predates rerender support/was cleaned -- is an expected, real failure
    mode here, not a bug, so callers should surface it directly rather than
    a traceback."""


class JobInfo(NamedTuple):
    """What `describe_job` reports about one job directory."""

    job_id: str
    source_video: Optional[str]
    frame_count: Optional[int]
    created_at: Optional[float]
    rerenderable: bool
    reason: Optional[str]  # None when rerenderable; otherwise names what's missing
    object_ids: Optional[list] = None


def _job_meta_path(job_dir):
    return os.path.join(str(job_dir), JOB_META_FILENAME)


def write_job_meta(job_dir, source_video, object_ids=None, prompts=None):
    """Record `source_video`'s path (resolved to absolute, so it stays
    correct even if the working directory changes before a later
    `rerender`) alongside a creation timestamp.

    Called once, by `run_pipeline`, right after extract -- this is the only
    bookkeeping a job needs beyond its masks/motion.npz to be re-renderable
    later without re-running SAM2.

    For multi-object jobs, `object_ids` (a list of object IDs being tracked)
    is also recorded, enabling per-object mask/motion rerenderability checks.

    `prompts`, when given, is the raw list of per-object seed data (points,
    labels, prompt_frame -- whatever the caller was given to track with),
    recorded purely for debugging: when tracking locks onto the wrong thing,
    this is what tells you where the click/line that caused it actually
    landed, instead of having to reverse-engineer it from the resulting
    mask. It is never read back by the pipeline itself."""
    meta = {
        "source_video": os.path.abspath(str(source_video)),
        "created_at": time.time(),
    }
    if object_ids is not None:
        meta["object_ids"] = list(object_ids)
    if prompts is not None:
        meta["prompts"] = prompts
    with open(_job_meta_path(job_dir), "w") as f:
        json.dump(meta, f)
    return meta


def read_job_meta(job_dir):
    """Return the dict written by `write_job_meta`, or None if this job
    directory has none -- it predates rerender support, or extract never
    completed."""
    path = _job_meta_path(job_dir)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _read_frame_count(job_dir):
    video_meta_path = os.path.join(str(job_dir), "video_meta.txt")
    if not os.path.exists(video_meta_path):
        return None
    try:
        with open(video_meta_path) as f:
            f.readline()  # fps -- not needed here
            return int(f.readline())
    except (ValueError, OSError):
        return None


def describe_job(job_dir):
    """Report everything `rerender`/`jobs` need to know about one job
    directory: is it re-renderable, and if not, exactly what's missing.

    A job is re-renderable if it still has masks/ (in either mask format --
    see blade.py's compressed/legacy fallback), motion.npz, video_meta.txt,
    and a source clip that still exists on disk at its recorded path.
    frames/ is deliberately NOT required: it is re-extracted from the source
    clip, which costs seconds against the hundreds of megabytes keeping it
    would cost (docs/design-notes.md, "Two storage trade-offs").

    For multi-object jobs (when `object_ids` is in the recorded metadata),
    this checks per-object mask directories (`masks/{obj_id}/`) and motion
    files (`motion/{obj_id}.npz`) instead of the flat layout."""
    job_dir = str(job_dir)
    job_id = os.path.basename(job_dir.rstrip(os.sep))
    meta = read_job_meta(job_dir)
    source_video = meta.get("source_video") if meta else None
    created_at = meta.get("created_at") if meta else None
    object_ids = meta.get("object_ids") if meta else None
    frame_count = _read_frame_count(job_dir)

    reasons = []
    if object_ids is not None:
        for obj_id in object_ids:
            # int() rather than str(): the ids come straight out of the job's
            # JSON, and a malformed one ("0 ", "../x") would otherwise build a
            # path that quietly does not exist and be reported as a missing
            # mask dir instead of as the bad metadata it is.
            obj_masks_dir = os.path.join(job_dir, "masks", str(int(obj_id)))
            if not os.path.isdir(obj_masks_dir) or not mask_frame_indices(obj_masks_dir):
                reasons.append(f"no masks/ for object {obj_id} (tracking was never run, or the job was cleaned)")
            if not os.path.exists(os.path.join(job_dir, "motion", f"{int(obj_id)}.npz")):
                reasons.append(f"no motion/{obj_id}.npz")
    else:
        masks_dir = os.path.join(job_dir, "masks")
        if not os.path.isdir(masks_dir) or not mask_frame_indices(masks_dir):
            reasons.append("no masks/ (tracking was never run, or the job was cleaned)")
        if not os.path.exists(os.path.join(job_dir, "motion.npz")):
            reasons.append("no motion.npz")

    if not os.path.exists(os.path.join(job_dir, "video_meta.txt")):
        reasons.append("no video_meta.txt")
    if source_video is None:
        reasons.append("no recorded source clip path (job predates rerender support)")
    elif not os.path.exists(source_video):
        reasons.append(f"source clip not found: {source_video} (it may have been moved or deleted)")

    return JobInfo(
        job_id=job_id,
        source_video=source_video,
        frame_count=frame_count,
        created_at=created_at,
        rerenderable=not reasons,
        reason="; ".join(reasons) if reasons else None,
        object_ids=object_ids,
    )


def require_rerenderable(job_dir):
    """Like `describe_job`, but raises `JobNotRerenderableError` (carrying
    the same clear reason) instead of returning a falsy result -- for
    callers (`rerender_pipeline`, the CLI, the web endpoint) that need to
    fail loudly with an actionable message rather than check a flag."""
    info = describe_job(job_dir)
    if not info.rerenderable:
        raise JobNotRerenderableError(f"Job {info.job_id!r} is not re-renderable: {info.reason}")
    return info
