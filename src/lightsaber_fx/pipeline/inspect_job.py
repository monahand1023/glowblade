"""Debugging aid for a rendered job.

Dumps sample frames, per-object mask overlays (on the raw source footage,
not the glow-composited output), and simple tracking-geometry checks, so a
render that "looks awful" can be diagnosed by reading a handful of files
and numbers instead of re-deriving them by hand -- extracting frames with
ffmpeg, loading masks.npz, computing bounding boxes -- every single time.

Not part of the render pipeline itself; only `lightsaber_fx.cli`'s
`inspect` command calls this.
"""

import os

import cv2
import numpy as np

from . import job_meta
from .blade import (
    LOW_ELONGATION_FRAC_THRESHOLD,
    MIN_ELONGATION,
    elongation_stats,
    load_mask_optional,
    load_motion,
)

# A tracked "blade" this wide, or with a tip-to-hilt span this large,
# relative to the frame, is not a blade -- these are the exact numbers
# that flagged job 08873c75 (mask == the background roofline, ~16% width,
# ~98% span) against a correctly-tracked sword (~3% width, well inside a
# single figure). They are deliberately loose: false positives here just
# mean glancing at a debug image that turns out fine, but a miss means
# doing the archaeology by hand again.
ANOMALY_WIDTH_FRAC = 0.5
ANOMALY_SPAN_FRAC = 0.85

# Width/span alone miss a real failure mode: job 01 (a Mixkit knights-
# battling test clip) locked onto a compact patch of tunic cloth for the
# whole clip -- never too wide, never spanning the frame, just the wrong
# shape entirely. blade.elongation_stats' MIN_ELONGATION (the bar a mask
# has to clear to be proposed as a blade in the first place) catches that:
# a mask this blob-like, this often (LOW_ELONGATION_FRAC_THRESHOLD), isn't
# a blade either, no matter when in the pipeline it turns up.


def _frame_size(job_dir):
    frame0 = os.path.join(job_dir, "frame0.jpg")
    img = cv2.imread(frame0) if os.path.exists(frame0) else None
    if img is None:
        cap = cv2.VideoCapture(os.path.join(job_dir, "input.mp4"))
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Could not read any frame from job {job_dir!r}")
    height, width = img.shape[:2]
    return width, height


def _sample_indices(n_frames_total, n_samples):
    if n_frames_total <= 0:
        return []
    n_samples = min(n_samples, n_frames_total)
    if n_samples <= 1:
        return [0]
    return sorted({round(i * (n_frames_total - 1) / (n_samples - 1)) for i in range(n_samples)})


def _analyze_object(motion_path, width):
    motion = load_motion(motion_path)
    tip, hilt, blade_width = motion["tip"], motion["hilt"], motion["width"]
    valid = ~np.isnan(blade_width)
    n_tracked = int(valid.sum())
    if n_tracked == 0:
        return {
            "n_tracked": 0, "max_width_frac": None, "max_span_frac": None,
            "mean_elongation": None, "low_elongation_frac": None,
            "anomalies": ["no valid tracked frames"],
        }

    max_width_frac = float(np.nanmax(blade_width)) / width
    span = np.abs(tip[:, 0] - hilt[:, 0])
    max_span_frac = float(np.nanmax(span)) / width
    mean_elongation, low_elongation_frac = elongation_stats(motion)

    anomalies = []
    if max_width_frac > ANOMALY_WIDTH_FRAC:
        anomalies.append(
            f"max tracked width is {max_width_frac:.0%} of frame width -- too thick for a blade, "
            "likely tracking a broad background region"
        )
    if max_span_frac > ANOMALY_SPAN_FRAC:
        anomalies.append(
            f"tip-to-hilt span reaches {max_span_frac:.0%} of frame width -- likely grabbed something "
            "spanning most of the frame (e.g. a background structure) instead of the object"
        )
    if low_elongation_frac is not None and low_elongation_frac > LOW_ELONGATION_FRAC_THRESHOLD:
        anomalies.append(
            f"{low_elongation_frac:.0%} of tracked frames have elongation below {MIN_ELONGATION:.0f} "
            f"(mean {mean_elongation:.1f}) -- looks blob-shaped, not blade-shaped, likely tracking "
            "the wrong thing"
        )
    return {
        "n_tracked": n_tracked,
        "max_width_frac": max_width_frac,
        "max_span_frac": max_span_frac,
        "mean_elongation": mean_elongation,
        "low_elongation_frac": low_elongation_frac,
        "anomalies": anomalies,
    }


def inspect_job(job_dir, out_dir, n_samples=6):
    """Analyze `job_dir` and write debug artifacts to `out_dir`.

    Returns a dict summarizing what was found -- the CLI's `inspect`
    command formats it for the terminal, but it's plain data so it can be
    driven from a script too."""
    job_dir = str(job_dir)
    meta = job_meta.read_job_meta(job_dir) or {}
    info = job_meta.describe_job(job_dir)
    width, height = _frame_size(job_dir)

    object_ids = meta.get("object_ids")
    legacy = object_ids is None
    ids_to_check = [None] if legacy else list(object_ids)

    os.makedirs(out_dir, exist_ok=True)

    object_reports = {}
    for oid in ids_to_check:
        motion_path = (
            os.path.join(job_dir, "motion.npz") if oid is None
            else os.path.join(job_dir, "motion", f"{oid}.npz")
        )
        if not os.path.exists(motion_path):
            object_reports[oid] = {"n_tracked": 0, "max_width_frac": None, "max_span_frac": None, "anomalies": ["no motion file"]}
            continue
        object_reports[oid] = _analyze_object(motion_path, width)

    video_path = os.path.join(job_dir, "final.mp4")
    if not os.path.exists(video_path):
        video_path = os.path.join(job_dir, "input.mp4")
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_idxs = _sample_indices(total, n_samples)
    sample_frames = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = os.path.join(out_dir, f"frame_{idx:05d}.png")
        cv2.imwrite(out_path, frame)
        sample_frames.append(out_path)
    cap.release()

    mask_overlays = []
    cap = cv2.VideoCapture(os.path.join(job_dir, "input.mp4"))
    for oid in ids_to_check:
        masks_dir = os.path.join(job_dir, "masks") if oid is None else os.path.join(job_dir, "masks", str(oid))
        if not os.path.isdir(masks_dir):
            continue
        label = "legacy" if oid is None else str(oid)
        for idx in sample_idxs:
            mask = load_mask_optional(masks_dir, idx)
            if mask is None:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            tinted = frame.copy()
            tinted[mask] = (0.4 * np.array([0, 255, 0]) + 0.6 * tinted[mask]).astype(np.uint8)
            out_path = os.path.join(out_dir, f"mask_{label}_{idx:05d}.png")
            cv2.imwrite(out_path, tinted)
            mask_overlays.append(out_path)
    cap.release()

    return {
        "job_dir": job_dir,
        "out_dir": out_dir,
        "meta": meta,
        "rerenderable": info.rerenderable,
        "reason": info.reason,
        "width": width,
        "height": height,
        "object_reports": object_reports,
        "sample_frames": sample_frames,
        "mask_overlays": mask_overlays,
    }
