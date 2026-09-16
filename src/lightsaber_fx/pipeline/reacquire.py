"""Recovering tracked-object identity after two tracked objects visually
cross and SAM2's shared multi-object tracking session loses the
distinction between them -- confirmed on real footage (a fencing bout)
where both objects' masks permanently converged onto the same blade after
a crossing and never recovered on their own.

Runs as a post-hoc reconciliation stage in the multi-object pipeline,
after `track.track_objects` finishes and before `blade.compute_motion`
runs: reads the masks `track_objects` already wrote, and where it finds a
crossing between exactly two tracked objects, patches the lost object's
mask files in place before anything downstream sees them.

See docs/superpowers/specs/2026-09-16-cross-object-identity-recovery-design.md.

All frame-count/threshold constants below are starting defaults, validated
only loosely against the one real clip this was diagnosed on -- tune as
more real footage is tested against this.
"""

import os

import cv2
import numpy as np

from .blade import fit_blade, load_mask, load_mask_optional, save_mask
from .detect import MAX_MASK_AREA_FRAC, _build_image_predictor, _points_on_axis
from .vision_detect import (
    DETECTION_PROMPT,
    DETECTION_SCHEMA,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_MS,
    _mask_iou,
    _parse_gemini_response,
    _validate_box_mask,
)

MERGE_IOU_THRESHOLD = 0.8
MERGE_SUSTAIN_FRAMES = 15
REFERENCE_LOOKBACK_FRAMES = 90
REACQUIRE_SEARCH_STEP_FRAMES = 10
REACQUIRE_SEARCH_CAP_FRAMES = 150
REACQUIRE_MAX_OVERLAP_IOU = 0.1


def detect_merge(masks_dir_a, masks_dir_b, frame_indices,
                  iou_threshold=MERGE_IOU_THRESHOLD, sustain_frames=MERGE_SUSTAIN_FRAMES):
    """The first frame of a sustained high-overlap run between two
    objects' masks, or None if no such run exists in `frame_indices`.

    The *sustained* requirement is what distinguishes a real, permanent
    identity mixup from two objects briefly touching and correctly
    separating again (e.g. normal sword contact, tracked correctly) --
    only a run of `sustain_frames` or more consecutive high-IoU frames
    counts.

    `frame_indices` must be sorted; "consecutive" is measured as
    consecutive *entries* in this list, not consecutive frame numbers -- a
    real gap (an object briefly lost) is rare enough not to special-case
    here. If either object is missing a mask file for a frame (object lost),
    that frame resets the current run, same as a sub-threshold-IoU frame.
    """
    run_start = None
    run_len = 0
    for idx in frame_indices:
        mask_a = load_mask_optional(masks_dir_a, idx)
        mask_b = load_mask_optional(masks_dir_b, idx)
        if mask_a is None or mask_b is None:
            run_start = None
            run_len = 0
        else:
            iou = _mask_iou(mask_a, mask_b)
            if iou >= iou_threshold:
                if run_start is None:
                    run_start = idx
                run_len += 1
                if run_len >= sustain_frames:
                    return run_start
            else:
                run_start = None
                run_len = 0
    return None


def find_clean_reference(masks_dir_a, masks_dir_b, merge_start_frame, frame_indices,
                          lookback_frames=REFERENCE_LOOKBACK_FRAMES):
    """The frame of maximum inter-object centroid separation within
    `lookback_frames` before `merge_start_frame`, or None if no frame in
    that window has valid geometry for both objects.

    Deliberately *not* "the last frame before the merge" -- that frame can
    already be mid-drift (confirmed on the real clip this was diagnosed
    on: the frame immediately before a confirmed merge was already down to
    7px separation, and using it as the identity reference produced a
    swapped match). The true local separation maximum is a meaningfully
    cleaner reference.

    Uses `load_mask_optional`, matching `detect_merge`: a candidate frame
    missing a mask file for either object (object briefly lost) is simply
    skipped, not treated as an error.
    """
    candidates = [idx for idx in frame_indices if merge_start_frame - lookback_frames <= idx < merge_start_frame]
    best_idx, best_dist = None, -1.0
    for idx in candidates:
        mask_a = load_mask_optional(masks_dir_a, idx)
        mask_b = load_mask_optional(masks_dir_b, idx)
        if mask_a is None or mask_b is None:
            continue
        geo_a = fit_blade(mask_a)
        geo_b = fit_blade(mask_b)
        if geo_a is None or geo_b is None:
            continue
        dist = float(np.hypot(geo_a.centroid[0] - geo_b.centroid[0], geo_a.centroid[1] - geo_b.centroid[1]))
        if dist > best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def _centroid_dist(c1, c2):
    return float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))


def match_detections_to_objects(detections, ref_centroid_a, ref_centroid_b):
    """Given exactly 2 detections (each a dict with a "centroid" key) and
    two reference centroids, return `(detection_for_a, detection_for_b)`:
    the assignment of the 2 detections to the 2 references that minimizes
    total centroid distance.

    Validated directly against real footage in the design spike: 43px and
    93px total-distance assignment, unambiguous.
    """
    d0, d1 = detections
    cost_keep_order = _centroid_dist(d0["centroid"], ref_centroid_a) + _centroid_dist(d1["centroid"], ref_centroid_b)
    cost_swap = _centroid_dist(d0["centroid"], ref_centroid_b) + _centroid_dist(d1["centroid"], ref_centroid_a)
    return (d0, d1) if cost_keep_order <= cost_swap else (d1, d0)


def patch_masks(lost_masks_dir, fresh_masks_dir, frozen_frame_idx, merge_start_frame, reacquire_frame, n_frames):
    """Rewrites `lost_masks_dir`'s files for frames `[merge_start_frame,
    n_frames)`:

    - `[merge_start_frame, reacquire_frame)`: a frozen copy of
      `lost_masks_dir`'s own mask at `frozen_frame_idx` (the clean
      reference frame from `find_clean_reference`) -- the recovered
      object holds its last known-good position through the crossing
      itself, rather than disappearing.
    - `[reacquire_frame, n_frames)`: copied from `fresh_masks_dir` (the
      output of a fresh `track_object` run), frame-index-aligned.

    Frames before `merge_start_frame` are untouched.
    """
    frozen_mask = load_mask(lost_masks_dir, frozen_frame_idx)
    for idx in range(merge_start_frame, reacquire_frame):
        save_mask(lost_masks_dir, idx, frozen_mask)
    for idx in range(reacquire_frame, n_frames):
        save_mask(lost_masks_dir, idx, load_mask(fresh_masks_dir, idx))


def _frame_path(frames_dir, frame_idx):
    return os.path.join(frames_dir, f"{frame_idx:05d}.jpg")


def _detections_at_frame(frame, predictor, client, max_mask_area):
    """Every Gemini-proposed box at this frame that passes the same
    shape/size validation `detect_blades_vlm` uses, as a list of
    `{"centroid": (x, y), "points": [[x, y], ...], "mask": <bool array>}`.
    """
    from google.genai import types

    height, width = frame.shape[:2]
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return []
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"),
            DETECTION_PROMPT,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=DETECTION_SCHEMA,
            http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
        ),
    )
    boxes = _parse_gemini_response(response.text, width, height)
    if not boxes:
        return []

    predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    detections = []
    for box in boxes:
        masks, _scores, _logits = predictor.predict(
            box=np.array(box, dtype=np.float32), multimask_output=False,
        )
        mask = np.asarray(masks)[0].astype(bool)
        result = _validate_box_mask(mask, max_mask_area)
        if result is None:
            continue
        _elongation, geometry = result
        detections.append({
            "centroid": geometry.centroid,
            "points": _points_on_axis(mask),
            "mask": mask,
        })
    return detections


def _two_separate_detections(detections, max_overlap_iou=REACQUIRE_MAX_OVERLAP_IOU):
    """The 2 detections if exactly 2 passed validation and their masks
    don't overlap past `max_overlap_iou`, else None. More or fewer than 2
    validated detections is treated as "not a clean re-acquisition frame
    yet" -- matches this project's existing decline-rather-than-guess-
    wrong philosophy (see `detect.py`'s elongation gate, `server.py`'s
    VLM-then-motion fallback)."""
    if len(detections) != 2:
        return None
    if _mask_iou(detections[0]["mask"], detections[1]["mask"]) > max_overlap_iou:
        return None
    return detections


def reacquire_pair(
    frames_dir, search_start_frame, checkpoint_path, config_name, device,
    client=None, search_step=REACQUIRE_SEARCH_STEP_FRAMES, search_cap=REACQUIRE_SEARCH_CAP_FRAMES,
):
    """Walk forward from `search_start_frame` in `search_step`
    increments, asking Gemini to find blade-like boxes at each frame
    checked. Returns `(reacquire_frame, [det_0, det_1])` at the first
    frame with exactly 2 validated, mutually non-overlapping detections,
    or `None` if the search window (`search_cap` frames from
    `search_start_frame`, or the end of the clip, whichever comes first)
    is exhausted without finding one. Each detection is
    `{"centroid": (x, y), "points": [[x, y], ...]}`.

    `client` is injectable (a `genai.Client`, or a test double) so tests
    never make a real network call -- same pattern as
    `vision_detect.detect_blades_vlm`.
    """
    if client is None:
        from google import genai
        client = genai.Client()

    predictor = _build_image_predictor(checkpoint_path, config_name, device)

    frame_idx = search_start_frame
    while frame_idx < search_start_frame + search_cap:
        frame = cv2.imread(_frame_path(frames_dir, frame_idx))
        if frame is None:
            return None  # ran past the end of the clip
        height, width = frame.shape[:2]
        detections = _detections_at_frame(frame, predictor, client, MAX_MASK_AREA_FRAC * width * height)
        separated = _two_separate_detections(detections)
        if separated is not None:
            return frame_idx, [
                {"centroid": d["centroid"], "points": d["points"]} for d in separated
            ]
        frame_idx += search_step
    return None
