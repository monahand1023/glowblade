"""Recovering tracked-object identity and geometry after two tracked
objects visually cross, two related but distinct failure modes:

- `reconcile_pair`: SAM2's shared multi-object tracking session loses the
  distinction between the two objects outright -- confirmed on real
  footage (a fencing bout) where both objects' masks permanently
  converged onto the same blade after a crossing and never recovered on
  their own. Runs as a post-hoc reconciliation stage after
  `track.track_objects` finishes and before `blade.compute_motion` runs:
  reads the masks `track_objects` already wrote, and where it finds a
  crossing, patches the lost object's mask files in place before anything
  downstream sees them.

- `retrack_overlap_runs`: a *partial* mask bleed between the two objects
  that never becomes the full, sustained, symmetric merge
  `reconcile_pair` detects. Runs after `blade.compute_motion` (needs
  finished motion.npz for both objects to find these runs) and before
  `blade.suppress_overlap_bleed`, attempting a real independent re-track
  through each run before falling back to that function's geometry
  interpolation.

See docs/superpowers/specs/2026-09-16-cross-object-identity-recovery-design.md.

All frame-count/threshold constants below are starting defaults, validated
only loosely against the handful of real clips this was diagnosed on --
tune as more real footage is tested against this.
"""

import logging
import os
import tempfile

import cv2
import numpy as np

from .blade import (
    _find_overlap_runs,
    _mask_iou,
    fit_blade,
    load_mask,
    load_mask_optional,
    load_motion,
    mask_frame_indices,
    save_mask,
)
from .detect import MAX_MASK_AREA_FRAC, _build_image_predictor, _points_on_axis
from .track import track_object
from .vision_detect import (
    DETECTION_PROMPT,
    DETECTION_SCHEMA,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_MS,
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

    All-or-nothing: every fresh mask is loaded (and would raise on a
    missing one) before any file in `lost_masks_dir` is written, so a
    gap in `fresh_masks_dir` never leaves `lost_masks_dir` half-patched.
    """
    frozen_mask = load_mask(lost_masks_dir, frozen_frame_idx)
    fresh_masks = [load_mask(fresh_masks_dir, idx) for idx in range(reacquire_frame, n_frames)]
    for idx in range(merge_start_frame, reacquire_frame):
        save_mask(lost_masks_dir, idx, frozen_mask)
    for idx, mask in zip(range(reacquire_frame, n_frames), fresh_masks, strict=True):
        save_mask(lost_masks_dir, idx, mask)


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
    VLM-then-motion fallback).

    Deliberate, recorded narrowing from the design doc's "returns >=2
    boxes" wording: this requires *exactly* 2, not >=2 with a
    disambiguation step. On a clip where Gemini's own duplicate-box
    behavior (see `vision_detect._dedupe_by_mask_iou`'s docstring) or a
    genuinely busier scene produces a 3rd validated detection at every
    checked frame within the search window, recovery will never fire --
    a silent no-op, not a crash, consistent with this module's fallback
    philosophy, but a real limitation on multi-person footage worth
    knowing about rather than discovering.
    """
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


def reconcile_pair(frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device, client=None):
    """Best-effort recovery from a detected crossing between exactly two
    tracked objects (object 0 and object 1). Returns `True` if either
    object's masks were changed, `False` if no merge was found or recovery
    failed at any step -- in the `False` case, both objects' masks are
    left completely untouched.

    A successful patch's `[reacquire_frame, n_frames)` portion is
    genuinely computed via SAM2 propagation, not frozen/approximated --
    but that is not the same guarantee as "accurate for its entire span."
    Confirmed on real footage: the re-tracked object can drift onto the
    other tracked object again later in that same span, a fresh problem
    this function has no way to know about at the time it returns. An
    earlier version of this function reported that span for a caller to
    exclude from `blade.suppress_overlap_bleed`'s own correction pass;
    that trusted the span more than it had earned and left a real, later
    overlap (confirmed via a full real end-to-end run: a completely
    missing blade for 162 frames) uncorrected. Only
    `retrack_overlap_runs`' `resolved_ranges` carries an actual accuracy
    check (drift against known-good geometry) and is safe to exclude that
    way -- this function's own output isn't.

    Never raises, structurally: the entire body runs inside one outer
    try/except, so this contract holds even for a failure this function
    doesn't specifically anticipate (not just the two Gemini/SAM2-
    dependent steps it already reasons about individually below).
    """
    try:
        return _reconcile_pair_impl(
            frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device, client,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "cross-object identity recovery failed unexpectedly, leaving today's tracking as-is",
            exc_info=True,
        )
        return False


def _reconcile_pair_impl(frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device, client):
    frame_indices = mask_frame_indices(masks_dir_0)

    merge_start = detect_merge(masks_dir_0, masks_dir_1, frame_indices)
    if merge_start is None:
        return False

    reference_frame = find_clean_reference(masks_dir_0, masks_dir_1, merge_start, frame_indices)
    if reference_frame is None:
        return False

    geo_0 = fit_blade(load_mask(masks_dir_0, reference_frame))
    geo_1 = fit_blade(load_mask(masks_dir_1, reference_frame))
    if geo_0 is None or geo_1 is None:
        return False

    try:
        result = reacquire_pair(frames_dir, merge_start, checkpoint_path, config_name, device, client=client)
    except Exception:
        logging.getLogger(__name__).warning(
            "cross-object identity recovery failed during re-acquisition, leaving today's tracking as-is",
            exc_info=True,
        )
        return False
    if result is None:
        return False
    reacquire_frame, detections = result

    # Confirm the two ORIGINAL (uncorrected) tracks are still merged at
    # reacquire_frame before touching anything. If they've already
    # separated on their own by then, this was a brief bind that the
    # tracker resolved correctly (routine sword contact, not the
    # permanent identity swap this module exists to fix) -- "fixing" it
    # would overwrite a track that was never actually broken. Declining
    # here also means a real merge later in the clip, if there is one,
    # isn't masked by a reconciliation attempt already spent on a false
    # positive.
    original_mask_0 = load_mask_optional(masks_dir_0, reacquire_frame)
    original_mask_1 = load_mask_optional(masks_dir_1, reacquire_frame)
    if original_mask_0 is None or original_mask_1 is None:
        return False
    if _mask_iou(original_mask_0, original_mask_1) < MERGE_IOU_THRESHOLD:
        return False

    det_for_0, det_for_1 = match_detections_to_objects(detections, geo_0.centroid, geo_1.centroid)

    # Sampled at reacquire_frame (the same instant as the detections),
    # not merge_start -- up to REACQUIRE_SEARCH_CAP_FRAMES apart, during
    # which the surviving blade can move. Comparing same-instant makes
    # the kept object's distance ~0 and the lost object's distance at
    # least the inter-blade separation, unambiguous by construction.
    merged_geo = fit_blade(original_mask_0)
    if merged_geo is None:
        return False
    dist_0 = _centroid_dist(det_for_0["centroid"], merged_geo.centroid)
    dist_1 = _centroid_dist(det_for_1["centroid"], merged_geo.centroid)
    # The "kept" object's fresh detection is close to where its
    # (corrupted, but still-tracking-*something*) mask currently sits;
    # the "lost" object's fresh detection is far from it.
    if dist_0 <= dist_1:
        lost_masks_dir, lost_points = masks_dir_1, det_for_1["points"]
    else:
        lost_masks_dir, lost_points = masks_dir_0, det_for_0["points"]

    with tempfile.TemporaryDirectory() as fresh_masks_dir:
        try:
            track_object(
                frames_dir, fresh_masks_dir, lost_points, [1] * len(lost_points),
                checkpoint_path, config_name, device, n_frames, prompt_frame=reacquire_frame,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "cross-object identity recovery failed during re-tracking, leaving today's tracking as-is",
                exc_info=True,
            )
            return False
        patch_masks(lost_masks_dir, fresh_masks_dir, reference_frame, merge_start, reacquire_frame, n_frames)

    return True


# How far (as a fraction of the object's own fitted length at the
# known-good "after" anchor) an independently re-tracked object's
# centroid may land from the true position there before the re-track is
# distrusted. Generous enough to tolerate ordinary tracking noise, but
# tight enough to catch the real failure mode this guards against: the
# independent session drifting onto the *other* tracked object during the
# very contact it's meant to track through, which lands it roughly a
# blade-length away, not a fraction of one.
RETRACK_MAX_DRIFT_FRAC = 0.5


def retrack_overlap_runs(frames_dir, masks_dir_0, masks_dir_1, motion_path_0, motion_path_1,
                          n_frames, checkpoint_path, config_name, device):
    """Best-effort upgrade over `blade.suppress_overlap_bleed`'s geometry
    interpolation: for each run of cross-object mask overlap that has a
    known-good frame on both sides (see `blade._find_overlap_runs`), try
    tracking each object through the run independently. A single-object
    SAM2 session, seeded from that object's own mask at the frame just
    before the run, has no *other* tracked object in its session to bleed
    into -- unlike the shared multi-object session that produced the
    bleed in the first place.

    Validated against the known-good geometry at the frame just after the
    run: only a re-track whose centroid lands close to the true position
    there (see `RETRACK_MAX_DRIFT_FRAC`) is trusted and spliced in. A run
    missing either anchor is always left alone -- there is nothing to
    validate an independent re-track against, same reasoning
    `suppress_overlap_bleed` uses to fall back to freezing there instead
    of interpolating.

    Runs after `compute_motion` has produced both objects' motion.npz (it
    reads them to find overlap runs, same as `suppress_overlap_bleed`) and
    before `suppress_overlap_bleed` itself, which remains the fallback for
    every run this can't fix -- failed validation, a missing anchor, or
    this function's own outer failure. The caller must re-run
    `compute_motion` for any object this patches before
    `suppress_overlap_bleed` runs, since this rewrites mask files, not
    motion.npz.

    Never raises, structurally, matching `reconcile_pair`: the entire body
    runs inside one outer try/except.

    Returns `(patched, resolved_ranges)`: `patched` is the set of object
    indices (a subset of `{0, 1}`) whose masks were changed; `resolved_ranges`
    is a list of `(run_start_frame, run_end_frame)` tuples for runs where
    *both* objects validated. A caller must pass `resolved_ranges` on to
    `suppress_overlap_bleed` as `exclude_frame_ranges` -- two blades in
    genuine, correctly-tracked contact still show high mask IoU (that's
    what real contact looks like), so without excluding them,
    `suppress_overlap_bleed`'s own re-detection would "fix" a run this
    function already got right, overwriting an accurate independent
    re-track with a worse interpolated approximation. A run where only one
    object validated is *not* included here -- the other object's data is
    still bad, so the run still needs `suppress_overlap_bleed`'s pass
    (which corrects both objects together; see its own docstring for why
    that's the right tradeoff).
    """
    try:
        return _retrack_overlap_runs_impl(
            frames_dir, masks_dir_0, masks_dir_1, motion_path_0, motion_path_1,
            n_frames, checkpoint_path, config_name, device,
        )
    except Exception:
        logging.getLogger(__name__).warning(
            "re-tracking through overlap failed unexpectedly, leaving masks as-is for "
            "suppress_overlap_bleed to interpolate",
            exc_info=True,
        )
        return set(), []


def _retrack_one_object(frames_dir, masks_dir, obj_idx, before_frame, after_frame,
                         run_start_frame, run_end_frame, n_frames, checkpoint_path, config_name, device):
    """Try re-tracking a single object through `[run_start_frame,
    run_end_frame]`, seeded from its own mask at `before_frame`. Returns
    True if the re-track validated and `masks_dir`'s files for that range
    were replaced, False otherwise (masks_dir is left untouched on any
    False return)."""
    logger = logging.getLogger(__name__)
    after_mask = load_mask(masks_dir, after_frame)
    true_after_geo = fit_blade(after_mask)
    if true_after_geo is None:
        return False

    before_mask = load_mask(masks_dir, before_frame)
    points = _points_on_axis(before_mask)

    with tempfile.TemporaryDirectory() as fresh_masks_dir:
        try:
            track_object(
                frames_dir, fresh_masks_dir, points, [1] * len(points),
                checkpoint_path, config_name, device, n_frames, prompt_frame=before_frame,
            )
        except Exception:
            logger.warning(
                "object %d: independent re-track through frames %d-%d failed, leaving to the "
                "interpolation fallback", obj_idx, run_start_frame, run_end_frame, exc_info=True,
            )
            return False

        fresh_after_mask = load_mask_optional(fresh_masks_dir, after_frame)
        fresh_after_geo = fit_blade(fresh_after_mask) if fresh_after_mask is not None else None
        if fresh_after_geo is None:
            logger.warning(
                "object %d: independent re-track lost the blade by frame %d, leaving to the "
                "interpolation fallback", obj_idx, after_frame,
            )
            return False

        drift = _centroid_dist(fresh_after_geo.centroid, true_after_geo.centroid)
        cap = RETRACK_MAX_DRIFT_FRAC * true_after_geo.length
        if drift > cap:
            logger.warning(
                "object %d: independent re-track drifted %.1fpx from the known-good frame %d position "
                "(cap %.1fpx) -- likely locked onto the other tracked object during contact, leaving to "
                "the interpolation fallback", obj_idx, drift, after_frame, cap,
            )
            return False

        fresh_run_masks = [load_mask(fresh_masks_dir, f) for f in range(run_start_frame, run_end_frame + 1)]
        for f, mask in zip(range(run_start_frame, run_end_frame + 1), fresh_run_masks, strict=True):
            save_mask(masks_dir, f, mask)

    logger.info(
        "object %d: independent re-track validated (drift %.1fpx, cap %.1fpx) -- replaced frames "
        "%d-%d with real tracked masks instead of interpolation",
        obj_idx, drift, cap, run_start_frame, run_end_frame,
    )
    return True


def _retrack_overlap_runs_impl(frames_dir, masks_dir_0, masks_dir_1, motion_path_0, motion_path_1,
                                n_frames, checkpoint_path, config_name, device):
    motion_0 = load_motion(motion_path_0)
    motion_1 = load_motion(motion_path_1)
    frame_indices = mask_frame_indices(masks_dir_0)
    runs = _find_overlap_runs(masks_dir_0, masks_dir_1, motion_0, motion_1)

    patched = set()
    resolved_ranges = []
    for run in runs:
        if run.before is None or run.after is None:
            continue
        before_frame = frame_indices[run.before]
        after_frame = frame_indices[run.after]
        run_start_frame = frame_indices[run.run_start]
        run_end_frame = frame_indices[run.run_end]

        validated_this_run = set()
        for obj_idx, masks_dir in ((0, masks_dir_0), (1, masks_dir_1)):
            ok = _retrack_one_object(
                frames_dir, masks_dir, obj_idx, before_frame, after_frame,
                run_start_frame, run_end_frame, n_frames, checkpoint_path, config_name, device,
            )
            if ok:
                patched.add(obj_idx)
                validated_this_run.add(obj_idx)

        if validated_this_run == {0, 1}:
            resolved_ranges.append((run_start_frame, run_end_frame))

    return patched, resolved_ranges
