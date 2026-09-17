"""Recovering each tracked object's hilt (hand/grip) position through a
sustained cross-object contact run via classical optical-flow tracking on
the raw video frames -- independent of SAM2 mask segmentation, which is
exactly what gets confused during sustained close contact (see
blade.suppress_overlap_bleed's confidence-weighted smoother, which has no
usable signal once the two objects' own raw fitted geometry coincides).

The fencers' hands/hilts (gripped, gloved) are comparatively high-texture
and, on real footage, rarely occupy the same pixels even when the blades
themselves cross -- exactly what makes this a different, complementary
technique to the mask-based approaches elsewhere in this pipeline.

See docs/superpowers/specs/2026-09-17-hilt-optical-flow-tracking-design.md.
"""

import os

import cv2
import numpy as np

from .blade import _find_overlap_runs, load_motion, mask_frame_indices

# Half-width (px) of the square window around a known-good hilt point
# searched for trackable corner features to seed optical flow from.
# Confirmed sufficient on the real job (frames 290/455, both tracked
# objects) -- a wider window (60, 80) found no more useful corners at the
# quality level below.
HILT_SEED_WINDOW_RADIUS_PX = 45

# Minimum number of good corner features required in the seed window to
# attempt tracking at all -- too few trackable points makes the per-frame
# median position estimate too noisy to trust. Comfortably cleared in
# practice on the real job (9-16 corners found per direction tested at
# HILT_SEED_WINDOW_RADIUS_PX / the quality level below); kept as a floor
# for a pathologically flat window, not because real seeding is marginal.
MIN_SEED_FEATURES = 4


def _frame_path(frames_dir, frame_idx):
    return os.path.join(frames_dir, f"{frame_idx:05d}.jpg")


def _seed_features(frame_gray, center, window_radius=HILT_SEED_WINDOW_RADIUS_PX, max_features=20):
    """Good corner features (cv2.goodFeaturesToTrack) within a square
    window of `window_radius` around `center` (x, y) on `frame_gray` (a
    single-channel uint8 image). Returns an (N, 1, 2) float32 array of
    points as cv2.calcOpticalFlowPyrLK expects, or None if fewer than
    MIN_SEED_FEATURES are found.

    qualityLevel=0.1 (not cv2's own 0.3 default) is deliberate --
    confirmed on the real job that 0.3 finds only 1-3 corners near a
    fencer's hilt (would decline immediately), while 0.1 finds 9-16.
    """
    h, w = frame_gray.shape
    cx, cy = center
    x0, x1 = max(0, int(cx - window_radius)), min(w, int(cx + window_radius))
    y0, y1 = max(0, int(cy - window_radius)), min(h, int(cy + window_radius))
    if x1 <= x0 or y1 <= y0:
        return None

    roi_mask = np.zeros_like(frame_gray, dtype=np.uint8)
    roi_mask[y0:y1, x0:x1] = 255
    points = cv2.goodFeaturesToTrack(
        frame_gray, maxCorners=max_features, qualityLevel=0.1, minDistance=5, mask=roi_mask,
    )
    if points is None or len(points) < MIN_SEED_FEATURES:
        return None
    return points


_LK_PARAMS = {
    "winSize": (21, 21), "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}


def _track_points_sequential(frames_dir, frame_indices, seed_points):
    """Track `seed_points` frame-to-frame across `frame_indices` (in the
    order given -- forward or backward, caller's choice) via
    cv2.calcOpticalFlowPyrLK. `frame_indices[0]` must be the frame
    `seed_points` was seeded on.

    Returns a dict {frame_idx: (x, y)} of the *median* (not mean, so one
    point drifting off the real feature doesn't skew the estimate)
    position of whichever points are still successfully tracked at each
    frame. Stops (returns whatever was tracked so far) once a frame is
    missing from disk or every point is lost.
    """
    positions = {}
    prev_gray = None
    points = seed_points

    for frame_idx in frame_indices:
        frame = cv2.imread(_frame_path(frames_dir, frame_idx))
        if frame is None:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            new_points, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, points, None, **_LK_PARAMS)
            if new_points is None:
                break
            keep = status.reshape(-1).astype(bool)
            points = new_points[keep]
            if len(points) == 0:
                break

        median = np.median(points.reshape(-1, 2), axis=0)
        positions[frame_idx] = (float(median[0]), float(median[1]))
        prev_gray = gray

    return positions


# How far (as a fraction of the *validating* anchor's own fitted blade
# length) a direction's tracked landing position may drift from that
# anchor's true position before the direction is declined entirely --
# mirrors reacquire.RETRACK_MAX_DRIFT_FRAC's already-proven
# validate-against-the-known-good-anchor philosophy, applied to a tracked
# hilt point instead of a whole re-tracked mask. Confirmed appropriate on
# the real job: all four directions tested (forward/backward x two
# objects) landed 20-29px from their true anchor, comfortably inside a
# 25-62px cap at this fraction.
HILT_TRACK_MAX_DRIFT_FRAC = 0.3


def _track_direction(frames_dir, all_frame_indices, start_frame, start_hilt,
                      validate_frame, validate_hilt, validate_length):
    """Seed at `start_frame`/`start_hilt`, track sequentially through
    every frame between `start_frame` and `validate_frame` (inclusive),
    and validate the landing position at `validate_frame` against the
    known-good `validate_hilt` -- must land within
    `HILT_TRACK_MAX_DRIFT_FRAC * validate_length` px.

    Returns the full {frame_idx: (x, y)} dict (covering every frame from
    `start_frame` to `validate_frame`) if validated, or None if seeding
    found no usable texture, tracking was lost before reaching
    `validate_frame`, or validation failed.
    """
    lo, hi = sorted((start_frame, validate_frame))
    path = [f for f in all_frame_indices if lo <= f <= hi]
    if start_frame == hi:
        path = list(reversed(path))
    # path now starts at start_frame and ends at validate_frame

    first_frame = cv2.imread(_frame_path(frames_dir, start_frame))
    if first_frame is None:
        return None
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    seed_points = _seed_features(first_gray, start_hilt)
    if seed_points is None:
        return None

    positions = _track_points_sequential(frames_dir, path, seed_points)
    if validate_frame not in positions:
        return None  # lost tracking (or ran out of frames) before reaching the far anchor

    landing = positions[validate_frame]
    drift = float(np.hypot(landing[0] - validate_hilt[0], landing[1] - validate_hilt[1]))
    if drift > HILT_TRACK_MAX_DRIFT_FRAC * validate_length:
        return None

    return positions


def track_hilt_through_run(
    frames_dir, frame_indices, run_start_frame, run_end_frame,
    before_frame, before_hilt, before_length,
    after_frame, after_hilt, after_length,
):
    """Recover per-frame hilt (x, y) positions for one tracked object
    through [run_start_frame, run_end_frame] (inclusive) by tracking
    forward from before_hilt (known-good, at before_frame) and backward
    from after_hilt (known-good, at after_frame), each validated against
    the *other* side's known-good anchor (see `_track_direction`).

    Where both directions validate, they are blended by
    `t = (frame - before_frame) / (after_frame - before_frame)`:
    `(1 - t) * forward + t * backward` -- at t=0 (the before_frame end)
    this is 100% the forward track, which started there and has had zero
    distance to drift; at t=1 it's 100% the backward track, for the same
    reason at its own end. Where only one direction validates, it alone
    is used. Where neither validates, returns {}.

    Returns {frame_number: (x, y)} for every frame in
    [run_start_frame, run_end_frame] a validated estimate exists for.
    """
    run_frames = [f for f in frame_indices if run_start_frame <= f <= run_end_frame]
    if not run_frames:
        return {}

    forward = _track_direction(
        frames_dir, frame_indices, before_frame, before_hilt, after_frame, after_hilt, after_length,
    )
    backward = _track_direction(
        frames_dir, frame_indices, after_frame, after_hilt, before_frame, before_hilt, before_length,
    )

    if forward is None and backward is None:
        return {}

    span = after_frame - before_frame
    result = {}
    for f in run_frames:
        fwd = forward.get(f) if forward is not None else None
        bwd = backward.get(f) if backward is not None else None
        if fwd is not None and bwd is not None:
            t = (f - before_frame) / span
            result[f] = ((1 - t) * fwd[0] + t * bwd[0], (1 - t) * fwd[1] + t * bwd[1])
        elif fwd is not None:
            result[f] = fwd
        elif bwd is not None:
            result[f] = bwd
    return result


def compute_hilt_overrides(
    frames_dir, masks_dir_a, masks_dir_b, motion_path_a, motion_path_b,
    exclude_frame_ranges=(),
):
    """For every cross-object overlap run (see blade._find_overlap_runs)
    not covered by `exclude_frame_ranges` (retrack_overlap_runs'
    already-validated resolved_ranges) and with both anchors present,
    attempt to recover each object's hilt position via
    `track_hilt_through_run`.

    Returns (hilt_overrides_a, hilt_overrides_b): two
    {frame_number: (x, y)} dicts (one per object), merged across every
    run processed. A run/object `track_hilt_through_run` couldn't
    validate simply contributes nothing to that dict -- no exception, no
    partial/unvalidated data.
    """
    motion_a = load_motion(motion_path_a)
    motion_b = load_motion(motion_path_b)
    frame_indices = mask_frame_indices(masks_dir_a)
    runs = _find_overlap_runs(masks_dir_a, masks_dir_b, motion_a, motion_b)

    overrides_a = {}
    overrides_b = {}
    for run_start, run_end, before, after, _max_iou in runs:
        if before is None or after is None:
            continue
        # Every frame strictly between the two anchors, not just the
        # narrower [run_start, run_end] IoU-overlap span -- confirmed on
        # real footage, a frame just below the overlap threshold (too
        # contaminated to trust as an anchor, see blade's
        # CROSS_OBJECT_ANCHOR_IOU_FRAC) but not yet part of the detected
        # run was left with its own uncorrected raw fit, visibly
        # diverging from the real blade right as contact began.
        # track_hilt_through_run already tracks this whole span
        # internally (forward/backward between the anchors) regardless;
        # this just stops throwing away the marginal frames' results.
        start_frame, end_frame = frame_indices[before + 1], frame_indices[after - 1]
        if any(start_frame <= ex_end and end_frame >= ex_start for ex_start, ex_end in exclude_frame_ranges):
            continue
        before_frame, after_frame = frame_indices[before], frame_indices[after]

        overrides_a.update(track_hilt_through_run(
            frames_dir, frame_indices, start_frame, end_frame,
            before_frame, tuple(motion_a["hilt"][before]), float(motion_a["length"][before]),
            after_frame, tuple(motion_a["hilt"][after]), float(motion_a["length"][after]),
        ))
        overrides_b.update(track_hilt_through_run(
            frames_dir, frame_indices, start_frame, end_frame,
            before_frame, tuple(motion_b["hilt"][before]), float(motion_b["length"][before]),
            after_frame, tuple(motion_b["hilt"][after]), float(motion_b["length"][after]),
        ))

    return overrides_a, overrides_b
