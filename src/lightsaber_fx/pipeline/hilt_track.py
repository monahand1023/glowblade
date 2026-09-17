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
