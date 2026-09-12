"""Per-frame blade geometry fitting from a binary mask, and the "motion"
pipeline stage that turns a job's tracked masks into a ``motion.npz``
artifact.

Pure numpy, no cv2/torch dependency, so it is cheap to unit-test in
isolation from tracking and rendering. ``compute_motion`` runs as its own
stage between tracking and rendering (see ``runner.py``) so both later
phases can *read* it:

- the visual phase rebuilds the blade as a capsule along ``axis`` between
  ``hilt`` and ``tip`` instead of tracing the raw mask silhouette, and uses
  the per-frame motion for directional motion blur;
- the audio phase drives swings from ``tip_speed``/``angular_speed`` instead
  of the mask centroid, which barely moves when a blade pivots in place.
"""

import os
from typing import NamedTuple, Optional

import numpy as np


class BladeGeometry(NamedTuple):
    """Fitted geometry of a blade-shaped mask for a single frame.

    All coordinates are (x, y) in pixel space; ``axis`` is a unit vector
    oriented from ``hilt`` to ``tip``; ``angle`` is ``atan2(axis[1],
    axis[0])`` of that oriented axis, in radians.
    """

    centroid: tuple
    axis: tuple
    tip: tuple
    hilt: tuple
    length: float
    width: float
    angle: float


def _axis_endpoint_extent(proj, perp, near_min, frac):
    """Perpendicular (max-min) extent of the points within `frac` of the
    axis's projection span, measured from the min-projection end if
    `near_min` else the max-projection end. 0.0 if the span or the region
    is empty."""
    lo, hi = proj.min(), proj.max()
    span = hi - lo
    if span <= 0:
        return 0.0
    if near_min:
        region = perp[proj <= lo + span * frac]
    else:
        region = perp[proj >= hi - span * frac]
    if len(region) == 0:
        return 0.0
    return float(region.max() - region.min())


def classify_tip_by_taper(proj, perp, frac=1.0 / 3.0):
    """Tip/hilt disambiguation heuristic.

    A lightsaber prop (a bat, a sword) tapers: it is wider at the hilt end
    than at the tip. Compare the perpendicular extent of the mask's points
    within the outer `frac` of the axis span at each end and call the
    narrower end the tip. This can't be decided from a single frame's axis
    direction alone, which is why it is a separate, directly testable
    function -- swap it out here if a better heuristic is found.

    Returns True if the tip is at the minimum-projection end, False if it's
    at the maximum-projection end. Ties (equal extents, e.g. no taper at
    all) resolve to the minimum end, arbitrarily but deterministically.
    """
    proj = np.asarray(proj, dtype=float)
    perp = np.asarray(perp, dtype=float)
    extent_min = _axis_endpoint_extent(proj, perp, near_min=True, frac=frac)
    extent_max = _axis_endpoint_extent(proj, perp, near_min=False, frac=frac)
    return extent_min <= extent_max


def _median_perpendicular_extent(proj, perp, n_bins=20):
    """Median perpendicular extent across bins along the axis.

    Using the median (rather than a single overall max-min) keeps the
    reported width representative of the blade's body even when one end
    is locally much wider (e.g. a bat's knob, or a sword's hilt guard).
    """
    lo, hi = proj.min(), proj.max()
    span = hi - lo
    if span <= 0:
        return float(perp.max() - perp.min()) if len(perp) else 0.0

    edges = np.linspace(lo, hi, n_bins + 1)
    bin_idx = np.clip(np.digitize(proj, edges) - 1, 0, n_bins - 1)
    extents = []
    for b in range(n_bins):
        vals = perp[bin_idx == b]
        if len(vals):
            extents.append(float(vals.max() - vals.min()))
    if not extents:
        return float(perp.max() - perp.min()) if len(perp) else 0.0
    return float(np.median(extents))


def fit_blade(mask, taper_frac=1.0 / 3.0, width_bins=20):
    """Fit blade geometry from a binary mask.

    Method: PCA over the mask's points gives the long axis; projecting all
    points onto that axis gives the two endpoints (min/max projection) and
    the perpendicular spread gives the width; `classify_tip_by_taper`
    disambiguates which endpoint is the tip.

    Returns None when the mask has no foreground pixels at all.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    points = np.stack([xs, ys], axis=1).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid

    if len(points) == 1:
        axis = np.array([1.0, 0.0])
    else:
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, np.argmax(eigvals)]
        norm = np.linalg.norm(axis)
        axis = axis / norm if norm > 0 else np.array([1.0, 0.0])

    perp_dir = np.array([-axis[1], axis[0]])
    proj = centered @ axis
    perp = centered @ perp_dir

    min_proj, max_proj = proj.min(), proj.max()
    end_min = centroid + min_proj * axis
    end_max = centroid + max_proj * axis

    if classify_tip_by_taper(proj, perp, frac=taper_frac):
        tip, hilt = end_min, end_max
    else:
        tip, hilt = end_max, end_min

    oriented_axis = tip - hilt
    oriented_norm = np.linalg.norm(oriented_axis)
    oriented_axis = oriented_axis / oriented_norm if oriented_norm > 0 else axis

    length = float(np.linalg.norm(tip - hilt))
    width = _median_perpendicular_extent(proj, perp, n_bins=width_bins)
    angle = float(np.arctan2(oriented_axis[1], oriented_axis[0]))

    return BladeGeometry(
        centroid=(float(centroid[0]), float(centroid[1])),
        axis=(float(oriented_axis[0]), float(oriented_axis[1])),
        tip=(float(tip[0]), float(tip[1])),
        hilt=(float(hilt[0]), float(hilt[1])),
        length=length,
        width=float(width),
        angle=angle,
    )


_FIELDS = ("centroid", "tip", "hilt", "axis", "length", "width", "angle")
_VECTOR_FIELDS = ("centroid", "tip", "hilt", "axis")


def save_motion(path, geometries):
    """Write a list of BladeGeometry (None entries allowed) to `path` as an
    .npz of parallel (N, ...) arrays, with NaN rows where a frame had no
    mask."""
    n = len(geometries)
    arrays = {}
    for field in _VECTOR_FIELDS:
        arrays[field] = np.full((n, 2), np.nan, dtype=np.float64)
    for field in _FIELDS:
        if field in _VECTOR_FIELDS:
            continue
        arrays[field] = np.full(n, np.nan, dtype=np.float64)

    for i, geo in enumerate(geometries):
        if geo is None:
            continue
        for field in _FIELDS:
            arrays[field][i] = getattr(geo, field)

    np.savez(path, **arrays)


def load_motion(path):
    """Load the .npz written by `save_motion` back into a dict of arrays."""
    with np.load(path) as data:
        return {k: data[k].copy() for k in data.files}


def compute_motion(masks_dir, motion_out_path, taper_frac=1.0 / 3.0, width_bins=20, progress_cb=None):
    """The motion pipeline stage: fit blade geometry for every tracked
    frame and write it to `motion_out_path` (see `save_motion`).

    This runs as its own stage between tracking and rendering. Motion is a
    first-class artifact that both the visual phase (capsule
    reconstruction, directional motion blur) and the audio phase
    (tip/angular speed) need to *read* -- render_glow producing it as a
    side effect of drawing was the wrong shape once both consumers exist.

    Processes every ``*.npy`` mask file present in `masks_dir`, in
    filename order (matching track_object's ``{idx:05d}.npy`` naming, one
    file per frame) -- a frame whose mask is empty (object lost that
    frame) gets a None geometry, which `save_motion` turns into a NaN row.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    mask_files = sorted(f for f in os.listdir(masks_dir) if f.endswith(".npy"))
    n = len(mask_files)
    geometries = []
    for i, fname in enumerate(mask_files):
        mask = np.load(os.path.join(masks_dir, fname))
        geometries.append(fit_blade(mask, taper_frac=taper_frac, width_bins=width_bins))
        report((i + 1) / n * 100, f"frame {i + 1}/{n}")

    save_motion(motion_out_path, geometries)
    if n == 0:
        report(100, "no masks found")


def wrap_axis_angle_delta(delta):
    """Wrap an axis-angle difference into (-pi/2, pi/2].

    An axis (a line, not a ray) is direction-agnostic modulo pi: angles
    that differ by exactly pi describe the same line. Wrapping into a
    half-pi-wide window means both ordinary angle wrap-around (e.g. just
    under +pi to just under -pi) and a hilt/tip relabeling flip at any
    orientation collapse to a small delta instead of a spurious spike.
    """
    return (np.asarray(delta) + np.pi / 2) % np.pi - np.pi / 2


def tip_speed(motion, fps):
    """Speed (px/sec) of the blade tip, frame to frame.

    NaN gaps (frames with no mask) are zeroed rather than propagated, so a
    single missing frame doesn't poison the rest of the output array.
    """
    tip = np.asarray(motion["tip"], dtype=np.float64)
    diffs = np.diff(tip, axis=0)
    dist = np.linalg.norm(diffs, axis=1)
    dist = np.nan_to_num(dist, nan=0.0)
    speed = np.zeros(len(tip), dtype=np.float64)
    speed[1:] = dist * fps
    return speed


def angular_speed(motion, fps):
    """Speed (rad/sec) of the blade's axis rotation, frame to frame.

    Catches pivot-in-place swings that barely move the centroid or tip.
    Deltas are wrapped mod pi (see `wrap_axis_angle_delta`) before scaling
    by fps, and NaN gaps are zeroed rather than propagated.
    """
    angle = np.asarray(motion["angle"], dtype=np.float64)
    diffs = np.diff(angle)
    wrapped = wrap_axis_angle_delta(diffs)
    wrapped = np.nan_to_num(wrapped, nan=0.0)
    speed = np.zeros(len(angle), dtype=np.float64)
    speed[1:] = np.abs(wrapped) * fps
    return speed
