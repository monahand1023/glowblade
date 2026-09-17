"""Per-frame blade geometry fitting from a binary mask, and the "motion"
pipeline stage that turns a job's tracked masks into a ``motion.npz``
artifact.

Pure numpy (plus scipy.ndimage for connected-component labeling), no
cv2/torch dependency, so it is cheap to unit-test in isolation from
tracking and rendering. ``compute_motion`` runs as its own stage between
tracking and rendering (see ``runner.py``) so both later phases can *read*
it:

- the visual phase rebuilds the blade as a capsule along ``axis`` between
  ``hilt`` and ``tip`` instead of tracing the raw mask silhouette, and uses
  the per-frame motion for directional motion blur;
- the audio phase drives swings from ``tip_speed``/``angular_speed`` instead
  of the mask centroid, which barely moves when a blade pivots in place.
"""

import logging
import os
from typing import NamedTuple

import numpy as np
from scipy import ndimage


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
    """Shape-only tip/hilt guess: the narrower end is the tip.

    This is a fine, honest thing for a single-frame heuristic to report --
    many lightsaber props (a sword: wide guard tapering to a point) really
    do taper narrower at the tip -- but it is not, by itself, the
    pipeline's tip/hilt decision. It is inverted for a bat-like object
    (thin handle, thick barrel), and a single frame has no way to tell the
    two cases apart from shape alone. `compute_motion` (via
    `_orient_by_motion`) decides tip/hilt for a whole clip from a
    physical signal instead -- which endpoint actually travels farther --
    and only falls back to this shape guess when that signal itself isn't
    trustworthy (too few frames, or a static clip with no motion to read).

    Compare the perpendicular extent of the mask's points within the outer
    `frac` of the axis span at each end and call the narrower end the tip.
    This can't be decided from a single frame's axis direction alone,
    which is why it is a separate, directly testable function -- swap it
    out here if a better per-frame heuristic is found.

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


def _largest_component(mask):
    """`mask`, reduced to its largest 8-connected blob -- dropping any
    smaller, disconnected ones.

    Measured on real footage: SAM2's per-frame video-tracking mask is
    usually one clean blob, but on a noisy frame (fast motion, an
    unrelated object nearby) it can include a second, disconnected chunk
    of foreground -- as little as a single stray pixel, in one case a
    fencer's body cord picked up 100+ px from the actual blade. That
    single far-away pixel doubled `fit_blade`'s reported length: PCA's
    second moment gives outsized leverage to points far from the
    centroid, so a speck under 0.1% of the mask's area can dominate the
    fitted axis. `detect.py` already treats a shattered, many-component
    mask as a sign of a bad segmentation at proposal time (see
    `_speckle_count`); this is the same principle applied per-frame,
    right before the geometry that `render_glow`'s `blade_extend`
    extrapolates from.
    """
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    largest_label = 1 + int(np.argmax(sizes))
    return labeled == largest_label


def fit_blade(mask, taper_frac=1.0 / 3.0, width_bins=20):
    """Fit blade geometry from a binary mask.

    Method: restrict to the mask's largest connected component (see
    `_largest_component`), then PCA over its points gives the long axis;
    projecting all points onto that axis gives the two endpoints (min/max
    projection) and the perpendicular spread gives the width;
    `classify_tip_by_taper` disambiguates which endpoint is the tip.

    Returns None when the mask has no foreground pixels at all.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    mask = _largest_component(mask)
    ys, xs = np.nonzero(mask)

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


# ---------------------------------------------------------------------------
# Sequence-level tip/hilt orientation.
#
# `fit_blade` (via `classify_tip_by_taper`) picks a tip from a single
# frame's shape alone: the narrower end. That is correct for a sword (wide
# guard tapering to a point) but backwards for a bat-like object (thin
# handle, thick barrel) -- the real regression this module was rewritten
# to fix. A single frame has no way to tell those two cases apart from
# shape; the whole sequence does, physically: in a swing the tip travels
# far more than the hilt, which sits near the pivot. `compute_motion` uses
# that -- decided once for the whole clip, not re-guessed per frame -- and
# only falls back to the per-frame taper guess when the motion signal
# itself isn't trustworthy.
# ---------------------------------------------------------------------------

# Fewer valid (non-None) frames than this and there isn't enough sequence
# to measure a path length from at all.
_MIN_VALID_FRAMES_FOR_MOTION = 2
# Below this many pixels of total travel, the longer-travelling endpoint
# isn't distinguishable from sensor/PCA noise -- treat the clip as static.
_MIN_TOTAL_PATH_PX = 1.0
# The longer track must have travelled at least this many times farther
# than the shorter one to be trusted as a real tip/hilt signal, rather
# than e.g. a pure translation (no rotation) where both ends of a rigid
# object move by the same amount and travelled-distance can't
# disambiguate them at all.
_MIN_PATH_RATIO = 1.5


def _track_endpoints(geometries):
    """Pair each frame's two (unordered) endpoints -- `geo.tip` and
    `geo.hilt`, whichever `fit_blade`/taper happened to call them -- to
    the nearer of the previous valid frame's two tracked positions, by
    nearest-neighbour. This builds two consistent point tracks across the
    sequence instead of inheriting the per-frame taper call's arbitrary
    (and sometimes flip-flopping) labelling.

    Returns `(path_a, path_b, assignments, n_valid)`: the two tracks'
    total path lengths, a per-frame `(point_for_a, point_for_b)` tuple (or
    None where `geometries[i]` is None), and the count of valid frames
    seen. Frames with no geometry (NaN gaps) are skipped when pairing --
    the next valid frame is compared against the last valid frame's
    positions, not a nonexistent intermediate one.
    """
    n = len(geometries)
    assignments = [None] * n
    path_a = 0.0
    path_b = 0.0
    prev_a = prev_b = None
    n_valid = 0

    for i, geo in enumerate(geometries):
        if geo is None:
            continue
        n_valid += 1
        p1 = np.array(geo.tip, dtype=np.float64)
        p2 = np.array(geo.hilt, dtype=np.float64)

        if prev_a is None:
            # Seed the two tracks arbitrarily on the first valid frame --
            # which physical end is "a" vs "b" doesn't matter, only that
            # it stays consistent from here on.
            a, b = p1, p2
        else:
            cost_keep = np.linalg.norm(p1 - prev_a) + np.linalg.norm(p2 - prev_b)
            cost_swap = np.linalg.norm(p2 - prev_a) + np.linalg.norm(p1 - prev_b)
            if cost_swap < cost_keep:
                a, b = p2, p1
            else:
                a, b = p1, p2
            path_a += float(np.linalg.norm(a - prev_a))
            path_b += float(np.linalg.norm(b - prev_b))

        assignments[i] = (a, b)
        prev_a, prev_b = a, b

    return path_a, path_b, assignments, n_valid


def _decide_tip_track(path_a, path_b, n_valid):
    """Return 'a' or 'b' for whichever endpoint track travelled farther,
    or None when the motion signal isn't trustworthy enough to say which
    physical end is the tip -- see `_orient_by_motion`, which still uses
    the (already continuous) `a`/`b` tracks even then.

    None covers: fewer than two valid frames to measure a path from; both
    tracks essentially static (a genuinely static clip, where the two ends
    are indistinguishable from position alone); or the two paths too close
    in length to call decisively. That last case is not rare -- confirmed
    on real fencing footage, a thrust translates the whole blade with the
    arm rather than pivoting it about a planted hilt, so the hilt end can
    legitimately travel nearly as far as the tip over a whole clip (a
    measured 9622px vs 8568px, ratio 1.12) with no reliable "tip travels
    farther" signal at all.
    """
    if n_valid < _MIN_VALID_FRAMES_FOR_MOTION:
        return None
    longer, shorter = (path_a, path_b) if path_a >= path_b else (path_b, path_a)
    if longer < _MIN_TOTAL_PATH_PX:
        return None
    ratio = longer / max(shorter, 1e-9)
    if ratio < _MIN_PATH_RATIO:
        return None
    return "a" if path_a > path_b else "b"


def _relabel_by_track(geometries, assignments, tip_track):
    """Rewrite `tip`/`hilt`/`axis`/`angle` on every frame to match
    `tip_track` ('a' or 'b', as decided by `_decide_tip_track`), leaving
    `centroid`/`length`/`width` untouched -- swapping which of two fixed
    points is called "tip" doesn't change the distance between them, the
    reported width, or the centroid."""
    relabeled = []
    for geo, assignment in zip(geometries, assignments, strict=False):
        if geo is None or assignment is None:
            relabeled.append(geo)
            continue

        a, b = assignment
        tip, hilt = (a, b) if tip_track == "a" else (b, a)

        axis_vec = tip - hilt
        norm = np.linalg.norm(axis_vec)
        if norm > 0:
            axis_vec = axis_vec / norm
        else:
            axis_vec = np.array(geo.axis, dtype=np.float64)
        angle = float(np.arctan2(axis_vec[1], axis_vec[0]))

        relabeled.append(geo._replace(
            tip=(float(tip[0]), float(tip[1])),
            hilt=(float(hilt[0]), float(hilt[1])),
            axis=(float(axis_vec[0]), float(axis_vec[1])),
            angle=angle,
        ))
    return relabeled


def _orient_by_motion(geometries):
    """Decide tip vs hilt for the whole sequence from where the motion
    actually is, rather than trusting each frame's independent taper
    guess (see the module-level comment above `_MIN_VALID_FRAMES_FOR_MOTION`).

    As a side effect this also removes frame-to-frame tip/hilt flip-flops:
    a per-frame-only shape guess can flip near a symmetric taper (a tie
    that `classify_tip_by_taper` breaks by picking the minimum-projection
    end, whose *physical* identity isn't guaranteed stable frame to frame
    since it follows PCA's arbitrary eigenvector sign) -- exactly what
    `glow.py`'s `_stabilize_tip_hilt` has been papering over downstream.
    Deciding once, globally, and applying it to every frame can't flip.

    That holds even when `_decide_tip_track` can't confidently say *which*
    end is the tip (returns None): `_track_endpoints` already built two
    continuous, non-flipping physical tracks ("a" and "b") by
    nearest-neighbour, entirely independent of that decision, so this
    picks 'a' -- an arbitrary but *fixed* choice, applied via the same
    `_relabel_by_track` the decisive case uses -- rather than leaving each
    frame's own taper guess in place. Confirmed as a real, not
    theoretical, gap on real footage: a tracked object's tip/hilt fully
    swapped ends between two adjacent frames whose actual mask barely
    changed shape, because the previous "leave it alone" fallback let
    per-frame taper's flip straight through. A downstream consumer that
    reads tip/hilt across multiple frames (e.g. `suppress_overlap_bleed`,
    or `tip_speed`/`angular_speed`) has no way to know a flip happened;
    fixing it here, once, before anything downstream ever sees the data,
    is far more robust than expecting every consumer to defend against it
    independently.
    """
    path_a, path_b, assignments, n_valid = _track_endpoints(geometries)
    tip_track = _decide_tip_track(path_a, path_b, n_valid) or "a"
    return _relabel_by_track(geometries, assignments, tip_track)


# ---------------------------------------------------------------------------
# Per-frame mask I/O
# ---------------------------------------------------------------------------
# track_object (pipeline/track.py) writes one boolean mask per frame;
# compute_motion (below) and render_glow (pipeline/glow.py) read them back.
# This is the single place that owns the on-disk format and the
# `{idx:05d}` frame-index naming, so no caller open-codes a mask filename
# or an `np.save`/`np.load` call of its own.
#
# A tracked blade mask is a thin, mostly-empty band (on the order of 2%
# foreground pixels on real footage), so `np.savez_compressed` beats plain
# `np.save` by two to three orders of magnitude -- far better than
# `np.packbits`, which only gets the fixed 8x of bit-packing and doesn't
# exploit the sparsity. That's what keeps a render's masks from being the
# dominant disk cost (see docs/design-notes.md, "Two storage
# trade-offs", for the measurements).
#
# Loading also transparently accepts the older, uncompressed `.npy` format
# written by earlier versions of `track_object`, so a job directory created
# before this change still loads correctly. New masks are never written in
# the old format.

_MASK_EXT = ".npz"
_MASK_EXT_LEGACY = ".npy"


def _mask_path(masks_dir, frame_idx, ext):
    return os.path.join(masks_dir, f"{frame_idx:05d}{ext}")


def _resolve_mask_path(masks_dir, frame_idx):
    """Return `(path, is_legacy)` for whichever format is on disk for
    `frame_idx`, or `(None, False)` if neither is present. Prefers the
    compressed format on the (should-never-happen, since we never write
    the legacy format) chance both exist."""
    npz_path = _mask_path(masks_dir, frame_idx, _MASK_EXT)
    if os.path.exists(npz_path):
        return npz_path, False
    legacy_path = _mask_path(masks_dir, frame_idx, _MASK_EXT_LEGACY)
    if os.path.exists(legacy_path):
        return legacy_path, True
    return None, False


def _load_mask_file(path, is_legacy):
    if is_legacy:
        return np.load(path)
    with np.load(path) as data:
        return data["mask"]


def save_mask(masks_dir, frame_idx, mask):
    """Write a single frame's boolean tracking mask to `masks_dir`,
    compressed, as the `{idx:05d}.npz` file this module's readers expect.
    Creates `masks_dir` if it doesn't already exist."""
    os.makedirs(masks_dir, exist_ok=True)
    np.savez_compressed(_mask_path(masks_dir, frame_idx, _MASK_EXT), mask=mask)


def load_mask(masks_dir, frame_idx):
    """Load the mask written by `save_mask` for `frame_idx`, falling back
    to the legacy uncompressed `.npy` format when present instead. Raises
    `FileNotFoundError` if neither is present."""
    path, is_legacy = _resolve_mask_path(masks_dir, frame_idx)
    if path is None:
        raise FileNotFoundError(f"No mask for frame {frame_idx} in {masks_dir}")
    return _load_mask_file(path, is_legacy)


def load_mask_optional(masks_dir, frame_idx):
    """Like `load_mask`, but returns `None` instead of raising when no mask
    file exists for `frame_idx` -- for callers such as render_glow, where a
    frame the tracker never produced a mask for (object lost) is expected,
    not an error."""
    path, is_legacy = _resolve_mask_path(masks_dir, frame_idx)
    if path is None:
        return None
    return _load_mask_file(path, is_legacy)


def mask_frame_indices(masks_dir):
    """Sorted list of frame indices with a saved mask present in
    `masks_dir`, in either format. Lets compute_motion discover exactly
    which frames track_object produced without hardcoding an extension."""
    indices = set()
    for fname in os.listdir(masks_dir):
        name, ext = os.path.splitext(fname)
        if ext in (_MASK_EXT, _MASK_EXT_LEGACY) and name.isdigit():
            indices.add(int(name))
    return sorted(indices)


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


# A fitted shape needs at least this much length-to-width ratio to be
# accepted as a blade rather than whatever else got fitted -- a body, a
# shield, a patch of cloth. Lives here (not in detect.py, which imports it)
# because elongation is a property of the geometry itself, used at both the
# point `detect.py` proposes a candidate and every point downstream
# (`inspect_job.py`'s diagnostics, `runner.py`'s in-flight quality check)
# asks the same question of an already-fitted BladeGeometry.
#
# 3.5 was too permissive, measured: a *standing person* fits at 3.7, which is
# how a sword clip came to propose the swordsman rather than his sword. Correct
# proposals, once every candidate is scored rather than the first acceptable
# one, come in far higher -- 15.1 on the baseball clip and 21.0 on a golf club.
# So the bar is set where a human body cannot reach it, and the cost (at
# proposal time) is that genuinely ambiguous footage returns None. That is the
# right trade there: None means "click it yourself", which is what the user
# would have done anyway, while a confident wrong guess costs them a full
# render to discover.
MIN_ELONGATION = 6.0


# How much of a tracked object's frames must be below MIN_ELONGATION before
# "sometimes blob-shaped" becomes "actually a blob" -- not a single frame,
# since a real blade can legitimately foreshorten toward the camera for a
# frame or two mid-swing, but a sustained majority is a different object
# entirely. Shared by `inspect_job.py`'s diagnostics and `runner.py`'s
# in-flight warning so the two can't quietly drift onto different bars for
# what is, underneath, the same question.
LOW_ELONGATION_FRAC_THRESHOLD = 0.3


def elongation_stats(motion):
    """Summarize how blade-shaped `motion` (a dict loaded by `load_motion`)
    looks across its tracked frames: `(mean_elongation, low_elongation_frac)`,
    where the latter is the fraction of frames with elongation (length /
    width) below `MIN_ELONGATION`.

    Computed only over frames with a valid, nonzero width -- a NaN width
    (never tracked) or a zero width (a real degenerate case `fit_blade` can
    produce, e.g. a single-row mask) makes elongation undefined rather than
    bad, so those frames are excluded instead of divided-by-zero or counted
    as an anomaly either way. Returns `(None, None)` if no frame qualifies.
    """
    width, length = motion["width"], motion["length"]
    valid = ~np.isnan(width) & (width > 0)
    if not valid.any():
        return None, None
    elongation = length[valid] / width[valid]
    return float(elongation.mean()), float((elongation < MIN_ELONGATION).sum() / valid.sum())


def _mask_iou(mask_a, mask_b):
    """Intersection-over-union of two boolean masks, 0.0 if both are empty.

    Lives here (not `vision_detect.py`, the original home of an identical
    function) because it's pure numpy with no cv2 dependency, same as
    everything else in this module -- and `suppress_overlap_bleed` below
    needs it without importing `vision_detect` (which itself imports
    `blade`, so the reverse import would be circular). `vision_detect.py`
    and `reacquire.py` both import this copy rather than keeping their own.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union else 0.0


# A different real-footage failure mode than the disconnected-speck case
# above: when two tracked objects' blades visually touch/cross without
# SAM2 losing their identities outright (that full, sustained, symmetric
# merge is `reacquire.py`'s `reconcile_pair` concern), one object's
# per-frame mask can still bleed into a *connected* extension covering
# part of the other object's blade for a stretch of frames. fit_blade then
# reports an honest but wrong PCA fit over that unioned shape -- there is
# no disconnected speck for `_largest_component` to drop.
#
# An earlier version of this guard compared each object's fitted length
# only against its own recent history (a growth-percentage cap). That
# could not be made reliable: on real fencing footage a blade's own fitted
# length legitimately swings from ~50px to ~420px within a few dozen
# frames as it points toward and away from the camera, which looks
# identical, from inside one object's own length history, to genuine mask
# corruption. The signal that actually separates the two, confirmed
# directly against real footage: the two objects' *masks* only overlap at
# all during a genuine bleed event (IoU up to 0.61 there) and are
# perfectly disjoint (IoU 0.0) during every one of the fast-foreshortening
# frames the growth heuristic falsely flagged. Below this threshold, two
# blade masks brushing past each other without actually bleeding measured
# at most 0.024 IoU on the same clip -- comfortably below this bar.
CROSS_OBJECT_OVERLAP_IOU_THRESHOLD = 0.1


# How much stricter than CROSS_OBJECT_OVERLAP_IOU_THRESHOLD a frame must be
# to serve as an interpolation *anchor*, rather than merely to stay
# uncorrected. Confirmed on real footage: IoU climbs gradually into a real
# overlap rather than jumping straight from zero (0.000 -> 0.007 -> 0.024
# -> 0.087 over 7 frames before crossing 0.1), so the frame immediately
# below the overlap threshold can already be a few frames into the same
# contamination -- not genuinely separated, just not (yet) over the bar
# that triggers a correction. Anchoring an interpolation there inherits
# that drift at exactly the point a wrong value matters most: the observed
# visual result was a blade whose glow started measurably off from the
# real hand and only converged onto it partway through the run. Every
# frame below this stricter bar measured a real, stable 0.000-0.008 IoU on
# the same footage -- comfortably clean.
CROSS_OBJECT_ANCHOR_IOU_FRAC = 0.2


def _good_frame_mask(motion_a, motion_b, ious, anchor_iou_threshold):
    """A frame is usable as an interpolation anchor when both objects have
    a real (non-NaN) fit and the masks are clean well below the
    overlap-detection threshold -- see CROSS_OBJECT_ANCHOR_IOU_FRAC."""
    valid = ~np.isnan(motion_a["length"]) & ~np.isnan(motion_b["length"])
    return valid & (ious <= anchor_iou_threshold)


def _interpolate_row(motion, i, before, after, t):
    """Set row `i` of `motion` by interpolating fraction `t` of the way
    from row `before` to row `after`, for `centroid`/`hilt`/`tip`/`width`,
    then re-deriving `axis`/`length`/`angle` from the interpolated
    `tip`/`hilt` so the geometry stays internally consistent (axis really
    is the unit vector from hilt to tip, length really is their
    distance) -- rather than blending all seven fields independently,
    which could disagree with each other."""
    for field in ("centroid", "hilt", "tip", "width"):
        motion[field][i] = (1 - t) * motion[field][before] + t * motion[field][after]
    axis_vec = motion["tip"][i] - motion["hilt"][i]
    norm = np.linalg.norm(axis_vec)
    motion["length"][i] = norm
    if norm > 0:
        motion["axis"][i] = axis_vec / norm
        motion["angle"][i] = np.arctan2(axis_vec[1], axis_vec[0])


def _freeze_row(motion, i, anchor):
    for field in motion:
        motion[field][i] = motion[field][anchor]


class OverlapRun(NamedTuple):
    """One run of consecutive frames where two tracked objects' masks
    overlapped past a threshold -- see `_find_overlap_runs`.

    `run_start`/`run_end` (inclusive) and `before`/`after` are *array*
    indices into `mask_frame_indices(masks_dir_a)`/the motion arrays, not
    raw frame numbers -- callers needing a frame number convert via
    `frame_indices[idx]`, same as `suppress_overlap_bleed` does internally.
    `before`/`after` are `None` when no clean frame exists on that side
    (the run starts at frame 0, or runs through the end of the clip).
    """

    run_start: int
    run_end: int
    before: int | None
    after: int | None
    max_iou: float


def _find_overlap_runs(masks_dir_a, masks_dir_b, motion_a, motion_b,
                        iou_threshold=CROSS_OBJECT_OVERLAP_IOU_THRESHOLD,
                        anchor_iou_threshold=None):
    """Every `OverlapRun` in `masks_dir_a`/`masks_dir_b`'s shared frame
    range, with the nearest clean anchor frame on each side (see
    `CROSS_OBJECT_ANCHOR_IOU_FRAC`) -- the shared detection step behind
    both `suppress_overlap_bleed`'s geometry interpolation and
    `reacquire.retrack_overlap_runs`' independent re-tracking attempt, so
    the two can't quietly disagree about where a run starts, ends, or
    which frames are trustworthy anchors.
    """
    if anchor_iou_threshold is None:
        anchor_iou_threshold = iou_threshold * CROSS_OBJECT_ANCHOR_IOU_FRAC
    frame_indices = mask_frame_indices(masks_dir_a)
    n = len(frame_indices)

    ious = np.array([
        _mask_iou(load_mask(masks_dir_a, frame_idx), load_mask(masks_dir_b, frame_idx))
        for frame_idx in frame_indices
    ])
    overlapping = ious > iou_threshold
    good = _good_frame_mask(motion_a, motion_b, ious, anchor_iou_threshold)
    good_indices = np.flatnonzero(good)

    runs = []
    i = 0
    while i < n:
        if not overlapping[i]:
            i += 1
            continue
        run_start = i
        while i < n and overlapping[i]:
            i += 1
        run_end = i - 1

        earlier = good_indices[good_indices < run_start]
        later = good_indices[good_indices > run_end]
        before = int(earlier[-1]) if len(earlier) else None
        after = int(later[0]) if len(later) else None

        runs.append(OverlapRun(
            run_start=run_start, run_end=run_end, before=before, after=after,
            max_iou=float(ious[run_start:run_end + 1].max()),
        ))
    return runs


def suppress_overlap_bleed(motion_path_a, masks_dir_a, motion_path_b, masks_dir_b,
                            iou_threshold=CROSS_OBJECT_OVERLAP_IOU_THRESHOLD,
                            anchor_iou_threshold=None, exclude_frame_ranges=()):
    """Patch two already-written motion.npz files in place: for every run of
    consecutive frames where the two tracked objects' raw masks overlap
    past `iou_threshold` (see `_find_overlap_runs`), replace *both*
    objects' fitted geometry for that run rather than trusting a per-frame
    fit_blade result computed from a mask that may have bled into the
    other object's blade.

    Prefers to *interpolate* linearly between the last good frame before
    the run and the first good frame after it, rather than freezing at a
    single value for the run's whole duration -- confirmed on real
    footage, a real hilt travels 150-235px across a ~3s run of overlapping
    frames (a real fencing exchange, not an instant touch), so a frozen
    geometry visibly detaches from the hand holding it ("the blade
    disembodies and floats") long before the run ends. Interpolating keeps
    both endpoints exactly right and approximates the motion between them
    -- not real tracking, but far closer to it than a dead hold, and either
    endpoint missing (the run starts at frame 0, or never ends before the
    clip does) falls back to freezing at whichever single endpoint exists.

    Runs after `compute_motion` has produced both objects' motion.npz (it
    patches, not produces, so it needs their finished output), after
    `reacquire.retrack_overlap_runs` has had a chance to replace a run with
    real independently-tracked masks instead (this is the fallback for
    whatever that couldn't fix), and only for a 2-object job -- see
    `runner.run_pipeline_multi`. Both objects are corrected together, not
    just whichever one looks more corrupted -- with the masks actually
    overlapping, neither per-frame fit can be trusted, and guessing which
    one is "more wrong" isn't necessary when correcting both is cheap and
    safe.

    `anchor_iou_threshold` (default `iou_threshold * CROSS_OBJECT_ANCHOR_IOU_FRAC`)
    is the stricter bar an anchor frame must clear -- see that constant.

    `exclude_frame_ranges` (a list of `(start_frame, end_frame)` tuples,
    inclusive) skips any detected run overlapping one entirely -- for runs
    `reacquire.retrack_overlap_runs` already resolved with a validated
    independent re-track for *both* objects. Two blades in genuine,
    correctly-tracked contact still show high mask IoU (that's what real
    contact looks like), so without this, re-detecting from the
    already-correct masks would "fix" a run that was never actually
    wrong, overwriting an accurate re-track with a worse interpolated
    approximation.

    Returns the number of frames patched (interpolated or held).
    """
    logger = logging.getLogger(__name__)
    motion_a = load_motion(motion_path_a)
    motion_b = load_motion(motion_path_b)
    frame_indices = mask_frame_indices(masks_dir_a)
    runs = _find_overlap_runs(masks_dir_a, masks_dir_b, motion_a, motion_b, iou_threshold, anchor_iou_threshold)

    n_held = 0
    for run_start, run_end, before, after, max_iou in runs:
        start_frame, end_frame = frame_indices[run_start], frame_indices[run_end]
        if any(start_frame <= ex_end and end_frame >= ex_start for ex_start, ex_end in exclude_frame_ranges):
            continue

        for j in range(run_start, run_end + 1):
            if before is not None and after is not None:
                t = (frame_indices[j] - frame_indices[before]) / (frame_indices[after] - frame_indices[before])
                _interpolate_row(motion_a, j, before, after, t)
                _interpolate_row(motion_b, j, before, after, t)
            elif before is not None:
                _freeze_row(motion_a, j, before)
                _freeze_row(motion_b, j, before)
            elif after is not None:
                _freeze_row(motion_a, j, after)
                _freeze_row(motion_b, j, after)
            # else: no anchor at all -- nothing better than the raw fit.

        if before is not None or after is not None:
            n_held += run_end - run_start + 1
            logger.warning(
                "frames %d-%d: tracked objects' masks overlapped (IoU up to %.2f, cap %.2f) -- %s "
                "both objects' geometry%s",
                frame_indices[run_start], frame_indices[run_end], max_iou,
                iou_threshold,
                "interpolated" if before is not None and after is not None else "held",
                (
                    f" between frames {frame_indices[before]} and {frame_indices[after]}"
                    if before is not None and after is not None
                    else f" at frame {frame_indices[before if before is not None else after]}'s values"
                ),
            )

    if n_held:
        np.savez(motion_path_a, **motion_a)
        np.savez(motion_path_b, **motion_b)
    return n_held


def compute_motion(masks_dir, motion_out_path, taper_frac=1.0 / 3.0, width_bins=20, progress_cb=None):
    """The motion pipeline stage: fit blade geometry for every tracked
    frame and write it to `motion_out_path` (see `save_motion`).

    This runs as its own stage between tracking and rendering. Motion is a
    first-class artifact that both the visual phase (capsule
    reconstruction, directional motion blur) and the audio phase
    (tip/angular speed) need to *read* -- render_glow producing it as a
    side effect of drawing was the wrong shape once both consumers exist.

    Processes every mask frame present in `masks_dir` (via
    `mask_frame_indices`/`load_mask`, which accept both the compressed
    `.npz` format `track_object` writes and the legacy uncompressed `.npy`
    format), in frame-index order -- a frame whose mask is empty (object
    lost that frame) gets a None geometry, which `save_motion` turns into a
    NaN row.

    Each frame's tip/hilt is initially guessed per-frame by `fit_blade`
    (shape/taper only), then `_orient_by_motion` re-decides tip vs hilt
    once for the whole sequence from which endpoint actually travelled
    farther -- see that function's docstring for why shape alone isn't
    enough (it's inverted for a bat) and when the motion-based decision
    falls back to the per-frame guess.

    Every valid per-frame fit is logged at DEBUG (length/width/centroid/
    angle) so a real run can be replayed from logs alone. This function
    only ever sees one object's masks, so it can't tell a mask that's
    bled into a nearby tracked object's blade from genuine fast motion --
    see `suppress_overlap_bleed`, which runs afterward with both objects'
    output in hand and can.

    Returns `(n_frames, n_with_blade)` so the caller can tell a good track
    from one that found nothing before paying for the glow stage -- see
    `runner._require_usable_track`.
    """
    logger = logging.getLogger(__name__)

    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    frame_indices = mask_frame_indices(masks_dir)
    n = len(frame_indices)
    geometries = []
    for i, frame_idx in enumerate(frame_indices):
        mask = load_mask(masks_dir, frame_idx)
        geo = fit_blade(mask, taper_frac=taper_frac, width_bins=width_bins)
        if geo is not None:
            logger.debug(
                "frame %d: length=%.1f width=%.1f centroid=(%.1f, %.1f) angle=%.2f",
                frame_idx, geo.length, geo.width, geo.centroid[0], geo.centroid[1], geo.angle,
            )
        geometries.append(geo)
        report((i + 1) / n * 100, f"frame {i + 1}/{n}")

    geometries = _orient_by_motion(geometries)

    save_motion(motion_out_path, geometries)
    if n == 0:
        report(100, "no masks found")
    return n, sum(1 for g in geometries if g is not None)


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
