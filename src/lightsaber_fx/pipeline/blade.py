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
    axis[0])`` of that oriented axis, in radians. ``bend``, when not
    ``None``, is a third control point (hilt, bend, tip form a quadratic
    Bezier) capturing real, measured bow during blade-on-blade contact --
    see ``_bend_offset`` and ``fit_blade``. Absent (``None``) on every
    ordinary frame; a default so every existing positional/keyword
    construction of this NamedTuple keeps working unchanged.
    """

    centroid: tuple
    axis: tuple
    tip: tuple
    hilt: tuple
    length: float
    width: float
    angle: float
    bend: tuple | None = None


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


# Threshold (px) above which fit_blade's own median midpoint-bin
# perpendicular offset is treated as real blade bow rather than PCA-fit
# noise. Calibrated against the entire real 506-frame job before this
# was implemented (see the design spec's Constants section): gated
# baseline noise ceiling was 3.2px (object 0) / 4.2px p99 (object 1),
# real signal 21.4-28.0px -- 8px sits with comfortable margin on both
# sides and produced zero false positives/negatives on that job's one
# real contact run.
BEND_SIGNIFICANCE_PX = 8


def _bend_offset(proj, perp, n_bins=20):
    """Median perpendicular offset of the points nearest the blade's
    midpoint projection -- the raw single-object signal for a candidate
    `bend` control point. Reuses `_median_perpendicular_extent`'s exact
    bin edges so the two stay consistent, but reports the **median**
    (not max-min extent) of a ~2-bin-wide window centered on the
    midpoint -- this exact window is what `BEND_SIGNIFICANCE_PX` was
    calibrated against; narrowing or widening it invalidates that
    calibration.

    Returns 0.0 if the axis span is degenerate or too few points fall in
    the window to trust a median from (fewer than 3) -- callers compare
    the *magnitude* of this value against `BEND_SIGNIFICANCE_PX`, and a
    same-signed false near-zero here is always safe (never registers as
    significant bow).
    """
    lo, hi = proj.min(), proj.max()
    span = hi - lo
    if span <= 0:
        return 0.0
    edges = np.linspace(lo, hi, n_bins + 1)
    mid = n_bins // 2
    lo_edge = edges[max(mid - 1, 0)]
    hi_edge = edges[min(mid + 1, n_bins)]
    window = perp[(proj >= lo_edge) & (proj <= hi_edge)]
    if len(window) < 3:
        return 0.0
    return float(np.median(window))


def _largest_component(mask, reference_point=None, max_jump_px=None):
    """`mask`, reduced to its most plausible 8-connected blob -- dropping
    any other, smaller-or-implausibly-located, disconnected ones.

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

    Picking the *largest* component alone isn't always enough: confirmed
    on real footage, one tracked object's mask carried a persistent
    secondary component for the majority of a whole clip -- almost
    certainly the fencer's own scoring cable, elongated and substantial
    enough to look blade-like -- comparable in size to the real blade
    throughout, and for a few separate, consecutive-frame stretches it
    briefly *exceeded* the real blade's own pixel count. Each time, the
    whole fitted blade snapped onto the cable's location instead:
    measured, the fitted centroid jumped 500+px in a single frame, then
    jumped back by a similar amount once the real blade regained the
    larger component -- against real per-frame blade motion in that same
    stretch of footage that never exceeded ~10px between frames (95th
    percentile 8.3px) outside these incidents. A wide, unambiguous gap.

    When `reference_point` (`(x, y)`, typically the previous frame's
    fitted centroid) is given: if the largest component's centroid is
    within `max_jump_px` (default `POSITION_GLITCH_JUMP_PX`) of it, it's
    used exactly as before. If not -- the largest component just jumped
    implausibly far -- this looks for a *smaller* component that IS
    within `max_jump_px` of `reference_point`, and uses that one
    instead. If no component is close to `reference_point` either, falls
    back to the largest component (matches the old behavior -- better
    than inventing a new guess when nothing looks trustworthy).
    `reference_point=None` (the default, and what every caller except
    `compute_motion`'s per-frame loop uses) always keeps the original
    largest-wins behavior.

    Whenever the two largest components are near-equal in size (second
    at least 90% of the largest's pixel count), that's logged at WARNING
    -- visible in the logs even on the frames this picks correctly, not
    only discoverable by rendering the result and watching for it. 0.9
    is calibrated, not guessed: on the real footage described above,
    every frame that actually picked wrong measured a 0.94-0.99 ratio,
    while the ordinary case (a real blade with a persistent-but-clearly-
    smaller secondary component most frames) sits at a median of 0.36 --
    a threshold anywhere in that gap avoids flooding the logs with the
    routine case while still catching every genuine near-tie.
    """
    logger = logging.getLogger(__name__)
    labeled, n = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, index=range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    largest_label = 1 + int(order[0])

    if n >= 2 and sizes[order[1]] > 0.9 * sizes[order[0]]:
        logger.warning(
            "mask has two nearly-equal-sized components (%.0f px and %.0f px, ratio %.2f) -- "
            "picking between them, not just taking the larger one blindly",
            sizes[order[0]], sizes[order[1]], sizes[order[1]] / sizes[order[0]],
        )

    if reference_point is None:
        return labeled == largest_label

    if max_jump_px is None:
        max_jump_px = POSITION_GLITCH_JUMP_PX

    def centroid_of(label_id):
        ys, xs = np.nonzero(labeled == label_id)
        return float(xs.mean()), float(ys.mean())

    largest_centroid = centroid_of(largest_label)
    largest_jump = _centroid_dist(largest_centroid, reference_point)
    if largest_jump <= max_jump_px:
        return labeled == largest_label

    for idx in order[1:]:
        label_id = 1 + int(idx)
        candidate_centroid = centroid_of(label_id)
        candidate_jump = _centroid_dist(candidate_centroid, reference_point)
        if candidate_jump <= max_jump_px:
            logger.warning(
                "largest mask component jumped %.0fpx from the previous frame's position "
                "(cap %.0fpx) -- using a smaller component %.0fpx away instead, likely the "
                "real blade with a persistent secondary component (e.g. a body cable) "
                "briefly larger than it",
                largest_jump, max_jump_px, candidate_jump,
            )
            return labeled == label_id

    return labeled == largest_label


def fit_blade(mask, taper_frac=1.0 / 3.0, width_bins=20, reference_point=None):
    """Fit blade geometry from a binary mask.

    Method: restrict to the mask's most plausible connected component
    (see `_largest_component`), then PCA over its points gives the long
    axis; projecting all points onto that axis gives the two endpoints
    (min/max projection) and the perpendicular spread gives the width;
    `classify_tip_by_taper` disambiguates which endpoint is the tip.

    `reference_point` (`(x, y)`, typically the previous frame's fitted
    centroid) is passed straight through to `_largest_component` -- see
    its docstring for why a mask with two comparably-sized components
    needs it to pick correctly.

    Returns None when the mask has no foreground pixels at all.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None

    mask = _largest_component(mask, reference_point=reference_point)
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

    bend_offset = _bend_offset(proj, perp, n_bins=width_bins)
    if abs(bend_offset) > BEND_SIGNIFICANCE_PX:
        mid_proj = (min_proj + max_proj) / 2.0
        bend_point = centroid + mid_proj * axis + bend_offset * perp_dir
        bend = (float(bend_point[0]), float(bend_point[1]))
    else:
        bend = None

    return BladeGeometry(
        centroid=(float(centroid[0]), float(centroid[1])),
        axis=(float(oriented_axis[0]), float(oriented_axis[1])),
        tip=(float(tip[0]), float(tip[1])),
        hilt=(float(hilt[0]), float(hilt[1])),
        length=length,
        width=float(width),
        angle=angle,
        bend=bend,
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


_FIELDS = ("centroid", "tip", "hilt", "axis", "length", "width", "angle", "bend")
_VECTOR_FIELDS = ("centroid", "tip", "hilt", "axis", "bend")


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
            value = getattr(geo, field)
            if value is None:
                continue  # bend can be None on a real frame; every
                          # other field is always populated when geo is
                          # not None, so this only ever fires for bend
            arrays[field][i] = value

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


def _curvature_matrix(times):
    """`(T, T+2)` matrix `L` such that `L @ y` -- for `y` of length `T+2`,
    with `y[0]`/`y[-1]` fixed boundary values and `y[1:-1]` the `T`
    interior unknowns -- gives each interior point's curvature
    (second-derivative) estimate under `times`' (length `T+2`) spacing,
    which need not be uniform. Standard three-point second-derivative
    estimate; reduces to the familiar `y[k-1] - 2*y[k] + y[k+1]` when
    consecutive `times` are evenly spaced.

    This is the building block of `_smooth_run_field`'s "prefer a
    physically smooth path" prior: a sequence with zero curvature
    everywhere is, by construction, the straight line between its two
    fixed endpoints.
    """
    t = np.asarray(times, dtype=np.float64)
    T = len(t) - 2
    L = np.zeros((T, T + 2))
    for k in range(1, T + 1):
        h0 = t[k] - t[k - 1]
        h1 = t[k + 1] - t[k]
        L[k - 1, k - 1] = 2.0 / (h0 * (h0 + h1))
        L[k - 1, k] = -2.0 / (h0 * h1)
        L[k - 1, k + 1] = 2.0 / (h1 * (h0 + h1))
    return L


# How strongly `_smooth_run_field` favors a physically-smooth (low
# curvature) path over chasing each frame's own raw fit within an overlap
# run. Calibrated against the real 293-454 (162-frame) contact run: swept
# over three orders of magnitude (10 to 100,000) and compared each
# candidate's resulting path against the raw per-frame data's own
# frame-to-frame acceleration (a jitter proxy, mean 10.64px/frame^2 on
# that run) and total travelled span (up to 282px on one channel). At
# this value the smoothed path's mean acceleration is ~0.07px/frame^2 --
# about 150x smoother than the raw data, visibly not chasing its jitter
# -- while still covering ~72% of the raw data's travelled span (202 of
# 282px) on the channel checked, versus the plain straight line's ~9%
# (26.6px). Because the weighted data term vanishes wherever confidence
# (see `_run_confidence_weights`) is exactly 0, this value has *no*
# effect on a run whose two objects' raw fits coincide throughout -- that
# degrades to the same straight line regardless of this constant (see
# `_curvature_matrix`'s docstring).
RUN_SMOOTHING_STRENGTH = 2000.0


def _smooth_run_field(raw, weights, all_times, before_value, after_value,
                       smoothing_strength=RUN_SMOOTHING_STRENGTH):
    """The `len(raw)` interior values that minimize weighted deviation
    from `raw` (weight `weights[i]` per frame) plus `smoothing_strength`
    times squared curvature (see `_curvature_matrix`), pinned to
    `before_value`/`after_value` at the two times just outside
    `all_times[1:-1]`'s span (`all_times[0]` and `all_times[-1]`).

    A weight of 0 for every frame reduces exactly to linear interpolation
    between `before_value` and `after_value` -- minimizing pure curvature
    with fixed endpoints has the straight line as its unique solution --
    so a run with no usable raw signal at all degrades to the same plain
    straight-line fallback this replaced, not to something worse. Where
    `weights` is nonzero, the solution bends toward `raw` there,
    proportional to how much that frame's mask overlap allows it to be
    trusted.

    A NaN entry in `raw` (a frame the tracker lost the object on
    entirely) is treated as if it were 0 there -- safe *only* because
    its own `weights` entry must independently be 0 too (a lost frame
    has no real position to be confident about); relying on that alone
    would fail on `0 * nan = nan`, silently propagating through the
    solve and returning an all-NaN result for every frame in the run,
    not just the missing one.
    """
    raw = np.nan_to_num(np.asarray(raw, dtype=np.float64), nan=0.0)
    weights = np.asarray(weights, dtype=np.float64)
    n_interior = len(raw)
    curvature = _curvature_matrix(all_times)
    curvature_int = curvature[:, 1:n_interior + 1]
    curvature_bnd = curvature[:, [0, n_interior + 1]]
    y_bnd = np.array([before_value, after_value], dtype=np.float64)

    a = smoothing_strength * (curvature_int.T @ curvature_int) + np.diag(weights)
    b = weights * raw - smoothing_strength * (curvature_int.T @ (curvature_bnd @ y_bnd))
    return np.linalg.solve(a, b)


def _run_confidence_weights(motion_a, motion_b, run_start, run_end, reference_length):
    """Per-frame confidence, in `[0, 1]`, for using each object's own raw
    fitted geometry as weak evidence while smoothing through an overlap
    run: the distance between the two objects' raw centroids at that
    frame, relative to `reference_length` (a typical blade length for
    this pair), clipped to `[0, 1]`.

    An earlier version of this used `1 - iou` (the cross-object mask IoU
    that detected the run in the first place) as the confidence signal
    instead. Confirmed wrong on real footage: at frame 410 of a real
    162-frame contact run, mask IoU was 0.845 -- comfortably below the
    run's 0.95 peak, so `1 - iou` reported a plausible-looking 0.155
    confidence -- yet the two objects' independently-fitted raw
    centroids landed 0.5px apart, both objects' `fit_blade` having
    latched onto essentially the same visible blade. Trusting that
    "confidence" pulled *both* objects' smoothed trajectories onto the
    same line for several consecutive frames, a worse and more visually
    obvious failure (both glow blades collapsing onto one, in the same
    color-blended spot) than the plain straight line it replaced. Two
    masks not being pixel-identical does not mean the *fitted geometry*
    disagrees -- a partially-merged mask can still leave PCA landing on
    the same shape for both objects -- so confidence has to measure that
    disagreement directly instead of inferring it from mask overlap.
    Two coincident raw fits (distance ~0, both objects visibly on the
    same blade) get 0 confidence regardless of what the mask pixels say;
    two fits a full blade-length or more apart get full confidence.

    A frame either object's tracker lost entirely (a NaN centroid, e.g.
    a marginal frame just outside the detected run -- see
    `suppress_overlap_bleed`) gets 0 confidence, not NaN: there's no
    real position to measure separation from, and an unhandled NaN here
    would silently propagate through `_smooth_run_field`'s solve and
    corrupt the whole run, not just that one frame.
    """
    n = run_end - run_start + 1
    if reference_length <= 0:
        return np.zeros(n)
    centroid_a = motion_a["centroid"][run_start:run_end + 1]
    centroid_b = motion_b["centroid"][run_start:run_end + 1]
    separation = np.linalg.norm(centroid_a - centroid_b, axis=1)
    return np.nan_to_num(np.clip(separation / reference_length, 0.0, 1.0), nan=0.0)


def _tip_confidence_weights(motion_a, motion_b, run_start, run_end, frame_indices,
                             hilt_overrides_a, hilt_overrides_b, reference_length):
    """Per-frame confidence, in `[0, 1]`, for trusting each object's own
    raw `tip` specifically -- distinct from `_run_confidence_weights`'
    centroid-based signal, which never looks at tip at all and can stay
    high even when tip individually bleeds onto the other tracked
    object's hand. Confirmed on real footage as a real, not theoretical,
    gap: on the real 293-454 contact run, both objects' raw *centroids*
    stayed well separated throughout (so `_run_confidence_weights` gave
    tip-smoothing reasonable confidence to lean on raw data), while
    several frames' raw `tip` values independently landed within a few
    px of the *other* object's real hand -- a smoothed curve pulled
    toward that contamination produced a rendered blade stretching
    across nearly the entire frame (measured: length more than doubled,
    ~200px to 388px, at one such frame).

    Requires a validated hilt-tracking override (see
    `hilt_track.compute_hilt_overrides`) for *both* objects at a frame --
    the only frames with a trustworthy reference for "whose hand is
    this." For those frames, confidence is how much closer this object's
    raw tip sits to its own real hilt than to the other object's,
    relative to `reference_length`, clipped to `[0, 1]` -- the same
    clipped-relative-distance shape `_run_confidence_weights` already
    uses, applied to the one thing that actually matters for tip.
    Frames missing an override for either object get 0 confidence -- no
    trustworthy signal available, so tip-smoothing there falls back to
    the same anchor-pinned curve it already would without this function
    (see `_smooth_run_field`'s zero-confidence guarantee). A frame with
    a validated hilt override but a NaN raw `tip` (the tracker lost the
    object that frame) also gets 0, not NaN -- an unhandled NaN here
    would silently propagate through `_smooth_run_field`'s solve and
    corrupt the whole run, not just that one frame.

    Returns `(weights_a, weights_b)`.
    """
    n = run_end - run_start + 1
    weights_a = np.zeros(n)
    weights_b = np.zeros(n)
    if reference_length <= 0:
        return weights_a, weights_b
    for offset in range(n):
        frame_num = frame_indices[run_start + offset]
        if frame_num not in hilt_overrides_a or frame_num not in hilt_overrides_b:
            continue
        hilt_a, hilt_b = hilt_overrides_a[frame_num], hilt_overrides_b[frame_num]
        raw_tip_a = motion_a["tip"][run_start + offset]
        raw_tip_b = motion_b["tip"][run_start + offset]
        if np.isnan(raw_tip_a).any() or np.isnan(raw_tip_b).any():
            continue
        weights_a[offset] = np.clip(
            (_centroid_dist(raw_tip_a, hilt_b) - _centroid_dist(raw_tip_a, hilt_a)) / reference_length, 0.0, 1.0,
        )
        weights_b[offset] = np.clip(
            (_centroid_dist(raw_tip_b, hilt_a) - _centroid_dist(raw_tip_b, hilt_b)) / reference_length, 0.0, 1.0,
        )
    return weights_a, weights_b


# _smooth_run_field's regularization strength, specifically for the
# `tip` field, when tip has its own confidence signal
# (_tip_confidence_weights) rather than the shared centroid-based one.
# Confirmed necessary on real footage, and confirmed NOT just "more
# smoothing is always safer": at RUN_SMOOTHING_STRENGTH (2000), a sparse
# cluster of real, correctly-high-confidence points concentrated near
# one end of a 162-frame run (frames 432-454, ramping to full
# confidence right before the "after" anchor) pulled the *entire* curve
# into a smooth but wildly wrong bow reaching 933px on a 1280px-wide
# frame -- neither anchor exceeds 751px. This is `_smooth_run_field`
# behaving as designed (a curvature-minimizing spline distributes
# curvature across its whole domain to reach a sparse, asymmetric pull
# smoothly, rather than "hooking in" locally near it) but is unsafe at
# this strength when the trustworthy signal is this asymmetric. Swept
# 2,000-1,000,000 directly against the real run: the object with the
# overshoot needs at least ~100,000 to keep the curve within its own
# anchors' range (measured max 750px, vs both anchors near 669-751px);
# the object that never had this problem is confirmed unaffected by the
# higher value (measured max changes by <1px across the whole sweep).
TIP_SMOOTHING_STRENGTH = 100000


def _smooth_interpolate_run(motion, run_start, run_end, before, after, frame_indices, weights,
                             smoothing_strength=RUN_SMOOTHING_STRENGTH, tip_weights=None,
                             tip_smoothing_strength=None):
    """Replace `motion`'s rows `run_start..run_end` (inclusive, array
    indices) with a smoothed trajectory anchored at `before`/`after`
    (also array indices): `centroid`/`hilt`/`tip`/`width` are each
    smoothed independently (per x/y coordinate for the vector fields) via
    `_smooth_run_field`, then `axis`/`length`/`angle` are re-derived from
    the smoothed `tip`/`hilt` so the geometry stays internally consistent
    (axis really is the unit vector from hilt to tip, length really is
    their distance) -- rather than smoothing all seven fields
    independently, which could disagree with each other.

    `tip_weights` (default `weights`) lets the `tip` field use a
    different confidence signal than `centroid`/`hilt`/`width` -- see
    `_tip_confidence_weights`, whose whole reason to exist is that a
    frame can have plenty of centroid-based confidence while its tip
    specifically has bled onto the other tracked object. Only when
    `tip_weights` is actually given does `tip` also default to its own,
    stronger `tip_smoothing_strength` (see `TIP_SMOOTHING_STRENGTH`)
    instead of the shared `smoothing_strength` -- the sparse-pull
    overshoot that constant's own comment documents is a property of the
    sharp, sparse confidence `_tip_confidence_weights` produces, not of
    the shared centroid-based `weights` (confirmed: `smoothing_strength`
    alone was already safe for every field before `tip_weights` existed,
    and stays that way when `tip_weights` is left unset).
    """
    using_tip_specific_weights = tip_weights is not None
    if tip_weights is None:
        tip_weights = weights
    if tip_smoothing_strength is None:
        tip_smoothing_strength = TIP_SMOOTHING_STRENGTH if using_tip_specific_weights else smoothing_strength
    all_times = [frame_indices[before], *frame_indices[run_start:run_end + 1], frame_indices[after]]
    raw = {field: motion[field][run_start:run_end + 1].copy() for field in ("centroid", "hilt", "tip", "width")}
    before_row = {field: motion[field][before] for field in ("centroid", "hilt", "tip", "width")}
    after_row = {field: motion[field][after] for field in ("centroid", "hilt", "tip", "width")}

    smoothed = {}
    for field in ("centroid", "hilt", "tip"):
        field_weights = tip_weights if field == "tip" else weights
        field_strength = tip_smoothing_strength if field == "tip" else smoothing_strength
        smoothed[field] = np.stack([
            _smooth_run_field(
                raw[field][:, c], field_weights, all_times, before_row[field][c], after_row[field][c],
                field_strength,
            )
            for c in (0, 1)
        ], axis=1)
    smoothed["width"] = _smooth_run_field(
        raw["width"], weights, all_times, before_row["width"], after_row["width"], smoothing_strength,
    )

    for field, values in smoothed.items():
        motion[field][run_start:run_end + 1] = values

    axis_vecs = motion["tip"][run_start:run_end + 1] - motion["hilt"][run_start:run_end + 1]
    norms = np.linalg.norm(axis_vecs, axis=1)
    motion["length"][run_start:run_end + 1] = norms
    nonzero = norms > 0
    row_idx = np.arange(run_start, run_end + 1)[nonzero]
    motion["axis"][row_idx] = axis_vecs[nonzero] / norms[nonzero, None]
    motion["angle"][row_idx] = np.arctan2(axis_vecs[nonzero, 1], axis_vecs[nonzero, 0])


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


def _cross_object_ious(masks_dir_a, masks_dir_b, frame_indices):
    """Per-frame cross-object mask IoU across `frame_indices`, in order --
    the same computation `_find_overlap_runs` uses to detect a run in the
    first place, extracted so `suppress_overlap_bleed`'s cross-object
    contamination gate (see that function) can reuse it across the whole
    clip without a second, possibly-inconsistent measurement of "are
    these two objects' masks colliding right now?" and without changing
    `_find_overlap_runs`'s own return type (which `reacquire.py` also
    depends on)."""
    return np.array([
        _mask_iou(load_mask(masks_dir_a, frame_idx), load_mask(masks_dir_b, frame_idx))
        for frame_idx in frame_indices
    ])


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
    ious = _cross_object_ious(masks_dir_a, masks_dir_b, frame_indices)
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


# A run this long (roughly 3.5s+ at typical frame rates) means neither the
# shared multi-object tracking session nor, if it was tried,
# `reacquire.retrack_overlap_runs`'s independent re-track could tell the
# two objects apart for a long stretch. `_smooth_interpolate_run`'s
# confidence-weighted smoother (see `_run_confidence_weights`) does
# meaningfully better than a plain straight line here -- it bends toward
# each frame's own raw fit wherever the two objects' independently
# fitted raw geometry is far enough apart to trust -- but on a span this
# long, a large fraction of frames can still have the two objects' raw
# fits essentially coincide (confirmed on real fencing footage: on the
# 162-frame 293-454 run, several multi-frame stretches had the two
# objects' raw centroids landing within a few pixels of each other even
# where mask IoU alone looked only moderately bad), where confidence is
# ~0 and the smoother has nothing but the boundary anchors and a
# smoothness prior to go on, same as a plain interpolation would. This
# constant exists so that fact is loud in the logs instead of blending
# into a routine per-run warning identical in shape to every short,
# well-approximated one.
LONG_INTERPOLATION_SPAN_FRAMES = 90


def _rederive_from_tip_hilt(motion, j):
    """Recompute row `j`'s `axis`/`length`/`angle` from its current
    `tip`/`hilt` -- shared by every place in this module that changes one
    endpoint and needs the other three fields to stay internally
    consistent (axis really is the unit vector from hilt to tip, length
    really is their distance)."""
    axis_vec = motion["tip"][j] - motion["hilt"][j]
    norm = np.linalg.norm(axis_vec)
    motion["length"][j] = norm
    if norm > 0:
        motion["axis"][j] = axis_vec / norm
        motion["angle"][j] = np.arctan2(axis_vec[1], axis_vec[0])


def _apply_hilt_overrides(motion, run_start, run_end, frame_indices, hilt_overrides):
    """For every frame in [run_start, run_end] with a validated entry in
    `hilt_overrides` ({frame number: (x, y)}, e.g. from
    `hilt_track.compute_hilt_overrides`), replace `motion`'s `hilt` row
    with it and re-derive `axis`/`length`/`angle` from the
    (already-smoothed) `tip` and the new `hilt`. `centroid`/`width` are
    left untouched -- hilt-tracking only has evidence about the hand's
    position, not the blade's overall shape.
    """
    for j in range(run_start, run_end + 1):
        frame_num = frame_indices[j]
        if frame_num not in hilt_overrides:
            continue
        motion["hilt"][j] = hilt_overrides[frame_num]
        _rederive_from_tip_hilt(motion, j)


def suppress_overlap_bleed(motion_path_a, masks_dir_a, motion_path_b, masks_dir_b,
                            iou_threshold=CROSS_OBJECT_OVERLAP_IOU_THRESHOLD,
                            anchor_iou_threshold=None, exclude_frame_ranges=(),
                            hilt_overrides_a=None, hilt_overrides_b=None):
    """Patch two already-written motion.npz files in place: for every run of
    consecutive frames where the two tracked objects' raw masks overlap
    past `iou_threshold` (see `_find_overlap_runs`), replace *both*
    objects' fitted geometry -- not just for the run itself, but for
    every frame strictly between its two anchors -- rather than trusting
    a per-frame fit_blade result computed from a mask that may have bled
    into the other object's blade.

    The correction deliberately reaches past the narrower [run_start,
    run_end] IoU-overlap span to cover the whole [before+1, after-1]
    gap: confirmed on real footage, a frame just below the overlap
    threshold (too contaminated to trust as an *anchor*, see
    `CROSS_OBJECT_ANCHOR_IOU_FRAC`) but not yet part of the detected run
    is not any more trustworthy left on its own -- IoU climbs gradually
    into a real overlap, so a frame at 0.087 (just under the 0.1 run
    threshold) can already be as contaminated as one at 0.15. Left
    uncorrected, that frame's own raw fit visibly diverged from the real
    blade right at the moment contact began -- exactly where a viewer's
    eye is drawn.

    Prefers to *smooth* a trajectory between the last good frame before the
    run and the first good frame after it, rather than freezing at a
    single value for the run's whole duration -- confirmed on real
    footage, a real hilt travels 150-235px across a ~3s run of overlapping
    frames (a real fencing exchange, not an instant touch), so a frozen
    geometry visibly detaches from the hand holding it ("the blade
    disembodies and floats") long before the run ends. The smoother
    (`_smooth_interpolate_run`) pins both endpoints exactly and fills the
    gap with a low-curvature path that also bends toward each frame's own
    raw fit wherever the two objects' raw fits are far enough apart from
    *each other* to trust (see `_run_confidence_weights`) -- confirmed on
    real footage to track a real, non-monotonic engagement (blades
    disengage and re-engage) far better than a straight line would, while
    degrading gracefully to exactly that straight line on a stretch where
    the two objects' raw fits coincide throughout and no raw signal is
    usable at all. Either endpoint missing (the run starts at frame 0, or
    never ends before the clip does) falls back to freezing at whichever
    single endpoint exists.

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

    `hilt_overrides_a`/`hilt_overrides_b` (each an optional
    `{frame_number: (x, y)}` dict, e.g. from
    `hilt_track.compute_hilt_overrides`) replace a smoothed run's `hilt`
    with a validated, independently-tracked position for whichever
    frames are present -- see `_apply_hilt_overrides`. `centroid`/`width`
    are left as the smoother produced them; only `hilt` (and
    `axis`/`length`/`angle`, re-derived from it) are affected.

    When *both* hilt overrides are given, `tip` is smoothed using its own
    confidence signal (see `_tip_confidence_weights`) instead of the
    shared centroid-based `weights`: confirmed on real footage, raw
    centroid separation can stay high (so the shared signal alone gives
    tip-smoothing real confidence to lean on raw data) while one object's
    individually-fitted raw `tip` has still bled almost exactly onto the
    *other* object's hand -- a blade stretching across nearly the whole
    frame, since centroid separation never looks at tip at all. Once both
    objects' real hand positions are known (from hilt tracking),
    tip-smoothing's confidence at a frame is instead how much closer that
    frame's raw tip sits to its own object's hilt than to the other
    object's.

    A smoothed run longer than `LONG_INTERPOLATION_SPAN_FRAMES` gets a
    second, more detailed WARNING beyond the routine per-run one --
    confirmed on real footage, a run this long can still spend most of its
    length with the two objects' raw fits essentially coinciding, where
    the smoother has no raw signal to lean on and falls back to the same
    straight line a plain interpolation would give, and that needs to be
    loud in the logs rather than looking like every other short,
    well-approximated run.

    Returns the number of frames patched (smoothed or held).
    """
    logger = logging.getLogger(__name__)
    motion_a = load_motion(motion_path_a)
    motion_b = load_motion(motion_path_b)
    frame_indices = mask_frame_indices(masks_dir_a)
    runs = _find_overlap_runs(masks_dir_a, masks_dir_b, motion_a, motion_b, iou_threshold, anchor_iou_threshold)

    n_held = 0
    for run_start, run_end, before, after, max_iou in runs:
        if before is not None and after is not None:
            # Correct every frame strictly between the two anchors, not
            # just the narrower [run_start, run_end] IoU-overlap span --
            # confirmed on real footage, a frame just below the overlap
            # threshold (too contaminated to trust as an anchor, see
            # CROSS_OBJECT_ANCHOR_IOU_FRAC) but not yet part of the
            # detected run was left with its own uncorrected raw fit,
            # visibly diverging from the real blade right as contact
            # began (measured: IoU 0.087 at that frame, just under the
            # 0.1 run threshold, already well past the 0.02 anchor bar).
            correct_start, correct_end = before + 1, after - 1
        else:
            correct_start, correct_end = run_start, run_end

        start_frame, end_frame = frame_indices[correct_start], frame_indices[correct_end]
        if any(start_frame <= ex_end and end_frame >= ex_start for ex_start, ex_end in exclude_frame_ranges):
            continue

        if before is not None and after is not None:
            reference_length = float(np.mean([
                motion_a["length"][before], motion_a["length"][after],
                motion_b["length"][before], motion_b["length"][after],
            ]))
            weights = _run_confidence_weights(motion_a, motion_b, correct_start, correct_end, reference_length)
            # tip gets its own confidence signal (and, with it, its own
            # stronger smoothing strength -- see TIP_SMOOTHING_STRENGTH)
            # only when both objects have validated hilt positions to
            # check it against -- see _tip_confidence_weights for why
            # centroid-based `weights` alone isn't enough. Computed
            # before _smooth_interpolate_run overwrites raw tip in
            # place. Left as None (not defaulted to `weights` here) when
            # unavailable, so _smooth_interpolate_run's own default
            # correctly falls back to the shared smoothing strength too.
            tip_weights_a = tip_weights_b = None
            if hilt_overrides_a and hilt_overrides_b:
                tip_weights_a, tip_weights_b = _tip_confidence_weights(
                    motion_a, motion_b, correct_start, correct_end, frame_indices,
                    hilt_overrides_a, hilt_overrides_b, reference_length,
                )
            _smooth_interpolate_run(motion_a, correct_start, correct_end, before, after, frame_indices, weights,
                                     tip_weights=tip_weights_a)
            _smooth_interpolate_run(motion_b, correct_start, correct_end, before, after, frame_indices, weights,
                                     tip_weights=tip_weights_b)
            if hilt_overrides_a:
                _apply_hilt_overrides(motion_a, correct_start, correct_end, frame_indices, hilt_overrides_a)
            if hilt_overrides_b:
                _apply_hilt_overrides(motion_b, correct_start, correct_end, frame_indices, hilt_overrides_b)
        else:
            for j in range(correct_start, correct_end + 1):
                if before is not None:
                    _freeze_row(motion_a, j, before)
                    _freeze_row(motion_b, j, before)
                elif after is not None:
                    _freeze_row(motion_a, j, after)
                    _freeze_row(motion_b, j, after)
                # else: no anchor at all -- nothing better than the raw fit.

        if before is not None or after is not None:
            span = correct_end - correct_start + 1
            n_held += span
            logger.warning(
                "frames %d-%d: tracked objects' masks overlapped (IoU up to %.2f, cap %.2f) -- %s "
                "both objects' geometry%s",
                frame_indices[correct_start], frame_indices[correct_end], max_iou,
                iou_threshold,
                "smoothed" if before is not None and after is not None else "held",
                (
                    f" between frames {frame_indices[before]} and {frame_indices[after]}"
                    if before is not None and after is not None
                    else f" at frame {frame_indices[before if before is not None else after]}'s values"
                ),
            )
            if before is not None and after is not None and span > LONG_INTERPOLATION_SPAN_FRAMES:
                logger.warning(
                    "frames %d-%d: this interpolated span is %d frames long -- long enough that a "
                    "straight line between its two endpoints likely does not track the real motion "
                    "well (see LONG_INTERPOLATION_SPAN_FRAMES). The confidence-weighted smoother "
                    "(_smooth_interpolate_run) bends toward each frame's own raw fit wherever the two "
                    "objects' raw fits are far enough apart to trust, but a run this long can still "
                    "spend most of its length with the two objects' raw fits essentially coinciding, "
                    "where there is no raw signal to lean on and it falls back to the same straight "
                    "line a plain interpolation would give; worth reviewing this stretch of the "
                    "render visually.",
                    frame_indices[correct_start], frame_indices[correct_end], span,
                )

    if n_held:
        np.savez(motion_path_a, **motion_a)
        np.savez(motion_path_b, **motion_b)
    return n_held


# A tracked object's fitted centroid jumping this far from BOTH its
# immediate accepted-neighbor and the next raw frame, while those two
# neighbors sit close to each other, is a single-frame (or short) SAM2
# tracking glitch -- not genuine motion, which moves the "after" position
# consistently *away* from "before" as it goes rather than snapping back
# to nearly the same spot a frame or two later. Confirmed on real footage,
# unrelated to any cross-object contact (IoU between the two tracked
# objects was 0 throughout): one object's centroid jumped to the opposite
# edge of a 1280px-wide frame for exactly one frame, sandwiched between
# two frames only 1px apart. A scan of one ~20s real clip found 8 such
# glitches across two tracked objects -- common enough to fix, not a
# one-off.
POSITION_GLITCH_JUMP_PX = 50.0


def _centroid_dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _interpolate_geometry(before, after, t):
    """A BladeGeometry `t` of the way from `before` to `after`: blend
    centroid/hilt/tip/width directly, then re-derive axis/length/angle
    from the interpolated hilt/tip so the result stays internally
    consistent instead of blending all seven fields independently."""
    centroid = (1 - t) * np.asarray(before.centroid) + t * np.asarray(after.centroid)
    hilt = (1 - t) * np.asarray(before.hilt) + t * np.asarray(after.hilt)
    tip = (1 - t) * np.asarray(before.tip) + t * np.asarray(after.tip)
    width = (1 - t) * before.width + t * after.width
    axis_vec = tip - hilt
    norm = np.linalg.norm(axis_vec)
    if norm > 0:
        axis = axis_vec / norm
        angle = float(np.arctan2(axis[1], axis[0]))
    else:
        axis = np.asarray(before.axis)
        angle = before.angle
    return BladeGeometry(
        centroid=(float(centroid[0]), float(centroid[1])),
        axis=(float(axis[0]), float(axis[1])),
        tip=(float(tip[0]), float(tip[1])),
        hilt=(float(hilt[0]), float(hilt[1])),
        length=float(norm), width=float(width), angle=angle,
    )


def _suppress_position_glitches(geometries, frame_indices, jump_px=POSITION_GLITCH_JUMP_PX):
    """Replace a frame's geometry with an interpolation between its
    neighbors when its centroid jumps far from both the last *accepted*
    frame and the very next raw frame, while those two sit close to each
    other -- see `POSITION_GLITCH_JUMP_PX`.

    Deliberately narrow: only catches a single bad frame immediately
    followed by a good one. A real multi-frame example on real footage
    (checked while building this) turned out ambiguous even on close
    inspection -- length alternated 140/250/140/250 across four frames
    before settling at 250, plausibly a genuine (if noisy) transition
    rather than a glitch -- so this does not try to resolve runs of
    consecutive bad frames; a wrong guess there costs more than the
    narrower scope. A frame whose jump doesn't match the correctable
    pattern (bracketed by two close neighbors) is left untouched, but
    still logged -- confirmed necessary on real footage: a multi-frame
    excursion this can't fix (the real blade's own mask losing out to a
    persistent secondary component -- see `_largest_component` -- for
    more than one consecutive frame) previously produced no log output
    at all, discoverable only by rendering the clip and watching for it.

    Processes forward, comparing each frame against the last *accepted*
    (already-corrected) frame rather than the last raw one, so a
    corrected frame becomes solid ground for the next comparison.

    Returns `(geometries, n_suppressed)`.
    """
    logger = logging.getLogger(__name__)
    result = list(geometries)
    n = len(result)
    n_suppressed = 0
    last_good_idx = None

    for i in range(n):
        geo = result[i]
        if geo is None:
            continue
        if last_good_idx is not None:
            after_idx = next((j for j in range(i + 1, n) if geometries[j] is not None), None)
            if after_idx is not None:
                prev_geo = result[last_good_idx]
                after_geo = geometries[after_idx]
                neighbor_dist = _centroid_dist(prev_geo.centroid, after_geo.centroid)
                dist_to_prev = _centroid_dist(geo.centroid, prev_geo.centroid)
                dist_to_after = _centroid_dist(geo.centroid, after_geo.centroid)
                if dist_to_prev > jump_px and dist_to_after > jump_px and neighbor_dist < jump_px:
                    t = (frame_indices[i] - frame_indices[last_good_idx]) / (
                        frame_indices[after_idx] - frame_indices[last_good_idx]
                    )
                    logger.warning(
                        "frame %d: centroid jumped %.0fpx from frame %d and %.0fpx from frame %d "
                        "(which are only %.0fpx apart) -- holding an interpolated position instead "
                        "of a likely tracking glitch",
                        frame_indices[i], dist_to_prev, frame_indices[last_good_idx],
                        dist_to_after, frame_indices[after_idx], neighbor_dist,
                    )
                    result[i] = _interpolate_geometry(prev_geo, after_geo, t)
                    n_suppressed += 1
                    continue
                if dist_to_prev > jump_px and dist_to_after <= jump_px:
                    logger.warning(
                        "frame %d: centroid jumped %.0fpx from frame %d (last accepted), but is "
                        "only %.0fpx from frame %d (next) -- doesn't match the bracketed "
                        "single-frame-glitch pattern this function corrects (see its own "
                        "docstring for why multi-frame runs are out of scope), so left as "
                        "tracked; worth checking whether this starts a real multi-frame "
                        "tracking problem",
                        frame_indices[i], dist_to_prev, frame_indices[last_good_idx],
                        dist_to_after, frame_indices[after_idx],
                    )
        last_good_idx = i

    return result, n_suppressed


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

    Each frame's tip/hilt is initially guessed per-frame by `fit_blade`,
    seeded with the previous valid frame's fitted centroid as
    `reference_point` so `_largest_component` can reject a comparably-
    sized but implausibly-located component (e.g. a tracked object's own
    body cable briefly outsizing the real blade) instead of just taking
    whichever component happens to have more pixels -- see that
    function's docstring. `_suppress_position_glitches` then holds an
    interpolated position for any *remaining* frame whose centroid jumps
    far from its neighbors and back -- an isolated SAM2 tracking glitch,
    not genuine motion -- before `_orient_by_motion` re-decides tip vs
    hilt once for the whole sequence from which endpoint actually
    travelled farther (run in that order so a wild single-frame glitch
    can't throw off `_orient_by_motion`'s own nearest-neighbour endpoint
    tracking too) -- see each function's docstring for more.

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
    last_good_centroid = None
    for i, frame_idx in enumerate(frame_indices):
        mask = load_mask(masks_dir, frame_idx)
        geo = fit_blade(mask, taper_frac=taper_frac, width_bins=width_bins, reference_point=last_good_centroid)
        if geo is not None:
            logger.debug(
                "frame %d: length=%.1f width=%.1f centroid=(%.1f, %.1f) angle=%.2f",
                frame_idx, geo.length, geo.width, geo.centroid[0], geo.centroid[1], geo.angle,
            )
            last_good_centroid = geo.centroid
        geometries.append(geo)
        report((i + 1) / n * 100, f"frame {i + 1}/{n}")

    geometries, n_glitches = _suppress_position_glitches(geometries, frame_indices)
    if n_glitches:
        logger.warning(
            "%s: held %d/%d frame(s) at an interpolated position due to isolated tracking glitches",
            masks_dir, n_glitches, n,
        )

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
