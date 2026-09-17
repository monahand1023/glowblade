# Hilt Optical-Flow Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover each tracked object's hilt (hand/grip) position through the "dead zone" of a sustained cross-object blade-contact run, where `blade.suppress_overlap_bleed`'s confidence-weighted smoother has zero usable per-object mask signal (~65% of the real 162-frame 293-454 run), by tracking the fencer's hand via classical optical flow directly on the raw video frames.

**Architecture:** A new module `pipeline/hilt_track.py`, independent of SAM2/masks/`blade.py` internals, tracks a small cluster of trackable corner features seeded near each object's known-good hilt position, forward from the run's "before" anchor and backward from its "after" anchor, validates each direction against the *other* anchor's known-good position, and blends the two when both validate. `blade.suppress_overlap_bleed` gains optional `hilt_overrides_a`/`hilt_overrides_b` parameters applied after its existing smoothing step. `runner.py` wires a new `compute_hilt_overrides` call between `retrack_overlap_runs` and `suppress_overlap_bleed`.

**Tech Stack:** Python, OpenCV (`cv2` -- already a dependency via `reacquire.py`; no new dependency), numpy, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-hilt-optical-flow-tracking-design.md`

## Global Constraints

- Position only -- this plan does not recover blade angle/length in the dead zone; those stay exactly as `_smooth_interpolate_run` already produces them.
- Only for cross-object overlap runs `retrack_overlap_runs` did *not* already resolve (i.e., runs not in `resolved_ranges`/`exclude_frame_ranges`).
- Must fail silently: any run/object this can't validate gets no override, never an exception, never a worse result than today.
- No new dependency -- `cv2` only, already used by `reacquire.py`.
- Calibrated constants (validated against the real job at frames 290/455 before this plan was written -- see spec's Constants section):
  `HILT_SEED_WINDOW_RADIUS_PX = 45`, `goodFeaturesToTrack(qualityLevel=0.1, minDistance=5, maxCorners=20)`,
  `MIN_SEED_FEATURES = 4`, `HILT_TRACK_MAX_DRIFT_FRAC = 0.3`,
  `calcOpticalFlowPyrLK(winSize=(21, 21), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))`.

---

### Task 1: Core seeding and sequential point tracking

**Files:**
- Create: `src/lightsaber_fx/pipeline/hilt_track.py`
- Test: Create `tests/pipeline/test_hilt_track.py`

**Interfaces:**
- Produces: `_frame_path(frames_dir, frame_idx) -> str`, `_seed_features(frame_gray, center, window_radius=HILT_SEED_WINDOW_RADIUS_PX, max_features=20) -> np.ndarray | None`, `_track_points_sequential(frames_dir, frame_indices, seed_points) -> dict[int, tuple[float, float]]`, constants `HILT_SEED_WINDOW_RADIUS_PX`, `MIN_SEED_FEATURES`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/pipeline/test_hilt_track.py
import os

import cv2
import numpy as np

from lightsaber_fx.pipeline.hilt_track import (
    MIN_SEED_FEATURES,
    _seed_features,
    _track_points_sequential,
)


def _write_checker_frame(path, canvas, center, patch_size=30, square=4, value=128):
    """A uniform gray frame with a small checkerboard patch at `center` --
    textured enough for cv2.goodFeaturesToTrack to find real corners,
    unlike a flat color."""
    frame = np.full((canvas[1], canvas[0], 3), value, dtype=np.uint8)
    cx, cy = center
    half = patch_size // 2
    y0, y1 = max(0, cy - half), min(canvas[1], cy + half)
    x0, x1 = max(0, cx - half), min(canvas[0], cx + half)
    for y in range(y0, y1):
        for x in range(x0, x1):
            block = ((x - x0) // square + (y - y0) // square) % 2
            frame[y, x] = (240, 240, 240) if block == 0 else (10, 10, 10)
    cv2.imwrite(str(path), frame)


def _write_flat_frame(path, canvas, value=128):
    frame = np.full((canvas[1], canvas[0], 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def _write_translating_checker_sequence(frames_dir, start_frame, end_frame, start_center, dx, dy,
                                         canvas=(200, 200), patch_size=30):
    """Writes one frame per index in [start_frame, end_frame], each with a
    checkerboard patch translated by (dx, dy) per frame from
    `start_center`. Returns {frame_idx: (x, y)} of the patch's true
    center at each frame, for test assertions."""
    os.makedirs(str(frames_dir), exist_ok=True)
    true_positions = {}
    for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        cx = int(round(start_center[0] + dx * i))
        cy = int(round(start_center[1] + dy * i))
        _write_checker_frame(
            os.path.join(str(frames_dir), f"{frame_idx:05d}.jpg"), canvas, (cx, cy), patch_size=patch_size,
        )
        true_positions[frame_idx] = (float(cx), float(cy))
    return true_positions


def test_seed_features_finds_corners_on_a_textured_patch(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    _write_checker_frame(str(frame_path), (200, 200), center=(100, 100))
    gray = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2GRAY)

    points = _seed_features(gray, center=(100, 100))

    assert points is not None
    assert len(points) >= MIN_SEED_FEATURES


def test_seed_features_declines_on_a_flat_patch(tmp_path):
    frame_path = tmp_path / "frame.jpg"
    _write_flat_frame(str(frame_path), (200, 200))
    gray = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2GRAY)

    points = _seed_features(gray, center=(100, 100))

    assert points is None


def test_track_points_sequential_follows_a_known_translation(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=10, start_center=(100, 100), dx=2, dy=1,
    )
    gray0 = cv2.cvtColor(cv2.imread(os.path.join(str(frames_dir), "00000.jpg")), cv2.COLOR_BGR2GRAY)
    seed_points = _seed_features(gray0, center=(100, 100))
    assert seed_points is not None

    positions = _track_points_sequential(str(frames_dir), list(range(0, 11)), seed_points)

    assert set(positions.keys()) == set(range(0, 11))
    for frame_idx, (x, y) in positions.items():
        true_x, true_y = true_positions[frame_idx]
        assert abs(x - true_x) < 3.0
        assert abs(y - true_y) < 3.0


def test_track_points_sequential_stops_early_when_a_frame_is_missing(tmp_path):
    frames_dir = tmp_path / "frames"
    _write_translating_checker_sequence(frames_dir, start_frame=0, end_frame=5, start_center=(100, 100), dx=1, dy=0)
    gray0 = cv2.cvtColor(cv2.imread(os.path.join(str(frames_dir), "00000.jpg")), cv2.COLOR_BGR2GRAY)
    seed_points = _seed_features(gray0, center=(100, 100))

    # asks for frames 0-10, but only 0-5 exist on disk
    positions = _track_points_sequential(str(frames_dir), list(range(0, 11)), seed_points)

    assert set(positions.keys()) == set(range(0, 6))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_hilt_track.py -v`
Expected: FAIL (or collection error) -- `lightsaber_fx.pipeline.hilt_track` does not exist yet.

- [ ] **Step 3: Create the module with the minimal implementation**

```python
# src/lightsaber_fx/pipeline/hilt_track.py
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


_LK_PARAMS = dict(
    winSize=(21, 21), maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_hilt_track.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run ruff**

Run: `ruff check src/lightsaber_fx/pipeline/hilt_track.py tests/pipeline/test_hilt_track.py`
Expected: no issues (fix any before continuing)

- [ ] **Step 6: Commit**

```bash
git add src/lightsaber_fx/pipeline/hilt_track.py tests/pipeline/test_hilt_track.py
git commit -m "Add hilt_track core: seed trackable corners and follow them via optical flow"
```

---

### Task 2: Direction validation and `track_hilt_through_run`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/hilt_track.py` (append)
- Test: `tests/pipeline/test_hilt_track.py` (append)

**Interfaces:**
- Consumes: `_frame_path`, `_seed_features`, `_track_points_sequential` (Task 1).
- Produces: `_track_direction(frames_dir, all_frame_indices, start_frame, start_hilt, validate_frame, validate_hilt, validate_length) -> dict[int, tuple[float, float]] | None`, `track_hilt_through_run(frames_dir, frame_indices, run_start_frame, run_end_frame, before_frame, before_hilt, before_length, after_frame, after_hilt, after_length) -> dict[int, tuple[float, float]]`, constant `HILT_TRACK_MAX_DRIFT_FRAC`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/pipeline/test_hilt_track.py
from lightsaber_fx.pipeline.hilt_track import _track_direction, track_hilt_through_run


def test_track_direction_validates_when_landing_is_close_to_the_known_good_anchor(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )

    positions = _track_direction(
        str(frames_dir), list(range(0, 21)),
        start_frame=0, start_hilt=true_positions[0],
        validate_frame=20, validate_hilt=true_positions[20], validate_length=100.0,
    )

    assert positions is not None
    assert 20 in positions
    assert abs(positions[20][0] - true_positions[20][0]) < 3.0
    assert abs(positions[20][1] - true_positions[20][1]) < 3.0


def test_track_direction_declines_when_landing_drifts_too_far(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )
    wrong_hilt = (true_positions[20][0] + 200.0, true_positions[20][1] + 200.0)

    positions = _track_direction(
        str(frames_dir), list(range(0, 21)),
        start_frame=0, start_hilt=true_positions[0],
        validate_frame=20, validate_hilt=wrong_hilt, validate_length=50.0,
    )

    assert positions is None


def test_track_direction_declines_when_seed_window_has_no_texture(tmp_path):
    frames_dir = tmp_path / "frames"
    os.makedirs(str(frames_dir), exist_ok=True)
    for i in range(21):
        _write_flat_frame(os.path.join(str(frames_dir), f"{i:05d}.jpg"), (200, 200))

    positions = _track_direction(
        str(frames_dir), list(range(0, 21)),
        start_frame=0, start_hilt=(80.0, 80.0),
        validate_frame=20, validate_hilt=(90.0, 90.0), validate_length=100.0,
    )

    assert positions is None


def test_track_hilt_through_run_blends_forward_and_backward_when_both_validate(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )

    overrides = track_hilt_through_run(
        str(frames_dir), list(range(0, 21)), run_start_frame=5, run_end_frame=15,
        before_frame=0, before_hilt=true_positions[0], before_length=100.0,
        after_frame=20, after_hilt=true_positions[20], after_length=100.0,
    )

    assert set(overrides.keys()) == set(range(5, 16))
    for frame_idx, (x, y) in overrides.items():
        true_x, true_y = true_positions[frame_idx]
        assert abs(x - true_x) < 3.0
        assert abs(y - true_y) < 3.0


def test_track_hilt_through_run_returns_empty_when_neither_direction_validates(tmp_path):
    frames_dir = tmp_path / "frames"
    os.makedirs(str(frames_dir), exist_ok=True)
    for i in range(21):
        _write_flat_frame(os.path.join(str(frames_dir), f"{i:05d}.jpg"), (200, 200))

    overrides = track_hilt_through_run(
        str(frames_dir), list(range(0, 21)), run_start_frame=5, run_end_frame=15,
        before_frame=0, before_hilt=(80.0, 80.0), before_length=100.0,
        after_frame=20, after_hilt=(90.0, 90.0), after_length=100.0,
    )

    assert overrides == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_hilt_track.py -v -k "track_direction or track_hilt_through_run"`
Expected: FAIL -- `_track_direction`/`track_hilt_through_run` not defined.

- [ ] **Step 3: Implement**

```python
# append to src/lightsaber_fx/pipeline/hilt_track.py

# How far (as a fraction of the *validating* anchor's own fitted blade
# length) a direction's tracked landing position may drift from that
# anchor's true position before the direction is declined entirely --
# mirrors reacquire.RETRACK_MAX_DRIFT_FRAC's already-proven
# validate-against-the-known-good-anchor philosophy, applied to a
# tracked hilt point instead of a whole re-tracked mask. Confirmed
# appropriate on the real job: all four directions tested
# (forward/backward x two objects) landed 20-29px from their true
# anchor, comfortably inside a 25-62px cap at this fraction.
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_hilt_track.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Run ruff**

Run: `ruff check src/lightsaber_fx/pipeline/hilt_track.py tests/pipeline/test_hilt_track.py`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add src/lightsaber_fx/pipeline/hilt_track.py tests/pipeline/test_hilt_track.py
git commit -m "Add track_hilt_through_run: validated forward/backward hilt tracking"
```

---

### Task 3: `compute_hilt_overrides` wrapper

**Files:**
- Modify: `src/lightsaber_fx/pipeline/hilt_track.py` (append)
- Test: `tests/pipeline/test_hilt_track.py` (append)

**Interfaces:**
- Consumes: `track_hilt_through_run` (Task 2); `blade._find_overlap_runs`, `blade.load_motion`, `blade.mask_frame_indices` (existing).
- Produces: `compute_hilt_overrides(frames_dir, masks_dir_a, masks_dir_b, motion_path_a, motion_path_b, exclude_frame_ranges=()) -> tuple[dict, dict]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/pipeline/test_hilt_track.py
from lightsaber_fx.pipeline.blade import BladeGeometry, save_mask, save_motion
from lightsaber_fx.pipeline.hilt_track import compute_hilt_overrides


def _write_overlap_run_fixture(tmp_path, true_positions, overlap_start=5, overlap_end=15, n=21):
    """Two objects whose masks are identical (IoU 1.0, a detected run)
    for [overlap_start, overlap_end] and disjoint everywhere else, with
    motion.npz built from `true_positions` (a {frame_idx: (x, y)} dict,
    e.g. from _write_translating_checker_sequence) for both objects."""
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    for i in range(n):
        mask_a = np.zeros((20, 20), dtype=bool)
        mask_a[0:4, 0:4] = True
        save_mask(str(masks_a), i, mask_a)
        mask_b = np.zeros((20, 20), dtype=bool)
        if overlap_start <= i <= overlap_end:
            mask_b[0:4, 0:4] = True
        else:
            mask_b[15:17, 15:17] = True
        save_mask(str(masks_b), i, mask_b)

    def geo(i):
        x, y = true_positions[i]
        return BladeGeometry(centroid=(x, y), axis=(1.0, 0.0), tip=(x + 50.0, y), hilt=(x, y),
                              length=50.0, width=5.0, angle=0.0)

    save_motion(str(motion_a), [geo(i) for i in range(n)])
    save_motion(str(motion_b), [geo(i) for i in range(n)])
    return masks_a, masks_b, motion_a, motion_b


def test_compute_hilt_overrides_returns_tracked_positions_for_an_unresolved_run(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )
    masks_a, masks_b, motion_a, motion_b = _write_overlap_run_fixture(tmp_path, true_positions)

    overrides_a, overrides_b = compute_hilt_overrides(
        str(frames_dir), str(masks_a), str(masks_b), str(motion_a), str(motion_b),
    )

    # the run is frames 5-15 (mask overlap); frames 0-4/16-20 are anchors
    # or clean, untouched.
    assert set(overrides_a.keys()) == set(range(5, 16))
    assert set(overrides_b.keys()) == set(range(5, 16))


def test_compute_hilt_overrides_skips_excluded_ranges(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )
    masks_a, masks_b, motion_a, motion_b = _write_overlap_run_fixture(tmp_path, true_positions)

    overrides_a, overrides_b = compute_hilt_overrides(
        str(frames_dir), str(masks_a), str(masks_b), str(motion_a), str(motion_b),
        exclude_frame_ranges=[(5, 15)],
    )

    assert overrides_a == {}
    assert overrides_b == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_hilt_track.py -v -k compute_hilt_overrides`
Expected: FAIL -- `compute_hilt_overrides` not defined.

- [ ] **Step 3: Implement**

```python
# append to src/lightsaber_fx/pipeline/hilt_track.py

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
    validate simply contributes nothing to that dict -- no exception,
    no partial/unvalidated data.
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
        start_frame, end_frame = frame_indices[run_start], frame_indices[run_end]
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_hilt_track.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Run ruff**

Run: `ruff check src/lightsaber_fx/pipeline/hilt_track.py tests/pipeline/test_hilt_track.py`
Expected: no issues

- [ ] **Step 6: Commit**

```bash
git add src/lightsaber_fx/pipeline/hilt_track.py tests/pipeline/test_hilt_track.py
git commit -m "Add compute_hilt_overrides: loop unresolved overlap runs per object"
```

---

### Task 4: Wire hilt overrides into `blade.suppress_overlap_bleed`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/blade.py:860-943` (function signature at line 860, insert after the existing `_smooth_interpolate_run` calls at lines 942-943)
- Test: Modify `tests/pipeline/test_blade.py` (append)

**Interfaces:**
- Consumes: nothing new from other tasks -- `hilt_overrides_a`/`hilt_overrides_b` are plain `{frame_number: (x, y)}` dicts, the same shape `hilt_track.compute_hilt_overrides` produces (Task 3), but this task's tests build them directly without importing `hilt_track` at all (keeps `blade.py` decoupled from `hilt_track.py` -- only `runner.py` imports both).
- Produces: `suppress_overlap_bleed(..., hilt_overrides_a=None, hilt_overrides_b=None)`, `_apply_hilt_overrides(motion, run_start, run_end, frame_indices, hilt_overrides)`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/pipeline/test_blade.py

def test_suppress_overlap_bleed_uses_a_hilt_override_when_provided(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 7
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2, 3, 4, 5})
    _write_lengths(motion_a, [100] * n)
    _write_lengths(motion_b, [100, 120, 150, 500, 150, 120, 100])

    # A hilt position far from anything the smoother alone would produce
    # at frame 3, to make the override's effect unambiguous.
    suppress_overlap_bleed(
        str(motion_a), str(masks_a), str(motion_b), str(masks_b),
        hilt_overrides_b={3: (9000.0, -9000.0)},
    )

    result_b = load_motion(str(motion_b))
    assert result_b["hilt"][3] == pytest.approx([9000.0, -9000.0])
    # axis/length/angle re-derived from the (smoothed) tip and the new hilt
    tip = result_b["tip"][3]
    expected_length = float(np.hypot(tip[0] - 9000.0, tip[1] - (-9000.0)))
    assert result_b["length"][3] == pytest.approx(expected_length)
    # frames without an override in the dict are unaffected by it
    assert result_b["hilt"][1] != pytest.approx([9000.0, -9000.0])


def test_suppress_overlap_bleed_leaves_centroid_and_width_untouched_by_a_hilt_override(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 7
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2, 3, 4, 5})
    _write_lengths(motion_a, [100] * n)
    _write_lengths(motion_b, [100, 120, 150, 500, 150, 120, 100])

    suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))
    baseline = load_motion(str(motion_b))

    _write_lengths(motion_b, [100, 120, 150, 500, 150, 120, 100])  # reset (suppress_overlap_bleed patches in place)
    suppress_overlap_bleed(
        str(motion_a), str(masks_a), str(motion_b), str(masks_b),
        hilt_overrides_b={3: (9000.0, -9000.0)},
    )
    overridden = load_motion(str(motion_b))

    assert overridden["centroid"][3] == pytest.approx(baseline["centroid"][3])
    assert overridden["width"][3] == pytest.approx(baseline["width"][3])


def test_suppress_overlap_bleed_default_hilt_overrides_behave_exactly_as_before(tmp_path):
    # Regression guard: omitting hilt_overrides_a/b entirely must produce
    # byte-identical output to every pre-existing suppress_overlap_bleed
    # test -- this is a pure addition, not a behavior change by default.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a, [100, 500, 600, 150])
    _write_lengths(motion_b, [200, 500, 600, 250])

    suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))
    result_a = load_motion(str(motion_a))

    assert result_a["length"][0] == pytest.approx(100.0)
    assert result_a["length"][3] == pytest.approx(150.0)
    assert 100.0 < result_a["length"][1] < result_a["length"][2] < 150.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_blade.py -v -k hilt_override`
Expected: FAIL -- `suppress_overlap_bleed()` got an unexpected keyword argument `hilt_overrides_b`.

- [ ] **Step 3: Implement**

In `src/lightsaber_fx/pipeline/blade.py`, change the `suppress_overlap_bleed` signature (currently at line 860):

```python
def suppress_overlap_bleed(motion_path_a, masks_dir_a, motion_path_b, masks_dir_b,
                            iou_threshold=CROSS_OBJECT_OVERLAP_IOU_THRESHOLD,
                            anchor_iou_threshold=None, exclude_frame_ranges=(),
                            hilt_overrides_a=None, hilt_overrides_b=None):
```

Add to the docstring, after the `exclude_frame_ranges` paragraph:

```
    `hilt_overrides_a`/`hilt_overrides_b` (each an optional
    `{frame_number: (x, y)}` dict, e.g. from
    `hilt_track.compute_hilt_overrides`) replace a smoothed run's `hilt`
    with a validated, independently-tracked position for whichever
    frames are present -- see `_apply_hilt_overrides`. `centroid`/`width`
    are left as the smoother produced them; only `hilt` (and
    `axis`/`length`/`angle`, re-derived from it) are affected.
```

Immediately after the existing two `_smooth_interpolate_run` calls (currently lines 942-943, inside the `if before is not None and after is not None:` block):

```python
            weights = _run_confidence_weights(motion_a, motion_b, run_start, run_end, reference_length)
            _smooth_interpolate_run(motion_a, run_start, run_end, before, after, frame_indices, weights)
            _smooth_interpolate_run(motion_b, run_start, run_end, before, after, frame_indices, weights)
            if hilt_overrides_a:
                _apply_hilt_overrides(motion_a, run_start, run_end, frame_indices, hilt_overrides_a)
            if hilt_overrides_b:
                _apply_hilt_overrides(motion_b, run_start, run_end, frame_indices, hilt_overrides_b)
```

Add the new helper immediately above `suppress_overlap_bleed`'s definition (right after `_smooth_interpolate_run`, which currently ends just before it):

```python
def _apply_hilt_overrides(motion, run_start, run_end, frame_indices, hilt_overrides):
    """For every frame in [run_start, run_end] with a validated entry in
    `hilt_overrides` ({frame number: (x, y)}, e.g. from
    `hilt_track.compute_hilt_overrides`), replace `motion`'s `hilt` row
    with it and re-derive `axis`/`length`/`angle` from the
    (already-smoothed) `tip` and the new `hilt` -- the same
    re-derive-from-tip-and-hilt pattern `_smooth_interpolate_run` already
    uses. `centroid`/`width` are left untouched -- hilt-tracking only has
    evidence about the hand's position, not the blade's overall shape.
    """
    for j in range(run_start, run_end + 1):
        frame_num = frame_indices[j]
        if frame_num not in hilt_overrides:
            continue
        motion["hilt"][j] = hilt_overrides[frame_num]
        axis_vec = motion["tip"][j] - motion["hilt"][j]
        norm = np.linalg.norm(axis_vec)
        motion["length"][j] = norm
        if norm > 0:
            motion["axis"][j] = axis_vec / norm
            motion["angle"][j] = np.arctan2(axis_vec[1], axis_vec[0])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_blade.py -v`
Expected: PASS (all tests, including the 3 new ones and every pre-existing `suppress_overlap_bleed` test unchanged)

- [ ] **Step 5: Run the full suite and ruff**

Run: `python -m pytest -q && ruff check src/lightsaber_fx/pipeline/blade.py tests/pipeline/test_blade.py`
Expected: all pass; no new ruff issues (the file already has one unrelated pre-existing `SIM210` issue in the test file -- do not fix it as part of this task, it predates this plan)

- [ ] **Step 6: Commit**

```bash
git add src/lightsaber_fx/pipeline/blade.py tests/pipeline/test_blade.py
git commit -m "Let suppress_overlap_bleed accept validated hilt-tracking overrides"
```

---

### Task 5: Wire `compute_hilt_overrides` into `runner.run_pipeline_multi`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/runner.py` (import list near line 17, and the block around lines 346-375)
- Test: Modify `tests/pipeline/test_runner.py` (append)

**Interfaces:**
- Consumes: `hilt_track.compute_hilt_overrides` (Task 3), `blade.suppress_overlap_bleed`'s new `hilt_overrides_a`/`hilt_overrides_b` params (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/pipeline/test_runner.py

def test_run_pipeline_multi_calls_compute_hilt_overrides_for_a_two_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.retrack_overlap_runs",
        lambda *a, **k: (set(), [(3, 5)]),
    )
    calls = []

    def fake_compute_hilt_overrides(frames_dir, masks_dir_a, masks_dir_b, motion_path_a, motion_path_b,
                                     exclude_frame_ranges=()):
        # Must run after both objects' compute_motion (it reads their
        # finished motion.npz) and before suppress_overlap_bleed.
        assert os.path.exists(motion_path_a)
        assert os.path.exists(motion_path_b)
        calls.append(list(exclude_frame_ranges))
        return {7: (1.0, 2.0)}, {}

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.compute_hilt_overrides", fake_compute_hilt_overrides
    )

    overload_calls = []
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.suppress_overlap_bleed",
        lambda *a, hilt_overrides_a=None, hilt_overrides_b=None, **k: overload_calls.append(
            (hilt_overrides_a, hilt_overrides_b)
        ) or 0,
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert calls == [[(3, 5)]]  # same resolved_ranges threaded through as exclude_frame_ranges
    assert overload_calls == [({7: (1.0, 2.0)}, {})]


def test_run_pipeline_multi_skips_compute_hilt_overrides_for_a_single_saber_job(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("compute_hilt_overrides should not run for a single-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.compute_hilt_overrides", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[{"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"}],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_runner.py -v -k compute_hilt_overrides`
Expected: FAIL -- `module 'lightsaber_fx.pipeline.runner' has no attribute 'compute_hilt_overrides'`.

- [ ] **Step 3: Implement**

In `src/lightsaber_fx/pipeline/runner.py`, add to the import block (near the existing `from .reacquire import reconcile_pair, retrack_overlap_runs` at line 17):

```python
from .hilt_track import compute_hilt_overrides
```

Replace the existing block (currently lines ~346-375):

```python
    if len(object_ids) == 2:
        # Try a real independent re-track through each cross-object
        # overlap run first -- strictly more accurate than
        # suppress_overlap_bleed's geometry interpolation when it
        # validates. Whichever objects it patches need compute_motion
        # re-run (it rewrites mask files, not motion.npz) before anything
        # downstream, including suppress_overlap_bleed itself, sees them.
        retracked, resolved_ranges = retrack_overlap_runs(
            paths["frames_dir"], paths["masks_dirs"][0], paths["masks_dirs"][1],
            paths["motion_paths"][0], paths["motion_paths"][1],
            n_frames, checkpoint_path, config_name, device,
        )
        for oid in retracked:
            track_counts[oid] = compute_motion(
                paths["masks_dirs"][oid], paths["motion_paths"][oid], progress_cb=stage_cb("motion"),
            )

        # Runs after both objects' motion.npz exist (it patches, not
        # produces, so it needs their finished output) and before the
        # usability check below, so a stretch of frames this corrects
        # doesn't spuriously trip the low-elongation warning. Remains the
        # fallback for whatever retrack_overlap_runs above couldn't fix;
        # exclude_frame_ranges excludes only what it already fixed and
        # validated -- see the reconcile_pair comment above for why
        # reconcile_pair's own output isn't included here too.
        suppress_overlap_bleed(
            paths["motion_paths"][0], paths["masks_dirs"][0],
            paths["motion_paths"][1], paths["masks_dirs"][1],
            exclude_frame_ranges=resolved_ranges,
        )
```

with:

```python
    if len(object_ids) == 2:
        # Try a real independent re-track through each cross-object
        # overlap run first -- strictly more accurate than
        # suppress_overlap_bleed's geometry interpolation when it
        # validates. Whichever objects it patches need compute_motion
        # re-run (it rewrites mask files, not motion.npz) before anything
        # downstream, including suppress_overlap_bleed itself, sees them.
        retracked, resolved_ranges = retrack_overlap_runs(
            paths["frames_dir"], paths["masks_dirs"][0], paths["masks_dirs"][1],
            paths["motion_paths"][0], paths["motion_paths"][1],
            n_frames, checkpoint_path, config_name, device,
        )
        for oid in retracked:
            track_counts[oid] = compute_motion(
                paths["masks_dirs"][oid], paths["motion_paths"][oid], progress_cb=stage_cb("motion"),
            )

        # For whatever overlap run retrack_overlap_runs above couldn't
        # resolve, try recovering each object's hilt (hand/grip) position
        # via optical flow on the raw frames -- a different technique
        # from anything else in this pipeline (it never touches SAM2
        # masks), so it can succeed exactly where the mask-based
        # approaches got confused. Same exclude_frame_ranges as
        # suppress_overlap_bleed below -- a run retrack_overlap_runs
        # already resolved needs nothing further.
        hilt_overrides_0, hilt_overrides_1 = compute_hilt_overrides(
            paths["frames_dir"], paths["masks_dirs"][0], paths["masks_dirs"][1],
            paths["motion_paths"][0], paths["motion_paths"][1],
            exclude_frame_ranges=resolved_ranges,
        )

        # Runs after both objects' motion.npz exist (it patches, not
        # produces, so it needs their finished output) and before the
        # usability check below, so a stretch of frames this corrects
        # doesn't spuriously trip the low-elongation warning. Remains the
        # fallback for whatever retrack_overlap_runs above couldn't fix;
        # exclude_frame_ranges excludes only what it already fixed and
        # validated -- see the reconcile_pair comment above for why
        # reconcile_pair's own output isn't included here too.
        suppress_overlap_bleed(
            paths["motion_paths"][0], paths["masks_dirs"][0],
            paths["motion_paths"][1], paths["masks_dirs"][1],
            exclude_frame_ranges=resolved_ranges,
            hilt_overrides_a=hilt_overrides_0, hilt_overrides_b=hilt_overrides_1,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_runner.py -v`
Expected: PASS (all tests, including the 2 new ones)

- [ ] **Step 5: Run the full suite and ruff**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest -q && ruff check src/lightsaber_fx/pipeline/ tests/pipeline/`
Expected: all pass; no new ruff issues

- [ ] **Step 6: Commit**

```bash
git add src/lightsaber_fx/pipeline/runner.py tests/pipeline/test_runner.py
git commit -m "Wire compute_hilt_overrides into run_pipeline_multi's two-saber path"
```

---

### Task 6: Real-footage validation, re-render, and finish

Not a TDD task -- this is the same real-data validation rigor every fix this session has used, plus the project's established end-of-cycle wrap-up.

**Files:** none (validation only; no code changes expected unless real footage surfaces a bug)

- [ ] **Step 1: Re-run compute_motion + retrack_overlap_runs equivalent stages against the real job**

The real job's masks (already fully tracked from the actual end-to-end run) live at:
```
JOB=/private/tmp/claude-501/-Users-danm-Development-lightsaber-fx/120e85ed-3352-4317-b8d2-346ddcba8e89/scratchpad/final_e2e_test/job
```
`$JOB/masks/0`, `$JOB/masks/1` are real, `$JOB/frames/` has the raw extracted JPEGs `hilt_track.py` reads directly. Re-generate fresh motion (mirroring the already-validated `motion_smoothed_fix` from the prior fix) and compute hilt overrides with WARNING logging on:

```python
import logging, sys, os
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
sys.path.insert(0, "src")
from lightsaber_fx.pipeline.blade import compute_motion, suppress_overlap_bleed
from lightsaber_fx.pipeline.hilt_track import compute_hilt_overrides

JOB = "/private/tmp/claude-501/-Users-danm-Development-lightsaber-fx/120e85ed-3352-4317-b8d2-346ddcba8e89/scratchpad/final_e2e_test/job"
outdir = f"{JOB}/motion_hilt_fix"
os.makedirs(outdir, exist_ok=True)
m0, m1 = f"{outdir}/0.npz", f"{outdir}/1.npz"
compute_motion(f"{JOB}/masks/0", m0)
compute_motion(f"{JOB}/masks/1", m1)

overrides_0, overrides_1 = compute_hilt_overrides(f"{JOB}/frames", f"{JOB}/masks/0", f"{JOB}/masks/1", m0, m1)
print("object 0 overrides:", len(overrides_0), "frames")
print("object 1 overrides:", len(overrides_1), "frames")

suppress_overlap_bleed(m0, f"{JOB}/masks/0", m1, f"{JOB}/masks/1",
                        hilt_overrides_a=overrides_0, hilt_overrides_b=overrides_1)
```

Expected: both override dicts cover a large majority of the 293-454 run (162 frames) -- the spike (done before this plan was written) validated both directions for both objects across this exact span. If either dict is empty or covers only a handful of frames, STOP and investigate why before continuing (re-check the real drift/seed-count numbers the same way the spike did, don't guess).

- [ ] **Step 2: Re-render and visually inspect broadly across the run**

Reuse the render pattern already established this session (`render_glow_multi` + `mux.encode`, see `/private/tmp/claude-501/.../scratchpad/rerender_smoothed_fix.py` for the exact structure to copy, pointing at `motion_hilt_fix` instead of `motion_smoothed_fix`). Render the full clip, then use the Read tool to visually inspect frames **throughout** the 293-454 run, not just its edges -- the previous fix's regression was only caught by checking a broad spread of frames including the deep middle (360-430), so check at least: 300, 320, 350, 360, 380, 400, 410, 420, 440. Confirm both glow blades stay anchored to each fencer's hand through the previously-dead middle (360-430), compared against both the raw source frames and the prior (`glow_frames_smoothed_fix`) render at the same frame numbers.

If any frame in the previously-dead zone still shows a floating/disconnected blade, check whether that frame has a hilt override at all (`overrides_0`/`overrides_1` from Step 1) -- if not, that's an honest remaining limit (hilt tracking itself couldn't validate there, e.g. hands also occluded/merged at that instant) and should be reported as such, not treated as a bug to force-fix.

- [ ] **Step 3: Run the full test suite and ruff one final time**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest -q && ruff check src/lightsaber_fx/pipeline/ tests/pipeline/`
Expected: all pass; only the pre-existing unrelated `SIM210` issue in `test_blade.py` (not introduced by this plan)

- [ ] **Step 4: Commit (only if real-data validation confirms a genuine improvement), push, update Desktop copy**

```bash
git add -A
git commit -m "$(cat <<'EOF'
<write based on actual Step 1/2 findings -- include the override
coverage numbers and which frames were visually confirmed fixed>
EOF
)"
git push origin master
cp <path to the new rendered mp4> ~/Desktop/lightsaber_fx_e2e_final_fixed.mp4
open -R ~/Desktop/lightsaber_fx_e2e_final_fixed.mp4
```

Report back to the user: override coverage (how much of the 293-454 run got a validated hilt position), which frames were visually confirmed fixed, and an honest account of anything that's still imperfect and why (matching the reporting style used for every fix this session).
