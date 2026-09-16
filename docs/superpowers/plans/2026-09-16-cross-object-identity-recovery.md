# Cross-Object Identity Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Recover a tracked object's identity after two tracked objects (e.g. two swords) visually cross and SAM2's shared tracking session permanently merges their identities, instead of silently rendering the wrong data or leaving the failure undocumented.

**Architecture:** A new, self-contained module (`pipeline/reacquire.py`) runs as a post-hoc reconciliation stage between `track_objects()` and `compute_motion()` in `runner.py`'s multi-object render path. It reads the masks `track_objects()` already wrote, detects a sustained merge via mask IoU, walks backward to a clean pre-merge reference frame, re-detects the lost object with Gemini once it's visually separable again, re-tracks it forward with a fresh SAM2 session, and patches its mask files in place. Every other pipeline stage is unmodified and has no awareness reconciliation happened.

**Tech Stack:** Python, NumPy, OpenCV, SAM2 (`sam2.build_sam`), `google-genai` (Gemini), pytest.

**Spec:** `docs/superpowers/specs/2026-09-16-cross-object-identity-recovery-design.md`

## Global Constraints

- Reconciliation runs **only** for jobs tracking exactly 2 objects (pairwise-only scope). Jobs with 1, 3, or 4 objects skip it entirely.
- Reconciliation **never raises** and never blocks a render. Any failure (no merge found, no `GEMINI_API_KEY`, a Gemini error, no clean separation found within the search window) leaves both objects' masks exactly as `track_objects()` produced them.
- The lost object's blade holds at its last known-good (frozen) position during the crossing itself, per the approved design -- it does not disappear.
- All frame-count/threshold defaults (IoU 0.8, 15-frame sustain, 90-frame lookback, 10-frame search step, 150-frame search cap, 0.1 max-overlap IoU for a clean re-acquisition) are named module-level constants in `reacquire.py`, not magic numbers inline -- tune later without hunting through the file.
- New code lives entirely in one new file, `src/lightsaber_fx/pipeline/reacquire.py`, plus one integration point in `runner.py` and its tests in `runner.py`'s existing test file. No other pipeline module is modified.
- Tests never make a real SAM2 or Gemini call (both are "far too slow for tests," per this codebase's existing testing notes). Every test either exercises a pure function directly, or injects a fake `client` (matching `detect_blades_vlm`'s exact `client=None` convention) and monkeypatches `_build_image_predictor`/`track_object` (matching `test_vision_detect.py`'s and `test_runner.py`'s exact existing conventions).

---

## Codebase reference (read this before starting any task)

These exact signatures and file locations are used throughout the tasks below. Look them up yourself if anything is unclear rather than guessing -- the line numbers below are for orientation, not a promise they won't have shifted by a line or two.

- `save_mask(masks_dir, frame_idx, mask)`, `load_mask(masks_dir, frame_idx)`, `load_mask_optional(masks_dir, frame_idx)`, `mask_frame_indices(masks_dir)` -- `src/lightsaber_fx/pipeline/blade.py`.
- `fit_blade(mask, taper_frac=1/3, width_bins=20) -> BladeGeometry | None` -- `src/lightsaber_fx/pipeline/blade.py`. `BladeGeometry` has fields `centroid, axis, tip, hilt, length, width, angle`; `centroid` is `(x, y)`.
- `track_object(frames_dir, masks_dir, points, labels, checkpoint_path, config_name, device, n_frames, prompt_frame=0, progress_cb=None)` -- `src/lightsaber_fx/pipeline/track.py:78`. Propagates both directions from `prompt_frame` and writes one mask per frame to `masks_dir`.
- `_build_image_predictor(checkpoint_path, config_name, device)`, `_points_on_axis(mask, fractions=(0.3, 0.5, 0.7)) -> list[[x, y], ...]`, `MAX_MASK_AREA_FRAC = 0.25` -- `src/lightsaber_fx/pipeline/detect.py`.
- `DETECTION_PROMPT`, `DETECTION_SCHEMA`, `GEMINI_MODEL`, `GEMINI_TIMEOUT_MS`, `_parse_gemini_response(response_text, frame_width, frame_height) -> list[[x0,y0,x1,y1]]`, `_validate_box_mask(mask, max_mask_area) -> (elongation, geometry) | None`, `_mask_iou(mask_a, mask_b) -> float` -- `src/lightsaber_fx/pipeline/vision_detect.py`.
- `extract_frames(video_path, frames_dir) -> (fps, n_frames)` writes `frames_dir/{i:05d}.jpg` for every frame -- `src/lightsaber_fx/pipeline/frames.py`. This is what `reacquire_pair` reads frames from (not the original input video).
- `run_pipeline_multi(...)` -- `src/lightsaber_fx/pipeline/runner.py:249`. Calls `track_objects(...)` around line 307, then loops `compute_motion(...)` per `oid` in `object_ids` around line 312. `object_ids = list(range(len(sabers)))`, so for a 2-saber job this is always `[0, 1]`. `paths["masks_dirs"]` is `{oid: path}`; `paths["frames_dir"]` is the shared frame-JPEG directory.
- Gemini client injection and SAM2 predictor stubbing conventions -- `tests/pipeline/test_vision_detect.py` (`_FakeGenaiResponse`, `_FakeGenaiClient`, `_FakePredictor`, and `monkeypatch.setattr("lightsaber_fx.pipeline.vision_detect._build_image_predictor", ...)`).
- `track_objects`/`track_object` stubbing convention in `runner.py` tests -- `tests/pipeline/test_runner.py` (`_fake_track_objects_writing`, `_blade`, `_blank`, per-test `fail_if_called`).
- `tiny_video_path` fixture (5-frame, 64x48, 10fps synthetic clip) -- `tests/conftest.py`. Too short for this plan's merge-detection tests (needs 15+ sustained frames); Task 8 writes its own longer synthetic clip.

---

### Task 1: `reacquire.py` module + `detect_merge`

**Files:**
- Create: `src/lightsaber_fx/pipeline/reacquire.py`
- Create: `tests/pipeline/test_reacquire.py`

**Interfaces:**
- Produces: `detect_merge(masks_dir_a, masks_dir_b, frame_indices, iou_threshold=MERGE_IOU_THRESHOLD, sustain_frames=MERGE_SUSTAIN_FRAMES) -> int | None`. Constants `MERGE_IOU_THRESHOLD = 0.8`, `MERGE_SUSTAIN_FRAMES = 15`.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_reacquire.py`:

```python
import numpy as np

from lightsaber_fx.pipeline.blade import save_mask
from lightsaber_fx.pipeline.reacquire import detect_merge


def _mask_at(x, width=120, height=80, bar_width=6):
    """A vertical bar mask, `bar_width` wide, centered at column `x`."""
    mask = np.zeros((height, width), dtype=bool)
    x0 = max(0, x - bar_width // 2)
    x1 = min(width, x + bar_width // 2)
    mask[10:70, x0:x1] = True
    return mask


def test_detect_merge_finds_the_first_frame_of_a_sustained_overlap(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    # Frames 0-9: two separate bars. Frames 10-29: identical (merged) bars.
    for i in range(30):
        x = 20 if i < 10 else 60
        save_mask(str(masks_a), i, _mask_at(x))
        save_mask(str(masks_b), i, _mask_at(60))

    merge_start = detect_merge(str(masks_a), str(masks_b), list(range(30)))

    assert merge_start == 10


def test_detect_merge_ignores_a_brief_touch_that_separates_again(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(30):
        # Bars touch (identical) for frames 10-14 only -- 5 frames, short of
        # the 15-frame sustain default -- then separate again.
        x = 60 if 10 <= i < 15 else 20
        save_mask(str(masks_a), i, _mask_at(x))
        save_mask(str(masks_b), i, _mask_at(60))

    merge_start = detect_merge(str(masks_a), str(masks_b), list(range(30)))

    assert merge_start is None


def test_detect_merge_returns_none_when_never_overlapping(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(20):
        save_mask(str(masks_a), i, _mask_at(20))
        save_mask(str(masks_b), i, _mask_at(90))

    assert detect_merge(str(masks_a), str(masks_b), list(range(20))) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v`
Expected: collection error or `ModuleNotFoundError: No module named 'lightsaber_fx.pipeline.reacquire'` (the module doesn't exist yet).

- [ ] **Step 3: Write the module and the minimal implementation**

Create `src/lightsaber_fx/pipeline/reacquire.py`:

```python
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

import numpy as np

from .blade import fit_blade, load_mask
from .vision_detect import _mask_iou

MERGE_IOU_THRESHOLD = 0.8
MERGE_SUSTAIN_FRAMES = 15


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
    here.
    """
    run_start = None
    run_len = 0
    for idx in frame_indices:
        iou = _mask_iou(load_mask(masks_dir_a, idx), load_mask(masks_dir_b, idx))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add src/lightsaber_fx/pipeline/reacquire.py tests/pipeline/test_reacquire.py
git commit -m "$(cat <<'EOF'
Add reacquire.py with detect_merge for cross-object identity recovery

First piece of the post-hoc reconciliation stage described in
docs/superpowers/specs/2026-09-16-cross-object-identity-recovery-design.md:
detects a sustained mask-IoU merge between two tracked objects, the
signal that SAM2's shared tracking session has lost the distinction
between them.
EOF
)"
```

---

### Task 2: `find_clean_reference`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/reacquire.py`
- Modify: `tests/pipeline/test_reacquire.py`

**Interfaces:**
- Consumes: `fit_blade`, `load_mask` (already imported in Task 1).
- Produces: `find_clean_reference(masks_dir_a, masks_dir_b, merge_start_frame, frame_indices, lookback_frames=REFERENCE_LOOKBACK_FRAMES) -> int | None`. Constant `REFERENCE_LOOKBACK_FRAMES = 90`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_reacquire.py`:

```python
from lightsaber_fx.pipeline.reacquire import find_clean_reference


def test_find_clean_reference_returns_the_frame_of_maximum_separation(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    # Object a drifts from x=100 toward object b (fixed at x=60) over 20
    # frames -- separation shrinks every frame, so frame 0 is the true
    # clean reference (max separation), not frame 19 (one before the
    # "merge" at frame 20).
    for i in range(20):
        save_mask(str(masks_a), i, _mask_at(100 - i * 2))
        save_mask(str(masks_b), i, _mask_at(60))

    ref = find_clean_reference(
        str(masks_a), str(masks_b), merge_start_frame=20,
        frame_indices=list(range(20)), lookback_frames=90,
    )

    assert ref == 0


def test_find_clean_reference_only_searches_within_the_lookback_window(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(20):
        save_mask(str(masks_a), i, _mask_at(100 - i * 2))
        save_mask(str(masks_b), i, _mask_at(60))

    # Lookback of only 5 frames: frame 0 (the true max) is out of range, so
    # the best candidate within [15, 20) is frame 15 (separation 10px, vs
    # 8/6/4/2 at frames 16-19).
    ref = find_clean_reference(
        str(masks_a), str(masks_b), merge_start_frame=20,
        frame_indices=list(range(20)), lookback_frames=5,
    )

    assert ref == 15


def test_find_clean_reference_returns_none_when_no_candidate_has_valid_geometry(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    empty = np.zeros((80, 120), dtype=bool)
    for i in range(20):
        save_mask(str(masks_a), i, empty)
        save_mask(str(masks_b), i, empty)

    ref = find_clean_reference(
        str(masks_a), str(masks_b), merge_start_frame=20,
        frame_indices=list(range(20)), lookback_frames=90,
    )

    assert ref is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v -k find_clean_reference`
Expected: `ImportError: cannot import name 'find_clean_reference'`.

- [ ] **Step 3: Write the minimal implementation**

In `src/lightsaber_fx/pipeline/reacquire.py`, add below `MERGE_SUSTAIN_FRAMES = 15`:

```python
REFERENCE_LOOKBACK_FRAMES = 90
```

And append the function after `detect_merge`:

```python
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
    """
    candidates = [idx for idx in frame_indices if merge_start_frame - lookback_frames <= idx < merge_start_frame]
    best_idx, best_dist = None, -1.0
    for idx in candidates:
        geo_a = fit_blade(load_mask(masks_dir_a, idx))
        geo_b = fit_blade(load_mask(masks_dir_b, idx))
        if geo_a is None or geo_b is None:
            continue
        dist = float(np.hypot(geo_a.centroid[0] - geo_b.centroid[0], geo_a.centroid[1] - geo_b.centroid[1]))
        if dist > best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v`
Expected: 6 passed (3 from Task 1 + 3 new).

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add src/lightsaber_fx/pipeline/reacquire.py tests/pipeline/test_reacquire.py
git commit -m "$(cat <<'EOF'
Add find_clean_reference for cross-object identity recovery

Walks backward from a detected merge to the true local-maximum-
separation frame, not just "the last frame before the merge" -- that
frame is often already contaminated by the same drift that caused the
merge, which produced a swapped identity match when validated against
real footage.
EOF
)"
```

---

### Task 3: `match_detections_to_objects`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/reacquire.py`
- Modify: `tests/pipeline/test_reacquire.py`

**Interfaces:**
- Produces: `match_detections_to_objects(detections, ref_centroid_a, ref_centroid_b) -> (dict, dict)`. `detections` is a list of exactly 2 dicts, each with at least a `"centroid": (x, y)` key. Returns `(detection_for_a, detection_for_b)` -- the same two input dicts, reordered.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_reacquire.py`:

```python
from lightsaber_fx.pipeline.reacquire import match_detections_to_objects


def test_match_detections_to_objects_matches_by_nearest_centroid():
    det_left = {"centroid": (100, 100), "points": [[100, 100]]}
    det_right = {"centroid": (500, 500), "points": [[500, 500]]}

    for_a, for_b = match_detections_to_objects(
        [det_right, det_left],  # input order shouldn't matter
        ref_centroid_a=(90, 90), ref_centroid_b=(510, 510),
    )

    assert for_a is det_left
    assert for_b is det_right


def test_match_detections_to_objects_handles_already_matching_order():
    det_a_like = {"centroid": (10, 10), "points": []}
    det_b_like = {"centroid": (200, 200), "points": []}

    for_a, for_b = match_detections_to_objects(
        [det_a_like, det_b_like], ref_centroid_a=(0, 0), ref_centroid_b=(210, 210),
    )

    assert for_a is det_a_like
    assert for_b is det_b_like
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v -k match_detections`
Expected: `ImportError: cannot import name 'match_detections_to_objects'`.

- [ ] **Step 3: Write the minimal implementation**

Append to `src/lightsaber_fx/pipeline/reacquire.py`, after `find_clean_reference`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add src/lightsaber_fx/pipeline/reacquire.py tests/pipeline/test_reacquire.py
git commit -m "$(cat <<'EOF'
Add match_detections_to_objects for cross-object identity recovery

Minimal-total-distance assignment of 2 freshly re-detected objects back
to their original tracked identities, given a clean reference frame's
centroids.
EOF
)"
```

---

### Task 4: `patch_masks`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/reacquire.py`
- Modify: `tests/pipeline/test_reacquire.py`

**Interfaces:**
- Consumes: `save_mask` (new import from `.blade`).
- Produces: `patch_masks(lost_masks_dir, fresh_masks_dir, frozen_frame_idx, merge_start_frame, reacquire_frame, n_frames) -> None`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_reacquire.py`:

```python
from lightsaber_fx.pipeline.blade import load_mask
from lightsaber_fx.pipeline.reacquire import patch_masks


def test_patch_masks_freezes_the_gap_and_copies_the_fresh_track(tmp_path):
    lost = tmp_path / "lost"
    fresh = tmp_path / "fresh"
    for i in range(10):
        save_mask(str(lost), i, _mask_at(20))  # the object's own track before/through the merge
    for i in range(6, 10):
        save_mask(str(fresh), i, _mask_at(80))  # the freshly re-tracked object, frames 6-9

    patch_masks(str(lost), str(fresh), frozen_frame_idx=4, merge_start_frame=5, reacquire_frame=6, n_frames=10)

    # Frames 0-4: untouched (still the object's own original track).
    for i in range(5):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(20))
    # Frame 5 (the gap): frozen copy of frame 4's mask.
    assert np.array_equal(load_mask(str(lost), 5), _mask_at(20))
    # Frames 6-9: the freshly re-tracked mask.
    for i in range(6, 10):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(80))


def test_patch_masks_handles_a_zero_length_gap(tmp_path):
    lost = tmp_path / "lost"
    fresh = tmp_path / "fresh"
    for i in range(5):
        save_mask(str(lost), i, _mask_at(20))
    for i in range(3, 5):
        save_mask(str(fresh), i, _mask_at(80))

    # reacquire_frame == merge_start_frame: no frozen gap at all.
    patch_masks(str(lost), str(fresh), frozen_frame_idx=2, merge_start_frame=3, reacquire_frame=3, n_frames=5)

    for i in range(3):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(20))
    for i in range(3, 5):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(80))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v -k patch_masks`
Expected: `ImportError: cannot import name 'patch_masks'`.

- [ ] **Step 3: Write the minimal implementation**

Update the import line in `src/lightsaber_fx/pipeline/reacquire.py` from:

```python
from .blade import fit_blade, load_mask
```

to:

```python
from .blade import fit_blade, load_mask, save_mask
```

Append the function after `match_detections_to_objects`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add src/lightsaber_fx/pipeline/reacquire.py tests/pipeline/test_reacquire.py
git commit -m "$(cat <<'EOF'
Add patch_masks for cross-object identity recovery

Pure file-level patch: freezes the lost object's last known-good mask
through the crossing gap, then splices in the freshly re-tracked masks
from the re-acquisition frame onward. No SAM2/Gemini dependency, so this
is directly unit-testable.
EOF
)"
```

---

### Task 5: `reacquire_pair`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/reacquire.py`
- Modify: `tests/pipeline/test_reacquire.py`

**Interfaces:**
- Consumes: `_build_image_predictor`, `_points_on_axis`, `MAX_MASK_AREA_FRAC` (from `.detect`); `DETECTION_PROMPT`, `DETECTION_SCHEMA`, `GEMINI_MODEL`, `GEMINI_TIMEOUT_MS`, `_parse_gemini_response`, `_validate_box_mask`, `_mask_iou` (from `.vision_detect`, `_mask_iou` already imported in Task 1).
- Produces: `reacquire_pair(frames_dir, search_start_frame, checkpoint_path, config_name, device, client=None, search_step=REACQUIRE_SEARCH_STEP_FRAMES, search_cap=REACQUIRE_SEARCH_CAP_FRAMES) -> (int, list[dict]) | None`. Each returned detection dict has `"centroid": (x, y)` and `"points": [[x, y], ...]`. Constants `REACQUIRE_SEARCH_STEP_FRAMES = 10`, `REACQUIRE_SEARCH_CAP_FRAMES = 150`, `REACQUIRE_MAX_OVERLAP_IOU = 0.1`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_reacquire.py`:

```python
import json

import cv2

from lightsaber_fx.pipeline.reacquire import reacquire_pair

FRAME_W, FRAME_H = 1280, 720


class _FakeGenaiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGenaiClient:
    """`text_for_call(n)` returns the response text for the n-th call
    (0-indexed), so a test can simulate different frames returning
    different detections as reacquire_pair walks forward."""

    def __init__(self, text_for_call):
        self.text_for_call = text_for_call
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        text = self.text_for_call(self.calls)
        self.calls += 1
        return _FakeGenaiResponse(text)


class _FakePredictor:
    def __init__(self, mask_for_box):
        self.mask_for_box = mask_for_box

    def set_image(self, image):
        pass

    def predict(self, box, multimask_output=False):
        mask = self.mask_for_box(box)
        return np.asarray([mask]), np.ones(1), None


def _write_frame(path, value=100):
    frame = np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def _line_mask_at(box, width=6):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cv2.line(mask, (x0, y0), (x1, y1), 1, width)
    return mask.astype(bool)


def _no_boxes_response():
    return json.dumps({"objects": []})


def _two_separated_boxes_response():
    # Two thin horizontal boxes, one on the left third of the frame, one
    # on the right third -- far enough apart their line masks won't overlap.
    return json.dumps({"objects": [
        {"box_2d": [340, 50, 360, 300], "label": "blade"},
        {"box_2d": [340, 700, 360, 950], "label": "blade"},
    ]})


def _two_overlapping_boxes_response():
    return json.dumps({"objects": [
        {"box_2d": [340, 50, 360, 300], "label": "blade"},
        {"box_2d": [340, 60, 360, 310], "label": "blade"},  # nearly identical -> overlapping masks
    ]})


def test_reacquire_pair_finds_two_separated_detections_on_the_first_checked_frame(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frame(frames_dir / "00050.jpg")

    client = _FakeGenaiClient(lambda call: _two_separated_boxes_response())
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client,
    )

    assert result is not None
    reacquire_frame, detections = result
    assert reacquire_frame == 50
    assert len(detections) == 2
    centroids_x = sorted(d["centroid"][0] for d in detections)
    assert centroids_x[0] < FRAME_W / 2 < centroids_x[1]  # one left-ish, one right-ish


def test_reacquire_pair_keeps_searching_past_frames_that_are_not_yet_separated(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in (50, 60, 70):
        _write_frame(frames_dir / f"{idx:05d}.jpg")

    # Frame 50 (1st check): nothing found. Frame 60 (2nd check): two boxes,
    # but still overlapping. Frame 70 (3rd check): finally separated.
    responses = [_no_boxes_response(), _two_overlapping_boxes_response(), _two_separated_boxes_response()]
    client = _FakeGenaiClient(lambda call: responses[call])
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client, search_step=10,
    )

    assert result is not None
    reacquire_frame, _detections = result
    assert reacquire_frame == 70


def test_reacquire_pair_returns_none_when_the_search_window_is_exhausted(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in (50, 60, 70):
        _write_frame(frames_dir / f"{idx:05d}.jpg")

    client = _FakeGenaiClient(lambda call: _no_boxes_response())
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client, search_step=10, search_cap=30,
    )

    assert result is None


def test_reacquire_pair_returns_none_when_it_runs_past_the_end_of_the_clip(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frame(frames_dir / "00050.jpg")
    # No frame 60 written -- the clip ends before the search budget does.

    client = _FakeGenaiClient(lambda call: _no_boxes_response())
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client, search_step=10, search_cap=150,
    )

    assert result is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v -k reacquire_pair`
Expected: `ImportError: cannot import name 'reacquire_pair'`.

- [ ] **Step 3: Write the minimal implementation**

Update the top of `src/lightsaber_fx/pipeline/reacquire.py`. Replace:

```python
import numpy as np

from .blade import fit_blade, load_mask, save_mask
from .vision_detect import _mask_iou
```

with:

```python
import os

import cv2
import numpy as np

from .blade import fit_blade, load_mask, save_mask
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
```

Add below `REFERENCE_LOOKBACK_FRAMES = 90`:

```python
REACQUIRE_SEARCH_STEP_FRAMES = 10
REACQUIRE_SEARCH_CAP_FRAMES = 150
REACQUIRE_MAX_OVERLAP_IOU = 0.1
```

Append to the end of the file:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v`
Expected: 14 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add src/lightsaber_fx/pipeline/reacquire.py tests/pipeline/test_reacquire.py
git commit -m "$(cat <<'EOF'
Add reacquire_pair for cross-object identity recovery

Walks forward from a detected merge, asking Gemini (same call shape as
detect_blades_vlm, pointed at a specific frame) to re-find both objects
once they're visually separable again. Client is injectable and SAM2's
image predictor is monkeypatchable, matching detect_blades_vlm's own
testing conventions -- no real network or SAM2 calls in tests.
EOF
)"
```

---

### Task 6: `reconcile_pair`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/reacquire.py`
- Modify: `tests/pipeline/test_reacquire.py`

**Interfaces:**
- Consumes: `mask_frame_indices` (new import from `.blade`); `track_object` (new import from `.track`); `detect_merge`, `find_clean_reference`, `match_detections_to_objects`, `reacquire_pair`, `patch_masks`, `fit_blade`, `load_mask` (already defined/imported in this module).
- Produces: `reconcile_pair(frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device, client=None) -> bool`. `True` if a patch was applied, `False` otherwise (masks left untouched). Never raises.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_reacquire.py`:

```python
from lightsaber_fx.pipeline.reacquire import reconcile_pair


def test_reconcile_pair_detects_and_patches_the_lost_object(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    # Frames 0-19: separate (object 0 at x=20, object 1 at x=80).
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))
    # Frames 20-49: both objects' tracking collapsed onto object 1's
    # target (x=80) -- object 0 is "lost".
    for i in range(20, 50):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))

    def fake_reacquire_pair(frames_dir, search_start_frame, checkpoint_path, config_name, device,
                             client=None, **kwargs):
        assert search_start_frame == 20  # merge_start_frame
        return 40, [
            {"centroid": (20.0, 45.0), "points": [[20, 40], [20, 45], [20, 50]]},
            {"centroid": (80.0, 45.0), "points": [[80, 40], [80, 45], [80, 50]]},
        ]

    def fake_track_object(frames_dir, out_masks_dir, points, labels, checkpoint_path, config_name, device,
                           n_frames, prompt_frame=0, progress_cb=None):
        # Recovered object 0 tracks at x=25 (distinct from both x=20 and
        # x=80, so the test can tell "frozen reference" and "freshly
        # tracked" apart) from prompt_frame onward.
        for i in range(prompt_frame, n_frames):
            save_mask(out_masks_dir, i, _mask_at(25))

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", fake_reacquire_pair)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.track_object", fake_track_object)

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=50,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is True
    # Object 1 (kept) is untouched throughout.
    for i in range(50):
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(80))
    # Object 0: original track for frames 0-19, frozen at its last-good
    # position (x=20) for the gap [20, 40), then freshly re-tracked (x=25)
    # for [40, 50).
    for i in range(20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20))
    for i in range(20, 40):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20))
    for i in range(40, 50):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(25))


def test_reconcile_pair_returns_false_when_no_merge_is_detected(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))

    called = []
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.reacquire.reacquire_pair",
        lambda *a, **k: called.append(True) or None,
    )

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=20,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is False
    assert called == []  # never even tried to re-acquire -- no merge was found
    for i in range(20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20))
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(80))


def test_reconcile_pair_returns_false_when_reacquisition_finds_nothing(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))
    for i in range(20, 40):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", lambda *a, **k: None)

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=40,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is False
    for i in range(40):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(80))  # untouched


def test_reconcile_pair_returns_false_instead_of_raising_when_reacquisition_errors(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))
    for i in range(20, 40):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))

    def raising_reacquire_pair(*a, **k):
        raise RuntimeError("Gemini network error")

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", raising_reacquire_pair)

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=40,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is False
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v -k reconcile_pair`
Expected: `ImportError: cannot import name 'reconcile_pair'`.

- [ ] **Step 3: Write the minimal implementation**

Update the top of `src/lightsaber_fx/pipeline/reacquire.py`. Replace:

```python
import os

import cv2
import numpy as np

from .blade import fit_blade, load_mask, save_mask
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
```

with:

```python
import logging
import os
import tempfile

import cv2
import numpy as np

from .blade import fit_blade, load_mask, mask_frame_indices, save_mask
from .detect import MAX_MASK_AREA_FRAC, _build_image_predictor, _points_on_axis
from .track import track_object
from .vision_detect import (
    DETECTION_PROMPT,
    DETECTION_SCHEMA,
    GEMINI_MODEL,
    GEMINI_TIMEOUT_MS,
    _mask_iou,
    _parse_gemini_response,
    _validate_box_mask,
)
```

Append to the end of the file:

```python
def reconcile_pair(frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device, client=None):
    """Best-effort recovery from a detected crossing between exactly two
    tracked objects (object 0 and object 1). Returns `True` if either
    object's masks were patched, `False` if no merge was found or
    recovery failed at any step -- in the `False` case, both objects'
    masks are left completely untouched.

    Never raises: any failure in the Gemini/SAM2-dependent steps
    (`reacquire_pair`, the fresh `track_object` call) is caught and
    treated the same as "recovery failed," matching this project's
    existing decline-rather-than-guess-wrong fallback philosophy (see
    `server.py`'s `_detect_proposals`).
    """
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

    det_for_0, det_for_1 = match_detections_to_objects(detections, geo_0.centroid, geo_1.centroid)

    merged_geo = fit_blade(load_mask(masks_dir_0, merge_start))
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_reacquire.py -v`
Expected: 18 passed.

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add src/lightsaber_fx/pipeline/reacquire.py tests/pipeline/test_reacquire.py
git commit -m "$(cat <<'EOF'
Add reconcile_pair, the top-level orchestrator for identity recovery

Wires detect_merge -> find_clean_reference -> reacquire_pair ->
match_detections_to_objects -> patch_masks together, determines which
of the two objects was actually lost by comparing each side's fresh
re-detection against where the merged track currently sits, and never
raises -- any failure anywhere in the chain leaves both objects' masks
exactly as track_objects() produced them.
EOF
)"
```

---

### Task 7: Wire `reconcile_pair` into `run_pipeline_multi`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/runner.py`
- Modify: `tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: `reconcile_pair(frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device, client=None) -> bool` (from Task 6).

- [ ] **Step 1: Write the failing tests**

In `tests/pipeline/test_runner.py`, add near the other `run_pipeline_multi` tests (after `test_run_pipeline_multi_end_to_end_with_stubbed_tracking`, defined around line 598-635):

```python
def test_run_pipeline_multi_calls_reconcile_pair_for_a_two_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    calls = []

    def fake_reconcile_pair(frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device,
                             client=None):
        calls.append((masks_dir_0, masks_dir_1, n_frames))
        return False

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.reconcile_pair", fake_reconcile_pair)

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

    assert len(calls) == 1
    masks_dir_0, masks_dir_1, _n_frames = calls[0]
    assert masks_dir_0 == str(job_dir / "masks" / "0")
    assert masks_dir_1 == str(job_dir / "masks" / "1")


def test_run_pipeline_multi_skips_reconcile_pair_for_a_single_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("reconcile_pair should not run for a single-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.reconcile_pair", fail_if_called)

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


def test_run_pipeline_multi_skips_reconcile_pair_for_a_four_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade, 2: _blade, 3: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("reconcile_pair should not run for a four-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.reconcile_pair", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "green", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_runner.py -v -k reconcile_pair`
Expected: `AttributeError: <module 'lightsaber_fx.pipeline.runner'> does not have the attribute 'reconcile_pair'` (the first test fails at the `monkeypatch.setattr` call since `runner.py` doesn't import it yet).

- [ ] **Step 3: Write the minimal implementation**

In `src/lightsaber_fx/pipeline/runner.py`, update the import block near the top. Replace:

```python
from .track import track_object, track_objects
```

with:

```python
from .reacquire import reconcile_pair
from .track import track_object, track_objects
```

Then, in `run_pipeline_multi`, insert the reconciliation call between the `track_objects(...)` call and the `compute_motion` loop. Find:

```python
    track_objects(
        paths["frames_dir"], prompts, checkpoint_path, config_name, device, n_frames,
        progress_cb=stage_cb("track"),
    )

    for oid in object_ids:
        n_tracked, n_with_blade = compute_motion(
```

and replace with:

```python
    track_objects(
        paths["frames_dir"], prompts, checkpoint_path, config_name, device, n_frames,
        progress_cb=stage_cb("track"),
    )

    if len(object_ids) == 2:
        reconcile_pair(
            paths["frames_dir"], paths["masks_dirs"][0], paths["masks_dirs"][1],
            n_frames, checkpoint_path, config_name, device,
        )

    for oid in object_ids:
        n_tracked, n_with_blade = compute_motion(
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_runner.py -v`
Expected: all tests in the file pass, including the 3 new ones (no regressions in the existing ~30+ tests in this file).

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add src/lightsaber_fx/pipeline/runner.py tests/pipeline/test_runner.py
git commit -m "$(cat <<'EOF'
Wire reconcile_pair into run_pipeline_multi for 2-saber jobs

Runs between track_objects() and compute_motion(), only for jobs
tracking exactly 2 objects (the pairwise-only scope this feature covers).
compute_motion() and everything after it are unmodified and unaware
reconciliation happened -- they just see whatever masks are on disk.
EOF
)"
```

---

### Task 8: End-to-end synthetic integration test through `run_pipeline_multi`

**Files:**
- Modify: `tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: `run_pipeline_multi` (existing), `reconcile_pair`'s monkeypatch points (`lightsaber_fx.pipeline.reacquire.reacquire_pair`, `lightsaber_fx.pipeline.reacquire.track_object`), `load_motion` (new import from `.blade`).

- [ ] **Step 1: Write the failing test**

Add `import cv2` to the top of `tests/pipeline/test_runner.py` (alongside the existing `import numpy as np`), and change:

```python
from lightsaber_fx.pipeline.blade import compute_motion, save_mask
```

to:

```python
from lightsaber_fx.pipeline.blade import compute_motion, load_motion, save_mask
```

Then append this test near the other `run_pipeline_multi` tests:

```python
def _write_longer_video(path, n_frames, width=64, height=48, fps=10.0):
    """Like the top-level `tiny_video_path` fixture's clip, but with a
    frame count this file controls -- merge detection needs 15+ sustained
    frames, more than that fixture's default 5."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_run_pipeline_multi_recovers_from_a_simulated_crossing_end_to_end(tmp_path, monkeypatch):
    """Full pipeline, reproducing the shape of the real bug this feature
    fixes: two objects track separately, then (simulating track_objects
    losing the distinction between them) both collapse onto the same
    target for a sustained stretch, and reconciliation recovers the lost
    one. Mirrors the design doc's real-footage spike; this is the
    synthetic, fast, CI-safe equivalent driven through the full pipeline
    entry point rather than reconcile_pair directly."""
    video_path = tmp_path / "longer.mp4"
    _write_longer_video(video_path, n_frames=40)

    def fake_track_objects(frames_dir, prompts, checkpoint_path, config_name, device, n_frames, progress_cb=None):
        for prompt in prompts:
            os.makedirs(prompt["masks_dir"], exist_ok=True)
            for i in range(n_frames):
                # Object 0 tracks separately (x=10) for frames 0-4, then
                # collapses onto object 1's target (x=40) from frame 5 on.
                x = 10 if (prompt["obj_id"] == 0 and i < 5) else 40
                mask = np.zeros((48, 64), dtype=bool)
                mask[10:34, x:x + 6] = True
                save_mask(prompt["masks_dir"], i, mask)

    def fake_reacquire_pair(frames_dir, search_start_frame, checkpoint_path, config_name, device,
                             client=None, **kwargs):
        return search_start_frame, [
            {"centroid": (10.0, 22.0), "points": [[10, 20], [10, 22], [10, 24]]},
            {"centroid": (40.0, 22.0), "points": [[40, 20], [40, 22], [40, 24]]},
        ]

    def fake_track_object_for_reacquire(frames_dir, out_masks_dir, points, labels, checkpoint_path, config_name,
                                         device, n_frames, prompt_frame=0, progress_cb=None):
        for i in range(prompt_frame, n_frames):
            mask = np.zeros((48, 64), dtype=bool)
            mask[10:34, 10:16] = True
            save_mask(out_masks_dir, i, mask)

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_objects", fake_track_objects)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", fake_reacquire_pair)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.track_object", fake_track_object_for_reacquire)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"

    run_pipeline_multi(
        input_video=str(video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert output_path.exists() and output_path.stat().st_size > 0
    motion_0 = load_motion(str(job_dir / "motion" / "0.npz"))
    # Object 0 was recovered: its centroid on the last tracked frame
    # should be back near its own target (x=10-16), not object 1's (x=40).
    assert motion_0["centroid"][-1][0] < 20
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_runner.py -v -k recovers_from_a_simulated_crossing`
Expected: FAIL. If Tasks 1-7 were completed correctly this should actually pass already; if it fails, read the failure carefully -- it most likely means one of `detect_merge`'s/`find_clean_reference`'s default thresholds isn't reached by this test's 40-frame clip (e.g. the 5-frame separate/35-frame merged split doesn't clear `MERGE_SUSTAIN_FRAMES=15`, or `REFERENCE_LOOKBACK_FRAMES=90` doesn't reach back far enough -- it does here, since all of frames 0-4 fall within a 90-frame lookback from merge_start=5). Do not change any threshold constant to make this pass; if it's failing for a reason other than a bug in this test's own setup, stop and re-check Tasks 1-7's implementations against this plan's Step 3 code before proceeding.

- [ ] **Step 3: No new implementation code**

This task adds no new production code -- Tasks 1-7 already implement everything this test exercises. If Step 2 passes immediately, that confirms the feature; there's nothing to change.

- [ ] **Step 4: Run the full pipeline test suite to verify no regressions**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/ -q`
Expected: all tests pass (the full suite, not just this file -- confirms nothing else broke).

- [ ] **Step 5: Commit**

```bash
cd /Users/danm/Development/lightsaber_fx
git add tests/pipeline/test_runner.py
git commit -m "$(cat <<'EOF'
Add end-to-end test for cross-object identity recovery through the full pipeline

Synthetic reproduction of the real bug's shape (two objects track
separately, then collapse onto one target, then get recovered), driven
through run_pipeline_multi rather than reconcile_pair directly -- proves
the whole chain (runner -> reconcile_pair -> reacquire_pair/track_object
-> patch_masks -> compute_motion -> render) fits together, not just each
piece in isolation.
EOF
)"
```

---

## Post-plan verification

After Task 8, run the complete suite and the lint check this session already used earlier, to confirm the new module introduces no regressions and no new lint findings:

```bash
cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/reacquire.py src/lightsaber_fx/pipeline/runner.py tests/pipeline/test_reacquire.py tests/pipeline/test_runner.py --select F,B,ASYNC,SIM,PERF
```

Both should be clean before considering this feature done. This plan does not include manual verification against the real job (`58a8f662`) as a task -- if you want that confirmation too, re-run the same kind of targeted script this session used for the original spike (see the conversation this plan came from), pointed at the now-real `reconcile_pair` instead of the spike's inline steps.
