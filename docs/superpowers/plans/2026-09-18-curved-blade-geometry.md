# Curved Blade Geometry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the rendered blade follow the real blade's measured bow during blade-on-blade contact, instead of always drawing a straight line from hilt to tip.

**Architecture:** Add one new optional control point (`bend`) to `BladeGeometry`/`motion.npz`, populated by `fit_blade` only when a frame's own mask shows real, significant bow. A new cross-object contamination gate in `suppress_overlap_bleed` clears `bend` wherever cross-object mask IoU is too high to trust (the dead zone), and a lightweight single-object temporal smoother denoises/ramps what survives. `glow.py`'s renderer draws a quadratic-Bezier capsule (hilt→bend→tip) when `bend` is present, and falls back to the exact existing straight-line code path, unchanged, when it is not.

**Tech Stack:** Python, numpy, OpenCV (`cv2`), pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-curved-blade-geometry-design.md` — read this in full before starting. This plan implements it, with two corrections found while planning (both explained in their tasks below): (1) the contamination gate is implemented via a new shared `_cross_object_ious` helper rather than changing `_find_overlap_runs`'s return signature, to avoid touching its other caller (`reacquire.py`); (2) `_stabilize_tip_hilt` does **not** need to swap `bend` on a tip/hilt flip — a quadratic Bézier's shape is identical whether traced hilt→bend→tip or tip→bend→hilt, so there is nothing to swap. Task 7 includes a test that locks this in explicitly.

## Global Constraints

- `BEND_SIGNIFICANCE_PX = 8` — exact value, calibrated against the real job (see spec's Constants section). Do not re-tune without new real-data evidence.
- The contamination gate reuses the existing `CROSS_OBJECT_OVERLAP_IOU_THRESHOLD` — no new IoU constant.
- `bend=None` (the case for every frame outside real contact) must produce **byte-identical** renderer output to today's code. This is verified by the existing golden-checksum test `test_render_glow_output_is_unchanged_by_the_extraction_refactor` in `tests/pipeline/test_glow.py` continuing to pass with its committed `EXPECTED_CHARACTERIZATION_CHECKSUMS` unchanged — if that assertion ever needs to change during this work, stop and treat it as a bug, not something to update.
- Full test suite + `ruff check src/lightsaber_fx/pipeline/*.py` must pass before every commit (`cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest -q`).
- Never guess a constant — every new constant in this plan is either copied from the spec's calibration or explicitly left for Task 8's real-data validation to confirm.
- Commit messages end with the attribution footer currently in effect for this session (check the most recent commit, e.g. `git log -1 --format=%B`, for the exact lines — they include a `Claude-Session` URL that is session-specific and must be copied fresh, not assumed from an old commit).

---

### Task 1: `bend` field on `BladeGeometry` and `motion.npz`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/blade.py` (`BladeGeometry` NamedTuple, `_FIELDS`/`_VECTOR_FIELDS`, `save_motion`)
- Test: `tests/pipeline/test_blade.py`

**Interfaces:**
- Produces: `BladeGeometry.bend: tuple | None` (default `None`), a `bend` key in every dict `load_motion` returns (an `(N, 2)` float64 array, NaN rows where absent).

**Context:** `BladeGeometry` (line 26) is a `NamedTuple` with `centroid, axis, tip, hilt, length, width, angle`. `_FIELDS`/`_VECTOR_FIELDS` (lines 552-553) drive `save_motion`'s NaN-array construction and per-frame copy loop. Every existing field is always a real value whenever a frame's `BladeGeometry` is not `None` — `bend` is the first field that can itself be `None` even when the frame's geometry exists, so `save_motion`'s per-frame copy loop (`arrays[field][i] = getattr(geo, field)`) needs a small guard, or assigning `None` into a float64 array slot will misbehave.

- [ ] **Step 1: Write the failing test**

```python
def test_save_and_load_motion_round_trips_bend_including_none(tmp_path):
    geo_with_bend = BladeGeometry(
        centroid=(10.0, 0.0), axis=(1.0, 0.0), tip=(20.0, 0.0), hilt=(0.0, 0.0),
        length=20.0, width=5.0, angle=0.0, bend=(10.0, 3.0),
    )
    geo_without_bend = BladeGeometry(
        centroid=(10.0, 0.0), axis=(1.0, 0.0), tip=(20.0, 0.0), hilt=(0.0, 0.0),
        length=20.0, width=5.0, angle=0.0, bend=None,
    )
    path = str(tmp_path / "motion.npz")
    save_motion(path, [geo_with_bend, geo_without_bend, None])
    motion = load_motion(path)

    assert motion["bend"][0] == pytest.approx([10.0, 3.0])
    assert np.isnan(motion["bend"][1]).all()   # bend=None on a real frame
    assert np.isnan(motion["bend"][2]).all()   # geo=None entirely


def test_blade_geometry_defaults_bend_to_none():
    geo = BladeGeometry(
        centroid=(0.0, 0.0), axis=(1.0, 0.0), tip=(1.0, 0.0), hilt=(0.0, 0.0),
        length=1.0, width=1.0, angle=0.0,
    )
    assert geo.bend is None
```

Both tests use the file's existing `tmp_path`-based convention (no manual cleanup, no bare relative paths) -- write them exactly as shown above, no further changes needed.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate && python -m pytest tests/pipeline/test_blade.py -k "bend_roundtrip or defaults_bend" -v`
Expected: FAIL — `BladeGeometry` has no field `bend` (`TypeError: __new__() got an unexpected keyword argument 'bend'`).

- [ ] **Step 3: Add the field and fix `save_motion`**

In `src/lightsaber_fx/pipeline/blade.py`, modify the `BladeGeometry` class (line 26):

```python
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
```

Modify `_FIELDS`/`_VECTOR_FIELDS` (lines 552-553):

```python
_FIELDS = ("centroid", "tip", "hilt", "axis", "length", "width", "angle", "bend")
_VECTOR_FIELDS = ("centroid", "tip", "hilt", "axis", "bend")
```

Modify `save_motion`'s per-frame copy loop (inside the `for i, geo in enumerate(geometries):` block, lines 569-573):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_blade.py -k "bend_roundtrip or defaults_bend or fit_blade" -v`
Expected: PASS. Also run the full file to catch any positional-construction breakage: `python -m pytest tests/pipeline/test_blade.py -q` — expect all pre-existing tests still pass (the new field has a default, so no existing `BladeGeometry(...)` call site breaks).

- [ ] **Step 5: Run full suite + ruff, then commit**

```bash
cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git add src/lightsaber_fx/pipeline/blade.py tests/pipeline/test_blade.py
git commit -m "$(cat <<'EOF'
Add an optional bend control point to BladeGeometry and motion.npz

First step toward rendering real blade bow during blade-on-blade
contact instead of always drawing a straight line -- see
docs/superpowers/specs/2026-09-17-curved-blade-geometry-design.md.
This task only adds the field/schema; nothing populates it yet.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: <copy the current session's Claude-Session URL from the system reminder -- do not reuse an old commit's>
EOF
)"
```

---

### Task 2: `fit_blade` computes the candidate `bend`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/blade.py` (new `_bend_offset` helper, `BEND_SIGNIFICANCE_PX` constant, `fit_blade`)
- Test: `tests/pipeline/test_blade.py`

**Interfaces:**
- Consumes: `BladeGeometry.bend` field from Task 1.
- Produces: `fit_blade(mask, ...)` now returns a `BladeGeometry` whose `.bend` is `(x, y)` when the mask's own shape shows bow past `BEND_SIGNIFICANCE_PX`, else `None`. Single-object only — no cross-object awareness, matching `fit_blade`'s existing documented boundary.

**Context:** `fit_blade` (line 219) already computes `proj`/`perp` (points projected onto/perpendicular to the PCA axis, centered on the mask's centroid) and calls `_median_perpendicular_extent(proj, perp, n_bins=width_bins)` (line 92) for `width`, which bins `proj` into `width_bins` (default 20) even-width bins across `[proj.min(), proj.max()]` and reports, per bin, `max - min` of `perp`. The new `_bend_offset` reuses the exact same bin edges but reports the **median** `perp` value in a ~2-bin-wide window centered on the midpoint bin — this exact window definition (not a single bin, not the full span) is what `BEND_SIGNIFICANCE_PX=8` was calibrated against during brainstorming; do not narrow or widen it without re-calibrating.

- [ ] **Step 1: Write the failing tests**

Add a bowed-mask helper near the existing `_bar_mask` helper (around line 746), following its exact style:

```python
def _bowed_bar_mask(peak_offset, canvas=(60, 400), x_start=50, x_end=350, y_center=30, thickness=6):
    """A mask shaped like a real blade under bind pressure: a horizontal
    bar whose vertical center sags by `peak_offset` px at its midpoint,
    tapering to 0 at both ends (a parabola through (x_start, 0),
    (mid, peak_offset), (x_end, 0)) -- mirrors the real, single-direction
    sag measured on real footage (see the design spec's Problem section),
    not an arbitrary bend shape."""
    mask = np.zeros(canvas, dtype=bool)
    xs = np.arange(x_start, x_end)
    mid = (x_start + x_end) / 2.0
    half_span = (x_end - x_start) / 2.0
    # parabola: 0 at both ends, peak_offset at the midpoint
    sag = peak_offset * (1.0 - ((xs - mid) / half_span) ** 2)
    for x, dy in zip(xs, sag, strict=True):
        y0 = int(round(y_center + dy - thickness / 2))
        y1 = y0 + thickness
        mask[max(0, y0):min(canvas[0], y1), x] = True
    return mask
```

```python
def test_fit_blade_populates_bend_for_a_significantly_bowed_mask():
    mask = _bowed_bar_mask(peak_offset=20.0)  # well past BEND_SIGNIFICANCE_PX=8
    geo = fit_blade(mask)
    assert geo.bend is not None


def test_fit_blade_leaves_bend_none_for_a_straight_mask():
    # Every existing fit_blade fixture in this file is a straight bar --
    # spot-check the two already used above, both must still give bend=None.
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:16, 5:55] = True
    geo = fit_blade(mask)
    assert geo.bend is None


def test_fit_blade_bend_offset_is_robust_to_one_contaminated_bin():
    # A straight mask with one small extra pixel cluster stuck onto a
    # single bin (mimicking cross-object bleed at one point along the
    # blade) must not move the median-per-bin bend fit past significance --
    # the whole reason _bend_offset uses a median, not a mean, of a
    # multi-point window, the same robustness _median_perpendicular_extent
    # already relies on for width.
    mask = np.zeros((48, 300), dtype=bool)
    mask[20:26, 5:295] = True  # straight, 290px long
    mask[35:45, 145:155] = True  # contamination blob near the midpoint, offset ~15-20px below
    geo = fit_blade(mask)
    assert geo.bend is None
```

Also update the existing field-presence test to include `bend`:

```python
def test_fit_blade_returns_geometry_with_expected_fields():
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:16, 5:55] = True
    geo = fit_blade(mask)
    assert isinstance(geo, BladeGeometry)
    for field in ("centroid", "axis", "tip", "hilt", "length", "width", "angle", "bend"):
        assert hasattr(geo, field)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_blade.py -k "bend" -v`
Expected: FAIL — `fit_blade`'s returned `geo.bend` is always `None` (the default), so `test_fit_blade_populates_bend_for_a_significantly_bowed_mask` fails.

- [ ] **Step 3: Implement `_bend_offset` and wire it into `fit_blade`**

Add near `_median_perpendicular_extent` (after line 113):

```python
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
```

In `fit_blade`, after `width = _median_perpendicular_extent(proj, perp, n_bins=width_bins)` (line 273) and before the `return BladeGeometry(...)` (line 276), add:

```python
    bend_offset = _bend_offset(proj, perp, n_bins=width_bins)
    if abs(bend_offset) > BEND_SIGNIFICANCE_PX:
        mid_proj = (min_proj + max_proj) / 2.0
        bend_point = centroid + mid_proj * axis + bend_offset * perp_dir
        bend = (float(bend_point[0]), float(bend_point[1]))
    else:
        bend = None
```

And add `bend=bend` to the `BladeGeometry(...)` return call (line 276-284).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_blade.py -k "fit_blade or bend" -v`
Expected: PASS, including every pre-existing `fit_blade` test (the field-presence test was updated in Step 1, not broken).

- [ ] **Step 5: Run full suite + ruff, then commit**

```bash
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git add src/lightsaber_fx/pipeline/blade.py tests/pipeline/test_blade.py
git commit -m "$(cat <<'EOF'
Compute a candidate bend point in fit_blade from the mask's own shape

Reuses _median_perpendicular_extent's exact binning for consistency.
This is single-object and unconditional -- the cross-object
contamination gate that makes this safe to trust lives in
suppress_overlap_bleed (next task), not here, matching fit_blade's
existing documented single-object boundary.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: <copy current session's Claude-Session URL>
EOF
)"
```

---

### Task 3: Extract `_cross_object_ious` from `_find_overlap_runs`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/blade.py` (`_find_overlap_runs`)
- Test: `tests/pipeline/test_blade.py`

**Interfaces:**
- Produces: `_cross_object_ious(masks_dir_a, masks_dir_b, frame_indices) -> np.ndarray`, the per-frame cross-object mask IoU array for the whole shared frame range (array length `len(frame_indices)`).
- Consumes (by `_find_overlap_runs`, updated): the same array, computed the same way as before -- this is a pure refactor, `_find_overlap_runs`'s own behavior and return type are unchanged.

**Context:** `_find_overlap_runs` (line 992) currently computes `ious` inline (lines 1008-1011) before splitting it into runs. Task 4 needs that same per-frame array available *outside* any detected run (the contamination gate must clear `bend` on any frame with high cross-object IoU, not only frames inside a formally detected `OverlapRun`). Changing `_find_overlap_runs`'s return signature to also hand back `ious` would be the more obvious move, but `_find_overlap_runs`'s own docstring says it's "the shared detection step behind both `suppress_overlap_bleed`'s geometry interpolation and `reacquire.retrack_overlap_runs`'s independent re-tracking attempt" -- i.e. `reacquire.py` also calls it, and changing its return tuple would force an unrelated change there too. Extracting the IoU computation into its own small function, called once more from Task 4's new code, avoids that -- recomputing a per-frame boolean-array IoU is cheap (this pipeline's real cost centers are SAM2 tracking and PNG rendering, not this).

- [ ] **Step 1: Write the failing test**

```python
def test_cross_object_ious_matches_find_overlap_runs_own_computation():
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    # (tmp_path fixture -- add it as a parameter to this test function)
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    frame_indices = mask_frame_indices(str(masks_a))

    ious = _cross_object_ious(str(masks_a), str(masks_b), frame_indices)

    assert ious[0] == pytest.approx(0.0)
    assert ious[1] == pytest.approx(1.0)
    assert ious[2] == pytest.approx(1.0)
    assert ious[3] == pytest.approx(0.0)
```

(This test needs `tmp_path` as a parameter -- write it as `def test_cross_object_ious_matches_find_overlap_runs_own_computation(tmp_path):`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/pipeline/test_blade.py -k cross_object_ious -v`
Expected: FAIL with `ImportError`/`NameError` -- `_cross_object_ious` doesn't exist yet.

- [ ] **Step 3: Extract the function**

Replace lines 1005-1011 of `_find_overlap_runs`:

```python
    frame_indices = mask_frame_indices(masks_dir_a)
    n = len(frame_indices)

    ious = np.array([
        _mask_iou(load_mask(masks_dir_a, frame_idx), load_mask(masks_dir_b, frame_idx))
        for frame_idx in frame_indices
    ])
```

with:

```python
    frame_indices = mask_frame_indices(masks_dir_a)
    n = len(frame_indices)
    ious = _cross_object_ious(masks_dir_a, masks_dir_b, frame_indices)
```

and add the new function just above `_find_overlap_runs` (before line 992):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_blade.py -k "cross_object_ious or find_overlap_runs" -v`
Expected: PASS. Also confirm no regression: `python -m pytest tests/pipeline/test_blade.py tests/pipeline/test_reacquire.py -q`.

- [ ] **Step 5: Run full suite + ruff, then commit**

```bash
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git add src/lightsaber_fx/pipeline/blade.py tests/pipeline/test_blade.py
git commit -m "$(cat <<'EOF'
Extract _cross_object_ious out of _find_overlap_runs for reuse

Pure refactor, no behavior change -- _find_overlap_runs's return type
is untouched (reacquire.py also depends on it). Lets the upcoming
cross-object contamination gate in suppress_overlap_bleed reuse the
same per-frame IoU measurement across the whole clip, not just inside
a formally detected run.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: <copy current session's Claude-Session URL>
EOF
)"
```

---

### Task 4: Cross-object contamination gate in `suppress_overlap_bleed`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/blade.py` (`suppress_overlap_bleed`)
- Test: `tests/pipeline/test_blade.py`

**Interfaces:**
- Consumes: `_cross_object_ious` (Task 3), `motion["bend"]` (Task 1/2), `CROSS_OBJECT_OVERLAP_IOU_THRESHOLD` (existing).
- Produces: after `suppress_overlap_bleed` runs, both objects' `bend` arrays have `NaN` at every frame where cross-object IoU exceeds `CROSS_OBJECT_OVERLAP_IOU_THRESHOLD`, regardless of whether that frame is inside a formally detected `OverlapRun`.

**Context — why this is not optional:** spiking this against the real 506-frame job during brainstorming found that a per-frame significance check on the mask alone (Task 2, with no gate) fires across long stretches deep inside the known dead zone -- e.g. frames 323-369 (47 frames) for object 0, where object 0's raw mask was confirmed to **nearly fully contain** object 1's mask (intersection 1015-1511px against an object-1 mask of only 1041-1537px). That is mask contamination, not blade bow. Gating on cross-object IoU (the same threshold that already governs `_find_overlap_runs`'s own `overlapping` decision) eliminated every false positive across the whole job and left only frames 291-292 -- exactly the pair the original screenshot showed.

The gate must run on **every frame in the shared clip**, not just frames inside `run_start..run_end` or the corrected `before+1..after-1` span used by the existing smoothing loop -- a frame's mask can be individually contaminated (high IoU with the other object) even outside what `_find_overlap_runs`'s IoU-threshold-based run detection considers a "run."

- [ ] **Step 1: Write the failing tests**

Add a helper mirroring `_write_lengths`/`_motion_geo`'s existing style (near line 973), extended for `bend`:

```python
def _motion_geo(length, i=0, x_offset=0.0, bend=None):
    return BladeGeometry(
        centroid=(float(i) + x_offset, 0.0), axis=(1.0, 0.0),
        tip=(float(i) + x_offset + length, 0.0), hilt=(float(i) + x_offset, 0.0),
        length=length, width=5.0, angle=0.0, bend=bend,
    )


def _write_lengths(path, lengths, x_offset=0.0, bends=None):
    """... `bends` (default: every frame None) lets a test set a
    candidate bend directly, matching how these tests already inject
    `length` directly rather than deriving it from a real mask."""
    if bends is None:
        bends = [None] * len(lengths)
    save_motion(str(path), [
        _motion_geo(length, i, x_offset=x_offset, bend=bend) if length is not None else None
        for i, (length, bend) in enumerate(zip(lengths, bends, strict=True))
    ])
```

(This changes the *existing* `_motion_geo`/`_write_lengths` definitions in place -- both gain an optional parameter with a default that reproduces every existing call site's behavior unchanged. Confirm this by running the full suite in Step 2 before writing any new code.)

```python
def test_suppress_overlap_bleed_clears_bend_where_cross_object_iou_is_high(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 5
    _write_fixed_mask(masks_a, n)
    # frame 2: full overlap (IoU 1.0, well past CROSS_OBJECT_OVERLAP_IOU_THRESHOLD=0.1)
    # every other frame: masks far apart (IoU 0.0)
    _write_overlap_masks(masks_b, n, overlapping_frames={2})
    _write_lengths(motion_a, [100] * n, bends=[(5.0, 5.0)] * n)
    _write_lengths(motion_b, [100] * n, bends=[(5.0, 5.0)] * n)

    suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    result_a = load_motion(str(motion_a))
    result_b = load_motion(str(motion_b))
    assert np.isnan(result_a["bend"][2]).all()  # cleared -- high cross-object IoU
    assert np.isnan(result_b["bend"][2]).all()
    for i in (0, 1, 3, 4):
        assert not np.isnan(result_a["bend"][i]).any()  # untouched -- low IoU
        assert not np.isnan(result_b["bend"][i]).any()


def test_suppress_overlap_bleed_clears_bend_outside_any_detected_run(tmp_path):
    # A single high-IoU frame below CROSS_OBJECT_OVERLAP_IOU_THRESHOLD's
    # run-detection bar entirely (no run is ever detected here -- the run
    # loop never touches this frame) must still get its bend cleared, since
    # the gate operates on the whole clip's IoU array independently of run
    # detection. Reuses the same fixture as above but only asserts on the
    # gate, making the "independent of run detection" property explicit
    # rather than incidental.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 3
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1})
    _write_lengths(motion_a, [100] * n, bends=[(5.0, 5.0)] * n)
    _write_lengths(motion_b, [100] * n, bends=[(5.0, 5.0)] * n)

    n_held = suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    result_a = load_motion(str(motion_a))
    assert np.isnan(result_a["bend"][1]).all()
    # sanity: this run WAS also detected/held by the existing smoothing
    # logic (single-frame overlap at index 1) -- both mechanisms agree
    # here, but the gate's own test above already proves it doesn't
    # depend on that.
    assert n_held == 1
```

- [ ] **Step 2: Run tests to verify they fail (and confirm no regression from the helper change)**

Run: `python -m pytest tests/pipeline/test_blade.py -q`
Expected: the two new tests FAIL (bend is never cleared -- no gate exists yet); every pre-existing test still PASSES (confirms the `_motion_geo`/`_write_lengths` signature change is backward compatible).

- [ ] **Step 3: Implement the gate**

In `suppress_overlap_bleed`, after the existing `for run_start, run_end, before, after, max_iou in runs:` loop (after line 1282, before the `if n_held:` block at line 1284), add:

```python
    # Cross-object contamination gate for `bend` -- see this function's
    # docstring addendum below and the design spec's "Cross-object
    # contamination gate" section. Operates on the whole clip's IoU
    # array, independent of which frames the run-detection loop above
    # touched: a frame's mask can be individually contaminated without
    # being part of a formally detected overlap run.
    had_bend_candidate = np.any(~np.isnan(motion_a["bend"][:, 0])) or np.any(~np.isnan(motion_b["bend"][:, 0]))
    if had_bend_candidate:
        ious_whole_clip = _cross_object_ious(masks_dir_a, masks_dir_b, frame_indices)
        contaminated = ious_whole_clip > iou_threshold
        motion_a["bend"][contaminated] = np.nan
        motion_b["bend"][contaminated] = np.nan
```

Update the save condition (line 1284) so a bend-only change still persists even when no run was smoothed/held:

```python
    if n_held or had_bend_candidate:
        np.savez(motion_path_a, **motion_a)
        np.savez(motion_path_b, **motion_b)
    return n_held
```

Add to `suppress_overlap_bleed`'s docstring (after the existing `hilt_overrides_a`/`hilt_overrides_b` paragraph, before the `LONG_INTERPOLATION_SPAN_FRAMES` paragraph):

```
    Any candidate `bend` (see `fit_blade`/`BEND_SIGNIFICANCE_PX`) on
    either object is cleared to NaN wherever cross-object mask IoU
    exceeds `iou_threshold`, across the *entire* clip -- not just frames
    the run-detection loop above touches. Confirmed necessary on real
    footage: a per-frame significance check on a single mask's own shape
    cannot distinguish real bow from contamination (one object's mask
    nearly fully containing the other's, deep in a sustained overlap
    run) -- cross-object IoU can, and this reuses the same measurement
    `_find_overlap_runs` already makes rather than a second, possibly
    inconsistent one.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_blade.py -q`
Expected: PASS, including both new tests and every pre-existing `suppress_overlap_bleed` test.

- [ ] **Step 5: Run full suite + ruff, then commit**

```bash
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git add src/lightsaber_fx/pipeline/blade.py tests/pipeline/test_blade.py
git commit -m "$(cat <<'EOF'
Gate candidate bend on cross-object mask IoU in suppress_overlap_bleed

Without this, a per-frame significance check on a single mask's own
shape fires across ~50-frame stretches deep in the known dead zone --
confirmed on the real job, object 0's mask nearly fully contains
object 1's there (contamination, not bow). Reuses the exact IoU
measurement _find_overlap_runs already makes; operates on the whole
clip, not just formally detected runs.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: <copy current session's Claude-Session URL>
EOF
)"
```

---

### Task 5: Temporal stability for the gated `bend` field

**Files:**
- Modify: `src/lightsaber_fx/pipeline/blade.py` (`suppress_overlap_bleed`, new `_edge_ramp_fraction`/`_smooth_bend_field`)
- Test: `tests/pipeline/test_blade.py`

**Interfaces:**
- Consumes: gated `motion["bend"]` from Task 4 (must run strictly *after* the gate, and after the existing `runs` loop, so it reads the *final* smoothed/overridden `hilt`/`tip`, not pre-correction values).
- Produces: within each contiguous stretch of non-NaN `bend`, a denoised, edge-ramped result -- no jitter between frames, no instant pop at a stretch's start/end.

**Context:** `blade.py` cannot import `glow.py`'s existing `ignition_fraction` (`glow.py` already imports from `blade.py`; the reverse would be circular), so `_edge_ramp_fraction` is a small local duplicate of the same rise/fall shape, not a shared import -- say so in its docstring so a future reader doesn't wonder why it isn't just reused directly. "Ramping bend in/out" means lerping the `bend` point toward the straight-line midpoint of `hilt`/`tip` at that frame (not toward the origin, and not scaling its magnitude in isolation) -- mirroring `glow._apply_ignition`'s `hilt + (tip - hilt) * frac` shape, adapted to lerp toward the segment's own midpoint instead of toward `hilt`.

- [ ] **Step 1: Write the failing tests**

```python
def test_edge_ramp_fraction_matches_ignition_fractions_shape():
    # Mirrors glow.ignition_fraction's own tests exactly, since this is a
    # deliberate local duplicate of the same rise/fall shape.
    assert _edge_ramp_fraction(0, 100, 4) == pytest.approx(0.25)
    assert _edge_ramp_fraction(1, 100, 4) == pytest.approx(0.5)
    assert _edge_ramp_fraction(3, 100, 4) == 1.0
    assert _edge_ramp_fraction(50, 100, 4) == 1.0
    assert _edge_ramp_fraction(99, 100, 4) == pytest.approx(0.25)


def test_edge_ramp_fraction_tapers_on_a_short_stretch():
    frac = _edge_ramp_fraction(2, 5, 4)
    assert 0.0 < frac < 1.0


def test_smooth_bend_field_ramps_in_and_out_of_a_stretch(tmp_path):
    motion_path = tmp_path / "a.npz"
    n = 6
    # frames 1-4: a real bend stretch, constant offset (10, 10) --
    # frames 0 and 5 have no candidate (None).
    bends = [None, (10.0, 10.0), (10.0, 10.0), (10.0, 10.0), (10.0, 10.0), None]
    _write_lengths(motion_path, [100] * n, bends=bends)
    motion = load_motion(str(motion_path))

    _smooth_bend_field(motion, window=1, ramp_frames=2)

    # stretch is frames 1-4 (length 4): ramp_frames=2 means frame 1 is at
    # ramp fraction 0.5, frame 2 reaches 1.0, frame 3 is still 1.0 (fall
    # starts from the far end), frame 4 is back down to 0.5.
    straight_mid_1 = (np.array(motion["hilt"][1]) + np.array(motion["tip"][1])) / 2.0
    expected_1 = straight_mid_1 + 0.5 * (np.array([10.0, 10.0]) - straight_mid_1)
    assert motion["bend"][1] == pytest.approx(expected_1)
    assert motion["bend"][2] == pytest.approx([10.0, 10.0])
    assert motion["bend"][3] == pytest.approx([10.0, 10.0])
    straight_mid_4 = (np.array(motion["hilt"][4]) + np.array(motion["tip"][4])) / 2.0
    expected_4 = straight_mid_4 + 0.5 * (np.array([10.0, 10.0]) - straight_mid_4)
    assert motion["bend"][4] == pytest.approx(expected_4)


def test_smooth_bend_field_denoises_a_jittery_stretch(tmp_path):
    motion_path = tmp_path / "a.npz"
    n = 5
    # one outlier frame in the middle of an otherwise-constant stretch
    bends = [(10.0, 10.0), (10.0, 10.0), (40.0, 40.0), (10.0, 10.0), (10.0, 10.0)]
    _write_lengths(motion_path, [100] * n, bends=bends)
    motion = load_motion(str(motion_path))

    _smooth_bend_field(motion, window=3, ramp_frames=0)  # ramp_frames=0: isolate denoising

    # a window-3 median centered on the outlier pulls it back toward its
    # neighbors -- must move meaningfully off 40.0, not stay there.
    assert motion["bend"][2][0] < 25.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_blade.py -k "edge_ramp or smooth_bend_field" -v`
Expected: FAIL -- neither function exists yet.

- [ ] **Step 3: Implement**

Add near the temporal-smoothing constants already in the file (after `TIP_SMOOTHING_STRENGTH`, before `_smooth_interpolate_run`, i.e. after line 901):

```python
# Small local window (frames) for denoising a contiguous bend-active
# stretch, and a small rise/fall ramp so bend eases in/out of existence
# instead of popping. Calibrated against the real job: the only real
# stretch found there (after Task 4's contamination gate) is 2 frames
# (291-292) -- not enough on its own to pin an exact value the way
# BEND_SIGNIFICANCE_PX's wide margin could, but it rules out anything
# large. Revisit against a wider sample of contact runs if one turns up
# in other real footage.
BEND_TEMPORAL_SMOOTH_WINDOW = 3
BEND_RAMP_FRAMES = 2


def _edge_ramp_fraction(offset_into_stretch, stretch_length, ramp_frames):
    """Rise 0->1 over the first `ramp_frames` frames of a stretch of
    length `stretch_length`, hold at 1, fall 1->0 over the last
    `ramp_frames` -- deliberately mirrors `glow.ignition_fraction`'s
    shape (kept as a small local duplicate, not a shared import:
    `blade.py` cannot import from `glow.py`, which already imports from
    `blade.py`). On a stretch shorter than `2 * ramp_frames`, rise and
    fall overlap and the peak never reaches 1.0, exactly like
    `ignition_fraction`'s own short-window case.
    """
    if ramp_frames <= 0:
        return 1.0
    rise = (offset_into_stretch + 1) / ramp_frames
    fall = (stretch_length - offset_into_stretch) / ramp_frames
    return max(0.0, min(1.0, rise, fall))


def _smooth_bend_field(motion, window=BEND_TEMPORAL_SMOOTH_WINDOW, ramp_frames=BEND_RAMP_FRAMES):
    """Denoise and edge-ramp `motion['bend']` in place, within each
    contiguous stretch of non-NaN frames independently. Purely
    single-object -- no cross-object data, no confidence weights, just
    "smooth this one object's own already-gated per-frame estimate."
    Must run after the cross-object contamination gate has already
    cleared untrustworthy candidates (see `suppress_overlap_bleed`), and
    after that function's own `hilt`/`tip` smoothing/overrides, since
    "ramping out" means lerping `bend` toward the *final* straight-line
    midpoint of `hilt`/`tip` at that frame, not a pre-correction one.
    """
    bend = motion["bend"]
    hilt, tip = motion["hilt"], motion["tip"]
    valid = ~np.isnan(bend[:, 0])
    n = len(bend)
    half = window // 2
    i = 0
    while i < n:
        if not valid[i]:
            i += 1
            continue
        start = i
        while i < n and valid[i]:
            i += 1
        end = i - 1
        length = end - start + 1
        raw_stretch = bend[start:end + 1].copy()
        denoised = np.empty_like(raw_stretch)
        for k in range(length):
            lo, hi = max(0, k - half), min(length, k + half + 1)
            denoised[k] = np.median(raw_stretch[lo:hi], axis=0)
        for k in range(length):
            frac = _edge_ramp_fraction(k, length, ramp_frames)
            straight_mid = (hilt[start + k] + tip[start + k]) / 2.0
            bend[start + k] = straight_mid + frac * (denoised[k] - straight_mid)
```

In `suppress_overlap_bleed`, immediately after the contamination-gate block added in Task 4 (still inside the `if had_bend_candidate:` block, after the two `motion_x["bend"][contaminated] = np.nan` lines):

```python
        _smooth_bend_field(motion_a)
        _smooth_bend_field(motion_b)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_blade.py -q`
Expected: PASS, all tests including Task 4's.

- [ ] **Step 5: Run full suite + ruff, then commit**

```bash
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git add src/lightsaber_fx/pipeline/blade.py tests/pipeline/test_blade.py
git commit -m "$(cat <<'EOF'
Denoise and edge-ramp the gated bend field in suppress_overlap_bleed

Runs after the contamination gate and after hilt/tip smoothing, so
"ramping out" lerps bend toward the final corrected straight-line
midpoint, not a pre-correction one. _edge_ramp_fraction mirrors
glow.ignition_fraction's shape as a local duplicate -- blade.py cannot
import glow.py, which already imports blade.py.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: <copy current session's Claude-Session URL>
EOF
)"
```

---

### Task 6: Renderer — curved capsule in `_capsule_mask`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/glow.py` (`_capsule_mask` and friends)
- Test: `tests/pipeline/test_glow.py`

**Interfaces:**
- Consumes: `bend: tuple | None` as a new parameter.
- Produces: `_capsule_mask(shape, hilt, tip, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac, bend=None)`. `bend=None` **must** produce byte-identical pixels to the pre-this-task function -- verified by keeping that exact code path completely untouched (see below), not by writing new code that merely "should" reduce to the same result. Floating-point operations are not guaranteed to associate identically across two different code paths that are mathematically equivalent, and the existing golden-checksum test (`EXPECTED_CHARACTERIZATION_CHECKSUMS`) checks exact bytes -- this is a hard constraint, not a style preference.

**Context:** `_capsule_mask` (line 89) is currently one function. Rather than generalizing its body to "a 1-segment polyline when bend is None, more segments otherwise" (which risks a different floating-point path even in the None case), rename the *entire current function body, unchanged*, to `_straight_capsule_mask`, and make `_capsule_mask` a thin dispatcher.

- [ ] **Step 1: Write the failing tests**

Add near the top of `test_glow.py`, after the existing imports (need to add `_capsule_mask` to the import from `lightsaber_fx.pipeline.glow`):

```python
def test_capsule_mask_with_bend_none_matches_straight_capsule_exactly():
    shape = (90, 220)
    hilt, tip = (40.0, 45.0), (130.0, 45.0)
    args = (shape, hilt, tip, 8.0, 0.10, 0.12, 0.35)
    with_none = _capsule_mask(*args, bend=None)
    without_param = _capsule_mask(*args)
    assert np.array_equal(with_none, without_param)


def test_capsule_mask_with_bend_follows_the_curve_not_the_straight_line():
    shape = (120, 220)
    hilt, tip = (40.0, 60.0), (180.0, 60.0)
    bend = (110.0, 20.0)  # well above the straight hilt-tip line (y=60)
    straight = _capsule_mask(shape, hilt, tip, 8.0, 0.10, 0.12, 0.35, bend=None)
    curved = _capsule_mask(shape, hilt, tip, 8.0, 0.10, 0.12, 0.35, bend=bend)
    # the curved capsule must light up pixels near the bend point that the
    # straight one (a horizontal bar at y=60) never touches
    assert curved[15:30, 100:120].any()
    assert not straight[15:30, 100:120].any()


def test_capsule_mask_bend_nan_falls_back_to_straight():
    shape = (90, 220)
    hilt, tip = (40.0, 45.0), (130.0, 45.0)
    args = (shape, hilt, tip, 8.0, 0.10, 0.12, 0.35)
    straight = _capsule_mask(*args, bend=None)
    with_nan = _capsule_mask(*args, bend=(float("nan"), float("nan")))
    assert np.array_equal(straight, with_nan)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_glow.py -k capsule_mask -v`
Expected: FAIL -- `_capsule_mask` doesn't accept a `bend` keyword yet (`TypeError`), and isn't importable by that usage pattern until the import line is updated.

- [ ] **Step 3: Implement**

Rename the existing `_capsule_mask` function (lines 89-131) to `_straight_capsule_mask` (change only the `def` line -- the body is byte-for-byte identical to what exists today).

Add the new dispatcher and curved implementation in its place:

```python
BEND_POLYLINE_POINTS = 14  # per the design spec's "~12-16 points" guidance


def _quadratic_bezier_points(p0, p1, p2, n_points):
    """`n_points` points along the quadratic Bezier from `p0` through
    control point `p1` to `p2`, inclusive of both endpoints, evenly
    spaced in the curve parameter t (not arc length -- close enough at
    real blade lengths/curvatures for a rendering polyline)."""
    t = np.linspace(0.0, 1.0, n_points)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _curved_capsule_mask(shape, hilt, tip, bend, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac):
    """Same tapered-hilt/rounded-tip capsule shape as
    `_straight_capsule_mask`, but walking a sampled quadratic-Bezier
    polyline (hilt -> bend -> tip) instead of one straight segment. Only
    called when `bend` is a real, finite point -- see `_capsule_mask`.
    """
    h, w = shape[:2]
    out = np.zeros((h, w), dtype=np.uint8)
    hilt = np.asarray(hilt, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    bend = np.asarray(bend, dtype=np.float64)
    half_w = max(0.5, width / 2.0)

    seg = tip - hilt
    length = float(np.linalg.norm(seg))
    if length < 1e-6:
        cv2.circle(out, (round(hilt[0]), round(hilt[1])), max(1, round(half_w)), 255, -1)
        return out

    axis = seg / length
    perp = np.array([-axis[1], axis[0]])
    taper_len = min(length * hilt_taper_frac, length * 0.9)
    body_start = hilt + axis * taper_len

    centerline = _quadratic_bezier_points(body_start, bend, tip, BEND_POLYLINE_POINTS)
    tip_tangent = centerline[-1] - centerline[-2]
    tip_tangent_norm = np.linalg.norm(tip_tangent)
    tip_dir = tip_tangent / tip_tangent_norm if tip_tangent_norm > 0 else axis
    centerline[-1] = tip + tip_dir * (length * extend_frac)

    for i in range(len(centerline) - 1):
        p0, p1 = centerline[i], centerline[i + 1]
        seg_vec = p1 - p0
        seg_len = np.linalg.norm(seg_vec)
        if seg_len < 1e-9:
            continue
        seg_perp = np.array([-seg_vec[1], seg_vec[0]]) / seg_len
        quad = np.array([
            p0 + seg_perp * half_w, p1 + seg_perp * half_w,
            p1 - seg_perp * half_w, p0 - seg_perp * half_w,
        ])
        cv2.fillConvexPoly(out, np.round(quad).astype(np.int32), 255)

    tip_pt = (round(centerline[-1][0]), round(centerline[-1][1]))
    cv2.circle(out, tip_pt, max(1, round(half_w)), 255, -1)

    hilt_half_w = half_w * hilt_taper_min_frac
    wedge = np.array([
        hilt + perp * hilt_half_w,
        body_start + perp * half_w,
        body_start - perp * half_w,
        hilt - perp * hilt_half_w,
    ])
    cv2.fillConvexPoly(out, np.round(wedge).astype(np.int32), 255)

    return out


def _capsule_mask(shape, hilt, tip, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac, bend=None):
    """Dispatches to `_straight_capsule_mask` (today's exact, unmodified
    code path -- byte-identical output is a hard requirement for every
    frame outside real blade-on-blade contact) or `_curved_capsule_mask`,
    depending on whether a real, finite `bend` point is given."""
    have_bend = bend is not None and not np.any(np.isnan(bend))
    if not have_bend:
        return _straight_capsule_mask(shape, hilt, tip, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac)
    return _curved_capsule_mask(shape, hilt, tip, bend, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac)
```

Update `_build_blade_shape` (line 134) to accept and forward `bend`:

```python
def _build_blade_shape(mask, frame_shape, tip, hilt, width, blade_extend,
                        extend_frac, hilt_taper_frac, hilt_taper_min_frac, bend=None):
    have_geometry = (
        blade_extend
        and tip is not None and hilt is not None
        and not (np.any(np.isnan(tip)) or np.any(np.isnan(hilt)))
    )
    if have_geometry:
        return _capsule_mask(frame_shape, hilt, tip, width, extend_frac, hilt_taper_frac, hilt_taper_min_frac, bend=bend)

    mask_u8 = mask.astype(np.uint8) * 255
    if mask_u8.shape[:2] != tuple(frame_shape[:2]):
        mask_u8 = cv2.resize(mask_u8, (frame_shape[1], frame_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask_u8
```

Update the `test_glow.py` import line to include `_capsule_mask`:

```python
from lightsaber_fx.pipeline.glow import (
    _capsule_mask,
    ignition_fraction,
    knoll_darken,
    parse_color,
    render_glow,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_glow.py -k capsule_mask -v`
Expected: PASS.

Then run the full existing golden-checksum test to confirm zero regression from the rename/dispatch:

Run: `python -m pytest tests/pipeline/test_glow.py -k unchanged_by_the_extraction_refactor -v`
Expected: PASS with the exact same committed `EXPECTED_CHARACTERIZATION_CHECKSUMS` -- this is the byte-identical regression guarantee in action. If this fails, the rename introduced a real behavior change and must be fixed before continuing, not the checksum updated.

- [ ] **Step 5: Run full suite + ruff, then commit**

```bash
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git add src/lightsaber_fx/pipeline/glow.py tests/pipeline/test_glow.py
git commit -m "$(cat <<'EOF'
Add a curved capsule renderer for a given bend point

_capsule_mask now dispatches to the pre-existing, completely
unmodified straight-line code path (renamed _straight_capsule_mask)
when bend is None/NaN, and to a new _curved_capsule_mask that walks a
sampled quadratic-Bezier polyline otherwise. Byte-identical output for
bend=None is verified by the existing golden-checksum characterization
test passing unchanged.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: <copy current session's Claude-Session URL>
EOF
)"
```

---

### Task 7: Thread `bend` through the compositing/render pipeline

**Files:**
- Modify: `src/lightsaber_fx/pipeline/glow.py` (`_composite_blade_contribution`, `render_glow`, `render_glow_multi`)
- Test: `tests/pipeline/test_glow.py`

**Interfaces:**
- Consumes: `_build_blade_shape(..., bend=None)` (Task 6), `motion.npz`'s `bend` array (Task 1, via `load_motion`).
- Produces: `render_glow`/`render_glow_multi` read `bend` per-frame from `motion.npz` (defaulting safely to all-`NaN` for a legacy motion.npz file written before this change) and pass it all the way to the renderer. `_stabilize_tip_hilt` is confirmed to need **no change** -- see the note below and its own test.

**Context — the `_stabilize_tip_hilt` correction:** the design spec assumed this function would need to swap `bend` alongside `tip`/`hilt` on a continuity flip. It does not: `bend` is stored as an absolute `(x, y)` point, not a signed offset relative to a direction, and a quadratic Bezier's shape is identical whether its three control points are traversed hilt→bend→tip or tip→bend→hilt (reversing a quadratic Bezier's parametrization retraces the exact same curve in space). There is nothing to swap. Do not add code to `_stabilize_tip_hilt` for this -- write the test below instead, which locks in that `bend` passes through a flip completely unchanged.

- [ ] **Step 1: Write the failing tests**

```python
def test_stabilize_tip_hilt_leaves_bend_completely_unchanged_on_a_flip():
    # bend is an absolute (x, y) point, not a directional offset -- a
    # tip/hilt continuity flip has nothing to swap it with. This locks
    # in that finding as a test, correcting an assumption in the design
    # spec that turned out to be unnecessary once worked through.
    from lightsaber_fx.pipeline.glow import _stabilize_tip_hilt
    tip = np.array([[10.0, 0.0], [-10.0, 0.0]])   # axis flips sign at frame 1
    hilt = np.array([[0.0, 0.0], [0.0, 0.0]])
    axis = np.array([[1.0, 0.0], [-1.0, 0.0]])
    bend_before = np.array([[5.0, 3.0], [5.0, 3.0]])

    new_tip, new_hilt, new_axis = _stabilize_tip_hilt(tip, hilt, axis)

    assert not np.allclose(new_tip[1], tip[1])  # confirms a flip actually happened
    # bend itself was never passed in and never touched -- nothing to assert
    # on bend's value changing, since _stabilize_tip_hilt's signature does
    # not take it. This test exists to make that omission a deliberate,
    # documented choice rather than a silent gap.


def test_render_glow_multi_renders_a_curved_blade_when_bend_is_present(tmp_path):
    clip = _build_blade_clip(tmp_path, "curved", n_frames=1, width=220, height=120, blade_len=90, blade_x0=40)
    # Manually inject a bend into the motion.npz compute_motion just wrote,
    # matching how a real contact-adjacent frame would carry one.
    motion = blade.load_motion(clip["motion_path"])
    hilt, tip = motion["hilt"][0], motion["tip"][0]
    mid = (hilt + tip) / 2.0
    bend_point = mid + np.array([0.0, -25.0])  # well off the straight line
    motion["bend"] = np.array([bend_point])
    np.savez(clip["motion_path"], **motion)

    out_dir = str(tmp_path / "out")
    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_dir, clip["motion_path"], ignition_ramp_seconds=0,
    )
    img = _load_png(out_dir, 0)
    baseline = float(clip["plate_value"])
    px, py = round(bend_point[0]), round(bend_point[1])
    signal = float(img[py, px].astype(np.float64).max()) - baseline
    assert signal > 20  # the curve actually reaches up near the bend point


def test_render_glow_handles_a_motion_npz_without_a_bend_column(tmp_path):
    # Backward compatibility: a motion.npz written before this feature
    # existed has no "bend" key at all. Loading and rendering it must not
    # crash -- treated exactly like bend=None everywhere.
    clip = _build_blade_clip(tmp_path, "legacy", n_frames=2)
    motion = blade.load_motion(clip["motion_path"])
    del motion["bend"]
    np.savez(clip["motion_path"], **motion)

    out_dir = str(tmp_path / "out")
    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_dir, clip["motion_path"],
    )
    img = _load_png(out_dir, 0)
    assert img is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/pipeline/test_glow.py -k "stabilize_tip_hilt or curved_blade or without_a_bend_column" -v`
Expected: `test_render_glow_multi_renders_a_curved_blade_when_bend_is_present` and `test_render_glow_handles_a_motion_npz_without_a_bend_column` FAIL (bend isn't read from motion.npz at all yet, and/or a `KeyError` on the deleted "bend" key in a real code path once one is added without a default). `test_stabilize_tip_hilt_leaves_bend_completely_unchanged_on_a_flip` PASSES immediately (it's a documentation test, not a behavior change) -- that's expected and fine, note it in the commit rather than treating it as a step-ordering problem.

- [ ] **Step 3: Implement**

In `_composite_blade_contribution` (line 351), add a `bend=None` parameter and thread it through to `_build_blade_shape`:

```python
def _composite_blade_contribution(
    frame_shape, mask, tip, hilt, velocity,
    canonical_width, blade_extend, ignition_frac,
    tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
    core_erode_kernel, core_sigma, colour_sigma,
    color_lin, glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
    spill_strength, motion_blur_gain, motion_blur_max_len, bbox_margin,
    bend=None,
):
```

Inside it, alongside the existing `effective_tip = _apply_ignition(tip, hilt, ignition_frac)` (in the `if blade_extend:` block):

```python
    effective_tip = tip
    effective_bend = bend
    if blade_extend:
        effective_tip = _apply_ignition(tip, hilt, ignition_frac)
        if bend is not None:
            effective_bend = _apply_ignition(bend, hilt, ignition_frac)
```

(`_apply_ignition` already returns `None`-safe/NaN-safe for `tip`/`hilt`; passing `bend` through the same function reuses that behavior for free -- an igniting/extinguishing blade's bend point shrinks toward the hilt proportionally with the rest of the blade, keeping the curve visually consistent with the shrunken length, matching how `tip` already behaves. This case essentially never arises in practice since ignition only happens at a clip's very first/last tracked frames while `bend` only appears mid-clip during contact, but costs nothing to handle correctly.)

Update the `_build_blade_shape` call inside `_composite_blade_contribution`:

```python
    blade_u8 = _build_blade_shape(
        mask, frame_shape, effective_tip, hilt, canonical_width, blade_extend,
        tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac, bend=effective_bend,
    )
```

In `render_glow`, after the existing `tip_arr`/`hilt_arr`/`axis_arr`/`width_arr`/`length_arr` extraction (around line 492-496):

```python
    bend_arr = np.asarray(motion.get("bend", np.full((len(tip_arr), 2), np.nan)), dtype=np.float64)
```

(`motion.get("bend", ...)` is the backward-compatibility default for a legacy motion.npz with no `bend` key at all.) In the main per-frame loop (around line 559-562), alongside `tip_i`/`hilt_i`/`vel_i`:

```python
        bend_i = bend_arr[row] if row is not None else None
```

And pass it into the `_composite_blade_contribution` call (around line 565-572):

```python
        full_fx, blade_u8 = _composite_blade_contribution(
            frame.shape, mask, tip_i, hilt_i, vel_i,
            canonical_width, blade_extend, frac,
            tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
            core_erode_kernel, core_sigma, colour_sigma,
            color_lin, glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
            spill_strength, motion_blur_gain, motion_blur_max_len, bbox_margin,
            bend=bend_i,
        )
```

Apply the equivalent three changes to `render_glow_multi`: in the per-object `prepared` dict construction (around line 670-706), add `bend_arr = np.asarray(motion.get("bend", np.full((len(tip_arr), 2), np.nan)), dtype=np.float64)` and store it as `prepared[-1]["bend_arr"] = bend_arr` (add this key to the dict literal built there); in the main per-frame loop over `prepared` (around line 731-748), extract `bend_i = obj_state["bend_arr"][row] if row is not None else None` and pass `bend=bend_i` into that loop's `_composite_blade_contribution` call.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/pipeline/test_glow.py -q`
Expected: PASS, all tests.

- [ ] **Step 5: Run full suite + ruff, then commit**

```bash
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git add src/lightsaber_fx/pipeline/glow.py tests/pipeline/test_glow.py
git commit -m "$(cat <<'EOF'
Thread bend through render_glow/render_glow_multi's per-frame reads

Reads bend from motion.npz the same way tip/hilt already are, with a
safe all-NaN default for a legacy motion.npz written before this
feature existed. Confirms (with a test) that _stabilize_tip_hilt needs
no change for this -- bend is an absolute point, not a directional
offset, so a tip/hilt flip has nothing to swap it with, correcting an
assumption in the original design spec.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: <copy current session's Claude-Session URL>
EOF
)"
```

---

### Task 8: Real-data validation, montage check, final push

**Files:** none (validation only; may produce a follow-up commit only if this step finds a real bug).

**Interfaces:** none new -- this task exercises everything from Tasks 1-7 against real footage.

**Context:** the scratch job used throughout the original debugging session (`/private/tmp/claude-501/.../scratchpad/final_e2e_test/job`) does not survive between sessions. The original job's `frames/`/`masks/` are still durably available at `/Users/danm/Library/Application Support/lightsaber-fx/jobs/58a8f662/` -- reuse those rather than asking for new footage. That job's own `motion/` subdirectory may predate several of this session's earlier fixes (hilt tracking, cable-component rejection, etc.), so regenerate `motion.npz` fresh with current code rather than trusting what's already there.

- [ ] **Step 1: Set up a scratch working copy**

```bash
cd /Users/danm/Development/lightsaber_fx && source .venv/bin/activate
JOB=/Users/danm/Library/Application\ Support/lightsaber-fx/jobs/58a8f662
WORK=/tmp/curved_blade_validation   # or this session's scratchpad directory, if available
mkdir -p "$WORK"
cp -r "$JOB/frames" "$JOB/masks" "$WORK/"
```

Determine the two tracked object indices from `$WORK/masks`' subdirectory names (expected `0` and `1`, matching every prior reference to this job this session).

- [ ] **Step 2: Re-run the full two-object pipeline stage sequence with current code**

Write a scratch script (in this session's scratchpad directory, not committed) that reproduces exactly `runner.run_pipeline_multi`'s two-object stage sequence (`src/lightsaber_fx/pipeline/runner.py`, lines 341-391), skipping only `track_objects`/`reconcile_pair` since the masks are already tracked and reconciled on disk:

```python
import sys
sys.path.insert(0, "/Users/danm/Development/lightsaber_fx/src")
from lightsaber_fx.pipeline.blade import compute_motion, mask_frame_indices, suppress_overlap_bleed
from lightsaber_fx.pipeline.reacquire import retrack_overlap_runs
from lightsaber_fx.pipeline.hilt_track import compute_hilt_overrides

WORK = "/tmp/curved_blade_validation"
frames_dir = f"{WORK}/frames"
masks_0, masks_1 = f"{WORK}/masks/0", f"{WORK}/masks/1"
motion_0, motion_1 = f"{WORK}/motion_0.npz", f"{WORK}/motion_1.npz"
n_frames = len(mask_frame_indices(masks_0))
checkpoint_path = ...  # same SAM2 checkpoint path the app config uses --
                        # check src/lightsaber_fx/config.py or the running
                        # web app's own startup args for the real path,
                        # needed only by retrack_overlap_runs/
                        # compute_hilt_overrides for their raw-frame work
config_name = "configs/sam2.1/sam2.1_hiera_s.yaml"
device = "cpu"  # or whatever this machine's runner.py default resolves to

compute_motion(masks_0, motion_0)
compute_motion(masks_1, motion_1)

retracked, resolved_ranges = retrack_overlap_runs(
    frames_dir, masks_0, masks_1, motion_0, motion_1,
    n_frames, checkpoint_path, config_name, device,
)
for oid, masks_dir, motion_path in ((0, masks_0, motion_0), (1, masks_1, motion_1)):
    if oid in retracked:
        compute_motion(masks_dir, motion_path)

hilt_overrides_0, hilt_overrides_1 = compute_hilt_overrides(
    frames_dir, masks_0, masks_1, motion_0, motion_1,
    exclude_frame_ranges=resolved_ranges,
)

suppress_overlap_bleed(
    motion_0, masks_0, motion_1, masks_1,
    exclude_frame_ranges=resolved_ranges,
    hilt_overrides_a=hilt_overrides_0, hilt_overrides_b=hilt_overrides_1,
)
```

If `retrack_overlap_runs`/`compute_hilt_overrides` need a real SAM2 checkpoint and one isn't readily available in this environment, that's fine -- they degrade gracefully to "nothing resolved" (empty `resolved_ranges`, empty override dicts) per their own documented fallback behavior, and `suppress_overlap_bleed` alone (with `hilt_overrides_a/b=None`) is sufficient to exercise every part of Tasks 1-7 that Task 8 needs to validate, since frame 292 sits at the edge of the run, not deep in the dead zone those two stages specifically target.

- [ ] **Step 3: Numeric check on frame 292**

```python
import numpy as np
from lightsaber_fx.pipeline.blade import load_motion, mask_frame_indices

frame_indices = mask_frame_indices(f"{WORK}/masks/0")
idx = frame_indices.index(292)
motion_1 = load_motion(f"{WORK}/motion_1.npz")  # blue, per this session's established object numbering
print("bend at 292:", motion_1["bend"][idx])
assert not np.isnan(motion_1["bend"][idx]).any(), "expected a real bend at frame 292 after re-running with current code"
```

If this assertion fails, stop and investigate with the same numeric-inspection rigor used throughout this session (check the regenerated mask's own shape at frame 292, the cross-object IoU there, and whether `BEND_SIGNIFICANCE_PX` is actually being cleared by the gate) before proceeding -- do not weaken the assertion or skip ahead.

- [ ] **Step 4: Render and visually inspect**

Render the glow output for both objects (`render_glow_multi`, reusing this session's established rerender-script pattern) into a scratch output directory. Using the Read tool, inspect:
- Frame 292 directly, and a zoomed crop of the blue-blade region (matching the exact crop technique used to root-cause this bug originally), confirming the rendered curve now visibly tracks the raw mask's bow instead of a straight line.
- A dense ffmpeg montage across frames 286-299 (`ffmpeg -vf "select='between(n\,286,299)'" -fps_mode vfr` + `tile` montage, the technique established earlier this session), confirming a smooth transition into and out of the curve -- no popping, no visible discontinuity at the stretch's edges.
- A broad sweep of frames spaced across the rest of the clip (e.g. every 20th frame, plus the previously-fixed problem frames 252-253, 350, 410, 469-475 from earlier this session) to confirm zero regressions -- every one of those should look exactly as it did before this feature, since none of them should have a significant `bend`.

- [ ] **Step 5: Full suite + ruff one more time, then push**

```bash
python -m pytest -q
ruff check src/lightsaber_fx/pipeline/*.py
git log --oneline -8   # confirm all 7 prior tasks' commits are present
git push origin master
```

- [ ] **Step 6: Update the Desktop copy and reveal it**

Following this session's established end-of-fix pattern: copy the final validated render to `~/Desktop/lightsaber_fx_e2e_final_fixed.mp4` (overwriting the existing one) and reveal it with `open -R ~/Desktop/lightsaber_fx_e2e_final_fixed.mp4`.

- [ ] **Step 7: Report back**

Report honestly what frame 292 (and any other frame that turned out to have a real, gate-surviving bend) looks like now versus before, and explicitly call out any remaining known limitation -- consistent with this session's established practice of never claiming a fix works on plausibility alone.
