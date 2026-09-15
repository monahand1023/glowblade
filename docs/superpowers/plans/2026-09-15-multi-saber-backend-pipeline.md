# Multi-Saber Backend Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the web app's render pipeline track, composite, and mix audio for
1-4 simultaneous sabers in one clip, each with its own color/intensity/voice
-- no UI changes and no interactive correction yet (those are separate,
later plans). After this plan, the feature is only reachable via the API
directly (e.g. `curl`/`TestClient`), not through the browser.

**Architecture:** Every per-frame pipeline artifact (masks, motion) moves
from one shared file/directory per job to one per tracked object, reusing
`blade.py`'s existing per-object I/O helpers unchanged. `glow.py`'s per-blade
compositing math is extracted into a new shared helper so the existing
single-object `render_glow` and the new `render_glow_multi` both call the
same code -- the N=1 case must produce pixel-identical output to today's
`render_glow`. New sibling orchestrator functions (`track_objects`,
`render_glow_multi`, `mix_hums`, `run_pipeline_multi`,
`rerender_pipeline_multi`) sit alongside the existing single-object ones,
which stay completely untouched (the CLI keeps using them as-is).

**Tech Stack:** Python, SAM2 (`sam2.build_sam.build_sam2_video_predictor`),
OpenCV, NumPy, soundfile, FastAPI, pytest.

**Spec:** `docs/superpowers/specs/2026-09-15-multi-saber-tracking-design.md`

## Global Constraints

- Up to 4 objects per job (`len(sabers) in range(1, 5)`, validated at the API layer).
- Every saber prompt uses `prompt_frame=0` -- auto-detect and mid-clip
  prompting are not extended to multi-object in this plan (see spec,
  "Explicitly out of scope"). `propagate_in_video` runs forward-only, once,
  for the whole shared session.
- The CLI (`lightsaber-fx run`/`rerender`) is untouched -- it keeps calling
  `track_object`/`render_glow`/`rerender_pipeline` exactly as today.
- New job directory layout (`masks/{obj_id}/`, `motion/{obj_id}.npz`,
  `object_ids` in `job_meta.json`) is **not** backward compatible with job
  directories created before this change -- those become non-rerenderable.
  Acceptable per spec; no migration code.
- `render_glow` (existing, single-object) must remain byte-for-byte
  unchanged in its public signature and behavior. Its existing 20 tests in
  `tests/pipeline/test_glow.py` must all still pass, unmodified, after the
  Task 1 refactor.
- `blade_extend` stays one job-level toggle shared by every object -- never
  per-object.

---

### Task 1: Extract shared per-blade compositing out of `render_glow`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/glow.py`
- Test: `tests/pipeline/test_glow.py`

**Interfaces:**
- Produces: `_composite_blade_contribution(frame_shape, mask, tip, hilt, velocity, canonical_width, blade_extend, ignition_frac, tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac, core_erode_kernel, core_sigma, colour_sigma, color_lin, glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac, spill_strength, motion_blur_gain, motion_blur_max_len, bbox_margin) -> (full_fx: np.ndarray[h,w,3] float32, blade_u8: np.ndarray[h,w] uint8)`, a private module-level function in `glow.py`. `full_fx` is zero everywhere outside this object's local bounding box; `blade_u8` is zero everywhere the blade shape doesn't cover.

This is a pure refactor -- no behavior change. `render_glow`'s frame loop
currently computes one object's core/colour/wide-glow/motion-blur contribution
inline; Task 2 needs that exact logic reusable for N objects, so it's
extracted first, proven identical via a characterization test, *then* reused.

- [ ] **Step 1: Write a characterization test pinning `render_glow`'s current output**

Add to `tests/pipeline/test_glow.py`:

```python
def test_render_glow_output_is_unchanged_by_the_extraction_refactor(tmp_path):
    # Characterization test for the Task 1 refactor in the multi-saber
    # backend plan: pins render_glow's exact pixel output on a
    # representative clip (extension, core/colour/glow, motion blur, and
    # the trail all exercised) before _composite_blade_contribution is
    # extracted, so the refactor can be verified byte-for-byte.
    clip = _build_blade_clip(tmp_path, "characterize", n_frames=6, dx=12, blade_len=60)
    out_dir = str(tmp_path / "out")
    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_dir, clip["motion_path"], color=(255, 90, 60), intensity=0.4,
    )
    frames = [_load_png(out_dir, i) for i in range(clip["n_frames"])]
    checksums = [int(f.astype(np.uint64).sum()) for f in frames]
    # A committed golden value, not a live re-comparison against another
    # render -- if this assertion ever needs to change, that means
    # render_glow's real output changed, which must be a deliberate,
    # reviewed decision, not an accidental refactor side effect.
    assert checksums == EXPECTED_CHARACTERIZATION_CHECKSUMS
```

Before filling in `EXPECTED_CHARACTERIZATION_CHECKSUMS`, run a tiny throwaway
script to print the real checksums produced by the *current, unrefactored*
`render_glow` on this exact fixture, then paste that literal list in as the
constant (defined near the top of the test file, alongside the other
fixtures). This captures "whatever `render_glow` produces today" as the
pinned baseline -- the golden values are specific to this repo's current
code, not something to derive by hand.

- [ ] **Step 2: Run it and confirm it passes against today's code**

Run: `.venv/bin/python -m pytest tests/pipeline/test_glow.py -k characterization -v`
Expected: PASS (this is pinning current behavior, not testing new behavior --
it must pass immediately since `render_glow` hasn't changed yet).

- [ ] **Step 3: Extract `_composite_blade_contribution`**

In `src/lightsaber_fx/pipeline/glow.py`, add the new function just above
`render_glow`:

```python
def _composite_blade_contribution(
    frame_shape, mask, tip, hilt, velocity,
    canonical_width, blade_extend, ignition_frac,
    tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
    core_erode_kernel, core_sigma, colour_sigma,
    color_lin, glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
    spill_strength, motion_blur_gain, motion_blur_max_len, bbox_margin,
):
    """One object's core/colour/wide-glow/motion-blur contribution for one
    frame, confined to a local bounding box. Both `render_glow` (single
    object) and `render_glow_multi` (N objects) call this once per object
    per frame and sum the `full_fx` results -- already in additive linear
    light -- before the shared trail/knoll-darken/tonemap steps that follow.
    Returns full-frame-sized arrays so callers can sum/OR them directly
    without tracking per-object offsets themselves.
    """
    h, w = frame_shape[:2]
    full_fx = np.zeros((h, w, 3), dtype=np.float32)
    blade_u8 = np.zeros((h, w), dtype=np.uint8)

    if mask is None or not mask.any():
        return full_fx, blade_u8

    effective_tip = tip
    if blade_extend:
        effective_tip = _apply_ignition(tip, hilt, ignition_frac)

    blade_u8 = _build_blade_shape(
        mask, frame_shape, effective_tip, hilt, canonical_width, blade_extend,
        tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
    )
    ys, xs = np.nonzero(blade_u8)
    if not len(xs):
        return full_fx, blade_u8

    x0 = max(0, int(xs.min()) - bbox_margin)
    y0 = max(0, int(ys.min()) - bbox_margin)
    x1 = min(w, int(xs.max()) + bbox_margin + 1)
    y1 = min(h, int(ys.max()) + bbox_margin + 1)

    blade01_local = blade_u8[y0:y1, x0:x1].astype(np.float32) / 255.0
    core_local = _make_core(blade01_local, core_erode_kernel, core_sigma)
    colour_local = _make_colour_band(blade01_local, colour_sigma)
    glow_local = _make_wide_glow(
        blade01_local, canonical_width, color_lin,
        glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
    ) * spill_strength

    fx_local = np.repeat(core_local[..., None], 3, axis=2)
    fx_local += colour_local[..., None] * color_lin[None, None, :]
    fx_local += glow_local

    kernel = _directional_kernel(velocity, motion_blur_gain, motion_blur_max_len)
    if kernel is not None:
        fx_local = cv2.filter2D(fx_local, -1, kernel)

    full_fx[y0:y1, x0:x1] = fx_local
    return full_fx, blade_u8
```

Then replace the body of `render_glow`'s frame loop (the block from
`if has_mask:` through `full_fx[y0:y1, x0:x1] = fx_local`) with a call to
this new function:

```python
    for n, fname in enumerate(frame_files):
        idx = int(os.path.splitext(fname)[0])
        frame = cv2.imread(os.path.join(frames_dir, fname))
        mask = load_mask_optional(masks_dir, idx)

        row = n if n < n_motion else None
        tip_i = tip_arr[row] if row is not None else None
        hilt_i = hilt_arr[row] if row is not None else None
        vel_i = velocity[row] if row is not None else np.zeros(2)
        frac = ignition_fraction(n, first_active, last_active, ignition_ramp_frames)

        full_fx, blade_u8 = _composite_blade_contribution(
            frame.shape, mask, tip_i, hilt_i, vel_i,
            canonical_width, blade_extend, frac,
            tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
            core_erode_kernel, core_sigma, colour_sigma,
            color_lin, glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
            spill_strength, motion_blur_gain, motion_blur_max_len, bbox_margin,
        )

        plate_lin = _srgb_to_linear(frame)
        trail = np.maximum(trail * trail_decay, full_fx)
        darkened_plate, _ = knoll_darken(
            plate_lin, blade_u8, knoll_dilate_px, knoll_darken_factor, knoll_feather_sigma,
            dilate_kernel=knoll_dilate_kernel,
        )
        wrap = _light_wrap(blade_u8, color_lin, wrap_dilate_kernel, wrap_blur_sigma, light_wrap_strength)
        jitter = 1.0 + float(rng.uniform(-flicker_strength, flicker_strength))
        combined = darkened_plate + (trail + wrap) * jitter
        combined = _soft_tonemap(combined)
        out = _linear_to_srgb(combined)
        cv2.imwrite(
            os.path.join(output_frames_dir, f"{idx:05d}.png"),
            out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
        report((n + 1) / total * 100, f"frame {n + 1}/{total}")
```

Remove the now-dead `has_mask` variable and the inline block this replaced.
`_apply_ignition` and `ignition_fraction` stay exactly as they are (Task 1
just moves *where* `_apply_ignition` is called from -- inside the new helper
instead of inline in the loop -- `ignition_fraction` itself is still called
once per frame in the loop, same as before).

- [ ] **Step 4: Run the full glow test suite**

Run: `.venv/bin/python -m pytest tests/pipeline/test_glow.py -v`
Expected: all tests PASS, including the new characterization test and all
20 pre-existing tests, unmodified.

- [ ] **Step 5: Commit**

```bash
git add src/lightsaber_fx/pipeline/glow.py tests/pipeline/test_glow.py
git commit -m "Extract per-blade compositing out of render_glow for reuse by multi-saber rendering"
```

---

### Task 2: Add `render_glow_multi`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/glow.py`
- Test: `tests/pipeline/test_glow.py`

**Interfaces:**
- Consumes: `_composite_blade_contribution` (Task 1), `ignition_fraction`, `IGNITION_RAMP_SECONDS`, `knoll_darken`, `_light_wrap`, `_srgb_to_linear`, `_linear_to_srgb`, `_soft_tonemap`, `_robust_median`, `load_mask_optional`, `load_motion`, `_stabilize_tip_hilt` (all existing, unchanged).
- Produces: `render_glow_multi(frames_dir, objects, video_meta_path, output_frames_dir, blade_extend=True, tip_extend_frac=0.10, hilt_taper_frac=0.12, hilt_taper_min_frac=0.35, core_erode_frac=0.45, core_blur_frac=0.18, colour_blur_frac=0.35, glow_scales=(0.5, 1.0, 2.0), glow_falloff_tau=1.0, glow_crush=4.0, knoll_darken_factor=0.7, knoll_dilate_frac=1.5, knoll_feather_frac=0.8, trail_decay=0.7, ignition_ramp_seconds=IGNITION_RAMP_SECONDS, motion_blur_gain=0.35, motion_blur_max_len=24, chromatic_bloom_frac=0.10, flicker_strength=0.04, rng_seed=12345, light_wrap_strength=0.15, light_wrap_dilate_frac=1.5, progress_cb=None) -> None`, where `objects: list[dict]`, each `{"masks_dir": str, "motion_path": str, "color": tuple[int,int,int] (BGR 0-255), "intensity": float}`. Every keyword defaults to the exact same value `render_glow` uses for that parameter -- these are shared visual-style knobs, not per-object.

- [ ] **Step 1: Write the failing test -- two independently-colored blades both render**

```python
def test_render_glow_multi_composites_two_independently_colored_blades(tmp_path):
    # Two synthetic objects, far apart, moving independently, each its own
    # color -- the core claim of multi-saber rendering.
    clip_a = _build_blade_clip(tmp_path, "objA", n_frames=5, blade_x0=10, dx=2, blade_len=30, blade_height=8)
    clip_b = _build_blade_clip(tmp_path, "objB", n_frames=5, blade_x0=150, dx=2, blade_len=30, blade_height=8, width=220)

    output_frames_dir = tmp_path / "glow_frames"
    from lightsaber_fx.pipeline.glow import render_glow_multi

    render_glow_multi(
        clip_a["frames_dir"],
        [
            {"masks_dir": clip_a["masks_dir"], "motion_path": clip_a["motion_path"], "color": (255, 0, 0), "intensity": 0.4},
            {"masks_dir": clip_b["masks_dir"], "motion_path": clip_b["motion_path"], "color": (0, 255, 0), "intensity": 0.4},
        ],
        clip_a["video_meta_path"], str(output_frames_dir),
        ignition_ramp_seconds=0,
    )

    img = _load_png(str(output_frames_dir), 0).astype(np.float64)
    baseline = float(clip_a["plate_value"])

    mask_a = blade.load_mask(clip_a["masks_dir"], 0)
    geo_a = blade.fit_blade(mask_a)
    px_a, py_a = round(geo_a.centroid[0]), round(geo_a.centroid[1])

    mask_b = blade.load_mask(clip_b["masks_dir"], 0)
    geo_b = blade.fit_blade(mask_b)
    px_b, py_b = round(geo_b.centroid[0]), round(geo_b.centroid[1])

    # Blue channel (index 0 in BGR) carries object A's pure-red-in-BGR-terms
    # signal; green channel (index 1) carries object B's.
    assert img[py_a, px_a, 2] - baseline > 20  # object A's red shows at A's own position
    assert img[py_b, px_b, 1] - baseline > 20  # object B's green shows at B's own position
    assert img[py_a, px_a, 1] - baseline < 5   # B's color doesn't bleed to A's position
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/pipeline/test_glow.py -k two_independently_colored -v`
Expected: FAIL with `ImportError` or `AttributeError: module has no attribute 'render_glow_multi'`.

- [ ] **Step 3: Implement `render_glow_multi`**

Add to `src/lightsaber_fx/pipeline/glow.py`, after `render_glow`:

```python
def render_glow_multi(
    frames_dir,
    objects,
    video_meta_path,
    output_frames_dir,
    blade_extend=True,
    tip_extend_frac=0.10,
    hilt_taper_frac=0.12,
    hilt_taper_min_frac=0.35,
    core_erode_frac=0.45,
    core_blur_frac=0.18,
    colour_blur_frac=0.35,
    glow_scales=(0.5, 1.0, 2.0),
    glow_falloff_tau=1.0,
    glow_crush=4.0,
    knoll_darken_factor=0.7,
    knoll_dilate_frac=1.5,
    knoll_feather_frac=0.8,
    trail_decay=0.7,
    ignition_ramp_seconds=IGNITION_RAMP_SECONDS,
    motion_blur_gain=0.35,
    motion_blur_max_len=24,
    chromatic_bloom_frac=0.10,
    flicker_strength=0.04,
    rng_seed=12345,
    light_wrap_strength=0.15,
    light_wrap_dilate_frac=1.5,
    progress_cb=None,
):
    """Like `render_glow`, but for `len(objects)` (1-4) simultaneously
    tracked sabers, each with its own mask/motion/color/intensity, summed
    into one composited PNG sequence. See `_composite_blade_contribution`'s
    docstring for how a single object's contribution is computed; this
    function's job is purely to do that once per object per frame, sum the
    results, and run the shared (object-count-agnostic) trail/knoll-darken/
    light-wrap/tonemap steps exactly once per frame -- the same steps
    `render_glow` already runs on its own single contribution.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    os.makedirs(output_frames_dir, exist_ok=True)
    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    with open(video_meta_path) as f:
        fps = float(f.readline())

    first = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w = first.shape[:2]
    total = len(frame_files)
    ignition_ramp_frames = round(fps * ignition_ramp_seconds)

    # Per-object setup: everything render_glow computes once from its one
    # shared masks_dir/motion_path, computed once per object here instead.
    prepared = []
    max_canonical_width = 1.0
    for obj in objects:
        masks_dir = obj["masks_dir"]
        motion = load_motion(obj["motion_path"])
        tip_arr = np.asarray(motion.get("tip", np.zeros((0, 2))), dtype=np.float64)
        hilt_arr = np.asarray(motion.get("hilt", np.zeros((0, 2))), dtype=np.float64)
        axis_arr = np.asarray(motion.get("axis", np.zeros((0, 2))), dtype=np.float64)
        width_arr = np.asarray(motion.get("width", np.zeros(0)), dtype=np.float64)
        tip_arr, hilt_arr, axis_arr = _stabilize_tip_hilt(tip_arr, hilt_arr, axis_arr)
        velocity = np.zeros_like(tip_arr)
        if len(tip_arr) > 1:
            velocity[1:] = np.diff(tip_arr, axis=0)
        velocity = np.nan_to_num(velocity, nan=0.0)

        canonical_width = _robust_median(width_arr, default=6.0)
        if canonical_width <= 0:
            canonical_width = 6.0
        max_canonical_width = max(max_canonical_width, canonical_width)

        mask_present = []
        for fname in frame_files:
            idx = int(os.path.splitext(fname)[0])
            m = load_mask_optional(masks_dir, idx)
            mask_present.append(m is not None and m.any())
        active_indices = [n for n, present in enumerate(mask_present) if present]

        color_lin = (np.asarray(obj["color"], dtype=np.float32) / 255.0) ** _GAMMA
        core_erode_px = max(1, round(canonical_width * core_erode_frac))
        prepared.append({
            "masks_dir": masks_dir,
            "tip_arr": tip_arr, "hilt_arr": hilt_arr, "velocity": velocity,
            "n_motion": len(tip_arr),
            "canonical_width": canonical_width,
            "first_active": active_indices[0] if active_indices else None,
            "last_active": active_indices[-1] if active_indices else None,
            "color_lin": color_lin,
            "spill_strength": obj["intensity"],
            "core_erode_kernel": np.ones((core_erode_px, core_erode_px), np.uint8),
            "core_sigma": max(0.6, canonical_width * core_blur_frac),
            "colour_sigma": max(0.6, canonical_width * colour_blur_frac),
        })

    knoll_dilate_px = max(1, round(max_canonical_width * knoll_dilate_frac))
    knoll_dilate_kernel = np.ones((knoll_dilate_px, knoll_dilate_px), np.uint8)
    knoll_feather_sigma = max(1.0, max_canonical_width * knoll_feather_frac)
    wrap_dilate_px = max(1, round(max_canonical_width * light_wrap_dilate_frac))
    wrap_dilate_kernel = np.ones((wrap_dilate_px, wrap_dilate_px), np.uint8)
    wrap_blur_sigma = max(1.0, max_canonical_width)

    max_scale = max(glow_scales) * (1.0 + chromatic_bloom_frac)
    blur_reach = int(np.ceil(max_canonical_width * max_scale * 3.5))
    bbox_margin = blur_reach + motion_blur_max_len + 5

    rng = np.random.default_rng(rng_seed)
    trail = np.zeros((h, w, 3), dtype=np.float32)

    for n, fname in enumerate(frame_files):
        idx = int(os.path.splitext(fname)[0])
        frame = cv2.imread(os.path.join(frames_dir, fname))
        plate_lin = _srgb_to_linear(frame)

        combined_fx = np.zeros((h, w, 3), dtype=np.float32)
        combined_blade_u8 = np.zeros((h, w), dtype=np.uint8)

        for obj_state in prepared:
            mask = load_mask_optional(obj_state["masks_dir"], idx)
            row = n if n < obj_state["n_motion"] else None
            tip_i = obj_state["tip_arr"][row] if row is not None else None
            hilt_i = obj_state["hilt_arr"][row] if row is not None else None
            vel_i = obj_state["velocity"][row] if row is not None else np.zeros(2)
            frac = ignition_fraction(
                n, obj_state["first_active"], obj_state["last_active"], ignition_ramp_frames,
            )

            full_fx, blade_u8 = _composite_blade_contribution(
                frame.shape, mask, tip_i, hilt_i, vel_i,
                obj_state["canonical_width"], blade_extend, frac,
                tip_extend_frac, hilt_taper_frac, hilt_taper_min_frac,
                obj_state["core_erode_kernel"], obj_state["core_sigma"], obj_state["colour_sigma"],
                obj_state["color_lin"], glow_scales, glow_falloff_tau, glow_crush, chromatic_bloom_frac,
                obj_state["spill_strength"], motion_blur_gain, motion_blur_max_len, bbox_margin,
            )
            combined_fx += full_fx
            combined_blade_u8 = np.maximum(combined_blade_u8, blade_u8)

        trail = np.maximum(trail * trail_decay, combined_fx)
        darkened_plate, _ = knoll_darken(
            plate_lin, combined_blade_u8, knoll_dilate_px, knoll_darken_factor, knoll_feather_sigma,
            dilate_kernel=knoll_dilate_kernel,
        )
        wrap_total = np.zeros((h, w, 3), dtype=np.float32)
        for obj_state in prepared:
            wrap_total += _light_wrap(
                combined_blade_u8, obj_state["color_lin"], wrap_dilate_kernel,
                wrap_blur_sigma, light_wrap_strength / len(prepared),
            )
        jitter = 1.0 + float(rng.uniform(-flicker_strength, flicker_strength))
        combined = darkened_plate + (trail + wrap_total) * jitter
        combined = _soft_tonemap(combined)
        out = _linear_to_srgb(combined)
        cv2.imwrite(
            os.path.join(output_frames_dir, f"{idx:05d}.png"),
            out, [int(cv2.IMWRITE_PNG_COMPRESSION), 3],
        )
        report((n + 1) / total * 100, f"frame {n + 1}/{total}")
```

Note the light-wrap step: with N objects it's computed once per object
(each tinted by that object's own color) against the shared union mask, each
scaled down by `1/len(prepared)` so N overlapping wraps don't sum to a much
stronger effect than the single-object case -- a deliberate, simple choice;
revisit if it looks wrong on a real multi-saber render.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pipeline/test_glow.py -k two_independently_colored -v`
Expected: PASS.

- [ ] **Step 5: Add a single-object equivalence test**

```python
def test_render_glow_multi_with_one_object_matches_render_glow(tmp_path):
    # The N=1 case must agree with today's render_glow -- not byte-for-byte
    # (light-wrap's 1/len(prepared) scaling is a no-op at N=1, but summing
    # order/floating point can still differ trivially), but materially the
    # same rendered result.
    clip = _build_blade_clip(tmp_path, "equiv", n_frames=4)
    out_single = str(tmp_path / "out_single")
    out_multi = str(tmp_path / "out_multi")

    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_single, clip["motion_path"], color=(40, 40, 255), intensity=0.35,
        ignition_ramp_seconds=0,
    )
    render_glow_multi(
        clip["frames_dir"],
        [{"masks_dir": clip["masks_dir"], "motion_path": clip["motion_path"], "color": (40, 40, 255), "intensity": 0.35}],
        clip["video_meta_path"], out_multi,
        ignition_ramp_seconds=0,
    )

    for i in range(clip["n_frames"]):
        a = _load_png(out_single, i).astype(np.int16)
        b = _load_png(out_multi, i).astype(np.int16)
        assert np.abs(a - b).max() <= 2  # allow trivial floating-point rounding differences
```

- [ ] **Step 6: Run it, confirm it passes**

Run: `.venv/bin/python -m pytest tests/pipeline/test_glow.py -k equivalence -v`
Expected: PASS. If it fails, the difference is almost always the light-wrap
`/len(prepared)` division (should be `/1` at N=1, a no-op) -- check that
first before anything else.

- [ ] **Step 7: Run the full glow suite and commit**

Run: `.venv/bin/python -m pytest tests/pipeline/test_glow.py -v`
Expected: all PASS (23 tests now).

```bash
git add src/lightsaber_fx/pipeline/glow.py tests/pipeline/test_glow.py
git commit -m "Add render_glow_multi for compositing up to 4 independently-colored sabers"
```

---

### Task 3: Add `track_objects`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/track.py`
- Test: `tests/pipeline/test_track.py`

**Interfaces:**
- Produces: `track_objects(frames_dir, prompts, checkpoint_path, config_name, device, n_frames, progress_cb=None) -> None`, where `prompts: list[dict]`, each `{"obj_id": int, "masks_dir": str, "points": list[[x, y]], "labels": list[int]}`. All objects are prompted on frame 0 and propagated forward once, in one shared SAM2 session.

- [ ] **Step 1: Write the failing test**

```python
@requires_sam2_checkpoint
def test_track_objects_tracks_two_objects_independently(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    n_frames = 3
    for i in range(n_frames):
        frame = np.zeros((64, 128, 3), dtype=np.uint8)
        frame[20:30, 10 + i * 3:20 + i * 3] = (255, 255, 255)   # object A, left side
        frame[20:30, 90 + i * 3:100 + i * 3] = (255, 255, 255)  # object B, right side
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), frame)

    masks_dir_a = tmp_path / "masks" / "0"
    masks_dir_b = tmp_path / "masks" / "1"

    from lightsaber_fx.pipeline.track import track_objects

    track_objects(
        str(frames_dir),
        [
            {"obj_id": 0, "masks_dir": str(masks_dir_a), "points": [[15, 25]], "labels": [1]},
            {"obj_id": 1, "masks_dir": str(masks_dir_b), "points": [[95, 25]], "labels": [1]},
        ],
        checkpoint_path=str(paths.get_checkpoint_path()),
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cpu",
        n_frames=n_frames,
    )

    for masks_dir in (masks_dir_a, masks_dir_b):
        assert sorted(os.listdir(masks_dir)) == [f"{i:05d}.npz" for i in range(n_frames)]
        for i in range(n_frames):
            assert load_mask(str(masks_dir), i).any()

    # The two objects' masks must stay on their own sides of the frame,
    # not bleed into or duplicate each other.
    mask_a0 = load_mask(str(masks_dir_a), 0)
    mask_b0 = load_mask(str(masks_dir_b), 0)
    assert not np.any(mask_a0 & mask_b0)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/bin/python -m pytest tests/pipeline/test_track.py -k two_objects -v`
Expected: FAIL with `ImportError` (only if the checkpoint is installed
locally -- otherwise it's SKIPPED, which is fine; this step is only
meaningful on a machine with `lightsaber-fx setup` already run).

- [ ] **Step 3: Implement `track_objects`**

Add to `src/lightsaber_fx/pipeline/track.py`, after `track_object`:

```python
def track_objects(
    frames_dir,
    prompts,
    checkpoint_path,
    config_name,
    device,
    n_frames,
    progress_cb=None,
):
    """Like `track_object`, but for `len(prompts)` (1-4) objects tracked
    together in one shared SAM2 session -- cheaper than N separate sessions,
    since each frame's image features are encoded once regardless of object
    count. Every object is prompted at frame 0 (see the multi-saber backend
    plan's Global Constraints: auto-detect and mid-clip prompting are not
    supported for multi-object tracking), so propagation runs forward-only,
    once, covering the whole clip in a single pass.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=device)

    state = predictor.init_state(video_path=frames_dir)
    for prompt in prompts:
        predictor.add_new_points_or_box(
            state,
            frame_idx=0,
            obj_id=prompt["obj_id"],
            points=np.array(prompt["points"], dtype=np.float32),
            labels=np.array(prompt["labels"], dtype=np.int32),
        )

    masks_dir_by_obj_id = {p["obj_id"]: p["masks_dir"] for p in prompts}
    for masks_dir in masks_dir_by_obj_id.values():
        os.makedirs(masks_dir, exist_ok=True)

    written = 0
    total_writes = n_frames * len(prompts)
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        for i, obj_id in enumerate(obj_ids):
            mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
            save_mask(masks_dir_by_obj_id[obj_id], frame_idx, mask)
            written += 1
        report(min(written / total_writes * 100, 100.0), f"frame {frame_idx + 1}/{n_frames}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/pipeline/test_track.py -k two_objects -v`
Expected: PASS (only on a machine with the SAM2 checkpoint installed).

- [ ] **Step 5: Run the full track suite and commit**

Run: `.venv/bin/python -m pytest tests/pipeline/test_track.py -v`
Expected: all PASS (or SKIPPED for the `@requires_sam2_checkpoint` ones, if
run somewhere without the checkpoint -- that's the existing, expected
behavior for this file, unchanged).

```bash
git add src/lightsaber_fx/pipeline/track.py tests/pipeline/test_track.py
git commit -m "Add track_objects for tracking up to 4 sabers in one shared SAM2 session"
```

---

### Task 4: Add `mix_hums` to `audio.py`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/audio.py`
- Test: `tests/pipeline/test_audio.py`

**Interfaces:**
- Consumes: `_soft_limit(x, ceiling=0.95)` (existing, unchanged), `soundfile` (already imported as `sf` in `audio.py`).
- Produces: `mix_hums(wav_paths, out_path) -> None`, where `wav_paths: list[str]` (1-4 paths to WAV files of identical length/sample rate -- guaranteed by the caller, since every object's `synthesize_audio` call reads duration from the same shared `video_meta_path`).

- [ ] **Step 1: Write the failing test**

```python
def test_mix_hums_sums_multiple_tracks_without_clipping(tmp_path):
    sr = 44100
    n = sr  # 1 second
    t = np.linspace(0, 1, n, endpoint=False)

    # Two loud, in-phase stereo tones -- naive summing would clip badly.
    tone_a = 0.9 * np.sin(2 * np.pi * 220 * t)
    tone_b = 0.9 * np.sin(2 * np.pi * 220 * t)
    path_a = str(tmp_path / "a.wav")
    path_b = str(tmp_path / "b.wav")
    sf.write(path_a, np.stack([tone_a, tone_a], axis=-1).astype(np.float32), sr)
    sf.write(path_b, np.stack([tone_b, tone_b], axis=-1).astype(np.float32), sr)

    out_path = str(tmp_path / "mixed.wav")
    from lightsaber_fx.pipeline.audio import mix_hums
    mix_hums([path_a, path_b], out_path)

    mixed, out_sr = sf.read(out_path)
    assert out_sr == sr
    assert mixed.shape == (n, 2)
    assert np.isfinite(mixed).all()
    assert np.abs(mixed).max() <= 1.0  # no clipping/overflow
    assert np.abs(mixed).max() > 0.5   # but not silenced into nothing either


def test_mix_hums_with_one_track_is_effectively_a_passthrough(tmp_path):
    sr = 44100
    n = 1000
    tone = (0.3 * np.sin(2 * np.pi * 100 * np.linspace(0, 1, n, endpoint=False))).astype(np.float32)
    path_a = str(tmp_path / "a.wav")
    sf.write(path_a, np.stack([tone, tone], axis=-1), sr)

    out_path = str(tmp_path / "mixed.wav")
    from lightsaber_fx.pipeline.audio import mix_hums
    mix_hums([path_a], out_path)

    mixed, _ = sf.read(out_path)
    original, _ = sf.read(path_a)
    assert np.abs(mixed - original).max() < 0.01  # soft-limit is near-identity well under ceiling
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_audio.py -k mix_hums -v`
Expected: FAIL with `ImportError: cannot import name 'mix_hums'`.

- [ ] **Step 3: Implement `mix_hums`**

Add to `src/lightsaber_fx/pipeline/audio.py`, near `synthesize_audio`:

```python
def mix_hums(wav_paths, out_path):
    """Sum 1-4 same-length, same-sample-rate stereo WAV files (one hum per
    tracked saber) into a single track, soft-limited (`_soft_limit`, the
    same tanh ceiling `synthesize_audio` already uses) so multiple sabers
    moving in sync don't clip. `wav_paths` are guaranteed equal-length by
    the caller -- every object's `synthesize_audio` call derives its
    duration from the same shared `video_meta_path`.
    """
    mixed = None
    sr = None
    for path in wav_paths:
        data, file_sr = sf.read(path)
        if mixed is None:
            mixed = np.zeros_like(data, dtype=np.float64)
            sr = file_sr
        mixed += data
    mixed = _soft_limit(mixed, ceiling=0.95)
    mixed = np.nan_to_num(mixed, nan=0.0, posinf=0.95, neginf=-0.95)
    sf.write(out_path, mixed.astype(np.float32), sr)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_audio.py -k mix_hums -v`
Expected: PASS.

- [ ] **Step 5: Run the full audio suite and commit**

Run: `.venv/bin/python -m pytest tests/pipeline/test_audio.py -v`
Expected: all PASS.

```bash
git add src/lightsaber_fx/pipeline/audio.py tests/pipeline/test_audio.py
git commit -m "Add mix_hums to combine per-saber audio into one soft-limited track"
```

---

### Task 5: Extend `job_meta.py` for multi-object jobs

**Files:**
- Modify: `src/lightsaber_fx/pipeline/job_meta.py`
- Test: `tests/pipeline/test_job_meta.py`

**Interfaces:**
- Produces: `write_job_meta(job_dir, source_video, object_ids=None) -> dict` (existing signature gains one optional parameter); `JobInfo` gains an `object_ids: Optional[list[int]]` field; `describe_job`/`require_rerenderable` check per-object masks (`masks/{obj_id}/`) instead of the flat `masks/` when `object_ids` is present in the job's meta.

- [ ] **Step 1: Write the failing tests**

Add to `tests/pipeline/test_job_meta.py`:

```python
def test_write_job_meta_records_object_ids_when_given(tmp_path):
    meta = write_job_meta(str(tmp_path), source_video="/tmp/x.mp4", object_ids=[0, 1, 2])
    assert meta["object_ids"] == [0, 1, 2]
    assert read_job_meta(str(tmp_path))["object_ids"] == [0, 1, 2]


def test_write_job_meta_omits_object_ids_when_not_given(tmp_path):
    # Legacy/CLI single-object jobs never pass object_ids -- the key must
    # not appear at all, not be written as null, so describe_job's
    # "is this a multi-object job" check can be a plain `in` test.
    meta = write_job_meta(str(tmp_path), source_video="/tmp/x.mp4")
    assert "object_ids" not in meta


def test_describe_job_checks_per_object_masks_for_a_multi_object_job(tmp_path):
    job_dir = tmp_path
    (job_dir / "masks" / "0").mkdir(parents=True)
    (job_dir / "masks" / "1").mkdir(parents=True)
    save_mask(str(job_dir / "masks" / "0"), 0, np.ones((4, 4), dtype=bool))
    save_mask(str(job_dir / "masks" / "1"), 0, np.ones((4, 4), dtype=bool))
    (job_dir / "motion" ).mkdir()
    compute_motion(str(job_dir / "masks" / "0"), str(job_dir / "motion" / "0.npz"))
    compute_motion(str(job_dir / "masks" / "1"), str(job_dir / "motion" / "1.npz"))
    (job_dir / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(str(job_dir), source_video=str(tmp_path / "src.mp4"), object_ids=[0, 1])
    (tmp_path / "src.mp4").write_bytes(b"x")

    info = describe_job(str(job_dir))

    assert info.rerenderable, info.reason
    assert info.object_ids == [0, 1]


def test_describe_job_reports_missing_object_masks_for_a_multi_object_job(tmp_path):
    job_dir = tmp_path
    (job_dir / "masks" / "0").mkdir(parents=True)
    save_mask(str(job_dir / "masks" / "0"), 0, np.ones((4, 4), dtype=bool))
    # object 1's masks/ dir is entirely missing
    (job_dir / "motion").mkdir()
    compute_motion(str(job_dir / "masks" / "0"), str(job_dir / "motion" / "0.npz"))
    (job_dir / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(str(job_dir), source_video=str(tmp_path / "src.mp4"), object_ids=[0, 1])
    (tmp_path / "src.mp4").write_bytes(b"x")

    info = describe_job(str(job_dir))

    assert not info.rerenderable
    assert "object 1" in info.reason
```

`tests/pipeline/test_job_meta.py` already imports `numpy as np` and
`from lightsaber_fx.pipeline.blade import compute_motion, save_mask` --
no new imports needed for these tests.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_job_meta.py -v`
Expected: the 4 new tests FAIL (`write_job_meta() got an unexpected keyword
argument 'object_ids'` / `AttributeError: 'JobInfo' object has no attribute
'object_ids'`).

- [ ] **Step 3: Implement the changes**

In `src/lightsaber_fx/pipeline/job_meta.py`:

```python
class JobInfo(NamedTuple):
    job_id: str
    source_video: Optional[str]
    frame_count: Optional[int]
    created_at: Optional[float]
    rerenderable: bool
    reason: Optional[str]
    object_ids: Optional[list] = None


def write_job_meta(job_dir, source_video, object_ids=None):
    meta = {
        "source_video": os.path.abspath(str(source_video)),
        "created_at": time.time(),
    }
    if object_ids is not None:
        meta["object_ids"] = list(object_ids)
    with open(_job_meta_path(job_dir), "w") as f:
        json.dump(meta, f)
    return meta
```

Update `describe_job` to branch on whether `object_ids` is present:

```python
def describe_job(job_dir):
    job_dir = str(job_dir)
    job_id = os.path.basename(job_dir.rstrip(os.sep))
    meta = read_job_meta(job_dir)
    source_video = meta.get("source_video") if meta else None
    created_at = meta.get("created_at") if meta else None
    object_ids = meta.get("object_ids") if meta else None
    frame_count = _read_frame_count(job_dir)

    reasons = []
    if object_ids is not None:
        for obj_id in object_ids:
            obj_masks_dir = os.path.join(job_dir, "masks", str(obj_id))
            if not os.path.isdir(obj_masks_dir) or not mask_frame_indices(obj_masks_dir):
                reasons.append(f"no masks/ for object {obj_id} (tracking was never run, or the job was cleaned)")
            if not os.path.exists(os.path.join(job_dir, "motion", f"{obj_id}.npz")):
                reasons.append(f"no motion/{obj_id}.npz")
    else:
        masks_dir = os.path.join(job_dir, "masks")
        if not os.path.isdir(masks_dir) or not mask_frame_indices(masks_dir):
            reasons.append("no masks/ (tracking was never run, or the job was cleaned)")
        if not os.path.exists(os.path.join(job_dir, "motion.npz")):
            reasons.append("no motion.npz")

    if not os.path.exists(os.path.join(job_dir, "video_meta.txt")):
        reasons.append("no video_meta.txt")
    if source_video is None:
        reasons.append("no recorded source clip path (job predates rerender support)")
    elif not os.path.exists(source_video):
        reasons.append(f"source clip not found: {source_video} (it may have been moved or deleted)")

    return JobInfo(
        job_id=job_id,
        source_video=source_video,
        frame_count=frame_count,
        created_at=created_at,
        rerenderable=not reasons,
        reason="; ".join(reasons) if reasons else None,
        object_ids=object_ids,
    )
```

`require_rerenderable` needs no changes -- it already just calls
`describe_job` and raises on `not info.rerenderable`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_job_meta.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/lightsaber_fx/pipeline/job_meta.py tests/pipeline/test_job_meta.py
git commit -m "Teach job_meta about multi-object jobs (per-object mask/motion rerenderability checks)"
```

---

### Task 6: Add `run_pipeline_multi`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/runner.py`
- Test: `tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: `track_objects` (Task 3), `render_glow_multi` (Task 2), `mix_hums` (Task 4), `write_job_meta(..., object_ids=...)` (Task 5), `extract_frames`, `compute_motion`, `synthesize_audio`, `encode` (all existing, unchanged), `parse_color` (existing).
- Produces: `run_pipeline_multi(input_video, sabers, output_path, job_dir, checkpoint_path, device, config_name="configs/sam2.1/sam2.1_hiera_s.yaml", blade_extend=True, progress_cb=None) -> str`, where `sabers: list[dict]`, each `{"points": [[x,y],...], "labels": [1,...], "color": "red"|"#RRGGBB", "intensity": float, "voice": "neutral"|"jedi"|"sith"}`. Returns `output_path`.

- [ ] **Step 1: Write the failing test**

```python
@requires_ffmpeg
def test_run_pipeline_multi_end_to_end_with_stubbed_tracking(tmp_path, monkeypatch, tiny_video_path):
    def fake_track_objects(frames_dir, prompts, checkpoint_path, config_name, device, n_frames, progress_cb=None):
        for prompt in prompts:
            os.makedirs(prompt["masks_dir"], exist_ok=True)
            for i in range(n_frames):
                x = 5 + i * 3
                mask = np.zeros((48, 64), dtype=bool)
                mask[10:20, x:x + 6] = True
                save_mask(prompt["masks_dir"], i, mask)
            if progress_cb:
                progress_cb(100, "done")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_objects", fake_track_objects)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"

    result = run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert result == str(output_path)
    assert output_path.exists() and output_path.stat().st_size > 0
    meta = job_meta.read_job_meta(str(job_dir))
    assert meta["object_ids"] == [0, 1]
    assert os.path.isdir(job_dir / "masks" / "0")
    assert os.path.isdir(job_dir / "masks" / "1")
    assert (job_dir / "motion" / "0.npz").exists()
    assert (job_dir / "motion" / "1.npz").exists()


def test_run_pipeline_multi_validates_every_saber_color_before_tracking(tmp_path, monkeypatch, tiny_video_path):
    def fail_if_called(*a, **k):
        raise AssertionError("extract_frames should not run before every color is validated")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.extract_frames", fail_if_called)
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(ValueError):
        run_pipeline_multi(
            input_video=str(tiny_video_path),
            sabers=[
                {"points": [[1, 1]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
                {"points": [[1, 1]], "labels": [1], "color": "not-a-color", "intensity": 0.35, "voice": "neutral"},
            ],
            output_path=str(tmp_path / "final.mp4"),
            job_dir=str(job_dir),
            checkpoint_path="unused",
            device="cpu",
        )
```

`tests/pipeline/test_runner.py` does not currently import `os` or `save_mask`
-- add these two lines to its existing import block:

```python
import os

from lightsaber_fx.pipeline.blade import save_mask
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runner.py -k multi -v`
Expected: FAIL with `ImportError: cannot import name 'run_pipeline_multi'`.

- [ ] **Step 3: Implement `run_pipeline_multi`**

Add to `src/lightsaber_fx/pipeline/runner.py`, after `run_pipeline`:

```python
def _multi_job_paths(job_dir, object_ids):
    """Per-object mask/motion paths for a multi-saber job, plus the shared
    (object-count-agnostic) paths every job already uses."""
    return {
        "frames_dir": os.path.join(job_dir, "frames"),
        "video_meta_path": os.path.join(job_dir, "video_meta.txt"),
        "glow_frames_dir": os.path.join(job_dir, "glow_frames"),
        "masks_dirs": {oid: os.path.join(job_dir, "masks", str(oid)) for oid in object_ids},
        "motion_paths": {oid: os.path.join(job_dir, "motion", f"{oid}.npz") for oid in object_ids},
        "audio_paths": {oid: os.path.join(job_dir, f"saber_audio_{oid}.wav") for oid in object_ids},
        "mixed_audio_path": os.path.join(job_dir, "saber_audio.wav"),
    }


def run_pipeline_multi(
    input_video,
    sabers,
    output_path,
    job_dir,
    checkpoint_path,
    device,
    config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
    blade_extend=True,
    progress_cb=None,
):
    """Like `run_pipeline`, but for 1-4 simultaneously tracked sabers, each
    with its own color/intensity/voice. Every saber is prompted at frame 0
    (see Global Constraints in the multi-saber backend plan) and tracked
    together in one SAM2 session via `track_objects`.
    """
    stage_cb = _make_stage_cb(progress_cb)
    color_bgrs = [parse_color(s["color"]) for s in sabers]  # validate every color up front

    object_ids = list(range(len(sabers)))
    paths = _multi_job_paths(job_dir, object_ids)
    os.makedirs(os.path.dirname(paths["masks_dirs"][0]), exist_ok=True)
    os.makedirs(os.path.dirname(paths["motion_paths"][0]), exist_ok=True)

    if progress_cb:
        progress_cb("extract", 0, "extracting frames")
    fps, n_frames = extract_frames(input_video, paths["frames_dir"])
    with open(paths["video_meta_path"], "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    job_meta.write_job_meta(job_dir, source_video=input_video, object_ids=object_ids)
    if progress_cb:
        progress_cb("extract", 100, f"{n_frames} frames at {fps:.2f} fps")

    prompts = [
        {"obj_id": oid, "masks_dir": paths["masks_dirs"][oid], "points": s["points"], "labels": s["labels"]}
        for oid, s in zip(object_ids, sabers)
    ]
    track_objects(
        paths["frames_dir"], prompts, checkpoint_path, config_name, device, n_frames,
        progress_cb=stage_cb("track"),
    )

    for oid in object_ids:
        n_tracked, n_with_blade = compute_motion(
            paths["masks_dirs"][oid], paths["motion_paths"][oid], progress_cb=stage_cb("motion"),
        )
        _require_usable_track(n_tracked, n_with_blade, stage_cb("motion"))

    objects = [
        {
            "masks_dir": paths["masks_dirs"][oid],
            "motion_path": paths["motion_paths"][oid],
            "color": color_bgrs[i],
            "intensity": sabers[i]["intensity"],
        }
        for i, oid in enumerate(object_ids)
    ]
    render_glow_multi(
        paths["frames_dir"], objects, paths["video_meta_path"], paths["glow_frames_dir"],
        blade_extend=blade_extend, progress_cb=stage_cb("glow"),
    )

    for i, oid in enumerate(object_ids):
        synthesize_audio(
            paths["motion_paths"][oid], paths["video_meta_path"], paths["audio_paths"][oid],
            voice=sabers[i]["voice"], progress_cb=stage_cb("audio"),
        )
    mix_hums(list(paths["audio_paths"].values()), paths["mixed_audio_path"])

    encode(paths["glow_frames_dir"], fps, paths["mixed_audio_path"], output_path, progress_cb=stage_cb("mux"))
    shutil.rmtree(paths["glow_frames_dir"], ignore_errors=True)

    return output_path
```

Add the two new imports at the top of `runner.py`:

```python
from .audio import mix_hums, synthesize_audio
from .glow import render_glow_multi
from .track import track_objects
```

(`synthesize_audio` is likely already imported -- check before duplicating.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runner.py -k multi -v`
Expected: PASS.

- [ ] **Step 5: Run the full runner suite and commit**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runner.py -v`
Expected: all PASS.

```bash
git add src/lightsaber_fx/pipeline/runner.py tests/pipeline/test_runner.py
git commit -m "Add run_pipeline_multi orchestrating multi-saber tracking through to a mixed render"
```

---

### Task 7: Add `rerender_pipeline_multi`

**Files:**
- Modify: `src/lightsaber_fx/pipeline/runner.py`
- Test: `tests/pipeline/test_runner.py`

**Interfaces:**
- Consumes: `_multi_job_paths` (Task 6), `job_meta.require_rerenderable` (Task 5), `render_glow_multi`, `mix_hums`, `synthesize_audio`, `encode`, `extract_frames`, `parse_color`.
- Produces: `rerender_pipeline_multi(job_dir, output_path, sabers, blade_extend=True, progress_cb=None) -> str`, where `sabers: list[dict]`, each `{"color": ..., "intensity": ..., "voice": ...}` (no `points`/`labels` -- tracking is never re-run). `len(sabers)` must equal the job's recorded `object_ids` count.

- [ ] **Step 1: Write the failing test**

```python
def test_rerender_pipeline_multi_reuses_cached_masks_for_a_new_color(tmp_path, monkeypatch, tiny_video_path):
    job_dir = tmp_path / "job"
    for oid in (0, 1):
        masks_dir = job_dir / "masks" / str(oid)
        for i in range(5):
            mask = np.zeros((48, 64), dtype=bool)
            mask[10:20, 5 + i * 3:11 + i * 3] = True
            save_mask(str(masks_dir), i, mask)
        motion_dir = job_dir / "motion"
        motion_dir.mkdir(exist_ok=True)
        compute_motion(str(masks_dir), str(motion_dir / f"{oid}.npz"))
    (job_dir / "video_meta.txt").write_text("10.0\n5\n")
    job_meta.write_job_meta(str(job_dir), source_video=str(tiny_video_path), object_ids=[0, 1])

    output_path = tmp_path / "final.mp4"
    result = rerender_pipeline_multi(
        job_dir=str(job_dir),
        output_path=str(output_path),
        sabers=[
            {"color": "green", "intensity": 0.6, "voice": "neutral"},
            {"color": "blue", "intensity": 0.3, "voice": "jedi"},
        ],
    )

    assert result == str(output_path)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_rerender_pipeline_multi_rejects_a_saber_count_mismatch(tmp_path, tiny_video_path):
    job_dir = tmp_path / "job"
    masks_dir = job_dir / "masks" / "0"
    for i in range(3):
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:20, 5:11] = True
        save_mask(str(masks_dir), i, mask)
    motion_dir = job_dir / "motion"
    motion_dir.mkdir(exist_ok=True)
    compute_motion(str(masks_dir), str(motion_dir / "0.npz"))
    (job_dir / "video_meta.txt").write_text("10.0\n3\n")
    job_meta.write_job_meta(str(job_dir), source_video=str(tiny_video_path), object_ids=[0])

    with pytest.raises(ValueError, match="1 tracked object"):
        rerender_pipeline_multi(
            job_dir=str(job_dir),
            output_path=str(tmp_path / "final.mp4"),
            sabers=[
                {"color": "red", "intensity": 0.35, "voice": "neutral"},
                {"color": "blue", "intensity": 0.35, "voice": "neutral"},
            ],
        )
```

This test file needs `@requires_ffmpeg` on the first test (matching the
existing `rerender_pipeline` tests' pattern) -- add it if the decorator
isn't already applied via a class or module-level marker.

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runner.py -k rerender_multi -v`
Expected: FAIL with `ImportError`/`NameError`.

- [ ] **Step 3: Implement `rerender_pipeline_multi`**

Add to `src/lightsaber_fx/pipeline/runner.py`, after `rerender_pipeline`:

```python
def rerender_pipeline_multi(
    job_dir,
    output_path,
    sabers,
    blade_extend=True,
    progress_cb=None,
):
    """Re-render an existing multi-saber job with new color/intensity/voice
    per saber, reusing every object's cached tracking masks -- `track_objects`
    never runs. Mirrors `rerender_pipeline`'s re-extract-frames-but-reuse-
    masks trade-off, generalized to N objects.
    """
    stage_cb = _make_stage_cb(progress_cb)
    color_bgrs = [parse_color(s["color"]) for s in sabers]  # validate before touching the job dir

    info = job_meta.require_rerenderable(job_dir)
    object_ids = info.object_ids
    if object_ids is None:
        raise ValueError("This job has no recorded object_ids -- it isn't a multi-saber job")
    if len(sabers) != len(object_ids):
        raise ValueError(
            f"This job has {len(object_ids)} tracked object(s), but {len(sabers)} saber(s) were given"
        )

    paths = _multi_job_paths(job_dir, object_ids)

    if progress_cb:
        progress_cb("extract", 0, "re-extracting frames from source clip")
    fps, n_frames = extract_frames(info.source_video, paths["frames_dir"])
    with open(paths["video_meta_path"], "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    if progress_cb:
        progress_cb("extract", 100, f"{n_frames} frames at {fps:.2f} fps")

    objects = [
        {
            "masks_dir": paths["masks_dirs"][oid],
            "motion_path": paths["motion_paths"][oid],
            "color": color_bgrs[i],
            "intensity": sabers[i]["intensity"],
        }
        for i, oid in enumerate(object_ids)
    ]
    render_glow_multi(
        paths["frames_dir"], objects, paths["video_meta_path"], paths["glow_frames_dir"],
        blade_extend=blade_extend, progress_cb=stage_cb("glow"),
    )

    for i, oid in enumerate(object_ids):
        synthesize_audio(
            paths["motion_paths"][oid], paths["video_meta_path"], paths["audio_paths"][oid],
            voice=sabers[i]["voice"], progress_cb=stage_cb("audio"),
        )
    mix_hums(list(paths["audio_paths"].values()), paths["mixed_audio_path"])

    encode(paths["glow_frames_dir"], fps, paths["mixed_audio_path"], output_path, progress_cb=stage_cb("mux"))
    shutil.rmtree(paths["glow_frames_dir"], ignore_errors=True)

    return output_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runner.py -k rerender_multi -v`
Expected: PASS.

- [ ] **Step 5: Run the full runner suite and commit**

Run: `.venv/bin/python -m pytest tests/pipeline/test_runner.py -v`
Expected: all PASS.

```bash
git add src/lightsaber_fx/pipeline/runner.py tests/pipeline/test_runner.py
git commit -m "Add rerender_pipeline_multi for re-rendering a multi-saber job with new colors"
```

---

### Task 8: Update `/api/jobs/{id}/points` for multiple sabers

**Files:**
- Modify: `src/lightsaber_fx/web/server.py`
- Test: `tests/web/test_server.py`

**Interfaces:**
- Consumes: `run_pipeline_multi` (Task 6).
- Produces: `POST /api/jobs/{job_id}/points` now expects body `{"sabers": [{"points": [[x,y,label],...], "prompt_frame": 0, "color": "red", "intensity": 0.35, "voice": "neutral"}, ...], "blade_extend": true}` -- 1 to 4 entries in `sabers`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/web/test_server.py`:

```python
def test_points_accepts_multiple_sabers_and_starts_a_multi_object_job(client, tiny_video_bytes, monkeypatch):
    captured = {}

    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"multi render bytes")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", fake_run_pipeline_multi)

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={
            "sabers": [
                {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"},
                {"points": [[20, 20, 1]], "color": "blue", "intensity": 0.5, "voice": "sith"},
            ],
        },
    )

    assert resp.status_code == 200
    server_module.manager.wait(timeout=2)
    assert len(captured["sabers"]) == 2
    assert captured["sabers"][0]["color"] == "red"
    assert captured["sabers"][1]["voice"] == "sith"


def test_points_rejects_zero_sabers(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(f"/api/jobs/{job_id}/points", json={"sabers": []})

    assert resp.status_code == 400


def test_points_rejects_more_than_four_sabers(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[1, 1, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}] * 5},
    )

    assert resp.status_code == 400


def test_points_rejects_a_saber_with_no_include_point(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[1, 1, 0]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )

    assert resp.status_code == 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/web/test_server.py -k "multi_object_job or rejects_zero_sabers or more_than_four or saber_with_no_include" -v`
Expected: FAIL (old single-`points`-list body shape still in place;
`test_points_accepts_multiple_sabers...` fails because `captured["sabers"]`
doesn't exist yet, the reject tests fail because today's validation doesn't
know about `sabers` at all and the existing single-object 400 rule doesn't
apply the same way).

- [ ] **Step 3: Update the endpoint**

Replace `submit_points` in `src/lightsaber_fx/web/server.py`. Keep
`_parse_render_params` for the existing single-object `/rerender` endpoint
(Task 9 changes `/rerender` separately) but add a new per-saber parser:

```python
def _parse_saber_specs(body: dict):
    """Validate and extract the list of per-saber specs from a `/points`
    request body: 1-4 entries, each needing at least one include point and
    a valid color/intensity/voice, using the exact same per-field rules
    `_parse_render_params` already enforces for the single-object endpoints."""
    sabers = body.get("sabers", [])
    if not 1 <= len(sabers) <= 4:
        raise HTTPException(status_code=400, detail="sabers must have between 1 and 4 entries")

    parsed = []
    for i, saber in enumerate(sabers):
        points_and_labels = saber.get("points", [])
        if not any(p[2] == 1 for p in points_and_labels):
            raise HTTPException(status_code=400, detail=f"saber {i}: at least one include point is required")
        color, intensity, _, voice = _parse_render_params(saber)
        parsed.append({
            "points": [[p[0], p[1]] for p in points_and_labels],
            "labels": [p[2] for p in points_and_labels],
            "color": color,
            "intensity": intensity,
            "voice": voice,
        })
    return parsed


@app.post("/api/jobs/{job_id}/points")
async def submit_points(job_id: str, body: dict):
    _validate_job_id(job_id)
    if not paths.get_checkpoint_path().exists():
        raise HTTPException(
            status_code=400,
            detail="SAM2 is not installed yet -- run `lightsaber-fx setup` first.",
        )
    job_dir = paths.get_jobs_dir() / job_id
    input_path = job_dir / "input.mp4"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    sabers = _parse_saber_specs(body)
    blade_extend = bool(body.get("blade_extend", True))

    output_path = job_dir / "final.mp4"
    device = select_device()

    def pipeline_fn(progress_cb):
        return run_pipeline_multi(
            input_video=str(input_path),
            sabers=sabers,
            output_path=str(output_path),
            job_dir=str(job_dir),
            checkpoint_path=str(paths.get_checkpoint_path()),
            device=device,
            blade_extend=blade_extend,
            progress_cb=progress_cb,
        )

    try:
        manager.start(job_id, pipeline_fn)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"status": "started"}
```

Note `_parse_render_params(saber)` is reused per-saber for its existing
color/intensity/voice validation (it ignores the `blade_extend` key inside
each saber dict since that's read separately, job-level, from `body`) --
this deliberately keeps exactly one place that knows what counts as a valid
`--intensity`/`--voice`, per the existing comment on `_parse_render_params`.

Add the import: `from ..pipeline.runner import rerender_pipeline_multi, run_pipeline_multi`
alongside the existing `rerender_pipeline, run_pipeline` import (Task 9 adds
`rerender_pipeline_multi`'s usage; import it here so both land together).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/web/test_server.py -k "multi_object_job or rejects_zero_sabers or more_than_four or saber_with_no_include" -v`
Expected: PASS.

- [ ] **Step 5: Run the full web test suite**

Run: `.venv/bin/python -m pytest tests/web/ -v`
Expected: **failures expected** in the pre-existing single-`points`-body-shape
tests (e.g. `test_points_then_events_then_result`,
`test_points_rejected_without_include_point`,
`test_points_rejected_with_out_of_range_intensity`,
`test_points_rejected_when_sam2_checkpoint_missing`,
`test_second_upload_returns_409_while_a_job_is_running`,
`test_second_points_submission_returns_409_while_job_is_running`,
`test_points_passes_prompt_frame_through_to_the_pipeline`,
`test_points_defaults_prompt_frame_to_zero`,
`test_points_rejects_a_negative_prompt_frame`) -- these all post the old flat
`{"points": [...], ...}` shape, which the new `_parse_saber_specs` rejects
with a 400 for having no `sabers` key. This is expected and handled in the
next step.

- [ ] **Step 6: Update the pre-existing single-shape tests to the new body**

Every test listed in Step 5 needs its `json={...}` body changed from the old
flat shape to `json={"sabers": [{...}]}`. For example,
`test_points_rejected_without_include_point`:

```python
def test_points_rejected_without_include_point(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 0]]}]},
    )

    assert resp.status_code == 400
```

Apply the same `{"points": [...], ...}` -> `{"sabers": [{"points": [...],
...}]}` wrapping to each test in the Step 5 list, keeping every other field
(`prompt_frame` moves inside the saber dict as before -- it's still a valid,
if currently-unused-by-`run_pipeline_multi`, per-saber field for future
detection work) and assertion unchanged. `test_points_passes_prompt_frame...`
and `test_points_defaults_prompt_frame_to_zero` specifically assert on
`captured["prompt_frame"]` from a stubbed `run_pipeline` -- since
`run_pipeline_multi` doesn't take a `prompt_frame` parameter at all (Global
Constraint: every saber prompts at frame 0), these two tests are replaced
rather than reshaped:

```python
def test_points_ignores_any_client_supplied_prompt_frame(client, tiny_video_bytes, monkeypatch):
    # Multi-saber tracking always prompts at frame 0 (see Global Constraints);
    # a client-supplied prompt_frame in a saber spec is accepted (forward
    # compatibility) but never reaches run_pipeline_multi, which has no such
    # parameter.
    captured = {}

    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"x")
        return output_path

    monkeypatch.setattr(server_module, "run_pipeline_multi", fake_run_pipeline_multi)
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 20, 1]], "prompt_frame": 17}]},
    )
    assert resp.status_code == 200
    server_module.manager.wait(timeout=10)

    assert "prompt_frame" not in captured
```

Delete `test_points_rejects_a_negative_prompt_frame` -- `prompt_frame`
validation was specifically about a value that reaches the pipeline; since
it no longer does, there's nothing left for that test to guard.

- [ ] **Step 7: Run the full web test suite and confirm all pass**

Run: `.venv/bin/python -m pytest tests/web/ -v`
Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/lightsaber_fx/web/server.py tests/web/test_server.py
git commit -m "Change /api/jobs/{id}/points to accept 1-4 sabers, each with its own color/intensity/voice"
```

---

### Task 9: Update `/api/jobs/{id}/rerender` for multiple sabers

**Files:**
- Modify: `src/lightsaber_fx/web/server.py`
- Test: `tests/web/test_server.py`

**Interfaces:**
- Consumes: `rerender_pipeline_multi` (Task 7), `job_meta.describe_job` (Task 5, via `require_rerenderable`).
- Produces: `POST /api/jobs/{job_id}/rerender` now expects body `{"sabers": [{"color": "red", "intensity": 0.35, "voice": "neutral"}, ...], "blade_extend": true}`, dispatching to `rerender_pipeline_multi` for every job going forward (all web-created jobs now record `object_ids`; the legacy single-object `rerender_pipeline` stays reachable only from the CLI).

- [ ] **Step 1: Write the failing tests**

Add to `tests/web/test_server.py`:

```python
def test_rerender_endpoint_accepts_multiple_sabers(client, tiny_video_bytes, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"first"))

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]
    points_resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [
            {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[20, 20, 1]], "color": "blue", "intensity": 0.35, "voice": "neutral"},
        ]},
    )
    assert points_resp.status_code == 200
    server_module.manager.wait(timeout=2)

    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": [0, 1]})())

    captured = {}

    def fake_rerender_pipeline_multi(*, output_path, progress_cb, **kwargs):
        captured.update(kwargs)
        with open(output_path, "wb") as f:
            f.write(b"rerendered")
        return output_path

    monkeypatch.setattr(server_module, "rerender_pipeline_multi", fake_rerender_pipeline_multi)

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [
            {"color": "green", "intensity": 0.6, "voice": "jedi"},
            {"color": "red", "intensity": 0.2, "voice": "sith"},
        ]},
    )

    assert resp.status_code == 200
    server_module.manager.wait(timeout=2)
    assert len(captured["sabers"]) == 2
    assert captured["sabers"][0]["voice"] == "jedi"


def test_rerender_endpoint_rejects_a_saber_count_mismatch(client, tiny_video_bytes, monkeypatch):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"x"))
    client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[1, 1, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )
    server_module.manager.wait(timeout=2)

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [
            {"color": "red", "intensity": 0.35, "voice": "neutral"},
            {"color": "blue", "intensity": 0.35, "voice": "neutral"},
        ]},
    )

    assert resp.status_code == 400
```

Add the small helper `_fake_run_pipeline_multi_writing` near the existing
`_fake_run_pipeline_writing` helper:

```python
def _fake_run_pipeline_multi_writing(content: bytes):
    def fake_run_pipeline_multi(*, output_path, progress_cb, **kwargs):
        for stage in ("extract", "track", "glow", "audio", "mux"):
            progress_cb(stage, 100, "done")
        with open(output_path, "wb") as f:
            f.write(content)
        return output_path
    return fake_run_pipeline_multi
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/web/test_server.py -k "rerender_endpoint_accepts_multiple or saber_count_mismatch" -v`
Expected: FAIL (endpoint still expects the old flat body / calls the old
single-object `rerender_pipeline`).

- [ ] **Step 3: Update the endpoint**

Replace `rerender_job` in `src/lightsaber_fx/web/server.py`:

```python
def _parse_saber_style_specs(body: dict):
    """Like `_parse_saber_specs`, for `/rerender` -- no points/labels here,
    tracking is never re-run, just color/intensity/voice per saber."""
    sabers = body.get("sabers", [])
    if not 1 <= len(sabers) <= 4:
        raise HTTPException(status_code=400, detail="sabers must have between 1 and 4 entries")
    parsed = []
    for saber in sabers:
        color, intensity, _, voice = _parse_render_params(saber)
        parsed.append({"color": color, "intensity": intensity, "voice": voice})
    return parsed


@app.post("/api/jobs/{job_id}/rerender")
async def rerender_job(job_id: str, body: dict):
    _validate_job_id(job_id)
    job_dir = paths.get_jobs_dir() / job_id
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found")

    sabers = _parse_saber_style_specs(body)
    blade_extend = bool(body.get("blade_extend", True))

    try:
        info = require_rerenderable(str(job_dir))
    except JobNotRerenderableError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if info.object_ids is None or len(sabers) != len(info.object_ids):
        expected = len(info.object_ids) if info.object_ids is not None else 1
        raise HTTPException(
            status_code=400,
            detail=f"This job has {expected} tracked object(s), but {len(sabers)} saber(s) were given",
        )

    output_path = job_dir / "final.mp4"

    def pipeline_fn(progress_cb):
        return rerender_pipeline_multi(
            job_dir=str(job_dir),
            output_path=str(output_path),
            sabers=sabers,
            blade_extend=blade_extend,
            progress_cb=progress_cb,
        )

    try:
        manager.start(job_id, pipeline_fn)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return {"status": "started"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/web/test_server.py -k "rerender_endpoint_accepts_multiple or saber_count_mismatch" -v`
Expected: PASS.

- [ ] **Step 5: Run the full web test suite**

Run: `.venv/bin/python -m pytest tests/web/ -v`
Expected: **failures expected** in 6 pre-existing `/rerender` tests using the
old flat body shape and the old `run_pipeline`/`rerender_pipeline` stub
targets. Replace each as follows:

```python
def test_rerender_endpoint_starts_job_and_produces_new_result(client, tiny_video_bytes, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"first render bytes"))

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    points_resp = client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )
    assert points_resp.status_code == 200
    server_module.manager.wait(timeout=2)

    first_result = client.get(f"/api/jobs/{job_id}/result")
    assert first_result.content == b"first render bytes"

    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": [0]})())

    def fake_rerender_pipeline_multi(*, output_path, progress_cb, **kwargs):
        for stage in ("extract", "glow", "audio", "mux"):
            progress_cb(stage, 100, "done")
        with open(output_path, "wb") as f:
            f.write(b"rerendered bytes")
        return output_path

    monkeypatch.setattr(server_module, "rerender_pipeline_multi", fake_rerender_pipeline_multi)

    rerender_resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"color": "blue", "intensity": 0.6, "voice": "neutral"}]},
    )
    assert rerender_resp.status_code == 200
    server_module.manager.wait(timeout=2)

    result_resp = client.get(f"/api/jobs/{job_id}/result")
    assert result_resp.status_code == 200
    assert result_resp.content == b"rerendered bytes"


def test_rerender_endpoint_404_for_unknown_job(client):
    resp = client.post("/api/jobs/doesnotexist/rerender", json={"sabers": []})
    assert resp.status_code == 404


def test_rerender_endpoint_400_when_job_has_no_masks_yet(client, tiny_video_bytes):
    # Upload only -- /points was never called, so there's no masks/,
    # motion/, or video_meta.txt for this job yet.
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"color": "blue", "intensity": 0.35, "voice": "neutral"}]},
    )

    assert resp.status_code == 400
    assert "masks" in resp.json()["detail"]


@pytest.mark.parametrize("intensity", [-1.0, 1.5, 200.0])
def test_rerender_endpoint_rejects_out_of_range_intensity(client, tiny_video_bytes, intensity):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"intensity": intensity}]},
    )

    assert resp.status_code == 400


def test_rerender_endpoint_rejects_invalid_voice(client, tiny_video_bytes):
    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]

    resp = client.post(
        f"/api/jobs/{job_id}/rerender",
        json={"sabers": [{"voice": "yoda"}]},
    )

    assert resp.status_code == 400


def test_rerender_endpoint_409_when_a_job_is_already_running(client, tiny_video_bytes, monkeypatch):
    monkeypatch.setattr(server_module, "run_pipeline_multi", _fake_run_pipeline_multi_writing(b"first"))

    upload_resp = client.post("/api/upload", files={"file": ("clip.mp4", tiny_video_bytes, "video/mp4")})
    job_id = upload_resp.json()["job_id"]
    client.post(
        f"/api/jobs/{job_id}/points",
        json={"sabers": [{"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"}]},
    )
    server_module.manager.wait(timeout=2)

    monkeypatch.setattr(server_module, "require_rerenderable",
                         lambda job_dir: type("Info", (), {"object_ids": [0]})())

    release = threading.Event()

    def slow_rerender_pipeline_multi(*, output_path, progress_cb, **kwargs):
        release.wait(timeout=2)
        with open(output_path, "wb") as f:
            f.write(b"slow")
        return output_path

    monkeypatch.setattr(server_module, "rerender_pipeline_multi", slow_rerender_pipeline_multi)

    first = client.post(f"/api/jobs/{job_id}/rerender", json={"sabers": [{"color": "blue", "intensity": 0.35, "voice": "neutral"}]})
    assert first.status_code == 200

    second = client.post(f"/api/jobs/{job_id}/rerender", json={"sabers": [{"color": "green", "intensity": 0.35, "voice": "neutral"}]})
    assert second.status_code == 409

    release.set()
    server_module.manager.wait(timeout=2)
```

`threading` is already imported at the top of `test_server.py` (used by
other tests in this same file) -- no new import needed.

- [ ] **Step 6: Run the full test suite and confirm all pass**

Run: `.venv/bin/python -m pytest tests/ -v`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/lightsaber_fx/web/server.py tests/web/test_server.py
git commit -m "Change /api/jobs/{id}/rerender to accept per-saber color/intensity/voice"
```

---

## After this plan

The API supports 1-4 tracked sabers end-to-end (upload -> track -> render ->
mix audio -> mux -> rerender), verified by automated tests, but there is no
browser UI for it yet (the existing picker still only sends one saber's
worth of points, so this plan's new multi-saber code path is reachable only
by calling the API directly) and no interactive correction when tracking
goes wrong. Two more plans follow:

- **Frontend multi-slot picker UI** -- lets a user actually add up to 4
  saber slots, pick points per slot, and choose each slot's color/intensity/
  voice, wired to the endpoints this plan built.
- **Interactive mid-render correction** -- the area-health heuristic,
  pause/resume mechanics, `/correction-frame` and `/correct` endpoints, and
  the correction UI screen, per the spec's "Tracking + mid-render
  correction" section. Depends on this plan's `track_objects`/masks-per-
  object layout, but changes `track_objects`'s single blocking call into a
  pausable generator loop -- a nontrivial follow-on change to the function
  this plan just wrote, not an additive one.
