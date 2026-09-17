import numpy as np
import pytest

from lightsaber_fx.pipeline.blade import (
    MIN_ELONGATION,
    BladeGeometry,
    _find_overlap_runs,
    _mask_iou,
    _smooth_run_field,
    _tip_confidence_weights,
    angular_speed,
    classify_tip_by_taper,
    compute_motion,
    elongation_stats,
    fit_blade,
    load_mask,
    load_mask_optional,
    load_motion,
    mask_frame_indices,
    save_mask,
    save_motion,
    suppress_overlap_bleed,
    tip_speed,
    wrap_axis_angle_delta,
)

# ---------------------------------------------------------------------------
# fit_blade
# ---------------------------------------------------------------------------

def test_fit_blade_returns_none_for_empty_mask():
    mask = np.zeros((48, 64), dtype=bool)
    assert fit_blade(mask) is None


def test_fit_blade_returns_geometry_with_expected_fields():
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:16, 5:55] = True
    geo = fit_blade(mask)
    assert isinstance(geo, BladeGeometry)
    for field in ("centroid", "axis", "tip", "hilt", "length", "width", "angle"):
        assert hasattr(geo, field)


def test_fit_blade_uniform_rectangle_axis_is_horizontal():
    # Uniform width rectangle: no taper, so the long axis must still come
    # out horizontal and the two endpoints must be the two ends of the
    # rectangle -- which one is "tip" is a genuine tie (see the dedicated
    # taper test below for that), so only check the axis and endpoint set.
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:16, 5:55] = True  # rows 10..15 (y), cols 5..54 (x)

    geo = fit_blade(mask)

    assert abs(geo.axis[1]) < 1e-6  # axis is (±1, 0): purely horizontal
    assert geo.length == pytest.approx(49.0, abs=1e-6)
    assert geo.width == pytest.approx(5.0, abs=1e-6)

    endpoints = {round(geo.tip[0], 3), round(geo.hilt[0], 3)}
    assert endpoints == {5.0, 54.0}
    assert geo.tip[1] == pytest.approx(12.5, abs=1e-6)
    assert geo.hilt[1] == pytest.approx(12.5, abs=1e-6)


def test_fit_blade_vertical_rectangle_axis_is_vertical():
    mask = np.zeros((80, 40), dtype=bool)
    mask[5:75, 15:21] = True  # tall and narrow: rows 5..74 (y), cols 15..20 (x)

    geo = fit_blade(mask)

    assert abs(geo.axis[0]) < 1e-6  # purely vertical axis
    assert geo.length == pytest.approx(69.0, abs=1e-6)
    endpoints = {round(geo.tip[1], 3), round(geo.hilt[1], 3)}
    assert endpoints == {5.0, 74.0}


def test_fit_blade_single_frame_taper_guess_picks_narrow_end_not_pipeline_truth():
    # fit_blade (via classify_tip_by_taper) is a *single-frame, shape-only*
    # guess: narrower end = tip. That's a fine, honest thing for it to
    # report in isolation -- a sword-like bulge (wide hilt, narrow far
    # end) really does taper that way -- but it is NOT the pipeline's
    # actual tip/hilt decision. For a bat-like object (thin handle, thick
    # barrel) this exact heuristic is backwards, which is precisely why
    # compute_motion (see test_compute_motion_bat_swing_puts_tip_at_barrel_
    # not_handle below) overrides it with a motion-based decision for the
    # whole clip and only falls back to this per-frame guess when motion
    # can't decide (see test_compute_motion_static_clip_falls_back_to_taper).
    mask = np.zeros((60, 100), dtype=bool)
    mask[27:33, 10:90] = True  # thin blade body, x 10..89, y 27..32
    mask[20:40, 10:20] = True  # wide hilt bulge, x 10..19, y 20..39

    geo = fit_blade(mask)

    assert geo.tip[0] > 70  # the heuristic calls the far, narrow end "tip"
    assert geo.hilt[0] < 25  # ...and the bulge "hilt"
    assert geo.width < 10  # median width reflects the thin body, not the bulge
    assert geo.width > 3


def test_fit_blade_single_pixel_mask_does_not_raise():
    mask = np.zeros((20, 20), dtype=bool)
    mask[10, 10] = True
    geo = fit_blade(mask)
    assert geo is not None
    assert geo.length == pytest.approx(0.0, abs=1e-9)
    assert not np.isnan(geo.angle)


def test_fit_blade_width_uses_median_not_max_extent():
    # Same bulge shape as above, but assert directly that the reported
    # width is nowhere near the bulge's ~19px extent, proving the median
    # (not a plain max-min over the whole mask) is what's reported.
    mask = np.zeros((60, 100), dtype=bool)
    mask[27:33, 10:90] = True
    mask[20:40, 10:20] = True

    geo = fit_blade(mask)

    assert geo.width < 10.0


def test_fit_blade_ignores_a_disconnected_speck_far_from_the_real_blade():
    # Reproduces a real failure seen on fencing footage: SAM2's per-frame
    # video-tracking mask for one object was mostly a clean blade, but on
    # several frames also contained a handful of stray pixels tens to
    # hundreds of pixels away (misclassified background, in that footage a
    # fencer's body cord lying on the floor). A single far-away pixel has
    # outsized leverage on PCA -- it doubled the fitted length and grossly
    # skewed the axis, even though it was under 0.1% of the mask's area.
    mask = np.zeros((60, 200), dtype=bool)
    mask[27:33, 10:90] = True  # the real blade: length 79, centered around y=30
    mask[5, 190] = True  # one disconnected pixel, 100+ px away

    geo = fit_blade(mask)

    assert geo.length == pytest.approx(79.0, abs=1.0)
    assert abs(geo.axis[1]) < 0.05  # still essentially horizontal, not pulled toward the speck


def test_fit_blade_keeps_the_largest_of_several_disconnected_components():
    # Same failure, worse: multiple stray blobs, one larger than a single
    # pixel. The real blade (a 6x80 = 480px rectangle) must still win over
    # a 20px speck.
    mask = np.zeros((60, 200), dtype=bool)
    mask[27:33, 10:90] = True  # real blade, 480px
    mask[45:50, 150:154] = True  # stray blob, 20px, disconnected

    geo = fit_blade(mask)

    assert geo.length == pytest.approx(79.0, abs=1.0)
    assert geo.centroid[0] < 100  # centered on the real blade, not pulled toward the blob


# ---------------------------------------------------------------------------
# classify_tip_by_taper -- the disambiguation heuristic, tested in isolation
# ---------------------------------------------------------------------------

def _tapered_points(tip_width, hilt_width, n_cols=30, span=30.0, samples_per_col=5):
    """Synthetic proj/perp arrays for a shape tapering from `tip_width` at
    the minimum-projection end to `hilt_width` at the maximum-projection end."""
    cols = np.linspace(-span / 2, span / 2, n_cols)
    proj, perp = [], []
    for c in cols:
        frac = (c - cols.min()) / (cols.max() - cols.min())
        width = tip_width + frac * (hilt_width - tip_width)
        half = width / 2
        for p in np.linspace(-half, half, samples_per_col):
            proj.append(c)
            perp.append(p)
    return np.array(proj), np.array(perp)


def test_classify_tip_by_taper_picks_narrow_min_end():
    proj, perp = _tapered_points(tip_width=2.0, hilt_width=10.0)
    assert classify_tip_by_taper(proj, perp) is True


def test_classify_tip_by_taper_picks_narrow_max_end():
    proj, perp = _tapered_points(tip_width=10.0, hilt_width=2.0)
    assert classify_tip_by_taper(proj, perp) is False


def test_classify_tip_by_taper_ties_resolve_to_min_end():
    proj, perp = _tapered_points(tip_width=6.0, hilt_width=6.0)
    assert classify_tip_by_taper(proj, perp) is True


# ---------------------------------------------------------------------------
# wrap_axis_angle_delta -- mod-pi wrapping for axis (line) angles
# ---------------------------------------------------------------------------

def test_wrap_axis_angle_delta_small_change_is_unaffected():
    assert wrap_axis_angle_delta(0.05) == pytest.approx(0.05)
    assert wrap_axis_angle_delta(-0.05) == pytest.approx(-0.05)


def test_wrap_axis_angle_delta_collapses_pi_flip_to_near_zero():
    # An axis line reported at +90 deg one frame and -90 deg the next is the
    # *same line* (mod pi) -- the flip must wrap to ~0, not ~pi.
    wrapped = wrap_axis_angle_delta(np.array([np.pi]))
    assert abs(wrapped[0]) < 1e-9


def test_wrap_axis_angle_delta_handles_array():
    deltas = np.array([0.1, np.pi, -np.pi, 2 * np.pi])
    wrapped = wrap_axis_angle_delta(deltas)
    assert np.all(np.abs(wrapped) <= np.pi / 2 + 1e-9)


# ---------------------------------------------------------------------------
# save_mask / load_mask -- per-frame mask I/O (compressed .npz, with a
# fallback that reads the legacy uncompressed .npy format track_object used
# to write). track.py, blade.py's own compute_motion, and glow.py's
# render_glow all go through these instead of open-coding np.save/np.load.
# ---------------------------------------------------------------------------

def test_save_mask_and_load_mask_roundtrip_preserves_dtype_and_values(tmp_path):
    masks_dir = tmp_path / "masks"
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:16, 5:55] = True

    save_mask(str(masks_dir), 3, mask)
    loaded = load_mask(str(masks_dir), 3)

    assert loaded.dtype == mask.dtype
    assert loaded.shape == mask.shape
    assert np.array_equal(loaded, mask)


def test_save_mask_and_load_mask_roundtrip_all_false(tmp_path):
    masks_dir = tmp_path / "masks"
    mask = np.zeros((48, 64), dtype=bool)

    save_mask(str(masks_dir), 0, mask)
    loaded = load_mask(str(masks_dir), 0)

    assert loaded.dtype == bool
    assert not loaded.any()
    assert np.array_equal(loaded, mask)


def test_save_mask_and_load_mask_roundtrip_all_true(tmp_path):
    masks_dir = tmp_path / "masks"
    mask = np.ones((48, 64), dtype=bool)

    save_mask(str(masks_dir), 0, mask)
    loaded = load_mask(str(masks_dir), 0)

    assert loaded.dtype == bool
    assert loaded.all()
    assert np.array_equal(loaded, mask)


def test_save_mask_creates_masks_dir_if_missing(tmp_path):
    masks_dir = tmp_path / "does_not_exist_yet"
    mask = np.zeros((10, 10), dtype=bool)
    save_mask(str(masks_dir), 0, mask)
    assert masks_dir.exists()
    assert (masks_dir / "00000.npz").exists()


def test_load_mask_reads_legacy_npy_format(tmp_path):
    # A job dir written before compression was added has raw .npy masks.
    # load_mask must read those transparently rather than failing.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    mask = np.zeros((20, 30), dtype=bool)
    mask[5:10, 5:10] = True
    np.save(masks_dir / "00007.npy", mask)  # legacy format, written directly

    loaded = load_mask(str(masks_dir), 7)

    assert loaded.dtype == bool
    assert np.array_equal(loaded, mask)


def test_load_mask_raises_when_frame_missing(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        load_mask(str(masks_dir), 0)


def test_load_mask_optional_returns_none_when_frame_missing(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    assert load_mask_optional(str(masks_dir), 0) is None


def test_load_mask_optional_returns_mask_when_present(tmp_path):
    masks_dir = tmp_path / "masks"
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    save_mask(str(masks_dir), 0, mask)
    loaded = load_mask_optional(str(masks_dir), 0)
    assert np.array_equal(loaded, mask)


def test_save_mask_never_writes_legacy_npy(tmp_path):
    masks_dir = tmp_path / "masks"
    mask = np.zeros((10, 10), dtype=bool)
    save_mask(str(masks_dir), 0, mask)
    assert not (masks_dir / "00000.npy").exists()
    assert (masks_dir / "00000.npz").exists()


def test_mask_frame_indices_sorted_numerically_and_mixed_formats(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    save_mask(str(masks_dir), 10, np.zeros((5, 5), dtype=bool))
    save_mask(str(masks_dir), 2, np.zeros((5, 5), dtype=bool))
    np.save(masks_dir / "00005.npy", np.zeros((5, 5), dtype=bool))  # legacy sibling

    assert mask_frame_indices(str(masks_dir)) == [2, 5, 10]


def test_save_mask_compressed_is_dramatically_smaller_than_raw_for_sparse_mask(tmp_path):
    # The whole point of this change: a sparse, blade-shaped mask (thin
    # diagonal band, small fraction of the frame) compresses far better
    # than plain np.save. Assert a conservative ratio (real measurements on
    # a representative blade mask were ~350x-470x -- see
    # docs/design-notes.md)
    # so this documents the intent without being brittle to numpy-version
    # compression differences.
    height, width = 1080, 1920
    mask = np.zeros((height, width), dtype=bool)
    # A thin diagonal-ish band, roughly a couple percent of the frame.
    for y in range(height):
        x0 = int(y * 0.3)
        mask[y, x0:x0 + 25] = True

    masks_dir = tmp_path / "masks"
    save_mask(str(masks_dir), 0, mask)
    compressed_size = (masks_dir / "00000.npz").stat().st_size

    raw_path = tmp_path / "raw.npy"
    np.save(raw_path, mask)
    raw_size = raw_path.stat().st_size

    assert compressed_size * 20 < raw_size, (
        f"expected >20x compression, got raw={raw_size} compressed={compressed_size}"
    )


# ---------------------------------------------------------------------------
# compute_motion -- the "motion" pipeline stage: reads masks, writes motion.npz
# ---------------------------------------------------------------------------

def test_compute_motion_writes_npz_for_every_mask_file(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    n_frames = 4
    for idx in range(n_frames):
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:16, 5:55] = True
        save_mask(str(masks_dir), idx, mask)
    motion_path = tmp_path / "motion.npz"
    progress_calls = []

    compute_motion(
        str(masks_dir), str(motion_path),
        progress_cb=lambda pct, msg: progress_calls.append(pct),
    )

    assert motion_path.exists()
    motion = load_motion(str(motion_path))
    assert motion["length"].shape == (n_frames,)
    assert not np.any(np.isnan(motion["length"]))
    assert progress_calls[-1] == 100


def test_compute_motion_empty_mask_yields_nan_row(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    save_mask(str(masks_dir), 0, np.zeros((48, 64), dtype=bool))  # object lost
    present = np.zeros((48, 64), dtype=bool)
    present[10:16, 5:55] = True
    save_mask(str(masks_dir), 1, present)
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))

    motion = load_motion(str(motion_path))
    assert np.isnan(motion["length"][0])
    assert np.all(np.isnan(motion["tip"][0]))
    assert not np.isnan(motion["length"][1])


def test_compute_motion_reads_legacy_npy_masks(tmp_path):
    # A job directory written before compression was added has masks in the
    # old, uncompressed `.npy` format. compute_motion must still read them
    # via the loader's fallback rather than silently finding nothing or
    # failing obscurely.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    n_frames = 3
    for idx in range(n_frames):
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:16, 5:55] = True
        np.save(masks_dir / f"{idx:05d}.npy", mask)  # legacy format, written directly
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))

    motion = load_motion(str(motion_path))
    assert motion["length"].shape == (n_frames,)
    assert not np.any(np.isnan(motion["length"]))


def test_compute_motion_no_masks_writes_empty_arrays(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))

    motion = load_motion(str(motion_path))
    assert motion["length"].shape == (0,)
    assert motion["tip"].shape == (0, 2)


# ---------------------------------------------------------------------------
# compute_motion -- tip/hilt orientation decided from motion over the whole
# clip, not per-frame shape (the actual regression: classify_tip_by_taper
# alone gets a bat backwards -- see the taper test above).
#
# All of these swing a mask about a fixed pivot with numpy only (no cv2):
# rotating a mask's foreground pixels about the pivot and rasterizing them
# onto a fresh canvas by nearest-pixel rounding. Rotation about a pivot
# preserves each point's *distance* from that pivot, so a physically
# correct, temporally-stable labelling should keep the hilt consistently
# near the pivot and the tip consistently far from it on every frame --
# any frame where that flips is either a wrong-end bug or a label flip.
# ---------------------------------------------------------------------------

_SWING_CANVAS = (900, 900)
_SWING_PIVOT = (450, 450)


def _rotate_points_mask(mask, pivot, angle_rad, canvas_shape):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(np.float64)
    pivot = np.asarray(pivot, dtype=np.float64)
    rel = pts - pivot
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rot = np.stack(
        [rel[:, 0] * c - rel[:, 1] * s, rel[:, 0] * s + rel[:, 1] * c], axis=1
    )
    new_pts = rot + pivot
    nx = np.round(new_pts[:, 0]).astype(int)
    ny = np.round(new_pts[:, 1]).astype(int)
    h, w = canvas_shape
    valid = (nx >= 0) & (nx < w) & (ny >= 0) & (ny < h)
    out = np.zeros(canvas_shape, dtype=bool)
    out[ny[valid], nx[valid]] = True
    return out


def _bat_base_mask():
    """Bat-shaped rest pose: thin handle starting at the pivot, thick
    barrel at the far end -- the exact regression shape (thin handle,
    thick barrel; taper alone calls the handle "tip")."""
    mask = np.zeros(_SWING_CANVAS, dtype=bool)
    px, py = _SWING_PIVOT
    mask[py - 5:py + 5, px:px + 240] = True  # handle: thin (10px)
    mask[py - 30:py + 30, px + 240:px + 300] = True  # barrel: thick (60px)
    return mask


def _sword_base_mask():
    """Sword-shaped rest pose: thick hilt at the pivot, tapering to a
    narrow point at the far end -- taper and motion agree here."""
    mask = np.zeros(_SWING_CANVAS, dtype=bool)
    px, py = _SWING_PIVOT
    mask[py - 30:py + 30, px:px + 60] = True  # hilt: thick (60px)
    mask[py - 5:py + 5, px + 60:px + 300] = True  # blade: thin (10px)
    return mask


def _dumbbell_base_mask():
    """Equal-width bulges at both ends of a thin shaft. classify_tip_by_
    taper sees a tie on every single frame, so its deterministic tie
    break ("minimum projection end") is at the mercy of PCA's arbitrary
    eigenvector sign -- which does NOT reliably track the same physical
    end from one frame to the next as the shape rotates, causing the
    per-frame-only heuristic to flip which end is "tip" almost every
    frame (verified empirically while building this fix)."""
    mask = np.zeros(_SWING_CANVAS, dtype=bool)
    px, py = _SWING_PIVOT
    mask[py - 5:py + 5, px:px + 300] = True  # shaft
    mask[py - 20:py + 20, px:px + 30] = True  # bulge near the pivot
    mask[py - 20:py + 20, px + 270:px + 300] = True  # bulge at the far end
    return mask


def _swing_sequence(base_mask, n_frames=10, max_angle_deg=80.0):
    """Rotate `base_mask` about `_SWING_PIVOT` in `n_frames` steps from 0
    to `max_angle_deg`, as if it were swung about that end."""
    angles = np.linspace(0.0, np.deg2rad(max_angle_deg), n_frames)
    return [_rotate_points_mask(base_mask, _SWING_PIVOT, a, _SWING_CANVAS) for a in angles]


def _write_masks(masks_dir, masks):
    for i, mask in enumerate(masks):
        save_mask(str(masks_dir), i, mask)


def _dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def test_compute_motion_bat_swing_puts_tip_at_barrel_not_handle(tmp_path):
    # The exact regression, reproduced end to end through compute_motion:
    # classify_tip_by_taper alone calls the thin handle "tip" (it's the
    # narrower end), which is backwards for a bat. Swung about the
    # handle, the handle barely moves while the barrel sweeps a wide arc
    # -- that physical signal must win. This test fails against the
    # taper-only code and passes once compute_motion decides from motion.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_masks(masks_dir, _swing_sequence(_bat_base_mask()))
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    assert len(motion["tip"]) == 10
    for i in range(len(motion["tip"])):
        hilt_dist = _dist(motion["hilt"][i], _SWING_PIVOT)
        tip_dist = _dist(motion["tip"][i], _SWING_PIVOT)
        assert hilt_dist < 20, f"frame {i}: hilt ({hilt_dist:.1f}px) should stay near the pivot/handle"
        assert tip_dist > 200, f"frame {i}: tip ({tip_dist:.1f}px) should be out at the barrel"


def test_compute_motion_sword_swing_keeps_tip_at_point(tmp_path):
    # Same motion-based mechanism, opposite shape: proves the fix isn't
    # simply inverting the taper heuristic. A sword's narrow point also
    # travels farthest when swung about the hilt, so it must still win.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_masks(masks_dir, _swing_sequence(_sword_base_mask()))
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    for i in range(len(motion["tip"])):
        hilt_dist = _dist(motion["hilt"][i], _SWING_PIVOT)
        tip_dist = _dist(motion["tip"][i], _SWING_PIVOT)
        assert hilt_dist < 20, f"frame {i}: hilt ({hilt_dist:.1f}px) should stay near the pivot/hilt"
        assert tip_dist > 200, f"frame {i}: tip ({tip_dist:.1f}px) should be out at the point"


def test_compute_motion_no_frame_to_frame_tip_hilt_flips(tmp_path):
    # A symmetric shape (equal-width ends) hands classify_tip_by_taper a
    # tie every frame; verified empirically, its deterministic tie-break
    # flips which physical end is "tip" on almost every frame of this
    # exact swing under a per-frame-only decision -- exactly what glow.py's
    # _stabilize_tip_hilt has been papering over downstream. Deciding once
    # from motion for the whole sequence must not flip.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_masks(masks_dir, _swing_sequence(_dumbbell_base_mask()))
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    axis = motion["axis"]
    for i in range(len(axis) - 1):
        assert np.dot(axis[i], axis[i + 1]) > 0, (
            f"axis direction flipped between frame {i} and {i + 1}"
        )
    for i in range(len(axis)):
        hilt_dist = _dist(motion["hilt"][i], _SWING_PIVOT)
        tip_dist = _dist(motion["tip"][i], _SWING_PIVOT)
        assert hilt_dist < 40, f"frame {i}: hilt ({hilt_dist:.1f}px) should stay near the pivot"
        assert tip_dist > 200, f"frame {i}: tip ({tip_dist:.1f}px) should stay out at the far end"


def test_compute_motion_static_clip_falls_back_to_taper(tmp_path):
    # No motion at all: the two ends are genuinely indistinguishable from
    # position alone, so the fallback's fixed-seed relabeling must land on
    # exactly what fit_blade's own per-frame taper guess reports for that
    # mask -- since an identical mask every frame never flips its own
    # taper call, "seed once, apply everywhere" and "trust each frame's
    # own guess" are the same thing here.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    static_mask = _bat_base_mask()
    _write_masks(masks_dir, [static_mask] * 5)
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    expected = fit_blade(static_mask)
    for i in range(5):
        assert motion["tip"][i] == pytest.approx(expected.tip, abs=1e-6)
        assert motion["hilt"][i] == pytest.approx(expected.hilt, abs=1e-6)
        assert motion["axis"][i] == pytest.approx(expected.axis, abs=1e-6)


def _translating_bar_mask(x0, canvas=(60, 300), length=150, y0=27, y1=33, wide_end=None, bump=6):
    """A uniform-width bar at `x0`, optionally with one end's rows widened
    by `bump` over the outer third of its length -- enough to flip
    `classify_tip_by_taper`'s tie-break for that one frame, while a plain
    rigid translation across frames (both ends moving by the same delta)
    keeps path_a/path_b equal, exactly the indecisive-ratio case."""
    mask = np.zeros(canvas, dtype=bool)
    mask[y0:y1, x0:x0 + length] = True
    if wide_end == "left":
        mask[y0 - bump:y1 + bump, x0:x0 + length // 3] = True
    elif wide_end == "right":
        mask[y0 - bump:y1 + bump, x0 + length - length // 3:x0 + length] = True
    return mask


def test_compute_motion_translating_object_does_not_flip_tip_hilt_mid_clip(tmp_path):
    # The real regression this guards, confirmed on real fencing footage:
    # a thrust translates the whole blade with the arm rather than
    # pivoting it about a planted hilt, so hilt and tip travel nearly
    # equal total distances (measured ratio 1.12, well under
    # _MIN_PATH_RATIO) -- `_decide_tip_track` correctly can't call it, but
    # the *previous* fallback ("leave each frame's own taper guess alone")
    # let one frame's shape-only taper flip go straight through even
    # though the mask barely changed shape, fully swapping tip and hilt
    # for that frame and every one after it. This bar translates rigidly
    # (ratio exactly 1.0) with one frame's taper deliberately flipped;
    # every frame's *labeled* tip/hilt must still land on the same
    # physical end regardless.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    masks = [
        _translating_bar_mask(50),
        _translating_bar_mask(55),
        _translating_bar_mask(60),
        _translating_bar_mask(65, wide_end="left"),  # taper flips here alone
        _translating_bar_mask(70),
        _translating_bar_mask(75),
    ]
    _write_masks(masks_dir, masks)
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    # the physical left end (tracked by its x-coordinate, which only ever
    # increases by 5px/frame under this translation) must be labeled the
    # same way -- tip or hilt -- on every single frame
    left_is_tip = motion["tip"][:, 0] < motion["hilt"][:, 0]
    assert np.all(left_is_tip) or np.all(~left_is_tip)
    # and the reported axis must never reverse direction frame to frame
    dots = np.sum(motion["axis"][:-1] * motion["axis"][1:], axis=1)
    assert np.all(dots > 0)


def test_compute_motion_bat_swing_with_nan_gap_still_orients_by_motion(tmp_path):
    # A dropped frame mid-swing must not break the global motion decision
    # for the frames around it, and must still leave a NaN row exactly at
    # the gap -- do not regress the existing NaN-gap handling.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    masks = _swing_sequence(_bat_base_mask())
    masks[4] = np.zeros(_SWING_CANVAS, dtype=bool)  # object lost this frame
    _write_masks(masks_dir, masks)
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    assert np.isnan(motion["length"][4])
    assert np.all(np.isnan(motion["tip"][4]))

    for i in range(len(motion["tip"])):
        if i == 4:
            continue
        hilt_dist = _dist(motion["hilt"][i], _SWING_PIVOT)
        tip_dist = _dist(motion["tip"][i], _SWING_PIVOT)
        assert hilt_dist < 20, f"frame {i}: hilt ({hilt_dist:.1f}px) should stay near the pivot/handle"
        assert tip_dist > 200, f"frame {i}: tip ({tip_dist:.1f}px) should be out at the barrel"


# ---------------------------------------------------------------------------
# compute_motion -- per-frame debug logging
# ---------------------------------------------------------------------------

def _bar_mask(x_end, canvas=(48, 400), x_start=50, y_start=20, y_end=26):
    mask = np.zeros(canvas, dtype=bool)
    mask[y_start:y_end, x_start:x_end] = True
    return mask


def test_compute_motion_debug_logs_per_frame_geometry(tmp_path, caplog):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    _write_masks(masks_dir, [_bar_mask(200)])
    motion_path = tmp_path / "motion.npz"

    with caplog.at_level("DEBUG", logger="lightsaber_fx.pipeline.blade"):
        compute_motion(str(masks_dir), str(motion_path))

    assert "length=" in caplog.text
    assert "centroid=" in caplog.text


# ---------------------------------------------------------------------------
# compute_motion -- isolated single-frame position-glitch suppression
#
# A third, distinct real-footage failure mode from the two above: an
# isolated SAM2 tracking glitch for one frame (the mask briefly jumps to
# an unrelated position, then the very next frame is back to normal),
# unrelated to any cross-object contact. Confirmed on real footage: one
# object's centroid jumped to the opposite edge of a 1280px frame for
# exactly one frame, sandwiched between two frames only 1px apart, with
# zero mask IoU against the other tracked object throughout. These tests
# exercise the guard that holds an interpolated position instead of
# trusting an isolated jump-and-back.
# ---------------------------------------------------------------------------

def test_compute_motion_interpolates_an_isolated_single_frame_position_glitch(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    masks = [
        _bar_mask(200),               # frame0: baseline, centroid x=124.5
        _bar_mask(200),               # frame1: baseline
        _bar_mask(390, x_start=350),  # frame2: glitch, centroid x=369.5
        _bar_mask(200),               # frame3: back to baseline
        _bar_mask(200),               # frame4: baseline
    ]
    _write_masks(masks_dir, masks)
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    # interpolated between frames 1 and 3, both baseline -- lands back on
    # the baseline position, nowhere near the raw glitch (x=369.5)
    assert motion["centroid"][2][0] == pytest.approx(motion["centroid"][1][0], abs=1.0)
    assert motion["centroid"][2][0] < 200


def test_compute_motion_does_not_suppress_sustained_fast_motion(tmp_path):
    # A real, consistent trend (each frame further than the last, never
    # snapping back) must pass through untouched, however large the
    # per-frame jump -- only a jump-and-return is a glitch's signature.
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    masks = [
        _bar_mask(200, x_start=50),    # centroid x=124.5
        _bar_mask(300, x_start=150),   # centroid x=224.5 (+100)
        _bar_mask(400, x_start=250),   # centroid x=324.5 (+100, keeps going)
    ]
    _write_masks(masks_dir, masks)
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    assert motion["centroid"][1][0] == pytest.approx(224.5, abs=1.0)


def test_compute_motion_position_glitch_skips_a_nan_gap_when_finding_last_good(tmp_path):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    masks = [
        _bar_mask(200),                          # frame0: baseline x=124.5
        np.zeros((48, 400), dtype=bool),         # frame1: object lost
        _bar_mask(390, x_start=350),              # frame2: glitch
        _bar_mask(200),                          # frame3: baseline
    ]
    _write_masks(masks_dir, masks)
    motion_path = tmp_path / "motion.npz"

    compute_motion(str(masks_dir), str(motion_path))
    motion = load_motion(str(motion_path))

    assert np.isnan(motion["centroid"][1][0])  # gap untouched
    assert motion["centroid"][2][0] == pytest.approx(motion["centroid"][0][0], abs=1.0)


def test_compute_motion_logs_a_warning_for_a_position_glitch(tmp_path, caplog):
    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    masks = [_bar_mask(200), _bar_mask(390, x_start=350), _bar_mask(200)]
    _write_masks(masks_dir, masks)
    motion_path = tmp_path / "motion.npz"

    with caplog.at_level("WARNING", logger="lightsaber_fx.pipeline.blade"):
        compute_motion(str(masks_dir), str(motion_path))

    assert "tracking glitch" in caplog.text
    assert "held 1/3" in caplog.text


# ---------------------------------------------------------------------------
# _mask_iou
# ---------------------------------------------------------------------------

def test_mask_iou_identical_masks_is_one():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    assert _mask_iou(mask, mask) == pytest.approx(1.0)


def test_mask_iou_disjoint_masks_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    a[0:3, 0:3] = True
    b = np.zeros((10, 10), dtype=bool)
    b[7:10, 7:10] = True
    assert _mask_iou(a, b) == 0.0


def test_mask_iou_partial_overlap_matches_expected_fraction():
    a = np.zeros((10, 10), dtype=bool)
    a[0:4, 0:4] = True  # 16px
    b = np.zeros((10, 10), dtype=bool)
    b[2:6, 2:6] = True  # 16px, overlapping a 2x2=4px corner
    assert _mask_iou(a, b) == pytest.approx(4 / 28)


def test_mask_iou_both_empty_is_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    assert _mask_iou(a, b) == 0.0


# ---------------------------------------------------------------------------
# suppress_overlap_bleed -- cross-object mask-overlap guard
#
# Confirmed directly on real footage: an earlier version of this guard
# compared each object's fitted length only against its own recent history
# (a growth-percentage cap), and could not tell genuine fast foreshortening
# (length legitimately swinging widely, no mask overlap) from actual
# cross-object mask bleed (mask overlap present) -- see the comment above
# CROSS_OBJECT_OVERLAP_IOU_THRESHOLD in blade.py. These tests exercise the
# corrected, overlap-gated version: two objects' raw masks are the trigger,
# not either object's own length history.
#
# A second real-footage finding after that fix landed: a run of overlapping
# frames can span several real seconds, and a real hilt travels 150-235px
# across one -- freezing at a single value for the whole run visibly
# detaches the glow from the hand holding it. These tests also cover the
# interpolation this drove: a run with a good frame on both sides gets
# interpolated between them; a run missing one side falls back to freezing
# at whichever side exists.
# ---------------------------------------------------------------------------

def _motion_geo(length, i=0, x_offset=0.0):
    return BladeGeometry(
        centroid=(float(i) + x_offset, 0.0), axis=(1.0, 0.0),
        tip=(float(i) + x_offset + length, 0.0), hilt=(float(i) + x_offset, 0.0),
        length=length, width=5.0, angle=0.0,
    )


def _write_lengths(path, lengths, x_offset=0.0):
    """A minimal valid motion.npz (a `None` entry becomes a NaN row, same
    as save_motion always has) -- only `length` matters to most of these
    tests, but the full field set is written so suppress_overlap_bleed's
    per-field copy has real arrays to work with, matching what
    compute_motion actually produces. `x_offset` shifts every frame's
    centroid/hilt/tip by a fixed amount -- used to give two objects a
    real raw-geometry separation for `_run_confidence_weights` (see
    `_motion_geo`); 0.0 (the default) matches every existing caller."""
    save_motion(str(path), [
        _motion_geo(length, i, x_offset=x_offset) if length is not None else None
        for i, length in enumerate(lengths)
    ])


def _write_overlap_masks(masks_dir, n_frames, overlapping_frames, canvas=(20, 20)):
    """Object A's mask sits at a fixed block; object B's mask sits on top
    of it (full overlap, IoU 1.0) on `overlapping_frames` and far away
    (IoU 0.0) everywhere else. The mask content is otherwise unrelated to
    any `length` a test writes via `_write_lengths` -- suppress_overlap_bleed
    reads masks only to compute IoU, never to refit geometry."""
    for i in range(n_frames):
        mask = np.zeros(canvas, dtype=bool)
        mask[0:4, 0:4] = True if i in overlapping_frames else False
        if i not in overlapping_frames:
            mask[15:17, 15:17] = True
        save_mask(str(masks_dir), i, mask)


def _write_fixed_mask(masks_dir, n_frames, canvas=(20, 20)):
    """Object A's own mask: a fixed block, always at the same place --
    what "overlapping" is measured against in `_write_overlap_masks`."""
    for i in range(n_frames):
        mask = np.zeros(canvas, dtype=bool)
        mask[0:4, 0:4] = True
        save_mask(str(masks_dir), i, mask)


def test_find_overlap_runs_reports_run_bounds_and_both_anchors(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a_path, motion_b_path = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a_path, [100, 500, 600, 150])
    _write_lengths(motion_b_path, [200, 500, 600, 250])
    motion_a, motion_b = load_motion(str(motion_a_path)), load_motion(str(motion_b_path))

    runs = _find_overlap_runs(str(masks_a), str(masks_b), motion_a, motion_b)

    assert len(runs) == 1
    run = runs[0]
    assert (run.run_start, run.run_end) == (1, 2)
    assert (run.before, run.after) == (0, 3)
    assert run.max_iou == pytest.approx(1.0)


def test_find_overlap_runs_reports_missing_anchors_as_none(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a_path, motion_b_path = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 3
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={0, 1})
    _write_lengths(motion_a_path, [500, 600, 120])
    _write_lengths(motion_b_path, [510, 610, 130])
    motion_a, motion_b = load_motion(str(motion_a_path)), load_motion(str(motion_b_path))

    runs = _find_overlap_runs(str(masks_a), str(masks_b), motion_a, motion_b)

    assert len(runs) == 1
    assert (runs[0].run_start, runs[0].run_end) == (0, 1)
    assert runs[0].before is None
    assert runs[0].after == 2


def test_find_overlap_runs_returns_empty_list_when_masks_never_overlap(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a_path, motion_b_path = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 3
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames=set())
    _write_lengths(motion_a_path, [100, 250, 90])
    _write_lengths(motion_b_path, [200, 220, 210])
    motion_a, motion_b = load_motion(str(motion_a_path)), load_motion(str(motion_b_path))

    assert _find_overlap_runs(str(masks_a), str(masks_b), motion_a, motion_b) == []


def test_suppress_overlap_bleed_interpolates_a_single_frame_between_its_neighbors(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 5
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={2})
    _write_lengths(motion_a, [100, 100, 500, 100, 100])  # frame 2: corrupted
    _write_lengths(motion_b, [200, 200, 999, 200, 200])

    n_held = suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert n_held == 1
    result_a = load_motion(str(motion_a))
    result_b = load_motion(str(motion_b))
    # both neighbors are 100/200, so the single interpolated frame lands
    # exactly there too -- this case can't tell interpolation and freezing
    # apart on its own; see the widening-gap test below for that.
    assert result_a["length"][2] == pytest.approx(100.0)
    assert result_b["length"][2] == pytest.approx(200.0)
    for i in (0, 1, 3, 4):
        assert result_a["length"][i] == pytest.approx([100, 100, 500, 100, 100][i])


def test_suppress_overlap_bleed_interpolates_toward_the_after_anchor_not_a_flat_hold(tmp_path):
    # The real-footage finding: a multi-frame run must move from the
    # "before" value toward the "after" value, not freeze at "before" for
    # the whole run (which is what visibly detached the glow from a moving
    # hand on real footage).
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a, [100, 500, 600, 150])
    _write_lengths(motion_b, [200, 500, 600, 250])

    suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    result_a = load_motion(str(motion_a))
    assert result_a["length"][0] == pytest.approx(100.0)  # before anchor, untouched
    assert result_a["length"][3] == pytest.approx(150.0)  # after anchor, untouched
    # strictly between the two anchors and moving monotonically toward
    # "after" -- not frozen flat at "before" (100) for both frames
    assert 100.0 < result_a["length"][1] < result_a["length"][2] < 150.0


def test_suppress_overlap_bleed_bends_toward_raw_data_when_raw_fits_are_well_separated(tmp_path):
    # The real-footage finding this drove: on a long run, a plain straight
    # line between the two anchors can badly miss real (non-monotonic)
    # motion. Here both anchors are identical (100), so a plain straight
    # line would hold flat at 100 for the whole run -- but object B's own
    # raw length spikes to 500 in the middle. Object B's whole geometry is
    # offset 300px from object A's (`x_offset`), well past the ~100px
    # reference length `_run_confidence_weights` compares against, so the
    # two objects' raw fits are confidently distinct throughout the run
    # (see `_run_confidence_weights` -- cross-object mask IoU alone was
    # confirmed on real footage *not* to be a reliable stand-in for this).
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 7
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2, 3, 4, 5})
    _write_lengths(motion_a, [100] * n)
    _write_lengths(motion_b, [100, 120, 150, 500, 150, 120, 100], x_offset=300.0)

    suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    result_b = load_motion(str(motion_b))
    assert result_b["length"][0] == pytest.approx(100.0)  # before anchor, untouched
    assert result_b["length"][6] == pytest.approx(100.0)  # after anchor, untouched
    # a plain straight line between two equal anchors would hold flat at
    # 100.0 for the whole run -- this must be measurably above that.
    assert result_b["length"][3] > 100.3


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
    # the same output as every pre-existing suppress_overlap_bleed test --
    # this is a pure addition, not a behavior change by default.
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


def _geo_with_tip(hilt, tip):
    axis_vec = np.array(tip) - np.array(hilt)
    length = float(np.linalg.norm(axis_vec))
    axis = tuple(axis_vec / length) if length > 0 else (1.0, 0.0)
    angle = float(np.arctan2(axis_vec[1], axis_vec[0])) if length > 0 else 0.0
    return BladeGeometry(centroid=tuple(hilt), axis=axis, tip=tuple(tip), hilt=tuple(hilt),
                          length=length, width=5.0, angle=angle)


def test_tip_confidence_weights_high_when_raw_tip_is_near_its_own_hilt_override(tmp_path):
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 7
    # Object A: hilt near (i, 0). Raw tip is 100px from its own hilt at
    # every frame *except* frame 3, where it has bled to (295, 0) --
    # almost exactly object B's hilt -- the real failure mode confirmed
    # on real footage.
    geoms_a = [
        _geo_with_tip((float(i), 0.0), (295.0, 0.0) if i == 3 else (float(i) + 100.0, 0.0))
        for i in range(n)
    ]
    # Object B: hilt near (300+i, 0), raw tip always safely further away.
    geoms_b = [_geo_with_tip((300.0 + i, 0.0), (300.0 + i + 100.0, 0.0)) for i in range(n)]
    save_motion(str(motion_a), geoms_a)
    save_motion(str(motion_b), geoms_b)

    hilt_overrides_a = {i: (float(i), 0.0) for i in range(1, 6)}
    hilt_overrides_b = {i: (300.0 + i, 0.0) for i in range(1, 6)}

    weights_a, weights_b = _tip_confidence_weights(
        load_motion(str(motion_a)), load_motion(str(motion_b)), 1, 5, list(range(n)),
        hilt_overrides_a, hilt_overrides_b, reference_length=100.0,
    )

    # frame 2 (offset 1 in run 1..5): raw tip 100px from own hilt, 200px
    # from the other's -- clearly closer to its own, full confidence.
    assert weights_a[1] == pytest.approx(1.0)
    # frame 3 (offset 2): raw tip 292px from its own hilt, 8px from the
    # other's -- the real bleed case, zero confidence.
    assert weights_a[2] == pytest.approx(0.0)
    # object B was never near object A at any frame -- full confidence throughout.
    assert (weights_b > 0.99).all()


def test_tip_confidence_weights_zero_without_both_hilt_overrides(tmp_path):
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_lengths(motion_a, [100, 200, 200, 100])
    _write_lengths(motion_b, [100, 200, 200, 100])

    # object B has no override at all -- no trustworthy reference for
    # "whose hand is this" anywhere in the run.
    weights_a, weights_b = _tip_confidence_weights(
        load_motion(str(motion_a)), load_motion(str(motion_b)), 1, 2, list(range(n)),
        {1: (0.0, 0.0), 2: (0.0, 0.0)}, {}, reference_length=100.0,
    )

    assert (weights_a == 0.0).all()
    assert (weights_b == 0.0).all()


def test_suppress_overlap_bleed_pulls_tip_toward_raw_data_only_where_it_is_not_contaminated(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 7
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2, 3, 4, 5})

    # Anchors (frames 0, 6) use a 50px length; every run frame's raw tip
    # is 100px from its own hilt except frame 3, which has bled to
    # object B's hilt -- same real-footage shape as the unit test above.
    geoms_a = []
    for i in range(n):
        if i in (0, 6):
            geoms_a.append(_geo_with_tip((float(i), 0.0), (float(i) + 50.0, 0.0)))
        elif i == 3:
            geoms_a.append(_geo_with_tip((float(i), 0.0), (295.0, 0.0)))
        else:
            geoms_a.append(_geo_with_tip((float(i), 0.0), (float(i) + 100.0, 0.0)))
    geoms_b = [_geo_with_tip((300.0 + i, 0.0), (300.0 + i + 100.0, 0.0)) for i in range(n)]
    save_motion(str(motion_a), geoms_a)
    save_motion(str(motion_b), geoms_b)

    hilt_overrides_a = {i: (float(i), 0.0) for i in range(1, 6)}
    hilt_overrides_b = {i: (300.0 + i, 0.0) for i in range(1, 6)}

    suppress_overlap_bleed(
        str(motion_a), str(masks_a), str(motion_b), str(masks_b),
        hilt_overrides_a=hilt_overrides_a, hilt_overrides_b=hilt_overrides_b,
    )

    result_a = load_motion(str(motion_a))
    # frame 3 (contaminated, zero tip confidence) must stay close to the
    # anchors' 50px -- nowhere near the contaminated raw tip's ~292px.
    assert result_a["length"][3] < 100.0
    # frame 2 (clean, full tip confidence) must be pulled measurably
    # above the anchors' flat 50px, toward its own 100px raw signal --
    # RUN_SMOOTHING_STRENGTH is calibrated for real-footage scale, so a
    # tiny toy run like this one only bends slightly even at full
    # confidence (see _smooth_run_field's own calibration tests for the
    # same effect); the point here is direction, not magnitude.
    assert result_a["length"][2] > 50.2


def test_suppress_overlap_bleed_skips_tip_confidence_weighting_without_both_hilt_overrides(tmp_path):
    # Regression guard: tip-specific weighting only applies when *both*
    # objects have hilt overrides (it needs both real hand positions to
    # judge "closer to which"). Without both, tip smoothing must fall
    # back to exactly the shared centroid-based `weights` -- i.e.
    # identical output to every pre-existing suppress_overlap_bleed test.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a, [100, 500, 600, 150])
    _write_lengths(motion_b, [200, 500, 600, 250])

    suppress_overlap_bleed(
        str(motion_a), str(masks_a), str(motion_b), str(masks_b),
        hilt_overrides_a={99: (9000.0, -9000.0)},  # frame 99 doesn't exist in this run -- a no-op override
    )
    result_a = load_motion(str(motion_a))

    assert result_a["length"][0] == pytest.approx(100.0)
    assert result_a["length"][3] == pytest.approx(150.0)
    assert 100.0 < result_a["length"][1] < result_a["length"][2] < 150.0


def test_smooth_run_field_reduces_to_linear_interpolation_when_weights_are_zero(tmp_path):
    # The key correctness property `_smooth_run_field`'s docstring
    # promises: a run with no usable raw signal at all (every confidence
    # weight 0, e.g. a fully-merged IoU-1.0 run) must degrade to exactly
    # the same result the old plain straight-line fallback gave, not to
    # something worse.
    raw = np.array([999.0, -50.0, 1e6, 3.0])  # deliberately irrelevant: weight is 0
    weights = np.zeros(4)
    all_times = [0, 10, 20, 30, 40, 50]  # 4 interior frames, non-uniform spacing
    before_value, after_value = 10.0, 210.0

    smoothed = _smooth_run_field(raw, weights, all_times, before_value, after_value, smoothing_strength=500.0)

    interior_times = np.array(all_times[1:-1], dtype=float)
    t0, t1 = all_times[0], all_times[-1]
    expected_linear = before_value + (after_value - before_value) * (interior_times - t0) / (t1 - t0)
    assert smoothed == pytest.approx(expected_linear, abs=1e-6)


def test_smooth_run_field_bends_toward_raw_data_when_confidently_weighted(tmp_path):
    # The mirror case: full confidence (weight 1) everywhere and a small
    # smoothing strength should pull the result close to the raw data's
    # own shape, not the straight line between the anchors.
    raw = np.array([10.0, 10.0, 100.0, 10.0, 10.0])
    weights = np.ones(5)
    all_times = [0, 1, 2, 3, 4, 5, 6]
    before_value, after_value = 10.0, 10.0  # a straight line would be flat at 10.0

    smoothed = _smooth_run_field(raw, weights, all_times, before_value, after_value, smoothing_strength=0.01)

    assert smoothed[2] > 50.0  # bends strongly toward the raw spike, not flat at 10.0


def test_suppress_overlap_bleed_leaves_wide_swings_untouched_when_masks_never_overlap(tmp_path):
    # The exact false-positive the growth-percentage version couldn't
    # avoid: large legitimate length swings with the masks never touching.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 3
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames=set())
    lengths = [100, 250, 90]
    _write_lengths(motion_a, lengths)
    _write_lengths(motion_b, [200, 220, 210])

    n_held = suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert n_held == 0
    result_a = load_motion(str(motion_a))
    assert result_a["length"] == pytest.approx(lengths)


def test_suppress_overlap_bleed_freezes_at_the_after_anchor_when_the_run_starts_at_frame_zero(tmp_path):
    # Overlap starting from frame 0: there is no pre-overlap frame to
    # interpolate from, so this falls back to freezing at the only anchor
    # that exists -- the first good frame once the overlap ends -- rather
    # than trusting a raw fit computed while the masks were still bled
    # together.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 3
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={0, 1})
    _write_lengths(motion_a, [500, 600, 120])
    _write_lengths(motion_b, [510, 610, 130])

    n_held = suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert n_held == 2
    result_a = load_motion(str(motion_a))
    assert result_a["length"] == pytest.approx([120.0, 120.0, 120.0])


def test_suppress_overlap_bleed_freezes_at_the_before_anchor_when_the_run_never_ends(tmp_path):
    # The mirror image: overlap that lasts through the end of the clip has
    # no "after" anchor to interpolate toward, so it falls back to
    # freezing at the last good frame before the run started.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 3
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a, [100, 500, 600])
    _write_lengths(motion_b, [200, 500, 600])

    n_held = suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert n_held == 2
    result_a = load_motion(str(motion_a))
    assert result_a["length"] == pytest.approx([100.0, 100.0, 100.0])


def test_suppress_overlap_bleed_does_not_use_a_nan_frame_as_an_anchor(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={2})
    _write_lengths(motion_a, [100, None, 999, 110])  # frame 1: object lost (NaN)
    _write_lengths(motion_b, [200, 210, 999, 220])

    suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    result_a = load_motion(str(motion_a))
    # interpolated between frame 0 (100) and frame 3 (110), skipping the
    # NaN gap at frame 1 -- not NaN itself, and not the raw corrupted 999
    assert not np.isnan(result_a["length"][2])
    assert 100.0 < result_a["length"][2] < 110.0


def test_suppress_overlap_bleed_skips_marginal_frames_when_picking_anchors(tmp_path):
    # Real footage shows IoU climbing gradually into a real overlap rather
    # than jumping straight from zero, so a frame just under the overlap
    # cap can still be a few frames into the same contamination -- not a
    # genuinely clean anchor. Frames 1 and 3 here sit under the 0.1
    # overlap cap (~0.053 IoU, a thin one-column sliver) but over the
    # stricter 0.02 anchor cap, so the true anchors must be frames 0 and 4
    # (IoU 0.0) instead -- not 1 and 3.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 5
    canvas = (20, 20)
    for i in range(n):
        mask = np.zeros(canvas, dtype=bool)
        mask[0:10, 0:10] = True
        save_mask(str(masks_a), i, mask)

    far = np.zeros(canvas, dtype=bool)
    far[10:20, 10:20] = True  # disjoint from A -- IoU 0.0
    marginal = np.zeros(canvas, dtype=bool)
    marginal[0:10, 9:19] = True  # one-column overlap with A -- IoU ~0.053
    overlapping = np.zeros(canvas, dtype=bool)
    overlapping[0:10, 0:10] = True  # identical to A -- IoU 1.0

    for i, mask in enumerate([far, marginal, overlapping, marginal, far]):
        save_mask(str(masks_b), i, mask)

    _write_lengths(motion_a, [100, 150, 999, 150, 200])
    _write_lengths(motion_b, [110, 160, 999, 160, 210])

    suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    result_a = load_motion(str(motion_a))
    assert result_a["length"][0] == pytest.approx(100.0)
    assert result_a["length"][4] == pytest.approx(200.0)
    assert 100.0 < result_a["length"][2] < 200.0
    # marginal frames are below the overlap cap, so left at their own raw
    # values -- just not trusted as anchors for frame 2's interpolation
    assert result_a["length"][1] == pytest.approx(150.0)
    assert result_a["length"][3] == pytest.approx(150.0)


def test_suppress_overlap_bleed_skips_a_run_covered_by_exclude_frame_ranges(tmp_path):
    # reacquire.retrack_overlap_runs already validated an independent
    # re-track for this exact run -- genuine, correctly-tracked contact
    # still reads as high mask IoU (that's what real contact looks like),
    # so re-detecting it here must not "fix" already-correct masks by
    # overwriting them with a worse interpolated approximation.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    lengths_a = [100, 500, 600, 150]
    _write_lengths(motion_a, lengths_a)
    _write_lengths(motion_b, [200, 500, 600, 250])

    n_held = suppress_overlap_bleed(
        str(motion_a), str(masks_a), str(motion_b), str(masks_b),
        exclude_frame_ranges=[(1, 2)],
    )

    assert n_held == 0
    result_a = load_motion(str(motion_a))
    assert result_a["length"] == pytest.approx(lengths_a)  # untouched, including frames 1-2


def test_suppress_overlap_bleed_still_corrects_a_run_outside_exclude_frame_ranges(tmp_path):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a, [100, 500, 600, 150])
    _write_lengths(motion_b, [200, 500, 600, 250])

    n_held = suppress_overlap_bleed(
        str(motion_a), str(masks_a), str(motion_b), str(masks_b),
        exclude_frame_ranges=[(10, 20)],  # doesn't overlap the actual run at all
    )

    assert n_held == 2
    result_a = load_motion(str(motion_a))
    assert 100.0 < result_a["length"][1] < result_a["length"][2] < 150.0


def test_suppress_overlap_bleed_leaves_raw_values_when_no_anchor_exists_anywhere(tmp_path):
    # The masks overlap for the entire clip -- no good frame exists on
    # either side to interpolate from or freeze at, so there is nothing
    # better than the raw (possibly wrong) fit to keep.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 3
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={0, 1, 2})
    lengths = [500, 600, 700]
    _write_lengths(motion_a, lengths)
    _write_lengths(motion_b, [510, 610, 710])

    n_held = suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert n_held == 0
    result_a = load_motion(str(motion_a))
    assert result_a["length"] == pytest.approx(lengths)


def test_suppress_overlap_bleed_logs_a_warning_for_a_held_run(tmp_path, caplog):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a, [100, 500, 600, 150])
    _write_lengths(motion_b, [200, 500, 600, 250])

    with caplog.at_level("WARNING", logger="lightsaber_fx.pipeline.blade"):
        suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert "overlapped" in caplog.text
    assert "frames 1-2" in caplog.text


def test_suppress_overlap_bleed_logs_an_extra_warning_for_a_long_interpolated_span(tmp_path, caplog):
    # Confirmed on real footage: a straight-line interpolation across a
    # long run (165 frames, 6.6s) can badly miss the real motion -- this
    # needs a distinct, loud warning, not just the routine per-run one
    # every short, well-approximated run also gets.
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 100
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames=set(range(2, 97)))  # 95-frame run
    _write_lengths(motion_a, [100] * n)
    _write_lengths(motion_b, [200] * n)

    with caplog.at_level("WARNING", logger="lightsaber_fx.pipeline.blade"):
        suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert "long enough that a straight line" in caplog.text
    assert "95 frames long" in caplog.text


def test_suppress_overlap_bleed_does_not_log_the_long_span_warning_for_a_short_run(tmp_path, caplog):
    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    n = 4
    _write_fixed_mask(masks_a, n)
    _write_overlap_masks(masks_b, n, overlapping_frames={1, 2})
    _write_lengths(motion_a, [100, 500, 600, 150])
    _write_lengths(motion_b, [200, 500, 600, 250])

    with caplog.at_level("WARNING", logger="lightsaber_fx.pipeline.blade"):
        suppress_overlap_bleed(str(motion_a), str(masks_a), str(motion_b), str(masks_b))

    assert "long enough that a straight line" not in caplog.text


# ---------------------------------------------------------------------------
# save_motion / load_motion
# ---------------------------------------------------------------------------

def _geo(cx, cy, ax, ay, tx, ty, hx, hy, length, width, angle):
    return BladeGeometry(
        centroid=(cx, cy), axis=(ax, ay), tip=(tx, ty), hilt=(hx, hy),
        length=length, width=width, angle=angle,
    )


def test_save_motion_writes_npz_file(tmp_path):
    path = tmp_path / "motion.npz"
    geometries = [_geo(0, 0, 1, 0, 1, 0, -1, 0, 2.0, 1.0, 0.0)]
    save_motion(str(path), geometries)
    assert path.exists()
    with open(path, "rb") as f:
        magic = f.read(2)
    assert magic == b"PK"  # npz is a zip archive


def test_save_motion_and_load_motion_roundtrip_with_none_gaps(tmp_path):
    geometries = [
        _geo(1, 2, 1, 0, 3, 2, -1, 2, 4.0, 1.5, 0.1),
        None,
        _geo(5, 6, 0, 1, 5, 10, 5, 2, 8.0, 2.0, 1.5708),
    ]
    path = tmp_path / "motion.npz"
    save_motion(str(path), geometries)
    motion = load_motion(str(path))

    for key in ("centroid", "tip", "hilt", "axis", "length", "width", "angle"):
        assert key in motion

    assert motion["centroid"].shape == (3, 2)
    assert motion["length"].shape == (3,)

    assert np.all(np.isnan(motion["centroid"][1]))
    assert np.isnan(motion["length"][1])
    assert np.isnan(motion["angle"][1])

    assert motion["centroid"][0] == pytest.approx([1, 2])
    assert motion["tip"][2] == pytest.approx([5, 10])
    assert motion["length"][2] == pytest.approx(8.0)
    assert motion["angle"][0] == pytest.approx(0.1)


def test_save_motion_all_none_produces_all_nan_rows(tmp_path):
    path = tmp_path / "motion.npz"
    save_motion(str(path), [None, None, None])
    motion = load_motion(str(path))
    assert motion["tip"].shape == (3, 2)
    assert np.all(np.isnan(motion["tip"]))
    assert np.all(np.isnan(motion["angle"]))


# ---------------------------------------------------------------------------
# tip_speed
# ---------------------------------------------------------------------------

def test_tip_speed_zero_for_static_tip():
    motion = {"tip": np.tile(np.array([10.0, 20.0]), (6, 1))}
    speed = tip_speed(motion, fps=30.0)
    assert speed.shape == (6,)
    assert np.allclose(speed, 0.0)


def test_tip_speed_reflects_known_displacement():
    fps = 10.0
    tip = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    speed = tip_speed({"tip": tip}, fps=fps)
    assert speed[0] == 0.0
    assert speed[1] == pytest.approx(50.0)  # 5px * 10fps
    assert speed[2] == pytest.approx(50.0)


def test_tip_speed_nan_gap_does_not_poison_whole_array():
    fps = 10.0
    tip = np.array([
        [0.0, 0.0],
        [5.0, 0.0],
        [np.nan, np.nan],
        [15.0, 0.0],
        [20.0, 0.0],
    ])
    speed = tip_speed({"tip": tip}, fps=fps)

    assert not np.any(np.isnan(speed))
    assert speed[1] == pytest.approx(50.0)
    # Entries touching the NaN frame are zeroed, not propagated as NaN...
    assert speed[2] == 0.0
    assert speed[3] == 0.0
    # ...and valid motion after the gap is still measured correctly.
    assert speed[4] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# angular_speed
# ---------------------------------------------------------------------------

def test_angular_speed_zero_for_static_angle():
    motion = {"angle": np.full(5, 0.3)}
    speed = angular_speed(motion, fps=30.0)
    assert speed.shape == (5,)
    assert np.allclose(speed, 0.0)


def test_angular_speed_reflects_true_rotation():
    fps = 10.0
    angle = np.array([0.0, 0.1, 0.2])
    speed = angular_speed({"angle": angle}, fps=fps)
    assert speed[0] == 0.0
    assert speed[1] == pytest.approx(1.0)  # 0.1 rad * 10fps
    assert speed[2] == pytest.approx(1.0)


def test_angular_speed_does_not_spike_on_pivot_through_vertical():
    # angle flips from just under +pi/2 to just under -pi/2 between frames --
    # the *line* barely moved (this is the mod-pi hilt/tip relabeling trap),
    # so angular speed must stay small, not register a ~pi-sized spike.
    fps = 30.0
    angle = np.array([np.pi / 2 - 0.01, -(np.pi / 2 - 0.01)])
    speed = angular_speed({"angle": angle}, fps=fps)
    assert speed[1] < 1.0  # not anywhere near (pi - 0.02) * fps ~= 94 rad/s


def test_angular_speed_nan_gap_does_not_poison_whole_array():
    fps = 10.0
    angle = np.array([0.0, 0.1, np.nan, 0.4, 0.5])
    speed = angular_speed({"angle": angle}, fps=fps)

    assert not np.any(np.isnan(speed))
    assert speed[1] == pytest.approx(1.0)
    assert speed[2] == 0.0
    assert speed[3] == 0.0
    assert speed[4] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# elongation_stats
# ---------------------------------------------------------------------------


def test_elongation_stats_all_elongated_frames_report_zero_low_fraction():
    motion = {"length": np.array([200.0, 220.0, 180.0]), "width": np.array([20.0, 20.0, 20.0])}
    mean_elongation, low_frac = elongation_stats(motion)
    assert mean_elongation == pytest.approx((10.0 + 11.0 + 9.0) / 3)
    assert low_frac == 0.0


def test_elongation_stats_flags_a_majority_of_blob_shaped_frames():
    # length=40, width=35 -> elongation ~1.14, well under MIN_ELONGATION (6).
    motion = {"length": np.full(10, 40.0), "width": np.full(10, 35.0)}
    mean_elongation, low_frac = elongation_stats(motion)
    assert mean_elongation < MIN_ELONGATION
    assert low_frac == 1.0


def test_elongation_stats_excludes_nan_and_zero_width_frames():
    # frame 0: never tracked (NaN). frame 1: a real degenerate fit_blade can
    # produce (zero width) -- elongation is undefined for it, not evidence of
    # anything, so it must not raise (divide-by-zero) or count as low.
    motion = {
        "length": np.array([np.nan, 100.0, 200.0, 220.0]),
        "width": np.array([np.nan, 0.0, 20.0, 20.0]),
    }
    mean_elongation, low_frac = elongation_stats(motion)
    assert mean_elongation == pytest.approx((10.0 + 11.0) / 2)
    assert low_frac == 0.0


def test_elongation_stats_returns_none_when_no_frame_has_a_usable_width():
    motion = {"length": np.array([np.nan, 100.0]), "width": np.array([np.nan, 0.0])}
    assert elongation_stats(motion) == (None, None)
