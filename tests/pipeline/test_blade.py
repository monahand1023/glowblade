import numpy as np
import pytest

from lightsaber_fx.pipeline.blade import (
    BladeGeometry,
    angular_speed,
    classify_tip_by_taper,
    compute_motion,
    fit_blade,
    load_mask,
    load_mask_optional,
    load_motion,
    mask_frame_indices,
    save_mask,
    save_motion,
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
    # position alone, so compute_motion must fall back explicitly to
    # fit_blade's own per-frame taper guess -- exactly what fit_blade
    # reports for that mask on its own -- rather than guess randomly.
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
