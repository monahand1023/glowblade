import numpy as np
import pytest

from lightsaber_fx.pipeline.blade import (
    BladeGeometry,
    angular_speed,
    classify_tip_by_taper,
    fit_blade,
    load_motion,
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


def test_fit_blade_disambiguates_tip_from_tapered_hilt_bulge():
    # A bat/sword shape: a long thin blade with a wide "knob" bulge at one
    # end (the hilt). The tip must be identified as the far, narrow end.
    mask = np.zeros((60, 100), dtype=bool)
    mask[27:33, 10:90] = True  # thin blade body, x 10..89, y 27..32
    mask[20:40, 10:20] = True  # wide hilt bulge, x 10..19, y 20..39

    geo = fit_blade(mask)

    assert geo.tip[0] > 70  # tip is near the far, narrow end
    assert geo.hilt[0] < 25  # hilt is near the bulge
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
