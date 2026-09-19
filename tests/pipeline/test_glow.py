import os

import cv2
import numpy as np
import pytest

from lightsaber_fx.pipeline import blade
from lightsaber_fx.pipeline.glow import (
    _capsule_mask,
    ignition_fraction,
    knoll_darken,
    parse_color,
    render_glow,
)

# ---------------------------------------------------------------------------
# parse_color / NAMED_COLORS -- unchanged public API, still depended on by
# the CLI and web layers.
# ---------------------------------------------------------------------------

def test_parse_color_named_colors():
    assert parse_color("red") == (40, 40, 255)
    assert parse_color("Blue") == (255, 90, 60)
    assert parse_color("GREEN") == (70, 220, 80)


def test_parse_color_hex():
    assert parse_color("#0000FF") == (255, 0, 0)  # pure blue, BGR order


def test_parse_color_rejects_unknown():
    with pytest.raises(ValueError):
        parse_color("not-a-color")


# ---------------------------------------------------------------------------
# B1.2 -- capsule mask with optional curved bend point
# ---------------------------------------------------------------------------

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
    # `bend` is a point the blade PASSES THROUGH, not a raw Bezier control
    # point -- _curved_capsule_mask solves for the control point that makes
    # B(0.5) land exactly on it. This test originally used bend=(110, -20),
    # which under the old (uncompensated) renderer drew a curve peaking at
    # y=40 -- half the requested offset -- and therefore had to sample a
    # window the curve reached rather than the bend point itself. Now that
    # the curve genuinely reaches `bend`, (110, -20) would put the curve's
    # peak 20px off the top of a 120-row canvas (verified by running:
    # nothing is lit anywhere in the old window). Moved the bend on-canvas
    # so the assertion can be the stronger, more direct one: the pixel AT
    # the bend point is lit. Verified this exact assertion fails under the
    # old uncompensated renderer (which peaks at y=40 here, 20px away).
    bend = (110.0, 20.0)  # well above the straight hilt-tip line (y=60)
    straight = _capsule_mask(shape, hilt, tip, 8.0, 0.10, 0.12, 0.35, bend=None)
    curved = _capsule_mask(shape, hilt, tip, 8.0, 0.10, 0.12, 0.35, bend=bend)
    # the curved capsule must light up the bend point itself, and pixels
    # around it that the straight one (a horizontal bar at y=60) never touches
    assert curved[round(bend[1]), round(bend[0])]
    assert curved[15:30, 100:120].any()
    assert not straight[15:30, 100:120].any()


def test_capsule_mask_bend_nan_falls_back_to_straight():
    shape = (90, 220)
    hilt, tip = (40.0, 45.0), (130.0, 45.0)
    args = (shape, hilt, tip, 8.0, 0.10, 0.12, 0.35)
    straight = _capsule_mask(*args, bend=None)
    with_nan = _capsule_mask(*args, bend=(float("nan"), float("nan")))
    assert np.array_equal(straight, with_nan)


def _bowed_bar_mask(peak_offset, canvas=(100, 400), x_start=50, x_end=350, y_center=50, thickness=6):
    """A bowed blade mask, same shape as test_blade.py's helper of the
    same name (a horizontal bar sagging by `peak_offset` px at its
    midpoint, tapering to 0 at both ends). Duplicated here rather than
    cross-imported between test modules: this test needs the *whole*
    chain (mask -> measured bow -> rendered curve) in one place to be
    readable as the end-to-end magnitude guard it is."""
    mask = np.zeros(canvas, dtype=bool)
    xs = np.arange(x_start, x_end)
    mid = (x_start + x_end) / 2.0
    half_span = (x_end - x_start) / 2.0
    sag = peak_offset * (1.0 - ((xs - mid) / half_span) ** 2)
    for x, dy in zip(xs, sag, strict=True):
        y0 = round(y_center + dy - thickness / 2)
        mask[max(0, y0):min(canvas[0], y0 + thickness), x] = True
    return mask


def test_rendered_curve_magnitude_matches_the_masks_own_measured_bow():
    """The design spec's acceptance criterion, as a number: the RENDERED
    curve's own perpendicular deviation from the straight hilt-tip line
    must match the bow `_bend_offset_from_mask` measured off the raw mask
    -- not merely "bend is non-NaN" or "some off-axis pixel is lit",
    which is all the other tests here check.

    This is the regression guard for the halving bug this test was
    written to close: `_curved_capsule_mask` used to feed `bend` straight
    in as the quadratic Bezier's control point, and a quadratic Bezier
    only reaches HALF its control point's own offset at t=0.5, so the
    rendered curve showed half the measured bow. Verified by computing it
    both ways: with the control-point solve, the rendered centerline sits
    24.0px off the straight line against a measured 24.0px; feeding
    `bend` in directly (the old behavior) gives 12.0px -- exactly half,
    and comfortably outside this test's 1.5px tolerance. On the real job
    this feature exists for, the same halving (compounded with the old
    BEND_RAMP_FRAMES=2 halving every real frame a second time) left
    frame 292's rendered curve peaking at 4.63px against an 18.51px bend
    point on a 227.8px blade; it now peaks at 18.51px.
    """
    canvas = (100, 400)
    hilt, tip = (50.0, 50.0), (350.0, 50.0)
    mask = _bowed_bar_mask(peak_offset=24.5, canvas=canvas, y_center=50)

    # Exactly what suppress_overlap_bleed does: measure the raw mask
    # against the straight hilt-tip line, then place `bend` that far off
    # the line's midpoint along its perpendicular.
    offset = blade._bend_offset_from_mask(mask, hilt, tip)
    assert abs(offset) > blade.BEND_SIGNIFICANCE_PX  # a real, significant bow
    seg = np.asarray(tip) - np.asarray(hilt)
    axis = seg / np.linalg.norm(seg)
    perp_dir = np.array([-axis[1], axis[0]])
    bend = (np.asarray(hilt) + np.asarray(tip)) / 2.0 + offset * perp_dir

    curved = _capsule_mask(canvas, hilt, tip, 10.0, 0.10, 0.12, 0.35, bend=bend)

    # The rendered capsule is a band of `width` around its centerline, so
    # read the centerline back as the midpoint of the lit rows in the
    # bend point's own column -- then compare that to the measured bow.
    col = curved[:, round(bend[0])]
    lit_rows = np.nonzero(col)[0]
    assert len(lit_rows) > 0, "nothing rendered at the bend point's column"
    rendered_centerline_y = (lit_rows.min() + lit_rows.max()) / 2.0
    rendered_deviation = rendered_centerline_y - hilt[1]  # straight line is y=50

    assert rendered_deviation == pytest.approx(offset, abs=1.5)
    # ...and the pixel at the bend point is genuinely inside the blade.
    assert curved[round(bend[1]), round(bend[0])]


def test_curved_capsule_hilt_wedge_has_no_seam_against_the_curve_body():
    """Regression guard for a cosmetic defect found on real footage: the
    hilt-side wedge's wide edge (at `body_start`) used the perpendicular
    of the *straight* hilt-tip axis, while the curve body's own first
    segment starts at that same point using the perpendicular of its
    local *tangent* -- which diverges from the straight axis as soon as
    `bend` pulls the centerline off the line. The two edges met at an
    angle instead of lining up, cutting a small notch into the blade's
    edge right at the wedge/body join. Confirmed visually (a zoomed PNG
    crop) and reproduced numerically here: this exact bend pulls the
    tangent at `body_start` far enough off-axis that pixel (98, 104) is
    inside the intended blade shape but was left unlit by the old
    straight-axis wedge.
    """
    shape = (200, 400)
    hilt, tip = (60.0, 100.0), (340.0, 100.0)
    bend = (200.0, 40.0)  # strong enough bow to visibly rotate the tangent

    curved = _capsule_mask(shape, hilt, tip, 16.0, 0.10, 0.12, 0.35, bend=bend)

    assert curved[104, 98], "notch: a pixel inside the intended blade shape is unlit at the wedge/body seam"


# ---------------------------------------------------------------------------
# Test helper: a synthetic clip with an elongated, blade-shaped mask (the
# shared synthetic_track_fixture's mask is a tiny near-square blob, fine for
# generic smoke tests but too small to give the capsule/core/colour-band/
# extension tests a meaningful margin). Reuses the real blade.compute_motion
# pipeline stage rather than faking motion.npz by hand.
# ---------------------------------------------------------------------------

def _build_blade_clip(tmp_path, name, n_frames, width=220, height=90, fps=24.0,
                       plate_value=210, blade_height=10, blade_x0=40, blade_len=90, dx=4,
                       drop_mask_frame=None):
    base = tmp_path / name
    frames_dir = base / "frames"
    masks_dir = base / "masks"
    frames_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    y0 = height // 2 - blade_height // 2
    y1 = y0 + blade_height

    for i in range(n_frames):
        frame = np.full((height, width, 3), plate_value, dtype=np.uint8)  # bright, sky-like plate
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), frame)

        if drop_mask_frame is not None and i == drop_mask_frame:
            mask = np.zeros((height, width), dtype=bool)  # "object lost" this frame
        else:
            x0 = blade_x0 + i * dx
            x1 = x0 + blade_len
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
        blade.save_mask(str(masks_dir), i, mask)

    video_meta_path = base / "video_meta.txt"
    video_meta_path.write_text(f"{fps}\n{n_frames}\n")

    motion_path = base / "motion.npz"
    blade.compute_motion(str(masks_dir), str(motion_path))

    return {
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "video_meta_path": str(video_meta_path),
        "motion_path": str(motion_path),
        "output_frames_dir": str(base / "glow_frames"),
        "n_frames": n_frames,
        "width": width,
        "height": height,
        "plate_value": plate_value,
        "y0": y0,
        "y1": y1,
    }


def _load_png(output_frames_dir, idx):
    path = os.path.join(output_frames_dir, f"{idx:05d}.png")
    img = cv2.imread(path)
    assert img is not None, f"missing/unreadable PNG: {path}"
    return img


# Golden checksums pinning render_glow's output on the characterization clip
# (6 frames, extending/motion/core/colour/glow/trail all exercised). Updated
# after Task 1 fix round to include ignition (on by default). Updated again
# after lowering the motion_blur_gain/motion_blur_max_len/trail_decay
# defaults (0.35->0.08, 24->6, 0.7->0.15) to cut the excessive blur/ghosting
# reported on real footage -- a deliberate, reviewed default change, not a
# refactor side effect.
EXPECTED_CHARACTERIZATION_CHECKSUMS = [
    12474070,  # frame 0
    12485576,  # frame 1
    12498112,  # frame 2
    12501893,  # frame 3
    12497118,  # frame 4
    12486148,  # frame 5
]


# ---------------------------------------------------------------------------
# Basic contract: PNG sequence, dimensions, progress, finite/in-range pixels
# ---------------------------------------------------------------------------

def test_render_glow_writes_png_sequence(tmp_path, synthetic_track_fixture):
    output_frames_dir = tmp_path / "glow_frames"
    motion_path = tmp_path / "motion.npz"
    blade.compute_motion(synthetic_track_fixture["masks_dir"], str(motion_path))

    progress_calls = []
    render_glow(
        synthetic_track_fixture["frames_dir"],
        synthetic_track_fixture["masks_dir"],
        synthetic_track_fixture["video_meta_path"],
        str(output_frames_dir),
        str(motion_path),
        progress_cb=lambda pct, msg: progress_calls.append(pct),
    )

    n = synthetic_track_fixture["n_frames"]
    pngs = sorted(f for f in os.listdir(output_frames_dir) if f.endswith(".png"))
    assert len(pngs) == n
    assert progress_calls[-1] == 100

    img = cv2.imread(os.path.join(output_frames_dir, pngs[0]))
    assert img.shape == (48, 64, 3)


def test_render_glow_output_is_finite_and_in_range(tmp_path, synthetic_track_fixture):
    output_frames_dir = tmp_path / "glow_frames"
    motion_path = tmp_path / "motion.npz"
    blade.compute_motion(synthetic_track_fixture["masks_dir"], str(motion_path))

    render_glow(
        synthetic_track_fixture["frames_dir"],
        synthetic_track_fixture["masks_dir"],
        synthetic_track_fixture["video_meta_path"],
        str(output_frames_dir),
        str(motion_path),
    )

    for i in range(synthetic_track_fixture["n_frames"]):
        img = _load_png(str(output_frames_dir), i)
        assert img.dtype == np.uint8
        arr = img.astype(np.float64)
        assert np.isfinite(arr).all()
        assert (arr >= 0).all() and (arr <= 255).all()


def test_render_glow_frame_with_missing_mask_still_renders(tmp_path):
    clip = _build_blade_clip(tmp_path, "dropframe", n_frames=6, drop_mask_frame=3)

    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        clip["output_frames_dir"], clip["motion_path"],
    )

    img = _load_png(clip["output_frames_dir"], 3)
    assert img.shape == (clip["height"], clip["width"], 3)
    arr = img.astype(np.float64)
    assert np.isfinite(arr).all()
    # every frame, including the dropped one, still gets written
    pngs = os.listdir(clip["output_frames_dir"])
    assert len(pngs) == clip["n_frames"]


# ---------------------------------------------------------------------------
# B1.2 -- blade extension beyond the mask's own extent along the axis
# ---------------------------------------------------------------------------

def test_blade_extend_lights_beyond_mask_extent(tmp_path):
    clip = _build_blade_clip(tmp_path, "extend", n_frames=3)

    # Ground truth geometry for frame 0, from the same fit_blade the
    # pipeline itself uses -- avoids hardcoding which end PCA calls "tip".
    mask0 = blade.load_mask(clip["masks_dir"], 0)
    geo = blade.fit_blade(mask0)
    tip_extend_frac = 0.10
    extended_tip = np.array(geo.tip) + np.array(geo.axis) * geo.length * tip_extend_frac
    px, py = round(extended_tip[0]), round(extended_tip[1])
    assert 0 <= px < clip["width"] and 0 <= py < clip["height"]

    out_true = str(tmp_path / "out_true")
    out_false = str(tmp_path / "out_false")
    for do_extend, out_dir in ((True, out_true), (False, out_false)):
        render_glow(
            clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
            out_dir, clip["motion_path"], ignition_ramp_seconds=0,
            blade_extend=do_extend, tip_extend_frac=tip_extend_frac,
        )

    img_true = _load_png(out_true, 0)
    img_false = _load_png(out_false, 0)
    baseline = float(clip["plate_value"])

    signal_true = float(img_true[py, px].astype(np.float64).max()) - baseline
    signal_false = float(img_false[py, px].astype(np.float64).max()) - baseline

    assert signal_true > 20  # solidly lit -- inside the extended capsule
    assert signal_true > signal_false + 10  # extension is doing something real


def test_no_blade_extend_falls_back_to_raw_mask(tmp_path):
    # A point well beyond the raw mask's own tip end must NOT light up at
    # all when blade_extend=False (raw-mask tracing only).
    clip = _build_blade_clip(tmp_path, "noextend", n_frames=2, blade_len=90)
    mask0 = blade.load_mask(clip["masks_dir"], 0)
    geo = blade.fit_blade(mask0)
    far_point = np.array(geo.tip) + np.array(geo.axis) * 40  # far past any plausible extension
    px, py = round(far_point[0]), round(far_point[1])
    assert 0 <= px < clip["width"] and 0 <= py < clip["height"]

    out_dir = str(tmp_path / "out")
    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_dir, clip["motion_path"], blade_extend=False,
    )
    img = _load_png(out_dir, 0)
    baseline = float(clip["plate_value"])
    signal = float(img[py, px].astype(np.float64).max()) - baseline
    assert signal < 3  # essentially untouched plate


# ---------------------------------------------------------------------------
# B1.3 -- eroded core narrower than the colour band
# ---------------------------------------------------------------------------

def test_core_is_narrower_than_colour_band(tmp_path):
    # Pure blue (BGR) means the green/red channels can only ever carry the
    # (achromatic, always-white) core -- colour band and glow are tinted
    # zero there. The blue channel carries core + colour + glow combined.
    # So comparing the green-channel profile's width against the blue-
    # channel profile's width isolates "core only" vs "core+colour+glow"
    # directly from real rendered pixels.
    clip = _build_blade_clip(tmp_path, "corewidth", n_frames=1, blade_height=14)
    out_dir = str(tmp_path / "out")
    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_dir, clip["motion_path"], color=(255, 0, 0), blade_extend=True,
        ignition_ramp_seconds=0,
    )
    img = _load_png(out_dir, 0).astype(np.float64)

    mask0 = blade.load_mask(clip["masks_dir"], 0)
    geo = blade.fit_blade(mask0)
    cx = round(geo.centroid[0])

    col_blue = img[:, cx, 0]
    col_green = img[:, cx, 1]
    baseline = float(clip["plate_value"])

    def half_max_width(profile):
        signal = profile - baseline
        peak = signal.max()
        if peak <= 1:
            return 0
        thresh = peak * 0.5
        return int((signal >= thresh).sum())

    core_width = half_max_width(col_green)
    combined_width = half_max_width(col_blue)

    assert core_width > 0
    assert core_width < combined_width


# ---------------------------------------------------------------------------
# B1.5 -- Knoll darkening (isolated unit test on the exported helper)
# ---------------------------------------------------------------------------

def test_knoll_darken_dims_plate_just_outside_the_blade():
    h, w = 80, 120
    plate = np.full((h, w, 3), 0.9, dtype=np.float32)  # bright, near-white plate
    blade_u8 = np.zeros((h, w), dtype=np.uint8)
    blade_u8[35:45, 40:80] = 255  # a blade-shaped rectangle

    darkened, feathered = knoll_darken(
        plate, blade_u8, dilate_px=9, darken_factor=0.7, feather_sigma=4.0,
    )

    # Just outside the raw blade rectangle (inside the dilated+feathered
    # region) the plate must be measurably darker than the original.
    assert darkened[30, 60, 0] < plate[30, 60, 0] - 0.01
    assert feathered[30, 60] > 0

    # Far away from the blade, the plate must be untouched.
    assert darkened[5, 5, 0] == pytest.approx(plate[5, 5, 0], abs=1e-6)
    assert feathered[5, 5] == pytest.approx(0.0, abs=1e-6)


def test_render_glow_darkens_plate_near_blade_end_to_end(tmp_path):
    # End-to-end version of the Knoll check: with color=(0, 0, 0) neither
    # the colour band nor the glow contribute any brightness (core stays
    # white but its blur is tight relative to Knoll's dilation), so a
    # point at the outer edge of the darkened region should read strictly
    # darker than the plain plate value.
    clip = _build_blade_clip(tmp_path, "knoll_e2e", n_frames=1, blade_height=10)
    out_dir = str(tmp_path / "out")
    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_dir, clip["motion_path"], color=(0, 0, 0), blade_extend=True,
        knoll_dilate_frac=2.5, knoll_darken_factor=0.6,
    )
    img = _load_png(out_dir, 0).astype(np.float64)

    # A point just above the blade band (inside dilation, outside the tight
    # core blur reach) and far in x from either end (inside the blade body).
    y = clip["y0"] - 6
    x = clip["width"] // 2
    assert img[y, x, 0] < clip["plate_value"] - 3


# ---------------------------------------------------------------------------
# B1.6 -- temporal trail: energy left behind at an earlier frame's position
# ---------------------------------------------------------------------------

def test_trail_leaves_energy_at_previous_position(tmp_path):
    clip = _build_blade_clip(tmp_path, "trail", n_frames=8, dx=15, blade_len=40)
    out_dir = str(tmp_path / "out")
    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_dir, clip["motion_path"], trail_decay=0.75,
        ignition_ramp_seconds=0,
    )

    mask0 = blade.load_mask(clip["masks_dir"], 0)
    geo0 = blade.fit_blade(mask0)
    px, py = round(geo0.centroid[0]), round(geo0.centroid[1])

    frame0 = _load_png(out_dir, 0).astype(np.float64)
    frame_later = _load_png(out_dir, 5).astype(np.float64)
    baseline = float(clip["plate_value"])

    signal_frame0 = frame0[py, px].max() - baseline
    signal_later = frame_later[py, px].max() - baseline

    assert signal_frame0 > 20  # blade was actually there at frame 0
    assert signal_later > 2  # decayed ghost still visible 5 frames later
    assert signal_later < signal_frame0  # but weaker than the original


# ---------------------------------------------------------------------------
# Reproducibility -- seeded flicker means identical renders are bit-identical
# ---------------------------------------------------------------------------

def test_render_glow_is_reproducible_with_same_seed(tmp_path):
    clip = _build_blade_clip(tmp_path, "seed", n_frames=3)
    out_a = str(tmp_path / "out_a")
    out_b = str(tmp_path / "out_b")
    for out_dir in (out_a, out_b):
        render_glow(
            clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
            out_dir, clip["motion_path"], rng_seed=777,
        )
    for i in range(clip["n_frames"]):
        a = _load_png(out_a, i)
        b = _load_png(out_b, i)
        assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Characterization test: golden baseline before Task 1's extraction refactor
# ---------------------------------------------------------------------------

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
        out_dir, clip["motion_path"], color=(255, 90, 60),
    )
    frames = [_load_png(out_dir, i) for i in range(clip["n_frames"])]
    checksums = [int(f.astype(np.uint64).sum()) for f in frames]
    # A committed golden value, not a live re-comparison against another
    # render -- if this assertion ever needs to change, that means
    # render_glow's real output changed, which must be a deliberate,
    # reviewed decision, not an accidental refactor side effect.
    assert checksums == EXPECTED_CHARACTERIZATION_CHECKSUMS


# ---------------------------------------------------------------------------
# Ignition/extinguish -- blade growth/shrinkage at clip start/end
# ---------------------------------------------------------------------------

def test_ignition_fraction_ramps_up_from_the_first_active_frame():
    assert ignition_fraction(0, 0, 100, 4) == pytest.approx(0.25)
    assert ignition_fraction(1, 0, 100, 4) == pytest.approx(0.5)
    assert ignition_fraction(3, 0, 100, 4) == 1.0


def test_ignition_fraction_ramps_down_toward_the_last_active_frame():
    assert ignition_fraction(100, 0, 100, 4) == pytest.approx(0.25)
    assert ignition_fraction(99, 0, 100, 4) == pytest.approx(0.5)
    assert ignition_fraction(97, 0, 100, 4) == 1.0


def test_ignition_fraction_is_full_in_the_steady_middle():
    assert ignition_fraction(50, 0, 100, 4) == 1.0


def test_ignition_fraction_tapers_instead_of_plateauing_on_a_short_window():
    # A 5-frame active window with a 4-frame ramp is too short for both the
    # rise and the fall to complete separately -- they must overlap, so the
    # peak never reaches 1.0 (a triangular taper, not a clipped plateau).
    frac = ignition_fraction(2, 0, 4, 4)
    assert 0.0 < frac < 1.0


def test_ignition_fraction_defaults_to_full_when_never_active():
    assert ignition_fraction(5, None, None, 4) == 1.0


def test_ignition_fraction_defaults_to_full_when_ramp_frames_is_zero():
    assert ignition_fraction(5, 0, 100, 0) == 1.0


def test_ignition_ramp_shortens_the_blade_at_the_start_of_the_clip(tmp_path):
    # 20 frames at 24fps gives an ~8-frame ramp (IGNITION_RAMP_SECONDS=0.35),
    # comfortably shorter than the clip -- frame 0 should be mid-ignition
    # while frame 10 sits in the steady middle.
    clip = _build_blade_clip(tmp_path, "ignition", n_frames=20, dx=3)

    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        clip["output_frames_dir"], clip["motion_path"],
    )

    baseline = float(clip["plate_value"])

    def signal_at_raw_tip(frame_idx):
        mask = blade.load_mask(clip["masks_dir"], frame_idx)
        geo = blade.fit_blade(mask)
        px, py = round(geo.tip[0]), round(geo.tip[1])
        img = _load_png(clip["output_frames_dir"], frame_idx)
        return float(img[py, px].astype(np.float64).max()) - baseline

    start_signal = signal_at_raw_tip(0)
    middle_signal = signal_at_raw_tip(10)

    assert middle_signal > 20  # solidly lit once ignition has ramped up
    assert start_signal < middle_signal - 10  # visibly shorter right at the start


# ---------------------------------------------------------------------------
# render_glow_multi -- compositing up to 4 independently-colored sabers
# ---------------------------------------------------------------------------

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
    signal = img - baseline  # (h, w, 3), BGR order

    mask_a = blade.load_mask(clip_a["masks_dir"], 0)
    geo_a = blade.fit_blade(mask_a)
    mask_b = blade.load_mask(clip_b["masks_dir"], 0)
    geo_b = blade.fit_blade(mask_b)

    # The exact centroid pixel saturates to solid white for either color
    # alike (core brightness clips there), so this searches a window
    # around each object's own centroid for the point of clearest
    # separation between its own color channel and the other object's,
    # instead of assuming one exact unsaturated offset.
    def best_own_color_pixel(cx, cy, own_channel, other_channel, radius=25):
        y0, y1 = max(0, cy - radius), min(img.shape[0], cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(img.shape[1], cx + radius + 1)
        region = signal[y0:y1, x0:x1]
        separation = region[..., own_channel] - region[..., other_channel]
        iy, ix = np.unravel_index(np.argmax(separation), separation.shape)
        return region[iy, ix]

    px_a, py_a = round(geo_a.centroid[0]), round(geo_a.centroid[1])
    px_b, py_b = round(geo_b.centroid[0]), round(geo_b.centroid[1])

    # BGR order: object A is pure blue (channel 0), object B is pure green (channel 1).
    best_a = best_own_color_pixel(px_a, py_a, own_channel=0, other_channel=1)
    best_b = best_own_color_pixel(px_b, py_b, own_channel=1, other_channel=0)

    assert best_a[0] > 15  # object A's own blue channel is clearly lit somewhere near its blade
    assert best_a[0] > best_a[1] + 10  # ...and clearly separated from B's color there
    assert best_b[1] > 15  # object B's own green channel is clearly lit somewhere near its blade
    assert best_b[1] > best_b[0] + 10  # ...and clearly separated from A's color there


def test_render_glow_multi_with_one_object_matches_render_glow(tmp_path):
    # The N=1 case must agree with today's render_glow. Now that light wrap
    # is per-object (each object wrapping its own blade at full strength)
    # there is no 1/N scaling left to differ at all, and this measures a max
    # absolute difference of exactly 0; the small tolerance below stays only
    # to absorb trivial summing-order/floating-point noise across platforms.
    clip = _build_blade_clip(tmp_path, "equiv", n_frames=4)
    out_single = str(tmp_path / "out_single")
    out_multi = str(tmp_path / "out_multi")

    render_glow(
        clip["frames_dir"], clip["masks_dir"], clip["video_meta_path"],
        out_single, clip["motion_path"], color=(40, 40, 255), spill_strength=0.35,
        ignition_ramp_seconds=0,
    )
    from lightsaber_fx.pipeline.glow import render_glow_multi
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


def test_render_glow_multi_light_wrap_uses_each_objects_own_color(tmp_path):
    # Light wrap spills a blade's color onto the plate around it. Wrapping
    # the union of every blade shape in every object's color -- which is what
    # this used to do, at 1/N strength each -- tints every blade's halo with
    # every other saber's color: a red-vs-blue duel comes out with two
    # magenta-ish halos. Each object's wrap must use only its own blade.
    #
    # Object A is pure blue in BGR, so nothing else in the pipeline can tell
    # its green channel from its red one: the plate is neutral gray, knoll
    # darkening is a grayscale multiplier, and the tonemap is the same curve
    # per channel. Green lifting above red next to A's blade therefore means
    # exactly one thing -- B's green leaked into A's wrap.
    clip_a = _build_blade_clip(tmp_path, "wrapA", n_frames=2, blade_x0=10, dx=2, blade_len=30, blade_height=8)
    clip_b = _build_blade_clip(tmp_path, "wrapB", n_frames=2, blade_x0=150, dx=2, blade_len=30, blade_height=8, width=220)

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
        # Above the 0.15 default purely to make the leak unmissable: the bug
        # scales with wrap strength (it measured G-R of 17 here at 0.8 versus
        # 4 at the default), so a decisive margin beats a marginal one.
        light_wrap_strength=0.8,
    )

    # Just past the tip of A's blade: outside both objects' masks, inside A's
    # wrap dilation radius (the 12px kernel reaches ~6px past the blade,
    # which ends at x=39), and ~100px clear of B, whose mask starts at x=150.
    sample_x, sample_y = 44, 45
    mask_a = blade.load_mask(clip_a["masks_dir"], 0)
    mask_b = blade.load_mask(clip_b["masks_dir"], 0)
    assert not mask_a[sample_y, sample_x], "sample point is inside object A's own mask"
    assert not mask_b[sample_y, sample_x], "sample point is inside object B's mask"

    img = _load_png(str(output_frames_dir), 0).astype(np.float64)
    b, g, r = img[sample_y, sample_x]

    # A is lit in its own color here...
    assert b - g > 30, f"object A's blue wrap is not visible at the sample point (BGR={b},{g},{r})"
    # ...and B's green has not come along with it. Pre-fix this read ~17.
    assert abs(g - r) <= 2, f"object B's green leaked into object A's wrap (BGR={b},{g},{r})"


def test_render_glow_multi_rejects_an_unsupported_object_count(tmp_path):
    # Zero objects used to render an untouched plate -- a silently wrong
    # result rather than an error.
    from lightsaber_fx.pipeline.glow import render_glow_multi

    clip = _build_blade_clip(tmp_path, "guard", n_frames=1)
    one = {
        "masks_dir": clip["masks_dir"], "motion_path": clip["motion_path"],
        "color": (40, 40, 255), "intensity": 0.35,
    }

    for objects in ([], [one] * 5):
        with pytest.raises(ValueError, match="1-4 objects"):
            render_glow_multi(
                clip["frames_dir"], objects, clip["video_meta_path"],
                str(tmp_path / "glow_frames"),
            )


# ---------------------------------------------------------------------------
# Task 7 -- threading `bend` from motion.npz through the per-frame reads in
# render_glow/render_glow_multi and into _composite_blade_contribution.
# ---------------------------------------------------------------------------

def test_stabilize_tip_hilt_leaves_bend_completely_unchanged_on_a_flip():
    # bend is an absolute (x, y) point, not a directional offset -- a
    # tip/hilt continuity flip has nothing to swap it with. This locks
    # in that finding as a test, correcting an assumption in the design
    # spec that turned out to be unnecessary once worked through.
    from lightsaber_fx.pipeline.glow import _stabilize_tip_hilt
    tip = np.array([[10.0, 0.0], [-10.0, 0.0]])   # axis flips sign at frame 1
    hilt = np.array([[0.0, 0.0], [0.0, 0.0]])
    axis = np.array([[1.0, 0.0], [-1.0, 0.0]])
    _bend_before = np.array([[5.0, 3.0], [5.0, 3.0]])

    new_tip, _new_hilt, _new_axis = _stabilize_tip_hilt(tip, hilt, axis)

    assert not np.allclose(new_tip[1], tip[1])  # confirms a flip actually happened
    # bend itself was never passed in and never touched -- nothing to assert
    # on bend's value changing, since _stabilize_tip_hilt's signature does
    # not take it. This test exists to make that omission a deliberate,
    # documented choice rather than a silent gap.


def test_render_glow_multi_never_renders_a_curved_blade_even_when_bend_is_present(tmp_path):
    # Lightsabers are rigid -- unlike the tracked prop swords, they never
    # bow, however real the measured contact bend is. render_glow_multi
    # (and render_glow) force bend to NaN right after loading motion.npz,
    # so a real bend value present in the data must have zero effect on
    # the render: this is the disable's own regression guard, replacing a
    # prior test with the opposite assertion (that bend USED to visibly
    # curve the render) now that the product decision is "never bend".
    clip = _build_blade_clip(tmp_path, "curved", n_frames=1, width=220, height=120, blade_len=90, blade_x0=40)
    motion = blade.load_motion(clip["motion_path"])
    hilt, tip = motion["hilt"][0], motion["tip"][0]
    mid = (hilt + tip) / 2.0
    bend_point = mid + np.array([0.0, -25.0])  # well off the straight line
    motion["bend"] = np.array([bend_point])
    np.savez(clip["motion_path"], **motion)

    out_dir = str(tmp_path / "out")
    from lightsaber_fx.pipeline.glow import render_glow_multi
    render_glow_multi(
        clip["frames_dir"],
        [{"masks_dir": clip["masks_dir"], "motion_path": clip["motion_path"],
          "color": (40, 40, 255), "intensity": 0.35}],
        clip["video_meta_path"], out_dir,
        ignition_ramp_seconds=0,
    )
    img = _load_png(out_dir, 0)
    baseline = float(clip["plate_value"])
    # Nothing is lit at the bend point -- were the blade curving toward
    # it, this pixel would show a strong signal (matching the old test's
    # own >20 threshold for "the curve reaches here").
    px, py = round(bend_point[0]), round(bend_point[1])
    signal = float(img[py, px].astype(np.float64).max()) - baseline
    assert signal < 20


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
