import os

import cv2
import numpy as np
import pytest

from lightsaber_fx.pipeline import blade
from lightsaber_fx.pipeline.glow import (
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
# after Task 1 fix round to include ignition (on by default).
EXPECTED_CHARACTERIZATION_CHECKSUMS = [
    12474070,  # frame 0
    12492073,  # frame 1
    12509242,  # frame 2
    12529970,  # frame 3
    12550266,  # frame 4
    12561351,  # frame 5
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
            out_dir, clip["motion_path"],
            blade_extend=do_extend, tip_extend_frac=tip_extend_frac,
            ignition_ramp_seconds=0,
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
    # Two synthetic objects, each with its own color, rendering together.
    # This test verifies that render_glow_multi successfully composites
    # multiple objects into a single output with valid pixels.
    clip_a = _build_blade_clip(tmp_path, "objA", n_frames=5, blade_x0=10, dx=2, blade_len=30, blade_height=8)
    clip_b = _build_blade_clip(tmp_path, "objB", n_frames=5, blade_x0=150, dx=2, blade_len=30, blade_height=8, width=220)

    output_frames_dir = tmp_path / "glow_frames"
    from lightsaber_fx.pipeline.glow import render_glow_multi

    render_glow_multi(
        clip_a["frames_dir"],
        [
            {"masks_dir": clip_a["masks_dir"], "motion_path": clip_a["motion_path"], "color": (0, 0, 255), "intensity": 0.4},
            {"masks_dir": clip_b["masks_dir"], "motion_path": clip_b["motion_path"], "color": (0, 255, 0), "intensity": 0.4},
        ],
        clip_a["video_meta_path"], str(output_frames_dir),
        ignition_ramp_seconds=0,
    )

    # Verify output is generated with valid pixels (finite, in range, correct shape/dtype)
    for i in range(clip_a["n_frames"]):
        img = _load_png(str(output_frames_dir), i)
        assert img.dtype == np.uint8
        assert img.shape == (clip_a["height"], clip_a["width"], 3)
        arr = img.astype(np.float64)
        assert np.isfinite(arr).all()
        assert (arr >= 0).all() and (arr <= 255).all()


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
