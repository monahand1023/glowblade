import os

import cv2
import numpy as np

from lightsaber_fx.pipeline.blade import BladeGeometry, save_mask, save_motion
from lightsaber_fx.pipeline.hilt_track import (
    BLADE_DIRECTION_SEED_FRAC,
    MIN_SEED_FEATURES,
    _seed_features,
    _track_direction,
    _track_points_sequential,
    compute_direction_overrides,
    compute_hilt_overrides,
    track_hilt_through_run,
)

# Offsets (from a moving center) of a handful of small, distinct, non-
# periodic dots -- confirmed on a quick spike to track via optical flow
# with sub-pixel accuracy, unlike a checkerboard (whose repeating pattern
# lets KLT alias onto a neighboring, identical-looking corner) or plain
# random noise (which JPEG's lossy compression smears enough to hurt
# tracking precision).
_DOT_OFFSETS = [(-10, -8), (7, -12), (-5, 9), (12, 5), (0, 0), (-13, 3), (9, 11)]


def _write_checker_frame(path, canvas, center, value=128):
    """A uniform gray frame with a handful of small distinct dots near
    `center` -- textured enough for cv2.goodFeaturesToTrack to find real,
    precisely-trackable corners, unlike a flat color."""
    frame = np.full((canvas[1], canvas[0], 3), value, dtype=np.uint8)
    cx, cy = center
    for i, (dx, dy) in enumerate(_DOT_OFFSETS):
        shade = 20 + i * 15
        cv2.circle(frame, (cx + dx, cy + dy), 3, (shade, shade, shade), -1)
    cv2.imwrite(str(path), frame)


def _write_flat_frame(path, canvas, value=128):
    frame = np.full((canvas[1], canvas[0], 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def _write_translating_checker_sequence(frames_dir, start_frame, end_frame, start_center, dx, dy,
                                         canvas=(200, 200)):
    """Writes one frame per index in [start_frame, end_frame], each with
    the dot pattern (see `_write_checker_frame`) translated by (dx, dy)
    per frame from `start_center`. Returns {frame_idx: (x, y)} of the
    pattern's true center at each frame, for test assertions."""
    os.makedirs(str(frames_dir), exist_ok=True)
    true_positions = {}
    for i, frame_idx in enumerate(range(start_frame, end_frame + 1)):
        cx = round(start_center[0] + dx * i)
        cy = round(start_center[1] + dy * i)
        _write_checker_frame(os.path.join(str(frames_dir), f"{frame_idx:05d}.jpg"), canvas, (cx, cy))
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

    positions = _track_points_sequential(str(frames_dir), list(range(11)), seed_points)

    assert set(positions.keys()) == set(range(11))
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
    positions = _track_points_sequential(str(frames_dir), list(range(11)), seed_points)

    assert set(positions.keys()) == set(range(6))


def test_track_direction_validates_when_landing_is_close_to_the_known_good_anchor(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )

    positions = _track_direction(
        str(frames_dir), list(range(21)),
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
        str(frames_dir), list(range(21)),
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
        str(frames_dir), list(range(21)),
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
        str(frames_dir), list(range(21)), run_start_frame=5, run_end_frame=15,
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
        str(frames_dir), list(range(21)), run_start_frame=5, run_end_frame=15,
        before_frame=0, before_hilt=(80.0, 80.0), before_length=100.0,
        after_frame=20, after_hilt=(90.0, 90.0), after_length=100.0,
    )

    assert overrides == {}


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


def test_compute_hilt_overrides_covers_marginal_frames_around_the_run_too(tmp_path):
    # Real footage shows IoU climbing gradually into a real overlap
    # rather than jumping straight from zero -- see
    # blade.CROSS_OBJECT_ANCHOR_IOU_FRAC. Frames 3 and 7 here sit in that
    # gap (IoU ~0.053: under the 0.1 run threshold, over the 0.02 anchor
    # threshold), so they're neither part of the detected run (4-6) nor
    # trusted as anchors -- track_hilt_through_run already tracks
    # through them (its forward/backward tracking spans the whole
    # before/after gap regardless), so their positions must be included
    # in the result, not silently dropped.
    frames_dir = tmp_path / "frames"
    n = 11
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=n - 1, start_center=(80, 80), dx=1, dy=0.5,
    )

    masks_a, masks_b = tmp_path / "masks_a", tmp_path / "masks_b"
    motion_a, motion_b = tmp_path / "a.npz", tmp_path / "b.npz"
    for i in range(n):
        mask_a = np.zeros((20, 20), dtype=bool)
        mask_a[0:10, 0:10] = True
        save_mask(str(masks_a), i, mask_a)
        mask_b = np.zeros((20, 20), dtype=bool)
        if 4 <= i <= 6:
            mask_b[0:10, 0:10] = True  # identical to A -- IoU 1.0, the detected run
        elif i in (3, 7):
            mask_b[0:10, 9:19] = True  # one-column overlap -- IoU ~0.053, marginal
        else:
            mask_b[10:20, 10:20] = True  # disjoint -- IoU 0.0, a clean anchor
        save_mask(str(masks_b), i, mask_b)

    def geo(i):
        x, y = true_positions[i]
        return BladeGeometry(centroid=(x, y), axis=(1.0, 0.0), tip=(x + 50.0, y), hilt=(x, y),
                              length=50.0, width=5.0, angle=0.0)

    save_motion(str(motion_a), [geo(i) for i in range(n)])
    save_motion(str(motion_b), [geo(i) for i in range(n)])

    overrides_a, overrides_b = compute_hilt_overrides(
        str(frames_dir), str(masks_a), str(masks_b), str(motion_a), str(motion_b),
    )

    # anchors are frames 2 and 8 (the nearest IoU-0.0 frames); every
    # frame strictly between them (3-7) must be covered, not just the
    # narrower detected run (4-6).
    assert set(overrides_a.keys()) == set(range(3, 8))
    assert set(overrides_b.keys()) == set(range(3, 8))


# ---------------------------------------------------------------------------
# compute_direction_overrides
#
# An accurate hilt override fixes the *base* of a run's interpolated tip
# path, but not its *angle* -- see BLADE_DIRECTION_SEED_FRAC's docstring
# for the real-footage gap this closes. Reuses track_hilt_through_run
# directly (no hilt-specific behavior in it, only naming), seeded at a
# point along the blade axis instead of the hilt itself.
# ---------------------------------------------------------------------------

def _write_overlap_run_fixture_for_direction(tmp_path, true_positions, overlap_start=5, overlap_end=15, n=21,
                                              length=125.0):
    """Like `_write_overlap_run_fixture`, but hilt/tip are placed so that
    `hilt + BLADE_DIRECTION_SEED_FRAC * (tip - hilt)` lands exactly on
    `true_positions[i]` -- the point compute_direction_overrides actually
    seeds tracking from, not the hilt itself."""
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
        hilt = (x - BLADE_DIRECTION_SEED_FRAC * length, y)
        tip = (hilt[0] + length, y)
        return BladeGeometry(centroid=hilt, axis=(1.0, 0.0), tip=tip, hilt=hilt,
                              length=length, width=5.0, angle=0.0)

    save_motion(str(motion_a), [geo(i) for i in range(n)])
    save_motion(str(motion_b), [geo(i) for i in range(n)])
    return masks_a, masks_b, motion_a, motion_b


def test_compute_direction_overrides_returns_tracked_positions_for_an_unresolved_run(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )
    masks_a, masks_b, motion_a, motion_b = _write_overlap_run_fixture_for_direction(tmp_path, true_positions)

    overrides_a, overrides_b = compute_direction_overrides(
        str(frames_dir), str(masks_a), str(masks_b), str(motion_a), str(motion_b),
    )

    assert set(overrides_a.keys()) == set(range(5, 16))
    assert set(overrides_b.keys()) == set(range(5, 16))
    for frame_idx, (x, y) in overrides_a.items():
        true_x, true_y = true_positions[frame_idx]
        assert abs(x - true_x) < 3.0
        assert abs(y - true_y) < 3.0


def test_compute_direction_overrides_skips_excluded_ranges(tmp_path):
    frames_dir = tmp_path / "frames"
    true_positions = _write_translating_checker_sequence(
        frames_dir, start_frame=0, end_frame=20, start_center=(80, 80), dx=1, dy=0.5,
    )
    masks_a, masks_b, motion_a, motion_b = _write_overlap_run_fixture_for_direction(tmp_path, true_positions)

    overrides_a, overrides_b = compute_direction_overrides(
        str(frames_dir), str(masks_a), str(masks_b), str(motion_a), str(motion_b),
        exclude_frame_ranges=[(5, 15)],
    )

    assert overrides_a == {}
    assert overrides_b == {}
