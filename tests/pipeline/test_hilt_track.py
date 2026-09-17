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
        cx = round(start_center[0] + dx * i)
        cy = round(start_center[1] + dy * i)
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
