import cv2
import numpy as np
import pytest


def _write_tiny_video(path, n_frames=5, width=64, height=48, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


@pytest.fixture
def tiny_video_path(tmp_path):
    path = tmp_path / "tiny.mp4"
    _write_tiny_video(path)
    return path


@pytest.fixture
def tiny_video_bytes(tiny_video_path):
    return tiny_video_path.read_bytes()


@pytest.fixture
def synthetic_track_fixture(tmp_path):
    """Builds frames/, masks/, and video_meta.txt like track_object would,
    without needing a real SAM2 checkpoint."""
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    masks_dir.mkdir()
    n_frames = 10
    fps = 24.0
    width, height = 64, 48
    for i in range(n_frames):
        frame = np.full((height, width, 3), 30, dtype=np.uint8)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), frame)
        mask = np.zeros((height, width), dtype=bool)
        x = 5 + i * 3
        mask[20:28, x:x + 4] = True
        np.save(masks_dir / f"{i:05d}.npy", mask)
    video_meta_path = tmp_path / "video_meta.txt"
    video_meta_path.write_text(f"{fps}\n{n_frames}\n")
    return {
        "frames_dir": str(frames_dir),
        "masks_dir": str(masks_dir),
        "video_meta_path": str(video_meta_path),
        "n_frames": n_frames,
        "fps": fps,
    }
