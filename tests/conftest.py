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
