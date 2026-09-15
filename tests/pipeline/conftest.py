"""Shared fixtures for tests/pipeline/*.

`rotating_bar_video` started as a fixture private to test_detect.py, but
test_vision_detect.py needs the same synthetic clip -- real, sampleable
motion plus static distractors a shape-only search would get wrong -- to
exercise `detect_blades_vlm`. Living here, both test files get it without
either duplicating it or importing test internals from one another.
"""

import cv2
import numpy as np
import pytest

WIDTH, HEIGHT = 320, 240
FPS = 30.0
PIVOT = (160, 140)
BAR_LENGTH = 70


def _write_video(path, draw_frame, n_frames=40):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT)
    )
    for i in range(n_frames):
        frame = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
        draw_frame(frame, i)
        writer.write(frame)
    writer.release()


def _static_distractors(frame, shift=0):
    """The things a shape-only search gets wrong: long, straight,
    high-contrast and perfectly still. A fence rail and a horizon are far
    more elongated than any bat, so a detector scoring on shape alone picks
    one of them. Scoring on motion is what makes them score zero."""
    cv2.line(frame, (-shift, 60), (WIDTH - shift, 60), (200, 200, 200), 3)
    cv2.line(frame, (-shift, 200), (WIDTH - shift, 190), (180, 180, 180), 4)
    cv2.line(frame, (30 - shift, 0), (30 - shift, HEIGHT), (170, 170, 170), 3)


def _rotating_bar(frame, i, n_frames=40):
    """An elongated object pivoting about one end -- rotation only, no
    travel. This is the case that broke the first version of the detector:
    optical-flow magnitude scales with radius, so only the bar's tip lights
    up, and the resulting flow blob has an elongation around 2. Any shape
    gate applied to the *flow* region rejects it."""
    angle = -np.pi / 2 + (i / n_frames) * np.pi
    tip = (
        int(PIVOT[0] + BAR_LENGTH * np.cos(angle)),
        int(PIVOT[1] + BAR_LENGTH * np.sin(angle)),
    )
    cv2.line(frame, PIVOT, tip, (240, 240, 240), 7)
    return tip


@pytest.fixture
def rotating_bar_video(tmp_path):
    path = tmp_path / "swing.mp4"
    _write_video(path, lambda frame, i: (_static_distractors(frame), _rotating_bar(frame, i)))
    return path
