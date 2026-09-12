import os

import cv2
import numpy as np
import pytest

from lightsaber_fx import paths
from lightsaber_fx.pipeline.track import pick_points_interactive, track_object

requires_sam2_checkpoint = pytest.mark.skipif(
    not paths.get_checkpoint_path().exists(),
    reason="SAM2 checkpoint not installed; run `lightsaber-fx setup` first",
)


def test_pick_points_interactive_records_include_and_exclude_clicks(tmp_path, monkeypatch):
    frame_path = tmp_path / "00000.jpg"
    cv2.imwrite(str(frame_path), np.zeros((100, 100, 3), dtype=np.uint8))

    callback_holder = {}

    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda window, cb: callback_holder.__setitem__("cb", cb))
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    state = {"ticks": 0}

    def fake_wait_key(delay):
        if state["ticks"] == 0:
            callback_holder["cb"](cv2.EVENT_LBUTTONDOWN, 10, 20, 0, None)
            callback_holder["cb"](cv2.EVENT_LBUTTONDOWN, 30, 40, cv2.EVENT_FLAG_SHIFTKEY, None)
            state["ticks"] += 1
            return -1
        return 13

    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)

    points, labels = pick_points_interactive(str(frame_path))

    assert points == [[10, 20], [30, 40]]
    assert labels == [1, 0]


@requires_sam2_checkpoint
def test_track_object_writes_a_mask_per_frame(tmp_path):
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    n_frames = 3
    for i in range(n_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x = 10 + i * 5
        frame[20:30, x:x + 10] = (255, 255, 255)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), frame)

    track_object(
        str(frames_dir), str(masks_dir),
        points=[[15, 25]], labels=[1],
        checkpoint_path=str(paths.get_checkpoint_path()),
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cpu",
        n_frames=n_frames,
    )

    written = sorted(os.listdir(masks_dir))
    assert written == [f"{i:05d}.npy" for i in range(n_frames)]
    for fname in written:
        mask = np.load(masks_dir / fname)
        assert mask.any()
