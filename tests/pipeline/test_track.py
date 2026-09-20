import os

import cv2
import numpy as np
import pytest

from glowblade import paths
from glowblade.pipeline.blade import load_mask
from glowblade.pipeline.detect import BladeProposal, MotionSeed
from glowblade.pipeline.track import (
    overlay_proposal,
    pick_points_interactive,
    track_object,
    track_objects,
)

requires_sam2_checkpoint = pytest.mark.skipif(
    not paths.get_checkpoint_path().exists(),
    reason="SAM2 checkpoint not installed; run `glowblade setup` first",
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
    assert written == [f"{i:05d}.npz" for i in range(n_frames)]
    for i in range(n_frames):
        mask = load_mask(str(masks_dir), i)
        assert mask.any()


def _headless_picker(monkeypatch, clicks):
    """Drive pick_points_interactive without a window. `clicks` is a list of
    (x, y, flags) delivered on the first tick; ENTER follows on the next."""
    holder = {}
    monkeypatch.setattr(cv2, "namedWindow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda window, cb: holder.__setitem__("cb", cb))
    monkeypatch.setattr(cv2, "imshow", lambda *a, **k: None)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    state = {"ticks": 0}

    def fake_wait_key(delay):
        if state["ticks"] == 0:
            for x, y, flags in clicks:
                holder["cb"](cv2.EVENT_LBUTTONDOWN, x, y, flags, None)
            state["ticks"] += 1
            return -1
        return 13

    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)


def _blade_proposal(shape=(100, 100)):
    mask = np.zeros(shape, dtype=bool)
    mask[48:52, 20:80] = True
    points = [[30, 50], [50, 50], [70, 50]]
    return BladeProposal(
        frame_index=42, points=points, labels=[1, 1, 1], mask=mask, elongation=9.0,
        seed=MotionSeed(frame_index=42, point=[70, 50], speed=8.0, area=200),
    )


def test_pick_points_interactive_returns_the_proposal_when_nothing_is_clicked(
    tmp_path, monkeypatch
):
    # Pressing ENTER with no clicks is the accept path -- the whole point of
    # propose-then-confirm. It must return the detected points verbatim.
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.zeros((100, 100, 3), dtype=np.uint8))
    proposal = _blade_proposal()
    _headless_picker(monkeypatch, clicks=[])

    points, labels = pick_points_interactive(str(frame_path), proposal=proposal)

    assert points == proposal.points
    assert labels == proposal.labels


def test_pick_points_interactive_discards_the_proposal_once_the_user_clicks(
    tmp_path, monkeypatch
):
    # A click means the user disagrees with what was detected, so the
    # proposal must be dropped rather than merged with their point. Merging
    # would keep whatever was wrong about it -- and if the detected mask was
    # on the wrong object, adding one correct point to it gives SAM2 two
    # contradictory prompts.
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.zeros((100, 100, 3), dtype=np.uint8))
    proposal = _blade_proposal()
    _headless_picker(monkeypatch, clicks=[(11, 22, 0)])

    points, labels = pick_points_interactive(str(frame_path), proposal=proposal)

    assert points == [[11, 22]]
    assert labels == [1]
    for detected in proposal.points:
        assert detected not in points


def test_pick_points_interactive_still_supports_exclude_clicks_over_a_proposal(
    tmp_path, monkeypatch
):
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.zeros((100, 100, 3), dtype=np.uint8))
    _headless_picker(monkeypatch, clicks=[(5, 6, 0), (7, 8, cv2.EVENT_FLAG_SHIFTKEY)])

    points, labels = pick_points_interactive(str(frame_path), proposal=_blade_proposal())

    assert points == [[5, 6], [7, 8]]
    assert labels == [1, 0]


def test_overlay_proposal_tints_the_mask_and_leaves_the_rest_alone():
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:14, 5:35] = True

    overlay_proposal(img, mask, points=[])

    assert img[12, 20].any(), "masked pixels were not tinted"
    assert not img[30, 20].any(), "unmasked pixels were modified"


def test_overlay_proposal_ignores_a_mask_that_does_not_match_the_frame():
    # The mask comes from detection and the image from disk; a size mismatch
    # means they describe different frames, and blending them would either
    # crash or silently tint the wrong pixels.
    img = np.zeros((40, 40, 3), dtype=np.uint8)

    overlay_proposal(img, np.ones((10, 10), dtype=bool), points=[])

    assert not img.any()


@requires_sam2_checkpoint
def test_track_object_covers_the_whole_clip_when_prompted_mid_way(tmp_path):
    # Automatic detection prompts the frame where the object was easiest to
    # find, which is usually mid-clip. Propagating forward only would leave
    # every earlier frame unmasked, so this asserts coverage on both sides of
    # the prompt frame.
    frames_dir = tmp_path / "frames"
    masks_dir = tmp_path / "masks"
    frames_dir.mkdir()
    n_frames = 5
    for i in range(n_frames):
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        x = 5 + i * 8
        frame[20:30, x:x + 10] = (255, 255, 255)
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), frame)

    prompt_frame = 2
    track_object(
        str(frames_dir), str(masks_dir),
        points=[[5 + prompt_frame * 8 + 5, 25]], labels=[1],
        checkpoint_path=str(paths.get_checkpoint_path()),
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cpu",
        n_frames=n_frames,
        prompt_frame=prompt_frame,
    )

    assert sorted(os.listdir(masks_dir)) == [f"{i:05d}.npz" for i in range(n_frames)]
    assert load_mask(str(masks_dir), 0).any(), "no mask before the prompt frame"
    assert load_mask(str(masks_dir), n_frames - 1).any(), "no mask after the prompt frame"


@requires_sam2_checkpoint
def test_track_objects_tracks_two_objects_independently(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    n_frames = 3
    for i in range(n_frames):
        frame = np.zeros((64, 128, 3), dtype=np.uint8)
        frame[20:30, 10 + i * 3:20 + i * 3] = (255, 255, 255)   # object A, left side
        frame[20:30, 90 + i * 3:100 + i * 3] = (255, 255, 255)  # object B, right side
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), frame)

    masks_dir_a = tmp_path / "masks" / "0"
    masks_dir_b = tmp_path / "masks" / "1"

    track_objects(
        str(frames_dir),
        [
            {"obj_id": 0, "masks_dir": str(masks_dir_a), "points": [[15, 25]], "labels": [1]},
            {"obj_id": 1, "masks_dir": str(masks_dir_b), "points": [[95, 25]], "labels": [1]},
        ],
        checkpoint_path=str(paths.get_checkpoint_path()),
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cpu",
        n_frames=n_frames,
    )

    for masks_dir in (masks_dir_a, masks_dir_b):
        assert sorted(os.listdir(masks_dir)) == [f"{i:05d}.npz" for i in range(n_frames)]
        for i in range(n_frames):
            assert load_mask(str(masks_dir), i).any()

    # The two objects' masks must stay on their own sides of the frame,
    # not bleed into or duplicate each other.
    mask_a0 = load_mask(str(masks_dir_a), 0)
    mask_b0 = load_mask(str(masks_dir_b), 0)
    assert not np.any(mask_a0 & mask_b0)


@requires_sam2_checkpoint
def test_track_objects_covers_the_whole_clip_when_prompted_mid_way(tmp_path):
    # The multi-object path is now the only path the web app uses, so it has
    # to honour what automatic detection reports: the frame an object was
    # easiest to find, which is usually mid-swing rather than frame 0.
    # Prompting at frame 0 regardless silently applies the points to a frame
    # the object has already left, and propagating forward-only would leave
    # every frame before the prompt unmasked.
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    n_frames = 5
    for i in range(n_frames):
        frame = np.zeros((64, 128, 3), dtype=np.uint8)
        frame[20:30, 10 + i * 6:20 + i * 6] = (255, 255, 255)   # object A, left side
        frame[20:30, 90 + i * 6:100 + i * 6] = (255, 255, 255)  # object B, right side
        cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), frame)

    masks_dir_a = tmp_path / "masks" / "0"
    masks_dir_b = tmp_path / "masks" / "1"
    prompt_frame = 2

    track_objects(
        str(frames_dir),
        [
            {"obj_id": 0, "masks_dir": str(masks_dir_a),
             "points": [[15 + prompt_frame * 6, 25]], "labels": [1], "prompt_frame": prompt_frame},
            {"obj_id": 1, "masks_dir": str(masks_dir_b),
             "points": [[95 + prompt_frame * 6, 25]], "labels": [1], "prompt_frame": prompt_frame},
        ],
        checkpoint_path=str(paths.get_checkpoint_path()),
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        device="cpu",
        n_frames=n_frames,
    )

    for masks_dir in (masks_dir_a, masks_dir_b):
        assert sorted(os.listdir(masks_dir)) == [f"{i:05d}.npz" for i in range(n_frames)]
        assert load_mask(str(masks_dir), 0).any(), "no mask before the prompt frame"
        assert load_mask(str(masks_dir), n_frames - 1).any(), "no mask after the prompt frame"

    # Still two distinct objects, not one mask duplicated across both ids.
    assert not np.any(load_mask(str(masks_dir_a), 0) & load_mask(str(masks_dir_b), 0))


# Deliberately NOT marked @requires_sam2_checkpoint: the guard has to fire
# before the predictor is built, so this passes a checkpoint path and a
# frames dir that do not exist. If the guard ever moved below
# build_sam2_video_predictor, this would fail with a different error (a
# missing checkpoint/import failure) rather than passing by luck on a
# machine that happens to have SAM2 installed.
def test_track_objects_rejects_objects_prompted_on_different_frames(tmp_path):
    # SAM2 conditions every object in one shared session, and mixing
    # conditioning frames breaks its memory attention: a BFloat16/Float dtype
    # RuntimeError on CPU, and on MPS a Metal assertion that kills the process
    # outright -- which no caller can catch or report. Refusing the call is
    # the only version of this that can be surfaced to a user.
    with pytest.raises(ValueError, match="same prompt_frame"):
        track_objects(
            str(tmp_path / "frames-that-do-not-exist"),
            [
                {"obj_id": 0, "masks_dir": str(tmp_path / "0"),
                 "points": [[15, 25]], "labels": [1], "prompt_frame": 0},
                {"obj_id": 1, "masks_dir": str(tmp_path / "1"),
                 "points": [[95, 25]], "labels": [1], "prompt_frame": 3},
            ],
            checkpoint_path=str(tmp_path / "checkpoint-that-does-not-exist.pt"),
            config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
            device="cpu",
            n_frames=5,
        )

    # Nothing was created on the way to the refusal.
    assert not (tmp_path / "0").exists()
    assert not (tmp_path / "1").exists()

