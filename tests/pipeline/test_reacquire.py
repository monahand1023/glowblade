import numpy as np

from lightsaber_fx.pipeline.blade import load_mask, save_mask
from lightsaber_fx.pipeline.reacquire import detect_merge, find_clean_reference, match_detections_to_objects, patch_masks


def _mask_at(x, width=120, height=80, bar_width=6):
    """A vertical bar mask, `bar_width` wide, centered at column `x`."""
    mask = np.zeros((height, width), dtype=bool)
    x0 = max(0, x - bar_width // 2)
    x1 = min(width, x + bar_width // 2)
    mask[10:70, x0:x1] = True
    return mask


def test_detect_merge_finds_the_first_frame_of_a_sustained_overlap(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    # Frames 0-9: two separate bars. Frames 10-29: identical (merged) bars.
    for i in range(30):
        x = 20 if i < 10 else 60
        save_mask(str(masks_a), i, _mask_at(x))
        save_mask(str(masks_b), i, _mask_at(60))

    merge_start = detect_merge(str(masks_a), str(masks_b), list(range(30)))

    assert merge_start == 10


def test_detect_merge_ignores_a_brief_touch_that_separates_again(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(30):
        # Bars touch (identical) for frames 10-14 only -- 5 frames, short of
        # the 15-frame sustain default -- then separate again.
        x = 60 if 10 <= i < 15 else 20
        save_mask(str(masks_a), i, _mask_at(x))
        save_mask(str(masks_b), i, _mask_at(60))

    merge_start = detect_merge(str(masks_a), str(masks_b), list(range(30)))

    assert merge_start is None


def test_detect_merge_returns_none_when_never_overlapping(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(20):
        save_mask(str(masks_a), i, _mask_at(20))
        save_mask(str(masks_b), i, _mask_at(90))

    assert detect_merge(str(masks_a), str(masks_b), list(range(20))) is None


def test_detect_merge_handles_missing_mask_gracefully(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(36):
        if i < 10:
            # Frames 0-9: separate masks, low overlap
            save_mask(str(masks_a), i, _mask_at(20))
            save_mask(str(masks_b), i, _mask_at(60))
        elif i == 20:
            # Frame 20: masks_b has no mask file (object lost)
            save_mask(str(masks_a), i, _mask_at(60))
            # Don't save masks_b[20] -- missing file
        else:
            # Frames 10-19 and 21-35: both have high-IoU masks
            save_mask(str(masks_a), i, _mask_at(60))
            save_mask(str(masks_b), i, _mask_at(60))

    # Frames 10-19 have 10 frames of overlap (not sustained, < 15).
    # Frame 20 has missing mask_b, resets the run.
    # Frames 21-35 have 15 frames of overlap (sustained), so merge_start = 21.
    merge_start = detect_merge(str(masks_a), str(masks_b), list(range(36)))

    assert merge_start == 21


def test_find_clean_reference_returns_the_frame_of_maximum_separation(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    # Object a drifts from x=100 toward object b (fixed at x=60) over 20
    # frames -- separation shrinks every frame, so frame 0 is the true
    # clean reference (max separation), not frame 19 (one before the
    # "merge" at frame 20).
    for i in range(20):
        save_mask(str(masks_a), i, _mask_at(100 - i * 2))
        save_mask(str(masks_b), i, _mask_at(60))

    ref = find_clean_reference(
        str(masks_a), str(masks_b), merge_start_frame=20,
        frame_indices=list(range(20)), lookback_frames=90,
    )

    assert ref == 0


def test_find_clean_reference_only_searches_within_the_lookback_window(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(20):
        save_mask(str(masks_a), i, _mask_at(100 - i * 2))
        save_mask(str(masks_b), i, _mask_at(60))

    # Lookback of only 5 frames: frame 0 (the true max) is out of range, so
    # the best candidate within [15, 20) is frame 15 (separation 10px, vs
    # 8/6/4/2 at frames 16-19).
    ref = find_clean_reference(
        str(masks_a), str(masks_b), merge_start_frame=20,
        frame_indices=list(range(20)), lookback_frames=5,
    )

    assert ref == 15


def test_find_clean_reference_returns_none_when_no_candidate_has_valid_geometry(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    empty = np.zeros((80, 120), dtype=bool)
    for i in range(20):
        save_mask(str(masks_a), i, empty)
        save_mask(str(masks_b), i, empty)

    ref = find_clean_reference(
        str(masks_a), str(masks_b), merge_start_frame=20,
        frame_indices=list(range(20)), lookback_frames=90,
    )

    assert ref is None


def test_find_clean_reference_skips_a_candidate_missing_a_mask_file(tmp_path):
    masks_a = tmp_path / "a"
    masks_b = tmp_path / "b"
    for i in range(20):
        save_mask(str(masks_a), i, _mask_at(100 - i * 2))
        if i != 15:
            save_mask(str(masks_b), i, _mask_at(60))
        # frame 15: masks_b has no file at all -- object briefly lost

    # Lookback window [15, 20): frame 15 (missing) would otherwise be the
    # best candidate (max separation); it must be skipped, falling back to
    # frame 16 (next-best separation).
    ref = find_clean_reference(
        str(masks_a), str(masks_b), merge_start_frame=20,
        frame_indices=list(range(20)), lookback_frames=5,
    )

    assert ref == 16


def test_match_detections_to_objects_matches_by_nearest_centroid():
    det_left = {"centroid": (100, 100), "points": [[100, 100]]}
    det_right = {"centroid": (500, 500), "points": [[500, 500]]}

    for_a, for_b = match_detections_to_objects(
        [det_right, det_left],  # input order shouldn't matter
        ref_centroid_a=(90, 90), ref_centroid_b=(510, 510),
    )

    assert for_a is det_left
    assert for_b is det_right


def test_match_detections_to_objects_handles_already_matching_order():
    det_a_like = {"centroid": (10, 10), "points": []}
    det_b_like = {"centroid": (200, 200), "points": []}

    for_a, for_b = match_detections_to_objects(
        [det_a_like, det_b_like], ref_centroid_a=(0, 0), ref_centroid_b=(210, 210),
    )

    assert for_a is det_a_like
    assert for_b is det_b_like


def test_patch_masks_freezes_the_gap_and_copies_the_fresh_track(tmp_path):
    lost = tmp_path / "lost"
    fresh = tmp_path / "fresh"
    for i in range(10):
        save_mask(str(lost), i, _mask_at(20))  # the object's own track before/through the merge
    for i in range(6, 10):
        save_mask(str(fresh), i, _mask_at(80))  # the freshly re-tracked object, frames 6-9

    patch_masks(str(lost), str(fresh), frozen_frame_idx=4, merge_start_frame=5, reacquire_frame=6, n_frames=10)

    # Frames 0-4: untouched (still the object's own original track).
    for i in range(5):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(20))
    # Frame 5 (the gap): frozen copy of frame 4's mask.
    assert np.array_equal(load_mask(str(lost), 5), _mask_at(20))
    # Frames 6-9: the freshly re-tracked mask.
    for i in range(6, 10):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(80))


def test_patch_masks_handles_a_zero_length_gap(tmp_path):
    lost = tmp_path / "lost"
    fresh = tmp_path / "fresh"
    for i in range(5):
        save_mask(str(lost), i, _mask_at(20))
    for i in range(3, 5):
        save_mask(str(fresh), i, _mask_at(80))

    # reacquire_frame == merge_start_frame: no frozen gap at all.
    patch_masks(str(lost), str(fresh), frozen_frame_idx=2, merge_start_frame=3, reacquire_frame=3, n_frames=5)

    for i in range(3):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(20))
    for i in range(3, 5):
        assert np.array_equal(load_mask(str(lost), i), _mask_at(80))


import json

import cv2

from lightsaber_fx.pipeline.reacquire import reacquire_pair

FRAME_W, FRAME_H = 1280, 720


class _FakeGenaiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGenaiClient:
    """`text_for_call(n)` returns the response text for the n-th call
    (0-indexed), so a test can simulate different frames returning
    different detections as reacquire_pair walks forward."""

    def __init__(self, text_for_call):
        self.text_for_call = text_for_call
        self.calls = 0
        self.models = self

    def generate_content(self, **kwargs):
        text = self.text_for_call(self.calls)
        self.calls += 1
        return _FakeGenaiResponse(text)


class _FakePredictor:
    def __init__(self, mask_for_box):
        self.mask_for_box = mask_for_box

    def set_image(self, image):
        pass

    def predict(self, box, multimask_output=False):
        mask = self.mask_for_box(box)
        return np.asarray([mask]), np.ones(1), None


def _write_frame(path, value=100):
    frame = np.full((FRAME_H, FRAME_W, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), frame)


def _line_mask_at(box, width=6):
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cv2.line(mask, (x0, y0), (x1, y1), 1, width)
    return mask.astype(bool)


def _no_boxes_response():
    return json.dumps({"objects": []})


def _two_separated_boxes_response():
    # Two thin horizontal boxes, one on the left third of the frame, one
    # on the right third -- far enough apart their line masks won't overlap.
    return json.dumps({"objects": [
        {"box_2d": [340, 50, 360, 300], "label": "blade"},
        {"box_2d": [340, 700, 360, 950], "label": "blade"},
    ]})


def _two_overlapping_boxes_response():
    return json.dumps({"objects": [
        {"box_2d": [340, 50, 360, 300], "label": "blade"},
        {"box_2d": [340, 60, 360, 310], "label": "blade"},  # nearly identical -> overlapping masks
    ]})


def test_reacquire_pair_finds_two_separated_detections_on_the_first_checked_frame(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frame(frames_dir / "00050.jpg")

    client = _FakeGenaiClient(lambda call: _two_separated_boxes_response())
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client,
    )

    assert result is not None
    reacquire_frame, detections = result
    assert reacquire_frame == 50
    assert len(detections) == 2
    centroids_x = sorted(d["centroid"][0] for d in detections)
    assert centroids_x[0] < FRAME_W / 2 < centroids_x[1]  # one left-ish, one right-ish


def test_reacquire_pair_keeps_searching_past_frames_that_are_not_yet_separated(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in (50, 60, 70):
        _write_frame(frames_dir / f"{idx:05d}.jpg")

    # Frame 50 (1st check): nothing found. Frame 60 (2nd check): two boxes,
    # but still overlapping. Frame 70 (3rd check): finally separated.
    responses = [_no_boxes_response(), _two_overlapping_boxes_response(), _two_separated_boxes_response()]
    client = _FakeGenaiClient(lambda call: responses[call])
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client, search_step=10,
    )

    assert result is not None
    reacquire_frame, _detections = result
    assert reacquire_frame == 70


def test_reacquire_pair_returns_none_when_the_search_window_is_exhausted(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    for idx in (50, 60, 70):
        _write_frame(frames_dir / f"{idx:05d}.jpg")

    client = _FakeGenaiClient(lambda call: _no_boxes_response())
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client, search_step=10, search_cap=30,
    )

    assert result is None


def test_reacquire_pair_returns_none_when_it_runs_past_the_end_of_the_clip(tmp_path, monkeypatch):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    _write_frame(frames_dir / "00050.jpg")
    # No frame 60 written -- the clip ends before the search budget does.

    client = _FakeGenaiClient(lambda call: _no_boxes_response())
    predictor = _FakePredictor(_line_mask_at)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._build_image_predictor", lambda *a, **k: predictor)

    result = reacquire_pair(
        str(frames_dir), search_start_frame=50, checkpoint_path="ckpt", config_name="cfg", device="cpu",
        client=client, search_step=10, search_cap=150,
    )

    assert result is None
