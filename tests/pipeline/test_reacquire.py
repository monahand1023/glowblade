import numpy as np

from lightsaber_fx.pipeline.blade import compute_motion, load_mask, save_mask
from lightsaber_fx.pipeline.reacquire import (
    detect_merge,
    find_clean_reference,
    match_detections_to_objects,
    patch_masks,
    retrack_overlap_runs,
)


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


from lightsaber_fx.pipeline.reacquire import reconcile_pair


def test_reconcile_pair_detects_and_patches_the_lost_object(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    # Frames 0-19: separate (object 0 at x=20, object 1 at x=80).
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))
    # Frames 20-49: both objects' tracking collapsed onto object 1's
    # target (x=80) -- object 0 is "lost".
    for i in range(20, 50):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))

    def fake_reacquire_pair(frames_dir, search_start_frame, checkpoint_path, config_name, device,
                             client=None, **kwargs):
        assert search_start_frame == 20  # merge_start_frame
        return 40, [
            {"centroid": (20.0, 45.0), "points": [[20, 40], [20, 45], [20, 50]]},
            {"centroid": (80.0, 45.0), "points": [[80, 40], [80, 45], [80, 50]]},
        ]

    def fake_track_object(frames_dir, out_masks_dir, points, labels, checkpoint_path, config_name, device,
                           n_frames, prompt_frame=0, progress_cb=None):
        # Recovered object 0 tracks at x=25 (distinct from both x=20 and
        # x=80, so the test can tell "frozen reference" and "freshly
        # tracked" apart) from prompt_frame onward.
        for i in range(prompt_frame, n_frames):
            save_mask(out_masks_dir, i, _mask_at(25))

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", fake_reacquire_pair)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.track_object", fake_track_object)

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=50,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is True
    # Object 1 (kept) is untouched throughout.
    for i in range(50):
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(80))
    # Object 0: original track for frames 0-19, frozen at its last-good
    # position (x=20) for the gap [20, 40), then freshly re-tracked (x=25)
    # for [40, 50).
    for i in range(20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20))
    for i in range(20, 40):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20))
    for i in range(40, 50):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(25))


def test_reconcile_pair_returns_false_when_no_merge_is_detected(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))

    called = []
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.reacquire.reacquire_pair",
        lambda *a, **k: called.append(True) or None,
    )

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=20,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is False
    assert called == []  # never even tried to re-acquire -- no merge was found
    for i in range(20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20))
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(80))


def test_reconcile_pair_returns_false_when_reacquisition_finds_nothing(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))
    for i in range(20, 40):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", lambda *a, **k: None)

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=40,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is False
    for i in range(20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20))  # untouched
    for i in range(20, 40):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(80))  # untouched


def test_reconcile_pair_returns_false_instead_of_raising_when_reacquisition_errors(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))
    for i in range(20, 40):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))

    def raising_reacquire_pair(*a, **k):
        raise RuntimeError("Gemini network error")

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", raising_reacquire_pair)

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=40,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is False


def test_reconcile_pair_declines_when_the_original_tracks_have_already_separated(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    # Frames 0-19: separate. Frames 20-34: a brief bind (15 frames, meets
    # the sustain threshold) that resolves on its own -- by frame 35 the
    # ORIGINAL tracker has already separated the two objects again. This
    # is a normal, correctly-tracked sword bind, not a permanent merge.
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))
    for i in range(20, 35):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))
    for i in range(35, 50):
        save_mask(str(masks_0), i, _mask_at(20))
        save_mask(str(masks_1), i, _mask_at(80))

    def fake_reacquire_pair(frames_dir, search_start_frame, checkpoint_path, config_name, device,
                             client=None, **kwargs):
        # Re-acquisition finds the two objects cleanly separated well
        # after the bind already resolved on its own.
        return 40, [
            {"centroid": (20.0, 45.0), "points": [[20, 40], [20, 45], [20, 50]]},
            {"centroid": (80.0, 45.0), "points": [[80, 40], [80, 45], [80, 50]]},
        ]

    called = []
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", fake_reacquire_pair)
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.reacquire.track_object",
        lambda *a, **k: called.append(True),
    )

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=50,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is False
    assert called == []  # never re-tracked -- declined before it got that far
    for i in range(50):
        expected_0 = _mask_at(20) if i < 20 or i >= 35 else _mask_at(80)
        assert np.array_equal(load_mask(str(masks_0), i), expected_0)
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(80))


def test_reconcile_pair_detects_and_patches_the_lost_object_when_it_is_object_1(tmp_path, monkeypatch):
    # Mirror of test_reconcile_pair_detects_and_patches_the_lost_object,
    # with object 1 collapsing onto object 0's stable target instead --
    # exercises the `dist_0 <= dist_1` branch (object 1 identified as
    # lost), never covered by the original test.
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    for i in range(20):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(20))
    for i in range(20, 50):
        save_mask(str(masks_0), i, _mask_at(80))
        save_mask(str(masks_1), i, _mask_at(80))

    def fake_reacquire_pair(frames_dir, search_start_frame, checkpoint_path, config_name, device,
                             client=None, **kwargs):
        assert search_start_frame == 20
        return 40, [
            {"centroid": (20.0, 45.0), "points": [[20, 40], [20, 45], [20, 50]]},
            {"centroid": (80.0, 45.0), "points": [[80, 40], [80, 45], [80, 50]]},
        ]

    def fake_track_object(frames_dir, out_masks_dir, points, labels, checkpoint_path, config_name, device,
                           n_frames, prompt_frame=0, progress_cb=None):
        for i in range(prompt_frame, n_frames):
            save_mask(out_masks_dir, i, _mask_at(25))

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", fake_reacquire_pair)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.track_object", fake_track_object)

    patched = reconcile_pair(
        "unused-frames-dir", str(masks_0), str(masks_1), n_frames=50,
        checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched is True
    # Object 0 (kept) is untouched throughout.
    for i in range(50):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(80))
    # Object 1: original track for 0-19, frozen at its reference position
    # (x=20) for [20, 40), freshly re-tracked (x=25) for [40, 50).
    for i in range(20):
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(20))
    for i in range(20, 40):
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(20))
    for i in range(40, 50):
        assert np.array_equal(load_mask(str(masks_1), i), _mask_at(25))


# ---------------------------------------------------------------------------
# retrack_overlap_runs -- independent single-object re-tracking through a
# partial cross-object mask bleed, tried before suppress_overlap_bleed's
# geometry interpolation. Real footage: a real hilt travels 150-235px
# across a multi-second overlap run, which a straight-line interpolation
# only approximates -- a validated independent re-track is strictly more
# accurate when it works.
# ---------------------------------------------------------------------------

def _write_bleed_masks(masks_0, masks_1, n_frames):
    """Object 0's own track: clean at x=20 before the run, bled onto
    object 1's position (x=60) for frames [10, 20), clean again at x=25
    after -- the exact partial-bleed shape retrack_overlap_runs exists
    to fix. Object 1 stays put at x=60 throughout, never itself bled."""
    for i in range(n_frames):
        x0 = 20 if i < 10 else (60 if i < 20 else 25)
        save_mask(str(masks_0), i, _mask_at(x0))
        save_mask(str(masks_1), i, _mask_at(60))


def _fake_track_object_threading_object_0(frames_dir, out_masks_dir, points, labels, checkpoint_path,
                                           config_name, device, n_frames, prompt_frame=0, progress_cb=None):
    """A correct independent re-track for whichever object's seed points
    this was called with: object 0's seed (x=20, from its own before-
    anchor mask) threads through x=22 during the contact and lands on its
    true recovered position (x=25); object 1's seed (x=60) stays there,
    since it was never actually displaced."""
    seed_x = points[0][0]
    for i in range(prompt_frame, n_frames):
        if seed_x < 40:
            x = 20 if i < 10 else (22 if i < 20 else 25)
        else:
            x = 60
        save_mask(out_masks_dir, i, _mask_at(x))


def test_retrack_overlap_runs_patches_masks_that_validate(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    n_frames = 30
    _write_bleed_masks(masks_0, masks_1, n_frames)

    motion_path_0 = tmp_path / "0.npz"
    motion_path_1 = tmp_path / "1.npz"
    compute_motion(str(masks_0), str(motion_path_0))
    compute_motion(str(masks_1), str(motion_path_1))

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.reacquire.track_object", _fake_track_object_threading_object_0,
    )

    patched, resolved_ranges = retrack_overlap_runs(
        "unused-frames-dir", str(masks_0), str(masks_1), str(motion_path_0), str(motion_path_1),
        n_frames, checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched == {0, 1}
    assert resolved_ranges == [(10, 19)]  # both objects validated -- fully resolved
    # Object 0's run-range masks now hold the fresh re-track (x=22), not
    # the original bled value (x=60).
    for i in range(10, 20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(22))
    # Untouched outside the run.
    for i in list(range(10)) + list(range(20, 30)):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(20 if i < 10 else 25))


def test_retrack_overlap_runs_leaves_masks_untouched_when_retrack_drifts_too_far(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    n_frames = 30
    _write_bleed_masks(masks_0, masks_1, n_frames)

    motion_path_0 = tmp_path / "0.npz"
    motion_path_1 = tmp_path / "1.npz"
    compute_motion(str(masks_0), str(motion_path_0))
    compute_motion(str(masks_1), str(motion_path_1))

    def fake_track_object_stuck_at_60(frames_dir, out_masks_dir, points, labels, checkpoint_path,
                                       config_name, device, n_frames, prompt_frame=0, progress_cb=None):
        # Every object's independent re-track drifts onto x=60 (object 1's
        # position) instead of finding its own true recovered position --
        # simulating a failed disentanglement.
        for i in range(prompt_frame, n_frames):
            save_mask(out_masks_dir, i, _mask_at(60))

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.reacquire.track_object", fake_track_object_stuck_at_60,
    )

    patched, resolved_ranges = retrack_overlap_runs(
        "unused-frames-dir", str(masks_0), str(masks_1), str(motion_path_0), str(motion_path_1),
        n_frames, checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    # Object 0's re-track (true after-position x=25) drifted to x=60 --
    # over the cap -- so it's declined. Object 1's re-track (true
    # after-position x=60) coincidentally matches x=60 exactly, so it
    # still validates.
    assert patched == {1}
    # Not fully resolved (only one of two objects validated) -- still
    # needs suppress_overlap_bleed's fallback pass.
    assert resolved_ranges == []
    for i in range(10, 20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(60))  # unchanged: still the raw bleed


def test_retrack_overlap_runs_skips_a_run_missing_an_anchor(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    n_frames = 20
    # The bleed runs from frame 0 (no "before" anchor exists at all).
    for i in range(n_frames):
        x0 = 60 if i < 10 else 25
        save_mask(str(masks_0), i, _mask_at(x0))
        save_mask(str(masks_1), i, _mask_at(60))

    motion_path_0 = tmp_path / "0.npz"
    motion_path_1 = tmp_path / "1.npz"
    compute_motion(str(masks_0), str(motion_path_0))
    compute_motion(str(masks_1), str(motion_path_1))

    def fail_if_called(*a, **k):
        raise AssertionError("track_object should not run for a run with no anchor to validate against")

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.track_object", fail_if_called)

    patched, resolved_ranges = retrack_overlap_runs(
        "unused-frames-dir", str(masks_0), str(masks_1), str(motion_path_0), str(motion_path_1),
        n_frames, checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched == set()
    assert resolved_ranges == []
    for i in range(10):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(60))


def test_retrack_overlap_runs_declines_a_run_whose_before_anchor_mask_is_empty(tmp_path, monkeypatch):
    # Confirmed on real footage (job 0eb4fda2, pair (1, 2)): an empty mask
    # at the "before" anchor frame reached `_points_on_axis`, whose PCA
    # over zero points raised an uncaught IndexError, only survived
    # because retrack_overlap_runs' outer try/except happened to catch it.
    # This declines the retrack the same clean way an empty after_mask
    # already does, before track_object (SAM2) ever runs.
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    n_frames = 30
    _write_bleed_masks(masks_0, masks_1, n_frames)
    # Object 0's before-anchor frame (9, the last frame before the run
    # starts at 10) has no foreground pixels at all.
    save_mask(str(masks_0), 9, np.zeros((80, 120), dtype=bool))

    motion_path_0 = tmp_path / "0.npz"
    motion_path_1 = tmp_path / "1.npz"
    compute_motion(str(masks_0), str(motion_path_0))
    compute_motion(str(masks_1), str(motion_path_1))

    def fail_if_called_for_object_0(frames_dir, out_masks_dir, points, labels, checkpoint_path,
                                     config_name, device, n_frames, prompt_frame=0, progress_cb=None):
        assert points[0][0] >= 40, "track_object should not run for object 0 -- its before anchor is empty"
        for i in range(prompt_frame, n_frames):
            save_mask(out_masks_dir, i, _mask_at(60))

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.reacquire.track_object", fail_if_called_for_object_0,
    )

    patched, resolved_ranges = retrack_overlap_runs(
        "unused-frames-dir", str(masks_0), str(masks_1), str(motion_path_0), str(motion_path_1),
        n_frames, checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert 0 not in patched
    assert resolved_ranges == []  # object 0 never validated, so the run isn't fully resolved
    for i in range(10, 20):
        assert np.array_equal(load_mask(str(masks_0), i), _mask_at(60))  # unchanged: still the raw bleed


def test_retrack_overlap_runs_never_raises_on_unexpected_failure(tmp_path, monkeypatch):
    masks_0 = tmp_path / "masks" / "0"
    masks_1 = tmp_path / "masks" / "1"
    n_frames = 30
    _write_bleed_masks(masks_0, masks_1, n_frames)

    motion_path_0 = tmp_path / "0.npz"
    motion_path_1 = tmp_path / "1.npz"
    compute_motion(str(masks_0), str(motion_path_0))
    compute_motion(str(masks_1), str(motion_path_1))

    def boom(*a, **k):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire._find_overlap_runs", boom)

    patched, resolved_ranges = retrack_overlap_runs(
        "unused-frames-dir", str(masks_0), str(masks_1), str(motion_path_0), str(motion_path_1),
        n_frames, checkpoint_path="ckpt", config_name="cfg", device="cpu",
    )

    assert patched == set()
    assert resolved_ranges == []
