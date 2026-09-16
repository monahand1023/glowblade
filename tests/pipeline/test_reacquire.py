import numpy as np

from lightsaber_fx.pipeline.blade import save_mask
from lightsaber_fx.pipeline.reacquire import detect_merge, find_clean_reference


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
