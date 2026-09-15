import json

import cv2
import numpy as np
import pytest

from lightsaber_fx.pipeline.vision_detect import _parse_gemini_response, _validate_box_mask

FRAME_W, FRAME_H = 1280, 720


def _response(objects):
    return json.dumps({"objects": objects})


def test_parse_gemini_response_converts_normalized_box_to_pixels():
    # ymin=100,xmin=200,ymax=300,xmax=400 on a 0-1000 scale, 1280x720 frame.
    text = _response([{"box_2d": [100, 200, 300, 400], "label": "sword"}])

    boxes = _parse_gemini_response(text, FRAME_W, FRAME_H)

    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert x0 == pytest.approx(200 / 1000 * FRAME_W)
    assert y0 == pytest.approx(100 / 1000 * FRAME_H)
    assert x1 == pytest.approx(400 / 1000 * FRAME_W)
    assert y1 == pytest.approx(300 / 1000 * FRAME_H)


def test_parse_gemini_response_handles_multiple_objects():
    text = _response([
        {"box_2d": [0, 0, 100, 100], "label": "sword"},
        {"box_2d": [200, 200, 500, 500], "label": "bat"},
    ])

    boxes = _parse_gemini_response(text, FRAME_W, FRAME_H)

    assert len(boxes) == 2


def test_parse_gemini_response_returns_empty_list_for_malformed_json():
    assert _parse_gemini_response("not json{{{", FRAME_W, FRAME_H) == []


def test_parse_gemini_response_returns_empty_list_when_objects_field_missing():
    assert _parse_gemini_response(json.dumps({"nope": []}), FRAME_W, FRAME_H) == []


def test_parse_gemini_response_returns_empty_list_for_empty_objects():
    assert _parse_gemini_response(_response([]), FRAME_W, FRAME_H) == []


def test_parse_gemini_response_drops_a_box_with_non_numeric_coordinates():
    text = _response([
        {"box_2d": ["a", "b", "c", "d"], "label": "sword"},
        {"box_2d": [0, 0, 100, 100], "label": "bat"},
    ])

    boxes = _parse_gemini_response(text, FRAME_W, FRAME_H)

    assert len(boxes) == 1  # the malformed entry is dropped, not fatal


def test_parse_gemini_response_drops_a_box_with_inverted_or_out_of_range_coordinates():
    text = _response([
        {"box_2d": [300, 100, 100, 300], "label": "inverted-y"},  # ymin > ymax
        {"box_2d": [0, 0, 1200, 100], "label": "out-of-range-x"},  # xmax > 1000
        {"box_2d": [0, 0, 100, 100], "label": "valid"},
    ])

    boxes = _parse_gemini_response(text, FRAME_W, FRAME_H)

    assert len(boxes) == 1


def test_parse_gemini_response_truncates_to_four_boxes():
    text = _response([{"box_2d": [i, i, i + 10, i + 10], "label": "x"} for i in range(6)])

    boxes = _parse_gemini_response(text, FRAME_W, FRAME_H)

    assert len(boxes) == 4


MAX_MASK_AREA = 0.25 * FRAME_W * FRAME_H


def _bar_mask(length=400, width=20):
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cv2.line(mask, (100, 100), (100 + length, 100), 1, width)
    return mask.astype(bool)


def _blob_mask(radius=60):
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cv2.circle(mask, (400, 300), radius, 1, -1)
    return mask.astype(bool)


def test_validate_box_mask_accepts_an_elongated_mask():
    result = _validate_box_mask(_bar_mask(), MAX_MASK_AREA)

    assert result is not None
    elongation, geometry = result
    assert elongation >= 6.0


def test_validate_box_mask_rejects_a_blob():
    assert _validate_box_mask(_blob_mask(), MAX_MASK_AREA) is None


def test_validate_box_mask_rejects_an_empty_mask():
    empty = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    assert _validate_box_mask(empty, MAX_MASK_AREA) is None


def test_validate_box_mask_rejects_a_mask_covering_most_of_the_frame():
    huge = np.ones((FRAME_H, FRAME_W), dtype=bool)
    assert _validate_box_mask(huge, MAX_MASK_AREA) is None


def test_validate_box_mask_rejects_a_speckled_mask():
    # Scattered single pixels: real edges a genuine object has, versus the
    # ragged shatter SAM2 returns for soft/out-of-focus background --
    # exactly the case _speckle_count exists to catch in detect.py itself.
    rng = np.random.default_rng(0)
    mask = np.zeros((FRAME_H, FRAME_W), dtype=bool)
    ys = rng.integers(0, FRAME_H, size=200)
    xs = rng.integers(0, FRAME_W, size=200)
    mask[ys, xs] = True

    assert _validate_box_mask(mask, MAX_MASK_AREA) is None
