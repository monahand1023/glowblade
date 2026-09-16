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
        {"box_2d": [0, 0, 100, 1200], "label": "out-of-range-x"},  # xmax > 1000
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


from lightsaber_fx.pipeline.vision_detect import detect_blades_vlm


class _FakeGenaiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGenaiClient:
    """Stands in for genai.Client. `response_text` is what
    models.generate_content returns; `calls` records each call's kwargs so
    a test can assert on the model name / schema used."""

    def __init__(self, response_text):
        self.response_text = response_text
        self.calls = []
        self.models = self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeGenaiResponse(self.response_text)


class _RaisingGenaiClient:
    def __init__(self, exc):
        self._exc = exc
        self.models = self

    def generate_content(self, **kwargs):
        raise self._exc


class _FakePredictor:
    """Stands in for SAM2's image predictor -- box-prompt only, matching
    what detect_blades_vlm actually calls. `masks_for_box` maps a box
    (rounded to ints, as a tuple) to the mask SAM2 would return for it."""

    def __init__(self, masks_for_box):
        self.masks_for_box = masks_for_box
        self.set_image_calls = 0

    def set_image(self, image):
        self.set_image_calls += 1

    def predict(self, box, multimask_output=False):
        key = tuple(int(round(v)) for v in box)
        mask = self.masks_for_box(key)
        return np.asarray([mask]), np.ones(1), None


def _bar_mask_at(x0, y0, x1, y1):
    mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
    cv2.rectangle(mask, (x0, y0), (x1, y1), 1, -1)
    return mask.astype(bool)


def test_detect_blades_vlm_returns_one_proposal_per_validated_box(
    monkeypatch, rotating_bar_video,
):
    text = json.dumps({"objects": [
        {"box_2d": [100, 100, 150, 900], "label": "sword"},
    ]})
    client = _FakeGenaiClient(text)
    predictor = _FakePredictor(lambda box: _bar_mask_at(*box))
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.vision_detect._build_image_predictor",
        lambda *a, **k: predictor,
    )

    proposals = detect_blades_vlm(
        str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client,
    )

    assert len(proposals) == 1
    assert proposals[0].elongation >= 6.0
    assert predictor.set_image_calls == 1  # built/set once, not per box


def test_detect_blades_vlm_calls_predict_once_per_box_on_one_predictor(
    monkeypatch, rotating_bar_video,
):
    text = json.dumps({"objects": [
        {"box_2d": [50, 50, 100, 900], "label": "a"},
        {"box_2d": [500, 50, 550, 900], "label": "b"},
    ]})
    client = _FakeGenaiClient(text)
    predictor = _FakePredictor(lambda box: _bar_mask_at(*box))
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.vision_detect._build_image_predictor",
        lambda *a, **k: predictor,
    )

    proposals = detect_blades_vlm(
        str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client,
    )

    assert len(proposals) == 2
    assert predictor.set_image_calls == 1  # still just once for both boxes


def test_detect_blades_vlm_drops_a_box_that_fails_validation(
    monkeypatch, rotating_bar_video,
):
    text = json.dumps({"objects": [
        {"box_2d": [100, 100, 150, 900], "label": "sword"},   # -> elongated bar
        {"box_2d": [400, 400, 600, 600], "label": "blob"},    # -> square blob
    ]})
    client = _FakeGenaiClient(text)

    def masks_for_box(box):
        x0, y0, x1, y1 = box
        if (x1 - x0) > (y1 - y0) * 3:  # the elongated one
            return _bar_mask_at(x0, y0, x1, y1)
        mask = np.zeros((FRAME_H, FRAME_W), dtype=np.uint8)
        cv2.circle(mask, ((x0 + x1) // 2, (y0 + y1) // 2), 60, 1, -1)
        return mask.astype(bool)

    predictor = _FakePredictor(masks_for_box)
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.vision_detect._build_image_predictor",
        lambda *a, **k: predictor,
    )

    proposals = detect_blades_vlm(
        str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client,
    )

    assert len(proposals) == 1  # the blob was dropped


def test_detect_blades_vlm_returns_empty_list_when_gemini_finds_nothing(
    rotating_bar_video,
):
    client = _FakeGenaiClient(json.dumps({"objects": []}))

    proposals = detect_blades_vlm(
        str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client,
    )

    assert proposals == []


def test_detect_blades_vlm_propagates_a_real_client_error(rotating_bar_video):
    # A network/auth failure is not "found nothing" -- it's an error the
    # caller (the /detect endpoint) needs to see, to decide whether to fall
    # back to motion-based detection. Swallowing it here would make that
    # decision invisible to the one place that needs to make it.
    client = _RaisingGenaiClient(RuntimeError("connection refused"))

    with pytest.raises(RuntimeError, match="connection refused"):
        detect_blades_vlm(str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client)


def test_detect_blades_vlm_raises_when_the_selected_frame_cannot_be_read(
    monkeypatch, rotating_bar_video,
):
    # A frame-read failure on the motion-selected frame happens before
    # Gemini is ever called -- it is a real technical failure, not "Gemini
    # searched and found nothing", so it must raise rather than return [].
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.vision_detect._read_frame", lambda cap, index: None,
    )
    client = _FakeGenaiClient(json.dumps({"objects": []}))

    with pytest.raises(ValueError, match="Could not read frame"):
        detect_blades_vlm(str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client)

    assert client.calls == []  # never got far enough to ask Gemini anything


def test_detect_blades_vlm_raises_when_the_frame_cannot_be_encoded(
    monkeypatch, rotating_bar_video,
):
    # Same reasoning as the unreadable-frame case: an imencode failure is a
    # real failure that happens before Gemini is called, not an empty result.
    monkeypatch.setattr(cv2, "imencode", lambda *a, **k: (False, None))
    client = _FakeGenaiClient(json.dumps({"objects": []}))

    with pytest.raises(RuntimeError, match="failed to encode frame"):
        detect_blades_vlm(str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client)

    assert client.calls == []  # never got far enough to ask Gemini anything


def test_detect_blades_vlm_all_proposals_share_one_frame_index(
    monkeypatch, rotating_bar_video,
):
    text = json.dumps({"objects": [
        {"box_2d": [50, 50, 100, 900], "label": "a"},
        {"box_2d": [500, 50, 550, 900], "label": "b"},
    ]})
    client = _FakeGenaiClient(text)
    predictor = _FakePredictor(lambda box: _bar_mask_at(*box))
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.vision_detect._build_image_predictor",
        lambda *a, **k: predictor,
    )

    proposals = detect_blades_vlm(
        str(rotating_bar_video), "ckpt", "cfg", "cpu", client=client,
    )

    assert len({p.frame_index for p in proposals}) == 1


import os

requires_gemini_key = pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"),
    reason="GEMINI_API_KEY not set",
)


@requires_gemini_key
def test_detect_blades_vlm_finds_multiple_swords_in_a_real_clip():
    # The one place this test file touches a real network call and real
    # SAM2 inference -- skipped everywhere except a machine with both a
    # Gemini key and the SAM2 checkpoint installed, matching how
    # test_track.py already skips its own real-SAM2 tests.
    from lightsaber_fx import paths
    from lightsaber_fx.device import select_device

    clip = "/Users/danm/Desktop/lightsaber/test-clips-mixkit/trimmed/01_knights_battling.mp4"
    if not os.path.exists(clip):
        pytest.skip("test clip not present on this machine")

    proposals = detect_blades_vlm(
        clip, str(paths.get_checkpoint_path()),
        "configs/sam2.1/sam2.1_hiera_s.yaml", select_device(),
    )

    assert len(proposals) >= 2
