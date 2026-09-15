# Vision-Assisted Multi-Object Auto-Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically find up to 4 swung props in a clip via Gemini vision, validated through the existing SAM2 + elongation gate, so busy multi-person clips populate the multi-slot picker without manual clicking -- falling back to today's single-object motion-based `detect_blade` whenever the vision path is unavailable.

**Architecture:** A new `vision_detect.py` module does one Gemini call on a motion-selected frame, parses normalized `[ymin,xmin,ymax,xmax]`/1000 boxes into pixel coordinates, and validates each through SAM2 + `fit_blade` + `MIN_ELONGATION` (reusing `detect.py`'s existing infrastructure, not duplicating it). The web `/detect` endpoint tries it first and falls back to `detect_blade` on any failure. The frontend generalizes its single `detectOverlay` to a per-slot field so N proposals populate N slots.

**Tech Stack:** Python, `google-genai` SDK (new dependency), existing SAM2/OpenCV pipeline, FastAPI, vanilla JS.

**Spec:** `docs/superpowers/specs/2026-09-15-vision-assisted-multi-object-detection-design.md`

## Global Constraints

- Gemini model: `gemini-3.8-flash` (the SDK's own current default/recommended model).
- Gemini coordinate format: response schema requests `box_2d: [ymin, xmin, ymax, xmax]` normalized to 0-1000 -- convert to absolute pixel `[x0, y0, x1, y1]` immediately after parsing; nothing past that conversion point works in normalized coordinates.
- Auth: `GEMINI_API_KEY` or `GOOGLE_API_KEY` read by the SDK itself from the environment. Never stored or handled by this codebase.
- Max 4 proposals per detection call (matches `MAX_SABERS` in `app.js`).
- `detect.py` is never modified by this plan -- it is the fallback, unchanged. Only its already-public helpers (`propose_motion_seeds`, `_read_frame`, `_build_image_predictor`, `_points_on_axis`, `_speckle_count`, `MIN_ELONGATION`, `MAX_MASK_AREA_FRAC`, `BladeProposal`) are imported and reused.
- SAM2 image predictor: build once and call `.set_image()` once per detection call; call `.predict(box=...)` once per candidate box on that same instance. Never rebuild the predictor per box.
- Partial success (found 1-3 of N actual objects) is still success -- never mixed with a `detect_blade` fallback in the same response. Fallback triggers only on zero validated VLM proposals or an exception.

---

### Task 1: `vision_detect.py` core -- response parsing and box validation

**Files:**
- Create: `src/lightsaber_fx/pipeline/vision_detect.py`
- Test: `tests/pipeline/test_vision_detect.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `_parse_gemini_response(response_text, frame_width, frame_height) -> list[list[float]]` (0-4 `[x0,y0,x1,y1]` pixel boxes), `_validate_box_mask(mask, max_mask_area) -> tuple[float, BladeGeometry] | None`, `DETECTION_PROMPT: str`, `DETECTION_SCHEMA: dict`.
- Consumes: `lightsaber_fx.pipeline.blade.fit_blade`, `lightsaber_fx.pipeline.detect._speckle_count`, `lightsaber_fx.pipeline.detect.MIN_ELONGATION`.

- [ ] **Step 1: Add the `google-genai` dependency**

Edit `pyproject.toml`'s `dependencies` list (currently ends `..., "platformdirs",`):

```toml
dependencies = [
    "opencv-python",
    "numpy",
    "scipy",
    "soundfile",
    "torch",
    "torchvision",
    "click",
    "fastapi",
    "uvicorn[standard]",
    "python-multipart",
    "platformdirs",
    "google-genai",
]
```

Run: `pip install -e .` (from the project's venv) and confirm `python -c "from google import genai"` succeeds with no error.

- [ ] **Step 2: Write the failing tests for `_parse_gemini_response`**

Create `tests/pipeline/test_vision_detect.py`:

```python
import json

import numpy as np
import pytest

from lightsaber_fx.pipeline.vision_detect import _parse_gemini_response

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
```

- [ ] **Step 2b: Run the tests to verify they fail correctly**

Run: `pytest tests/pipeline/test_vision_detect.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lightsaber_fx.pipeline.vision_detect'` (or `ImportError`) -- the module doesn't exist yet.

- [ ] **Step 3: Write `_parse_gemini_response`**

Create `src/lightsaber_fx/pipeline/vision_detect.py`:

```python
"""Vision-assisted multi-object detection: ask Gemini to find every swung
prop in one frame, then validate each candidate through the same SAM2 +
fit_blade quality gate `detect.py`'s own motion-based detection already
uses. Falls back to `detect.py`'s `detect_blade` (unmodified, untouched by
this module) whenever the vision path can't be used -- see `detect_blades_vlm`.

Different failure modes than `detect.py` -- network errors, API keys,
per-call cost -- so this lives in its own file rather than inside it.
"""

import json

import numpy as np

from .blade import fit_blade
from .detect import MAX_SPECKLES, MIN_ELONGATION, _speckle_count

DETECTION_PROMPT = (
    "Find every sword, bat, staff, or similar elongated object that a "
    "person in this image is holding or actively swinging. For each one, "
    "return a bounding box drawn tightly around just the object itself -- "
    "not the hand or arm holding it, and not the whole person. Ignore "
    "sheathed or holstered weapons, shields, and anything lying on the "
    "ground rather than being held. If you see none, return an empty list."
)

# Gemini's spatial grounding is trained on this exact shape: box_2d is
# [ymin, xmin, ymax, xmax], normalized to an integer 0-1000 scale regardless
# of the source image's actual resolution -- confirmed against Google's own
# documentation, not assumed. _parse_gemini_response converts to absolute
# pixel [x0,y0,x1,y1] immediately; nothing past that point knows this
# normalized format exists.
DETECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "box_2d": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "label": {"type": "string"},
                },
                "required": ["box_2d", "label"],
            },
        },
    },
    "required": ["objects"],
}

MAX_PROPOSALS = 4


def _parse_gemini_response(response_text, frame_width, frame_height):
    """Parse Gemini's JSON response into up to `MAX_PROPOSALS` absolute-
    pixel `[x0, y0, x1, y1]` boxes.

    Malformed JSON, a missing/wrong-shaped `objects` field, or one bad box
    (non-numeric, inverted, or out-of-range coordinates) is dropped rather
    than raised -- one bad entry from the model shouldn't discard the
    other, valid ones. Boxes beyond `MAX_PROPOSALS` are truncated (the
    model isn't asked to rank them, so "first N" is as good a cut as any).
    """
    try:
        data = json.loads(response_text)
    except (json.JSONDecodeError, TypeError):
        return []

    objects = data.get("objects") if isinstance(data, dict) else None
    if not isinstance(objects, list):
        return []

    boxes = []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        box = obj.get("box_2d")
        if not (isinstance(box, list) and len(box) == 4):
            continue
        try:
            ymin, xmin, ymax, xmax = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        if not (0 <= ymin < ymax <= 1000 and 0 <= xmin < xmax <= 1000):
            continue
        boxes.append([
            xmin / 1000 * frame_width,
            ymin / 1000 * frame_height,
            xmax / 1000 * frame_width,
            ymax / 1000 * frame_height,
        ])
        if len(boxes) == MAX_PROPOSALS:
            break
    return boxes
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/pipeline/test_vision_detect.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Write the failing tests for `_validate_box_mask`**

Append to `tests/pipeline/test_vision_detect.py`:

```python
import cv2

from lightsaber_fx.pipeline.vision_detect import _validate_box_mask

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
```

- [ ] **Step 6: Run to verify failure**

Run: `pytest tests/pipeline/test_vision_detect.py -v -k validate_box_mask`
Expected: FAIL with `ImportError: cannot import name '_validate_box_mask'`.

- [ ] **Step 7: Write `_validate_box_mask`**

Append to `src/lightsaber_fx/pipeline/vision_detect.py`:

```python
def _validate_box_mask(mask, max_mask_area):
    """Shape/size checks for a SAM2 mask segmented from a VLM-proposed box.

    The box-prompt analog of `detect.py`'s `_candidate_masks` -- but only
    the gates that don't need optical-flow data apply: absolute area
    bounds, `_speckle_count` (real edges vs. a ragged shatter), and
    `MIN_ELONGATION`. `_candidate_masks`' other two gates
    (`MAX_MASK_TO_MOTION_RATIO`, `_moving_fraction` against a seed's flow
    map) are keyed to a `MotionSeed` a VLM box simply doesn't have, so they
    don't apply here -- this is a deliberately smaller gate, not a bug.

    Returns `(elongation, geometry)` if the mask passes, `None` if not.
    """
    area = int(mask.sum())
    if area == 0 or area > max_mask_area:
        return None
    if _speckle_count(mask) > MAX_SPECKLES:
        return None
    geometry = fit_blade(mask)
    if geometry is None:
        return None
    elongation = geometry.length / max(geometry.width, 1.0)
    if elongation < MIN_ELONGATION:
        return None
    return elongation, geometry
```

- [ ] **Step 8: Run all vision_detect tests to verify they pass**

Run: `pytest tests/pipeline/test_vision_detect.py -v`
Expected: All 12 tests PASS.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/lightsaber_fx/pipeline/vision_detect.py tests/pipeline/test_vision_detect.py
git commit -m "Add vision_detect.py response parsing and box validation"
```

---

### Task 2: `detect_blades_vlm` orchestration

**Files:**
- Modify: `src/lightsaber_fx/pipeline/vision_detect.py`
- Modify: `tests/pipeline/test_vision_detect.py`

**Interfaces:**
- Consumes: Task 1's `_parse_gemini_response`, `_validate_box_mask`, `DETECTION_PROMPT`, `DETECTION_SCHEMA`; `detect.py`'s `propose_motion_seeds`, `_read_frame`, `_build_image_predictor`, `_points_on_axis`, `MAX_MASK_AREA_FRAC`, `BladeProposal`.
- Produces: `detect_blades_vlm(video_path, checkpoint_path, config_name, device, client=None) -> list[BladeProposal]`, raising on real errors (network, auth, missing SDK) rather than swallowing them -- the caller (Task 3's `/detect` endpoint) decides fallback behavior, matching how `detect_blade` already lets a truly unreadable video raise `ValueError` rather than hiding it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/pipeline/test_vision_detect.py`:

```python
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
```

`rotating_bar_video` is the existing fixture from `tests/pipeline/test_detect.py` (a synthetic clip with real, sampleable motion) -- confirm it's defined in a shared `conftest.py`; if it's only in `test_detect.py`'s own module scope, move it to `tests/pipeline/conftest.py` so both test files can use it without duplication.

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/pipeline/test_vision_detect.py -v -k detect_blades_vlm`
Expected: FAIL with `ImportError: cannot import name 'detect_blades_vlm'`.

- [ ] **Step 3: Write `detect_blades_vlm`**

Append to `src/lightsaber_fx/pipeline/vision_detect.py`, and add these imports at the top alongside the existing ones:

```python
import cv2
import numpy as np

from .blade import fit_blade
from .detect import (
    MAX_MASK_AREA_FRAC,
    MAX_SPECKLES,
    MIN_ELONGATION,
    BladeProposal,
    _build_image_predictor,
    _points_on_axis,
    _read_frame,
    _speckle_count,
    propose_motion_seeds,
)
```

```python
GEMINI_MODEL = "gemini-3.8-flash"


def detect_blades_vlm(video_path, checkpoint_path, config_name, device, client=None):
    """Ask Gemini to find every held, swung prop in one representative
    frame, then validate each candidate through SAM2 + fit_blade. Returns
    0-4 `BladeProposal` (the same type `detect.detect_blade` returns),
    sorted by elongation descending.

    Raises on a real failure (network error, bad auth, the SDK not
    installed) rather than swallowing it -- the caller decides whether to
    fall back to motion-based detection; this function's only "normal, not
    an error" empty case is Gemini genuinely finding nothing.

    `client` is injectable (a `genai.Client`, or a test double) so tests
    never make a real network call -- same pattern as `detect.py`'s own
    `_build_image_predictor`: "Isolated so tests can replace the whole
    dependency in one place."
    """
    if client is None:
        from google import genai
        client = genai.Client()

    from google.genai import types

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not read a frame from {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Not a blind middle-frame guess: reuse the same motion-scoring
        # detect_blade itself relies on (ranked fastest-first) to pick a
        # frame where the action is actually visible, not one that happens
        # to land on motion blur or an occlusion. This only uses the
        # winning seed's *frame*, not its point -- Gemini gets the whole
        # frame, no seed information leaks into the vision prompt.
        seeds = propose_motion_seeds(str(video_path))
        frame_index = seeds[0].frame_index if seeds else 0
        frame = _read_frame(cap, frame_index)
        if frame is None:
            return []

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return []

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"),
                DETECTION_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=DETECTION_SCHEMA,
            ),
        )
        boxes = _parse_gemini_response(response.text, width, height)
        if not boxes:
            return []

        max_mask_area = MAX_MASK_AREA_FRAC * width * height
        predictor = _build_image_predictor(checkpoint_path, config_name, device)
        predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        proposals = []
        for box in boxes:
            masks, _scores, _logits = predictor.predict(
                box=np.array(box, dtype=np.float32), multimask_output=False,
            )
            mask = np.asarray(masks)[0].astype(bool)
            result = _validate_box_mask(mask, max_mask_area)
            if result is None:
                continue
            elongation, _geometry = result
            points = _points_on_axis(mask)
            proposals.append(BladeProposal(
                frame_index=frame_index,
                points=points,
                labels=[1] * len(points),
                mask=mask,
                elongation=elongation,
                seed=None,  # no MotionSeed -- this proposal came from a VLM box, not motion
            ))
    finally:
        cap.release()

    proposals.sort(key=lambda p: p.elongation, reverse=True)
    return proposals
```

Note `DETECTION_PROMPT`/`DETECTION_SCHEMA` are already defined in this file from Task 1; this step only adds the new imports and `detect_blades_vlm`/`GEMINI_MODEL`.

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/pipeline/test_vision_detect.py -v`
Expected: All 18 tests PASS.

- [ ] **Step 5: Add the gated real-API smoke test**

Append to `tests/pipeline/test_vision_detect.py`:

```python
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
```

Run (only meaningful with a real key and checkpoint present): `GEMINI_API_KEY=... pytest tests/pipeline/test_vision_detect.py -v -k real_clip`
Expected: PASS, or a clean `SKIPPED` everywhere else -- confirm the skip fires with no key set: `pytest tests/pipeline/test_vision_detect.py -v -k real_clip` (no env var) shows `SKIPPED`.

- [ ] **Step 6: Full pipeline test suite**

Run: `pytest tests/ -q`
Expected: All tests pass (the new ones plus the full existing suite, unaffected).

- [ ] **Step 7: Commit**

```bash
git add src/lightsaber_fx/pipeline/vision_detect.py tests/pipeline/test_vision_detect.py
git commit -m "Add detect_blades_vlm orchestration"
```

---

### Task 3: Wire vision detection into the `/detect` endpoint

**Files:**
- Modify: `src/lightsaber_fx/web/server.py`
- Modify: `tests/web/test_server.py`

**Interfaces:**
- Consumes: Task 2's `detect_blades_vlm`; existing `detect_blade`.
- Produces: `/detect`'s response shape changes to `{"found": bool, "frame_index": int, "frame_url": str, "proposals": [{"points", "elongation", "mask_url"}, ...], "source": "vlm"|"motion", "width": int, "height": int}`; new `GET /api/jobs/{id}/detect-mask/{index}`.
- **Breaking change to 4 existing tests** (`test_detect_returns_a_proposal_with_its_frame_and_points`, `test_detect_serves_the_frame_the_points_refer_to_and_a_mask_overlay`, `test_detect_mask_overlay_is_transparent_outside_the_mask`, `test_detect_reports_not_found_without_erroring`) -- they assert on today's flat single-proposal shape and must be rewritten, not left alongside the new ones.

- [ ] **Step 1: Update the existing detect tests for the new response shape**

In `tests/web/test_server.py`, replace the 4 tests named above (currently around lines 488-556) with:

```python
def test_detect_returns_a_proposal_with_its_frame_and_points(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([_fake_proposal(3)], "motion"))
    job_id = _upload(client, tiny_video_bytes)

    resp = client.post(f"/api/jobs/{job_id}/detect")

    assert resp.status_code == 200
    data = resp.json()
    assert data["found"] is True
    assert data["frame_index"] == 3
    assert data["source"] == "motion"
    assert len(data["proposals"]) == 1
    proposal = data["proposals"][0]
    assert proposal["elongation"] == 8.4
    # Points come back in the same [x, y, label] shape /points takes, all
    # includes -- detection never proposes carving anything out.
    assert proposal["points"] == [[12, 24, 1], [32, 24, 1], [52, 24, 1]]


def test_detect_returns_multiple_proposals_from_the_vision_path(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(
        server_module, "_detect_proposals",
        lambda *a, **k: ([_fake_proposal(3), _fake_proposal(3)], "vlm"),
    )
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    assert data["found"] is True
    assert data["source"] == "vlm"
    assert len(data["proposals"]) == 2


def test_detect_serves_the_frame_the_points_refer_to_and_a_mask_overlay(
    client, tiny_video_bytes, monkeypatch
):
    # The frame matters as much as the points: a proposal from mid-swing is
    # meaningless drawn over frame 0, because the object has moved.
    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([_fake_proposal(3)], "motion"))
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    frame_resp = client.get(data["frame_url"])
    assert frame_resp.status_code == 200
    assert frame_resp.headers["content-type"] == "image/jpeg"

    mask_resp = client.get(data["proposals"][0]["mask_url"])
    assert mask_resp.status_code == 200
    assert mask_resp.headers["content-type"] == "image/png"


def test_detect_mask_overlay_is_transparent_outside_the_mask(
    client, tiny_video_bytes, monkeypatch
):
    # The page composites this over the frame, so anything outside the mask
    # must be fully transparent or it paints over the footage.
    import cv2
    import numpy as np

    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([_fake_proposal()], "motion"))
    job_id = _upload(client, tiny_video_bytes)
    client.post(f"/api/jobs/{job_id}/detect")

    overlay = cv2.imread(
        str(paths_module.get_jobs_dir() / job_id / "detect_mask_0.png"), cv2.IMREAD_UNCHANGED
    )
    assert overlay.shape[2] == 4, "overlay has no alpha channel"
    assert overlay[24, 32, 3] > 0, "masked pixels are transparent"
    assert overlay[5, 5, 3] == 0, "unmasked pixels are not transparent"
    assert np.count_nonzero(overlay[:, :, 3]) == 4 * 48


def test_detect_reports_not_found_without_erroring(client, tiny_video_bytes, monkeypatch):
    # "I couldn't find it" is a normal outcome, not a failure: the page falls
    # back to asking the user to click, which is what it did before detection
    # existed. Returning an error status would surface a scary message for
    # something entirely expected.
    monkeypatch.setattr(server_module, "_detect_proposals", lambda *a, **k: ([], "motion"))
    job_id = _upload(client, tiny_video_bytes)

    resp = client.post(f"/api/jobs/{job_id}/detect")

    assert resp.status_code == 200
    assert resp.json() == {"found": False}


def test_detect_falls_back_to_motion_when_vlm_raises(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(
        server_module, "detect_blades_vlm",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no api key")),
    )
    monkeypatch.setattr(server_module, "detect_blade", lambda *a, **k: _fake_proposal(3))
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    assert data["found"] is True
    assert data["source"] == "motion"
    assert len(data["proposals"]) == 1


def test_detect_falls_back_to_motion_when_vlm_finds_nothing(
    client, tiny_video_bytes, monkeypatch
):
    monkeypatch.setattr(server_module, "detect_blades_vlm", lambda *a, **k: [])
    monkeypatch.setattr(server_module, "detect_blade", lambda *a, **k: _fake_proposal(3))
    job_id = _upload(client, tiny_video_bytes)

    data = client.post(f"/api/jobs/{job_id}/detect").json()

    assert data["found"] is True
    assert data["source"] == "motion"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/web/test_server.py -v -k detect`
Expected: FAIL -- `server_module` has no attribute `_detect_proposals` (the rewritten tests reference a function that doesn't exist yet), and the unmodified `/detect` endpoint still returns the old flat shape.

- [ ] **Step 3: Rewrite the `/detect` endpoint**

In `src/lightsaber_fx/web/server.py`, add the import and replace the `detect` function:

```python
from ..pipeline.detect import _build_image_predictor, detect_blade
from ..pipeline.vision_detect import detect_blades_vlm
```

Replace the existing `detect` function (currently `@app.post("/api/jobs/{job_id}/detect") def detect(job_id: str): ...`) with:

```python
def _detect_proposals(input_path, device):
    """Try vision-assisted multi-object detection first, falling back to
    today's single-object motion-based detect_blade on any failure --
    missing API key, network error, bad response, or zero validated
    candidates. Returns `(list[BladeProposal], source)` where `source` is
    `"vlm"` or `"motion"`.

    Partial success is still success: if the VLM finds 2 of 3 actual
    objects, those 2 are returned as-is -- this never tops up a VLM result
    with a motion-detected one, since they could disagree about which
    frame to use and every saber in one render must share a prompt_frame.
    """
    try:
        proposals = detect_blades_vlm(
            input_path, str(paths.get_checkpoint_path()),
            "configs/sam2.1/sam2.1_hiera_s.yaml", device,
        )
        if proposals:
            return proposals, "vlm"
    except Exception:
        logging.getLogger(__name__).warning(
            "vision-assisted detection failed, falling back to motion detection", exc_info=True,
        )

    proposal = detect_blade(
        input_path, str(paths.get_checkpoint_path()),
        "configs/sam2.1/sam2.1_hiera_s.yaml", device,
    )
    return ([proposal] if proposal else []), "motion"


@app.post("/api/jobs/{job_id}/detect")
def detect(job_id: str):
    """Look for the swung object(s) and return proposals to confirm.

    Deliberately a *sync* route: detection runs (vision or motion) plus one
    or more SAM2 passes, several seconds of blocking CPU/network work, and
    FastAPI runs sync routes in a threadpool rather than on the event loop.
    Declaring this `async def` would stall every other request, including
    the progress stream, for the duration.

    Writes one clean frame plus one RGBA mask overlay per proposal into the
    job dir rather than returning pixels inline. The frame matters -- a
    proposal's points are meaningless against frame 0, since the object has
    moved by then.
    """
    _validate_job_id(job_id)
    job_dir = paths.get_jobs_dir() / job_id
    input_path = job_dir / "input.mp4"
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    if not paths.get_checkpoint_path().exists():
        raise HTTPException(
            status_code=400,
            detail="SAM2 is not installed yet — run `lightsaber-fx setup` first.",
        )

    proposals, source = _detect_proposals(str(input_path), select_device())
    if not proposals:
        return {"found": False}

    frame_index = proposals[0].frame_index  # all proposals share one frame -- see Global Constraints
    try:
        extract_frame_at(
            str(input_path), frame_index, str(job_dir / "detect_frame.jpg")
        )
    except ValueError:
        # Detection read this same file, so a frame it named should always be
        # seekable -- but a container whose index disagrees with its actual
        # frames is a real thing, and "found nothing" leaves the user clicking
        # the object as they would have anyway. A 500 here would instead break
        # a page that has a perfectly good fallback.
        return {"found": False}

    height, width = proposals[0].mask.shape[:2]
    result_proposals = []
    for i, proposal in enumerate(proposals):
        (job_dir / f"detect_mask_{i}.png").write_bytes(_mask_overlay_png(proposal.mask))
        result_proposals.append({
            "elongation": round(proposal.elongation, 1),
            "points": [[x, y, 1] for x, y in proposal.points],
            "mask_url": f"/api/jobs/{job_id}/detect-mask/{i}",
        })

    return {
        "found": True,
        "frame_index": frame_index,
        "frame_url": f"/api/jobs/{job_id}/detect-frame",
        "proposals": result_proposals,
        "source": source,
        "width": width,
        "height": height,
    }
```

Add `import logging` to the top imports if not already present.

- [ ] **Step 4: Replace the single-mask `/detect-mask` endpoint with an indexed one**

Replace the existing `get_detect_mask` function:

```python
@app.get("/api/jobs/{job_id}/detect-mask/{index}")
def get_detect_mask(job_id: str, index: int):
    _validate_job_id(job_id)
    if not 0 <= index < 4:
        raise HTTPException(status_code=404, detail="Job not found")
    path = paths.get_jobs_dir() / job_id / f"detect_mask_{index}.png"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Job not found")
    return FileResponse(path, media_type="image/png")
```

- [ ] **Step 5: Run to verify all detect tests pass**

Run: `pytest tests/web/test_server.py -v -k detect`
Expected: All PASS.

- [ ] **Step 6: Full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add src/lightsaber_fx/web/server.py tests/web/test_server.py
git commit -m "Wire vision-assisted detection into /detect with motion fallback"
```

---

### Task 4: Thread `source` through job bookkeeping and `inspect`

**Files:**
- Modify: `src/lightsaber_fx/web/server.py` (`_parse_saber_specs`)
- Modify: `src/lightsaber_fx/pipeline/runner.py` (`run_pipeline_multi`)
- Modify: `src/lightsaber_fx/pipeline/inspect_job.py`
- Modify: `src/lightsaber_fx/cli.py` (`inspect` command's print loop)
- Modify: `tests/pipeline/test_runner.py`, `tests/pipeline/test_inspect_job.py`

**Interfaces:**
- Consumes: nothing new externally.
- Produces: each saber spec dict gains an optional `"source"` key (default `"manual"`), threaded from the `/points` request body all the way into `job_meta.json`'s `prompts` list, and printed by `lightsaber-fx inspect`.

- [ ] **Step 1: Write the failing test for `_parse_saber_specs`**

In `tests/web/test_server.py`, find the existing tests for `_parse_saber_specs` (or `/points` validation) and add:

```python
def test_parse_saber_specs_defaults_source_to_manual():
    from lightsaber_fx.web.server import _parse_saber_specs

    parsed = _parse_saber_specs({"sabers": [
        {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral"},
    ]})

    assert parsed[0]["source"] == "manual"


def test_parse_saber_specs_passes_through_a_given_source():
    from lightsaber_fx.web.server import _parse_saber_specs

    parsed = _parse_saber_specs({"sabers": [
        {"points": [[10, 10, 1]], "color": "red", "intensity": 0.35, "voice": "neutral", "source": "vlm"},
    ]})

    assert parsed[0]["source"] == "vlm"
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/web/test_server.py -v -k parse_saber_specs_defaults_source`
Expected: FAIL with `KeyError: 'source'`.

- [ ] **Step 3: Update `_parse_saber_specs`**

In `src/lightsaber_fx/web/server.py`, inside `_parse_saber_specs`'s per-saber loop, add one line to the `parsed.append({...})` dict:

```python
        parsed.append({
            "points": [[p[0], p[1]] for p in points_and_labels],
            "labels": [p[2] for p in points_and_labels],
            "color": color,
            "intensity": intensity,
            "voice": voice,
            "prompt_frame": prompt_frame,
            "source": saber.get("source", "manual"),
        })
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/web/test_server.py -v -k parse_saber_specs`
Expected: PASS.

- [ ] **Step 5: Write the failing test for `run_pipeline_multi` threading `source` into `job_meta`**

In `tests/pipeline/test_runner.py`, add (near the other `run_pipeline_multi` + `job_meta` tests):

```python
def test_run_pipeline_multi_records_each_sabers_source_in_job_meta(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35,
             "voice": "neutral", "source": "vlm"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35,
             "voice": "neutral", "source": "manual"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    meta = job_meta.read_job_meta(str(job_dir))
    assert [p["source"] for p in meta["prompts"]] == ["vlm", "manual"]
```

- [ ] **Step 6: Run to verify failure**

Run: `pytest tests/pipeline/test_runner.py -v -k records_each_sabers_source`
Expected: FAIL with `KeyError: 'source'`.

- [ ] **Step 7: Update `run_pipeline_multi`**

In `src/lightsaber_fx/pipeline/runner.py`, find the `job_meta.write_job_meta` call inside `run_pipeline_multi` (its `prompts=` list comprehension) and add `"source"`:

```python
    job_meta.write_job_meta(
        job_dir, source_video=input_video, object_ids=object_ids,
        prompts=[
            {
                "points": s["points"], "labels": s["labels"],
                "prompt_frame": s.get("prompt_frame", 0),
                "source": s.get("source", "manual"),
            }
            for s in sabers
        ],
    )
```

- [ ] **Step 8: Run to verify it passes**

Run: `pytest tests/pipeline/test_runner.py -v -k records_each_sabers_source`
Expected: PASS.

- [ ] **Step 9: Write the failing test for the CLI surfacing `source`**

**There is no existing `inspect` command test to extend** -- `tests/test_cli.py` currently has zero coverage of the `inspect` command (confirmed: `grep -n "inspect" tests/test_cli.py` matches nothing). This step adds the first one, modeled on
`test_jobs_command_shows_the_object_count_for_a_multi_object_job`'s job-directory-setup style (the closest existing analog -- also builds a real multi-object job dir by hand).

`inspect_job()` needs a readable frame: `_frame_size` tries `frame0.jpg` first and raises `ValueError` if that's absent and `input.mp4` isn't readable either, so this fixture writes a real (tiny) `frame0.jpg` via `cv2.imwrite` rather than the `b"fake"` placeholder bytes the `jobs`-command tests get away with (that command never reads the video, only checks the source path exists).

Add `import cv2` to `tests/test_cli.py`'s imports, then add:

```python
def test_inspect_command_prints_each_sabers_source(tmp_path, monkeypatch, capsys):
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr("lightsaber_fx.cli.paths.get_jobs_dir", lambda: jobs_dir)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    job_dir = jobs_dir / "job1"
    job_dir.mkdir(parents=True)
    cv2.imwrite(str(job_dir / "frame0.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    for oid in (0, 1):
        save_mask(str(job_dir / "masks" / str(oid)), 0, mask)
        (job_dir / "motion").mkdir(exist_ok=True)
        compute_motion(str(job_dir / "masks" / str(oid)), str(job_dir / "motion" / f"{oid}.npz"))
    (job_dir / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(
        str(job_dir), source_video=str(video), object_ids=[0, 1],
        prompts=[
            {"points": [[1, 1]], "labels": [1], "prompt_frame": 0, "source": "vlm"},
            {"points": [[2, 2]], "labels": [1], "prompt_frame": 0, "source": "manual"},
        ],
    )

    result = CliRunner().invoke(main, ["inspect", "job1"])

    assert result.exit_code == 0, result.output
    assert "saber 0: source=vlm" in result.output
    assert "saber 1: source=manual" in result.output
```

- [ ] **Step 9b: Run to verify failure**

Run: `pytest tests/test_cli.py -v -k inspect_command_prints_each_sabers_source`
Expected: FAIL -- the current print line has no `source=` segment, so the assertion on exact output text fails (not an exception; `cli.py` hasn't been touched yet in this task).

- [ ] **Step 10: Update `cli.py`'s `inspect` command**

In `src/lightsaber_fx/cli.py`, find the `inspect` command's prompt-printing loop:

```python
    if prompts:
        click.echo("  prompts submitted:")
        for i, p in enumerate(prompts):
            click.echo(
                f"    saber {i}: prompt_frame={p.get('prompt_frame', 0)} "
                f"points={p.get('points')} labels={p.get('labels')}"
            )
```

Change to:

```python
    if prompts:
        click.echo("  prompts submitted:")
        for i, p in enumerate(prompts):
            click.echo(
                f"    saber {i}: source={p.get('source', 'manual')} "
                f"prompt_frame={p.get('prompt_frame', 0)} "
                f"points={p.get('points')} labels={p.get('labels')}"
            )
```

- [ ] **Step 11: Run the new test and the full suite**

Run: `pytest tests/test_cli.py -v -k inspect` then `pytest tests/ -q`
Expected: All PASS.

- [ ] **Step 12: Commit**

```bash
git add src/lightsaber_fx/web/server.py src/lightsaber_fx/pipeline/runner.py src/lightsaber_fx/cli.py tests/web/test_server.py tests/pipeline/test_runner.py tests/test_cli.py
git commit -m "Thread detection source (vlm/motion/manual) through job bookkeeping"
```

---

### Task 5: Frontend -- multi-slot auto-population

**Files:**
- Modify: `src/lightsaber_fx/web/static/app.js`

**Interfaces:**
- Consumes: Task 3's `/detect` response shape (`proposals`, `source`).
- Produces: `detect()` populates 1-4 saber slots instead of always writing into `sabers[0]`; `detectOverlay` becomes a per-saber `detectedMask` field; `pickerHint` explains a motion-fallback result.

No automated test coverage for this file -- matches this codebase's established pattern (manual browser verification only, per every `app.js` change earlier this session).

- [ ] **Step 1: Generalize `detectOverlay` to a per-slot field**

In `makeSaberSlot`, add `detectedMask: null`:

```javascript
function makeSaberSlot(color) {
  return { points: [], color: color || "red", intensity: 0.35, voice: "neutral", detectedMask: null };
}
```

Remove the module-level `let detectOverlay = null;` declaration and its accompanying comment block entirely.

- [ ] **Step 2: Update `redrawPoints`'s overlay branch**

Replace:

```javascript
  if (activeSaberIndex === 0 && detectOverlay) {
    ctx.drawImage(detectOverlay, 0, 0, canvas.width, canvas.height);
  } else if (selectionPreview) {
    ctx.drawImage(selectionPreview, 0, 0, canvas.width, canvas.height);
  }
```

with:

```javascript
  const activeDetectedMask = sabers[activeSaberIndex].detectedMask;
  if (activeDetectedMask) {
    ctx.drawImage(activeDetectedMask, 0, 0, canvas.width, canvas.height);
  } else if (selectionPreview) {
    ctx.drawImage(selectionPreview, 0, 0, canvas.width, canvas.height);
  }
```

- [ ] **Step 3: Update `setActiveSaber`'s detectOverlay reference**

Replace:

```javascript
  } else if (!(i === 0 && detectOverlay)) {
    pickerHint.textContent = MANUAL_HINT;
  }
```

with:

```javascript
  } else if (!sabers[i].detectedMask) {
    pickerHint.textContent = MANUAL_HINT;
  }
```

- [ ] **Step 4: Update the `mouseup` handler's discard-on-first-gesture logic**

Replace:

```javascript
  if (activeSaberIndex === 0 && detectOverlay) {
    detectOverlay = null;
    saber.points = [];
    pickerHint.textContent = MANUAL_HINT;
  }
```

with:

```javascript
  if (saber.detectedMask) {
    saber.detectedMask = null;
    saber.source = "manual";
    saber.points = [];
    pickerHint.textContent = MANUAL_HINT;
  }
```

(This also generalizes the *slot* the discard applies to -- from hardcoded slot 0 to whichever slot is active, since any slot can now carry a detected proposal.)

- [ ] **Step 5: Add a `source` field to `makeSaberSlot` and `renderRequestBody`**

Extend Step 1's `makeSaberSlot`:

```javascript
function makeSaberSlot(color) {
  return {
    points: [], color: color || "red", intensity: 0.35, voice: "neutral",
    detectedMask: null, source: "manual",
  };
}
```

In `renderRequestBody`, add `source` to each mapped saber:

```javascript
function renderRequestBody() {
  return {
    sabers: sabers.map((saber) => ({
      points: saber.points,
      prompt_frame: promptFrame,
      color: saber.color,
      intensity: saber.intensity,
      voice: saber.voice,
      source: saber.source,
    })),
    blade_extend: bladeExtendInput.checked,
  };
}
```

- [ ] **Step 6: Rewrite `detect()` to populate multiple slots**

Replace the whole `detect` function:

```javascript
async function detect() {
  const startedFor = jobId;
  pickerHint.textContent = "Looking for the swung object...";
  let data;
  try {
    const resp = await fetch(`/api/jobs/${jobId}/detect`, { method: "POST" });
    if (!resp.ok) throw new Error("detect failed");
    data = await resp.json();
  } catch {
    pickerHint.textContent = MANUAL_HINT;
    return;
  }
  // The user may have dropped another file while this was running.
  if (startedFor !== jobId) return;
  if (!data.found) {
    pickerHint.textContent = `Couldn't find it automatically. ${MANUAL_HINT}`;
    return;
  }

  promptFrame = data.frame_index;
  const masks = await Promise.all(data.proposals.map((p) => loadImage(p.mask_url)));
  if (startedFor !== jobId) return;

  sabers = data.proposals.map((proposal, i) => ({
    ...makeSaberSlot(DEFAULT_SABER_COLORS[i]),
    points: proposal.points,
    detectedMask: masks[i],
    source: data.source,
  }));
  activeSaberIndex = 0;

  await showFrame(data.frame_url, data.width, data.height);
  syncControlsToActiveSaber();
  renderSaberSlots();

  if (data.source === "vlm") {
    pickerHint.textContent = data.proposals.length > 1
      ? `Found ${data.proposals.length} objects via AI. Render them, or click any object yourself to override.`
      : "Found it via AI. Render it, or click the object yourself to override.";
  } else {
    pickerHint.textContent =
      `Found 1 object via motion detection (AI detection unavailable). ` +
      "Render it, add more sabers manually if there are others, or click to override.";
  }
}
```

- [ ] **Step 7: Manual verification in the browser**

Start the dev server (`lightsaber-fx serve --port 8010`, or confirm it's already running from earlier tonight and restart it to pick up the server.py changes). In an actual browser:

1. With `GEMINI_API_KEY` set in the server's environment: upload one of the Mixkit knights-battling clips (`/Users/danm/Desktop/lightsaber/test-clips-mixkit/trimmed/`). Confirm multiple slots auto-populate, each with its own visible mask overlay in the slot's own color once selected, and the hint text says "Found N objects via AI."
2. With `GEMINI_API_KEY` unset: upload the same clip. Confirm exactly one slot populates (today's existing single-object behavior) with the new fallback hint text explaining AI detection was unavailable.
3. In either case, click directly on the canvas for one slot: confirm its detected overlay/points clear and a fresh manual selection replaces them, without disturbing the other slots.
4. Submit a render with an AI-populated multi-slot selection through to completion; run `lightsaber-fx inspect <job_id>` afterward and confirm it prints `source=vlm` (or `source=motion`) per saber.

- [ ] **Step 8: Commit**

```bash
git add src/lightsaber_fx/web/static/app.js
git commit -m "Populate multiple saber slots from vision-assisted detection"
```

---

## Self-Review Notes (for whoever executes this plan)

- Task 2's test fixture `rotating_bar_video` needs to exist in a location both `test_detect.py` and `test_vision_detect.py` can import from (a shared `conftest.py`) -- check this first; if it's currently defined only inside `test_detect.py`'s module scope, moving it is a small, mechanical prerequisite of Task 2, not a separate task.
- Task 4's Step 9 CLI test is intentionally under-specified (it says to look at the neighboring existing test rather than inventing fixture setup blind) -- this is a real gap in this plan, not an oversight to silently paper over with invented code that might not match the actual existing test's conventions. Read `tests/test_cli.py`'s current `inspect` command test before writing it.
- Exact Gemini prompt wording (`DETECTION_PROMPT` in Task 1) is a first attempt, not a locked requirement -- the spec's own "Open questions" section flagged this needs iteration against real frames. If it under- or over-detects in Task 5's manual verification pass, adjusting the prompt text is expected, not a sign Task 1 was done wrong.
