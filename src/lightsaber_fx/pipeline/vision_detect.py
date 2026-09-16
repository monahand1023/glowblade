"""Vision-assisted multi-object detection: ask Gemini to find every swung
prop in one frame, then validate each candidate through the same SAM2 +
fit_blade quality gate `detect.py`'s own motion-based detection already
uses. Raises on a real technical failure, returns an empty list when Gemini
genuinely finds nothing -- callers (see `server.py`'s `_detect_proposals`)
decide whether either case falls back to `detect.py`'s `detect_blade`
(unmodified, untouched by this module).

Different failure modes than `detect.py` -- network errors, API keys,
per-call cost -- so this lives in its own file rather than inside it.
"""

import json

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


MASK_DEDUP_IOU_THRESHOLD = 0.5


def _mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return intersection / union if union else 0.0


def _dedupe_by_mask_iou(proposals):
    """Drops a proposal whose mask overlaps an already-kept one past
    `MASK_DEDUP_IOU_THRESHOLD` -- two Gemini boxes can legitimately segment
    the same swung object twice, and each would otherwise become its own
    tracked saber slot. Keeps the higher-elongation proposal of each
    overlapping pair."""
    kept = []
    for proposal in sorted(proposals, key=lambda p: p.elongation, reverse=True):
        if any(_mask_iou(proposal.mask, k.mask) > MASK_DEDUP_IOU_THRESHOLD for k in kept):
            continue
        kept.append(proposal)
    return kept


GEMINI_MODEL = "gemini-3.8-flash"
GEMINI_TIMEOUT_MS = 30000


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
            raise ValueError(f"Could not read frame {frame_index} from {video_path}")

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("failed to encode frame for Gemini")

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=encoded.tobytes(), mime_type="image/jpeg"),
                DETECTION_PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_json_schema=DETECTION_SCHEMA,
                http_options=types.HttpOptions(timeout=GEMINI_TIMEOUT_MS),
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

    return _dedupe_by_mask_iou(proposals)
