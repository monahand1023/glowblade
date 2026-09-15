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
