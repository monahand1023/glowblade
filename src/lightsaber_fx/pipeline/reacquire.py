"""Recovering tracked-object identity after two tracked objects visually
cross and SAM2's shared multi-object tracking session loses the
distinction between them -- confirmed on real footage (a fencing bout)
where both objects' masks permanently converged onto the same blade after
a crossing and never recovered on their own.

Runs as a post-hoc reconciliation stage in the multi-object pipeline,
after `track.track_objects` finishes and before `blade.compute_motion`
runs: reads the masks `track_objects` already wrote, and where it finds a
crossing between exactly two tracked objects, patches the lost object's
mask files in place before anything downstream sees them.

See docs/superpowers/specs/2026-09-16-cross-object-identity-recovery-design.md.

All frame-count/threshold constants below are starting defaults, validated
only loosely against the one real clip this was diagnosed on -- tune as
more real footage is tested against this.
"""

import numpy as np

from .blade import fit_blade, load_mask_optional
from .vision_detect import _mask_iou

MERGE_IOU_THRESHOLD = 0.8
MERGE_SUSTAIN_FRAMES = 15
REFERENCE_LOOKBACK_FRAMES = 90


def detect_merge(masks_dir_a, masks_dir_b, frame_indices,
                  iou_threshold=MERGE_IOU_THRESHOLD, sustain_frames=MERGE_SUSTAIN_FRAMES):
    """The first frame of a sustained high-overlap run between two
    objects' masks, or None if no such run exists in `frame_indices`.

    The *sustained* requirement is what distinguishes a real, permanent
    identity mixup from two objects briefly touching and correctly
    separating again (e.g. normal sword contact, tracked correctly) --
    only a run of `sustain_frames` or more consecutive high-IoU frames
    counts.

    `frame_indices` must be sorted; "consecutive" is measured as
    consecutive *entries* in this list, not consecutive frame numbers -- a
    real gap (an object briefly lost) is rare enough not to special-case
    here. If either object is missing a mask file for a frame (object lost),
    that frame resets the current run, same as a sub-threshold-IoU frame.
    """
    run_start = None
    run_len = 0
    for idx in frame_indices:
        mask_a = load_mask_optional(masks_dir_a, idx)
        mask_b = load_mask_optional(masks_dir_b, idx)
        if mask_a is None or mask_b is None:
            run_start = None
            run_len = 0
        else:
            iou = _mask_iou(mask_a, mask_b)
            if iou >= iou_threshold:
                if run_start is None:
                    run_start = idx
                run_len += 1
                if run_len >= sustain_frames:
                    return run_start
            else:
                run_start = None
                run_len = 0
    return None


def find_clean_reference(masks_dir_a, masks_dir_b, merge_start_frame, frame_indices,
                          lookback_frames=REFERENCE_LOOKBACK_FRAMES):
    """The frame of maximum inter-object centroid separation within
    `lookback_frames` before `merge_start_frame`, or None if no frame in
    that window has valid geometry for both objects.

    Deliberately *not* "the last frame before the merge" -- that frame can
    already be mid-drift (confirmed on the real clip this was diagnosed
    on: the frame immediately before a confirmed merge was already down to
    7px separation, and using it as the identity reference produced a
    swapped match). The true local separation maximum is a meaningfully
    cleaner reference.

    Uses `load_mask_optional`, matching `detect_merge`: a candidate frame
    missing a mask file for either object (object briefly lost) is simply
    skipped, not treated as an error.
    """
    candidates = [idx for idx in frame_indices if merge_start_frame - lookback_frames <= idx < merge_start_frame]
    best_idx, best_dist = None, -1.0
    for idx in candidates:
        mask_a = load_mask_optional(masks_dir_a, idx)
        mask_b = load_mask_optional(masks_dir_b, idx)
        if mask_a is None or mask_b is None:
            continue
        geo_a = fit_blade(mask_a)
        geo_b = fit_blade(mask_b)
        if geo_a is None or geo_b is None:
            continue
        dist = float(np.hypot(geo_a.centroid[0] - geo_b.centroid[0], geo_a.centroid[1] - geo_b.centroid[1]))
        if dist > best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx
