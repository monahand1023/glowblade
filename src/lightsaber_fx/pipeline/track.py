import os

import cv2
import numpy as np

from .blade import save_mask


def overlay_proposal(img, mask, points, tint=(0, 255, 0), alpha=0.35):
    """Draw a detected mask and its prompt points onto `img`, in place.

    Shared by the CLI picker and the web app's first-frame preview so a
    proposal looks the same wherever it is confirmed.
    """
    if mask is not None and mask.shape[:2] == img.shape[:2]:
        img[mask] = (alpha * np.array(tint) + (1 - alpha) * img[mask]).astype(img.dtype)
    for point in points:
        cv2.drawMarker(img, tuple(point), (0, 0, 255), cv2.MARKER_CROSS, 24, 3)
    return img


def pick_points_interactive(frame_path, proposal=None):
    """Show `frame_path` and collect click points.

    With a `proposal` (a `detect.BladeProposal`), its mask and points are
    drawn on the frame and returned as-is if the user just presses ENTER --
    the confirm half of detect-then-confirm. The first click discards the
    proposal entirely rather than adding to it: a detected mask the user is
    correcting is a mask they disagree with, and mixing their point into it
    would keep whatever was wrong about it.
    """
    points, labels = [], []
    proposal_points = list(proposal.points) if proposal is not None else []
    proposal_labels = list(proposal.labels) if proposal is not None else []
    proposal_mask = proposal.mask if proposal is not None else None
    state = {"overridden": False}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not state["overridden"] and proposal_points:
                print("Discarding the detected blade in favour of your clicks.")
                state["overridden"] = True
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                points.append([x, y])
                labels.append(0)
                print(f"Exclude point at ({x},{y})")
            else:
                points.append([x, y])
                labels.append(1)
                print(f"Include point at ({x},{y})")

    img = cv2.imread(frame_path)
    if img is None:
        raise ValueError(f"Could not read a frame from {frame_path}")
    clone = img.copy()
    if proposal is not None:
        window = "ENTER to accept the detected blade, or click to choose your own"
    else:
        window = "Click the object (shift-click to exclude), then press ENTER"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        disp = clone.copy()
        if not state["overridden"] and proposal is not None:
            overlay_proposal(disp, proposal_mask, proposal_points)
        for p, l in zip(points, labels):
            color = (0, 0, 255) if l == 1 else (255, 0, 0)
            cv2.circle(disp, tuple(p), 5, color, -1)
        cv2.imshow(window, disp)
        if (cv2.waitKey(20) & 0xFF) == 13:
            break
    cv2.destroyAllWindows()
    if points:
        return points, labels
    return proposal_points, proposal_labels


def track_object(
    frames_dir,
    masks_dir,
    points,
    labels,
    checkpoint_path,
    config_name,
    device,
    n_frames,
    prompt_frame=0,
    progress_cb=None,
):
    """Propagate `points`/`labels` given on `prompt_frame` across the clip.

    `prompt_frame` defaults to 0, which is what the interactive picker
    produces -- it shows the first frame. Automatic detection
    (`detect.detect_blade`) instead reports the frame where the swung
    object was easiest to find, which is usually mid-swing, so tracking
    runs in both directions from there.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    # NOTE: torch is already imported by this point (device.py imports it earlier
    # in the process), so this assignment is plausibly a no-op -- PYTORCH_ENABLE_MPS_FALLBACK
    # is normally read by torch at import time. Inherited from the original script;
    # a real MPS render has been verified working with this ordering, so it is left
    # as-is (known, unverified whether it does anything) rather than moved.
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=device)

    state = predictor.init_state(video_path=frames_dir)
    predictor.add_new_points_or_box(
        state,
        frame_idx=prompt_frame,
        obj_id=1,
        points=np.array(points, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
    )

    os.makedirs(masks_dir, exist_ok=True)
    # Propagate forward from the prompt frame, then backward, so prompting a
    # frame in the middle of the clip still covers all of it. Automatic
    # detection needs this: the frame where a swung object is easiest to
    # find is the frame where it is moving fastest, which is rarely frame 0
    # (on the 10 s test clip it is frame 135 of 300). With prompt_frame=0
    # the reverse pass has only that one frame to do and the behaviour is
    # unchanged from propagating forward alone.
    #
    # Both passes emit the prompt frame itself, so its mask is written
    # twice. That is a redundant write, not a conflict -- the same logits
    # produce the same mask -- and it keeps the loop free of special cases.
    written = 0
    for reverse in (False, True):
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state, reverse=reverse
        ):
            mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
            save_mask(masks_dir, frame_idx, mask)
            written += 1
            report(min(written / n_frames * 100, 100.0), f"frame {written}/{n_frames}")


def track_objects(
    frames_dir,
    prompts,
    checkpoint_path,
    config_name,
    device,
    n_frames,
    progress_cb=None,
):
    """Like `track_object`, but for `len(prompts)` (1-4) objects tracked
    together in one shared SAM2 session -- cheaper than N separate sessions,
    since each frame's image features are encoded once regardless of object
    count.

    Each prompt carries its own `prompt_frame` (default 0), so an object
    found by automatic detection is prompted on the frame it was actually
    found in -- usually mid-swing, rarely frame 0. Propagation then runs
    forward from those prompts and then backward, for the same reason
    `track_object` does both: a prompt in the middle of the clip would
    otherwise leave every earlier frame unmasked. When every prompt_frame
    is 0 the reverse pass has only that one frame to do, so the behaviour
    is unchanged from propagating forward alone.

    Both passes emit the prompt frames themselves, so those masks are
    written twice. That is a redundant write, not a conflict -- the same
    logits produce the same mask -- and it keeps the loop free of special
    cases.

    Known limitation, measured against the pinned SAM2 build: every object
    has to be prompted on the *same* frame. Conditioning objects on
    different frames within one shared session breaks SAM2's memory
    attention -- a BFloat16/Float dtype RuntimeError on CPU, and a hard
    (uncatchable) Metal assertion on MPS. Nothing here enforces it because
    nothing upstream can currently produce a mixed set: the web app sends
    one saber, and automatic detection reports one frame. A multi-slot
    picker that lets each saber be detected separately would need either
    one SAM2 session per prompt frame, or a check that rejects the mix.
    """
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(config_name, checkpoint_path, device=device)

    state = predictor.init_state(video_path=frames_dir)
    for prompt in prompts:
        predictor.add_new_points_or_box(
            state,
            frame_idx=prompt.get("prompt_frame", 0),
            obj_id=prompt["obj_id"],
            points=np.array(prompt["points"], dtype=np.float32),
            labels=np.array(prompt["labels"], dtype=np.int32),
        )

    masks_dir_by_obj_id = {p["obj_id"]: p["masks_dir"] for p in prompts}
    for masks_dir in masks_dir_by_obj_id.values():
        os.makedirs(masks_dir, exist_ok=True)

    written = 0
    total_writes = n_frames * len(prompts)
    for reverse in (False, True):
        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state, reverse=reverse):
            for i, obj_id in enumerate(obj_ids):
                mask = (mask_logits[i] > 0.0).cpu().numpy().squeeze()
                save_mask(masks_dir_by_obj_id[obj_id], frame_idx, mask)
                written += 1
            report(min(written / total_writes * 100, 100.0), f"frame {frame_idx + 1}/{n_frames}")
