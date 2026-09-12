import os

import cv2
import numpy as np

from .blade import save_mask


def pick_points_interactive(first_frame_path):
    points, labels = [], []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if flags & cv2.EVENT_FLAG_SHIFTKEY:
                points.append([x, y])
                labels.append(0)
                print(f"Exclude point at ({x},{y})")
            else:
                points.append([x, y])
                labels.append(1)
                print(f"Include point at ({x},{y})")

    img = cv2.imread(first_frame_path)
    if img is None:
        raise ValueError(f"Could not read a frame from {first_frame_path}")
    clone = img.copy()
    window = "Click the object (shift-click to exclude), then press ENTER"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        disp = clone.copy()
        for p, l in zip(points, labels):
            color = (0, 0, 255) if l == 1 else (255, 0, 0)
            cv2.circle(disp, tuple(p), 5, color, -1)
        cv2.imshow(window, disp)
        if (cv2.waitKey(20) & 0xFF) == 13:
            break
    cv2.destroyAllWindows()
    return points, labels


def track_object(
    frames_dir,
    masks_dir,
    points,
    labels,
    checkpoint_path,
    config_name,
    device,
    n_frames,
    progress_cb=None,
):
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
        frame_idx=0,
        obj_id=1,
        points=np.array(points, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
    )

    os.makedirs(masks_dir, exist_ok=True)
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        save_mask(masks_dir, frame_idx, mask)
        report((frame_idx + 1) / n_frames * 100, f"frame {frame_idx + 1}/{n_frames}")
