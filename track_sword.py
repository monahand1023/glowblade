"""
Step 1: Track the sword through the video using SAM2 (no markers needed).

Click once on the blade in the first frame. Hold SHIFT and click to mark
a point that should be EXCLUDED from the mask (e.g. the hilt or your
son's hand), if SAM2 grabs more than just the blade. Press ENTER when done.

Outputs:
  frames/          extracted video frames
  masks/*.npy      per-frame binary mask of the tracked object
  video_meta.txt   fps and frame count, used by later steps
"""
import os
import sys
import cv2
import numpy as np
import torch

VIDEO_PATH = "input.mp4"                    # <-- point this at your clip
FRAMES_DIR = "frames"
MASKS_DIR = "masks"
SAM2_CONFIG = "sam2.1_hiera_s.yaml"          # must match the checkpoint below
SAM2_CHECKPOINT = "checkpoints/sam2.1_hiera_small.pt"

points, labels = [], []


def extract_frames(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"), frame)
        i += 1
    cap.release()
    return fps, i


def on_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if flags & cv2.EVENT_FLAG_SHIFTKEY:
            points.append([x, y]); labels.append(0)   # exclude
            print(f"Exclude point at ({x},{y})")
        else:
            points.append([x, y]); labels.append(1)   # include
            print(f"Include point at ({x},{y})")


def pick_points(first_frame_path):
    img = cv2.imread(first_frame_path)
    clone = img.copy()
    window = "Click the blade (shift-click to exclude), then press ENTER"
    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)
    while True:
        disp = clone.copy()
        for p, l in zip(points, labels):
            color = (0, 0, 255) if l == 1 else (255, 0, 0)
            cv2.circle(disp, tuple(p), 5, color, -1)
        cv2.imshow(window, disp)
        if (cv2.waitKey(20) & 0xFF) == 13:  # Enter
            break
    cv2.destroyAllWindows()


def main():
    print("Extracting frames...")
    fps, n_frames = extract_frames(VIDEO_PATH, FRAMES_DIR)
    print(f"{n_frames} frames at {fps:.2f} fps")

    print("Click the blade in the popup window. Shift-click to exclude "
          "a spot (e.g. the hilt). Press Enter when done.")
    pick_points(os.path.join(FRAMES_DIR, "00000.jpg"))
    if not points:
        print("No points selected, exiting.")
        sys.exit(1)

    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # a few SAM2 ops aren't on MPS yet
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    from sam2.build_sam import build_sam2_video_predictor
    predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=device)

    state = predictor.init_state(video_path=FRAMES_DIR)
    predictor.add_new_points_or_box(
        state,
        frame_idx=0,
        obj_id=1,
        points=np.array(points, dtype=np.float32),
        labels=np.array(labels, dtype=np.int32),
    )

    os.makedirs(MASKS_DIR, exist_ok=True)
    print("Propagating mask through the video (slow part, be patient)...")
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
        np.save(os.path.join(MASKS_DIR, f"{frame_idx:05d}.npy"), mask)
        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{n_frames}")

    with open("video_meta.txt", "w") as f:
        f.write(f"{fps}\n{n_frames}\n")
    print("Done. Masks saved to", MASKS_DIR)


if __name__ == "__main__":
    main()
