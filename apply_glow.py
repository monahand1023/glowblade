"""
Step 2: Turn the tracked mask into a lightsaber glow and composite it back
onto the original footage, including a soft "light spill" that brightens
nearby surfaces the way a real glowing blade would.
"""
import os
import cv2
import numpy as np

FRAMES_DIR = "frames"
MASKS_DIR = "masks"
OUTPUT_VIDEO = "glow_video.mp4"
MOTION_OUT = "motion.npy"

BLADE_COLOR = (40, 40, 255)   # BGR (OpenCV order) -- this is red

CORE_BLUR = 5           # tight white-hot core
INNER_GLOW_BLUR = 25    # colored glow hugging the blade
SPILL_BLUR = 95         # wide, soft light thrown on nearby objects
SPILL_STRENGTH = 0.35   # 0-1, how strongly the spill brightens surroundings


def screen_blend(base, top):
    base = base.astype(np.float32) / 255.0
    top = top.astype(np.float32) / 255.0
    out = 1 - (1 - base) * (1 - top)
    return (out * 255).astype(np.uint8)


def make_glow_layers(mask, shape):
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.resize(mask_u8, (shape[1], shape[0]))

    core = cv2.GaussianBlur(mask_u8, (0, 0), CORE_BLUR)
    core_bgr = cv2.merge([core, core, core])

    colored = np.zeros((*shape[:2], 3), dtype=np.uint8)
    colored[mask_u8 > 0] = BLADE_COLOR
    inner = cv2.GaussianBlur(colored, (0, 0), INNER_GLOW_BLUR)

    spill = cv2.GaussianBlur(colored, (0, 0), SPILL_BLUR)
    spill = (spill.astype(np.float32) * SPILL_STRENGTH).astype(np.uint8)

    return core_bgr, inner, spill


def main():
    frame_files = sorted(f for f in os.listdir(FRAMES_DIR) if f.endswith(".jpg"))
    with open("video_meta.txt") as f:
        fps = float(f.readline())

    first = cv2.imread(os.path.join(FRAMES_DIR, frame_files[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    centroids = []
    for fname in frame_files:
        idx = int(os.path.splitext(fname)[0])
        frame = cv2.imread(os.path.join(FRAMES_DIR, fname))
        mask_path = os.path.join(MASKS_DIR, f"{idx:05d}.npy")

        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            ys, xs = np.where(mask)
            centroids.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
            core, inner, spill = make_glow_layers(mask, frame.shape)
            out = screen_blend(frame, spill)
            out = screen_blend(out, inner)
            out = screen_blend(out, core)
        else:
            centroids.append((np.nan, np.nan))
            out = frame

        writer.write(out)

    writer.release()
    np.save(MOTION_OUT, np.array(centroids))
    print(f"Wrote {OUTPUT_VIDEO} and {MOTION_OUT}")


if __name__ == "__main__":
    main()
