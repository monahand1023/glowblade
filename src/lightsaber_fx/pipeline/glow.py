import os

import cv2
import numpy as np

NAMED_COLORS = {
    "red": (40, 40, 255),
    "blue": (255, 90, 60),
    "green": (70, 220, 80),
}


def parse_color(spec):
    key = spec.strip().lower()
    if key in NAMED_COLORS:
        return NAMED_COLORS[key]
    if key.startswith("#") and len(key) == 7:
        r = int(key[1:3], 16)
        g = int(key[3:5], 16)
        b = int(key[5:7], 16)
        return (b, g, r)
    raise ValueError(f"Unrecognized color: {spec!r}. Use red, blue, green, or #RRGGBB.")


def screen_blend(base, top):
    base = base.astype(np.float32) / 255.0
    top = top.astype(np.float32) / 255.0
    out = 1 - (1 - base) * (1 - top)
    return (out * 255).astype(np.uint8)


def make_glow_layers(mask, shape, color, core_blur, inner_glow_blur, spill_blur, spill_strength):
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.resize(mask_u8, (shape[1], shape[0]))

    core = cv2.GaussianBlur(mask_u8, (0, 0), core_blur)
    core_bgr = cv2.merge([core, core, core])

    colored = np.zeros((*shape[:2], 3), dtype=np.uint8)
    colored[mask_u8 > 0] = color
    inner = cv2.GaussianBlur(colored, (0, 0), inner_glow_blur)

    spill = cv2.GaussianBlur(colored, (0, 0), spill_blur)
    spill = (spill.astype(np.float32) * spill_strength).astype(np.uint8)

    return core_bgr, inner, spill


def render_glow(
    frames_dir,
    masks_dir,
    video_meta_path,
    output_video_path,
    motion_out_path,
    color=(40, 40, 255),
    core_blur=5,
    inner_glow_blur=25,
    spill_blur=95,
    spill_strength=0.35,
    progress_cb=None,
):
    def report(pct, message):
        if progress_cb:
            progress_cb(pct, message)

    frame_files = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    with open(video_meta_path) as f:
        fps = float(f.readline())

    first = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    centroids = []
    total = len(frame_files)
    for n, fname in enumerate(frame_files):
        idx = int(os.path.splitext(fname)[0])
        frame = cv2.imread(os.path.join(frames_dir, fname))
        mask_path = os.path.join(masks_dir, f"{idx:05d}.npy")

        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            ys, xs = np.where(mask)
            centroids.append((xs.mean(), ys.mean()) if len(xs) else (np.nan, np.nan))
            core, inner, spill = make_glow_layers(
                mask, frame.shape, color, core_blur, inner_glow_blur, spill_blur, spill_strength
            )
            out = screen_blend(frame, spill)
            out = screen_blend(out, inner)
            out = screen_blend(out, core)
        else:
            centroids.append((np.nan, np.nan))
            out = frame

        writer.write(out)
        report((n + 1) / total * 100, f"frame {n + 1}/{total}")

    writer.release()
    np.save(motion_out_path, np.array(centroids))
