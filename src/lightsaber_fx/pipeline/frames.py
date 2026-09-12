import os

import cv2

# Near-lossless JPEG (C2): tracking and every later compositing step (the
# capsule glow, the Knoll darkening, the trail) reads from these frames, so
# the default OpenCV JPEG quality (~95, with visible blocking on hard edges)
# was baking a lossy generation in at the very first stage. This directory
# stays JPEG rather than switching to PNG -- SAM2's own frame loader
# (`sam2/utils/misc.py:load_video_frames`) globs for a literal `.jpg`/`.jpeg`
# extension and would silently see zero frames otherwise.
JPEG_QUALITY = 100


def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not read a frame from {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(
            os.path.join(frames_dir, f"{i:05d}.jpg"), frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
        )
        i += 1
    cap.release()
    if i == 0:
        raise ValueError(f"Could not read a frame from {video_path}")
    return fps, i


def extract_first_frame(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read a frame from {video_path}")
    cv2.imwrite(out_path, frame)


def extract_frame_at(video_path, index, out_path):
    """Write frame `index` to `out_path`.

    Needed because the frame a user confirms is not always the first one:
    automatic detection reports the frame where the swung object was easiest
    to find, which is usually mid-swing (frame 135 of 300 on the 10 s test
    clip). `index` 0 is equivalent to `extract_first_frame`.
    """
    cap = cv2.VideoCapture(video_path)
    if index:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError(f"Could not read frame {index} from {video_path}")
    cv2.imwrite(out_path, frame)
