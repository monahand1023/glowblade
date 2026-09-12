import os

import cv2


def extract_frames(video_path, frames_dir):
    os.makedirs(frames_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.jpg"), frame)
        i += 1
    cap.release()
    return fps, i


def extract_first_frame(video_path, out_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not read a frame from {video_path}")
    cv2.imwrite(out_path, frame)
