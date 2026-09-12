import os

from lightsaber_fx.pipeline.frames import extract_first_frame, extract_frames


def test_extract_frames_writes_one_jpg_per_frame(tmp_path, tiny_video_path):
    frames_dir = tmp_path / "frames"

    fps, n_frames = extract_frames(str(tiny_video_path), str(frames_dir))

    assert n_frames == 5
    assert abs(fps - 10.0) < 1.0
    written = sorted(os.listdir(frames_dir))
    assert written == [f"{i:05d}.jpg" for i in range(5)]


def test_extract_first_frame_writes_single_image(tmp_path, tiny_video_path):
    out_path = tmp_path / "frame0.jpg"

    extract_first_frame(str(tiny_video_path), str(out_path))

    assert out_path.exists()
    import cv2
    img = cv2.imread(str(out_path))
    assert img.shape == (48, 64, 3)
