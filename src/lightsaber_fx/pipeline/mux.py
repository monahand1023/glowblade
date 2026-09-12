import subprocess


def mux(video_path, audio_path, output_path, progress_cb=None):
    if progress_cb:
        progress_cb(0, "muxing video and audio")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            output_path,
        ],
        check=True,
        capture_output=True,
    )
    if progress_cb:
        progress_cb(100, "done")
