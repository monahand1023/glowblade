import subprocess

STDERR_TAIL_CHARS = 2000


def mux(video_path, audio_path, output_path, progress_cb=None):
    if progress_cb:
        progress_cb(0, "muxing video and audio")
    try:
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
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode(errors="replace")
        raise RuntimeError(f"ffmpeg failed: {stderr[-STDERR_TAIL_CHARS:]}") from exc
    if progress_cb:
        progress_cb(100, "done")
