import os
import subprocess

STDERR_TAIL_CHARS = 2000


def encode(frames_dir, fps, audio_path, output_path, progress_cb=None):
    """Encode the glow stage's lossless PNG sequence and the (stereo)
    synthesized audio into a single, shareable H.264/AAC file, in one ffmpeg
    pass.

    Replaces the old mpeg4-intermediate-then-``-c:v copy`` chain (three lossy
    generations: JPEG frame extraction, an OpenCV mpeg4 encode, then a
    stream copy that just carries the mediocre result through unchanged).
    Encoding once, straight from the lossless PNGs, at ``-crf 16`` is the
    deliberate quality-over-speed tradeoff this project's fidelity upgrade
    chose: it makes this stage slower, on purpose.

    - ``-c:v libx264 -crf 16 -pix_fmt yuv420p`` -- a high-quality encode in
      the pixel format every player actually supports (many players choke on
      4:4:4/4:2:2, which x264 would otherwise default to for some inputs).
    - ``-movflags +faststart`` -- moves the moov atom to the front so the
      file can start playing before it's fully downloaded.
    - ``-c:a aac -ac 2`` -- AAC audio pinned to 2 channels, so a stereo
      ``synthesize_audio`` render can't silently collapse to mono.
    - ``-shortest`` -- the video and audio durations are computed from the
      same frame count/fps and should already match; this just guards
      against a stray trailing frame or a rounding-length sample tail.

    `frames_dir` holds ``{idx:05d}.png`` files (as written by
    ``glow.render_glow``), numbered contiguously from 0.
    """
    if progress_cb:
        progress_cb(0, "encoding video and audio")
    frame_pattern = os.path.join(frames_dir, "%05d.png")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", frame_pattern,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "libx264",
                "-crf", "16",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-ac", "2",
                "-movflags", "+faststart",
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
