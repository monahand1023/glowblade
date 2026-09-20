import json
import os
import shutil
import subprocess

import cv2
import numpy as np
import pytest
import soundfile as sf

from glowblade.pipeline.mux import encode

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


def _write_png_sequence(frames_dir, n_frames=5, width=64, height=48):
    os.makedirs(frames_dir, exist_ok=True)
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 20) % 255, dtype=np.uint8)
        cv2.imwrite(os.path.join(frames_dir, f"{i:05d}.png"), frame)


def _probe_streams(path):
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(probe.stdout)["streams"]


@requires_ffmpeg
def test_encode_produces_h264_stereo_aac_with_faststart(tmp_path):
    frames_dir = tmp_path / "glow_frames"
    _write_png_sequence(str(frames_dir), n_frames=5, width=64, height=48)
    audio_path = tmp_path / "audio.wav"
    stereo = np.stack([np.sin(np.linspace(0, 440, 44100)), np.cos(np.linspace(0, 440, 44100))], axis=-1)
    sf.write(str(audio_path), stereo, 44100)
    output_path = tmp_path / "final.mp4"
    progress_calls = []

    encode(str(frames_dir), 10.0, str(audio_path), str(output_path),
           progress_cb=lambda pct, msg: progress_calls.append(pct))

    assert output_path.exists()
    streams = _probe_streams(output_path)
    by_type = {s["codec_type"]: s for s in streams}
    assert by_type["video"]["codec_name"] == "h264"
    assert by_type["video"]["pix_fmt"] == "yuv420p"
    assert by_type["video"]["width"] == 64
    assert by_type["video"]["height"] == 48
    assert by_type["audio"]["codec_name"] == "aac"
    assert by_type["audio"]["channels"] == 2
    assert progress_calls == [0, 100]


@requires_ffmpeg
def test_encode_preserves_frame_count_as_duration(tmp_path):
    frames_dir = tmp_path / "glow_frames"
    n_frames, fps = 20, 10.0
    _write_png_sequence(str(frames_dir), n_frames=n_frames, width=32, height=24)
    audio_path = tmp_path / "audio.wav"
    duration = n_frames / fps
    n_samples = round(duration * 44100)
    sf.write(str(audio_path), np.zeros((n_samples, 2)), 44100)
    output_path = tmp_path / "final.mp4"

    encode(str(frames_dir), fps, str(audio_path), str(output_path))

    streams = _probe_streams(output_path)
    video = next(s for s in streams if s["codec_type"] == "video")
    assert abs(float(video["duration"]) - duration) < 0.5


def test_encode_raises_actionable_error_on_ffmpeg_failure(tmp_path, monkeypatch):
    # A missing input (empty frames dir) or any other ffmpeg failure must
    # surface as a RuntimeError carrying ffmpeg's own stderr, not a bare
    # CalledProcessError -- this is the actionable-error behaviour a real
    # bad-encode or misconfigured-ffmpeg failure depends on.
    frames_dir = tmp_path / "glow_frames"
    frames_dir.mkdir()
    audio_path = tmp_path / "audio.wav"
    sf.write(str(audio_path), np.zeros((100, 2)), 44100)
    output_path = tmp_path / "final.mp4"

    def fake_run(cmd, check, capture_output):
        raise subprocess.CalledProcessError(1, cmd, output=b"", stderr=b"ffmpeg: no such file or pattern")

    monkeypatch.setattr("glowblade.pipeline.mux.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        encode(str(frames_dir), 10.0, str(audio_path), str(output_path))
