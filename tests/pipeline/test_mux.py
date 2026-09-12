import json
import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from lightsaber_fx.pipeline.mux import mux

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed",
)


@requires_ffmpeg
def test_mux_combines_video_and_audio_streams(tmp_path, tiny_video_path):
    audio_path = tmp_path / "audio.wav"
    sf.write(str(audio_path), np.sin(np.linspace(0, 440, 44100)), 44100)
    output_path = tmp_path / "final.mp4"
    progress_calls = []

    mux(str(tiny_video_path), str(audio_path), str(output_path),
        progress_cb=lambda pct, msg: progress_calls.append(pct))

    assert output_path.exists()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(output_path)],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    codec_types = {s["codec_type"] for s in streams}
    assert {"video", "audio"} <= codec_types
    assert progress_calls == [0, 100]
