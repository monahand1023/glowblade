import shutil

import numpy as np
import pytest

from lightsaber_fx.pipeline.runner import run_pipeline

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


@requires_ffmpeg
def test_run_pipeline_end_to_end_with_stubbed_tracking(tmp_path, monkeypatch, tiny_video_path):
    def fake_track_object(frames_dir, masks_dir, points, labels, checkpoint_path,
                           config_name, device, n_frames, progress_cb=None):
        import os
        os.makedirs(masks_dir, exist_ok=True)
        for i in range(n_frames):
            mask = np.zeros((48, 64), dtype=bool)
            mask[10:20, 10:20] = True
            np.save(os.path.join(masks_dir, f"{i:05d}.npy"), mask)
            if progress_cb:
                progress_cb((i + 1) / n_frames * 100, f"frame {i + 1}")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", fake_track_object)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"
    stages_seen = []

    result = run_pipeline(
        input_video=str(tiny_video_path),
        points=[[10, 10]],
        labels=[1],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        config_name="unused",
        device="cpu",
        progress_cb=lambda stage, pct, message: stages_seen.append(stage),
    )

    assert result == str(output_path)
    assert output_path.exists()
    assert {"extract", "track", "glow", "audio", "mux"} <= set(stages_seen)
