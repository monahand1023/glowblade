import inspect
import shutil

import numpy as np
import pytest

from lightsaber_fx.pipeline.runner import run_pipeline

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_default_config_name_is_the_full_relative_sam2_config_path():
    # Do-not-regress invariant: Hydra requires the full relative path, not a bare
    # filename (a bare filename fails with Hydra's MissingConfigException). This
    # only reliably runs on a machine with the SAM2 checkpoint installed
    # (tests/pipeline/test_track.py), which is skipped elsewhere -- so this fast,
    # always-on check pins the default directly against signature inspection.
    default = inspect.signature(run_pipeline).parameters["config_name"].default
    assert default == "configs/sam2.1/sam2.1_hiera_s.yaml"


def test_run_pipeline_validates_color_before_extracting_frames(tmp_path, monkeypatch, tiny_video_path):
    # A4: an invalid --color must fail immediately, before the (potentially
    # minutes-long) extract/track stages ever run.
    def fail_if_called(*a, **k):
        raise AssertionError("extract_frames should not run before color is validated")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.extract_frames", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(ValueError):
        run_pipeline(
            input_video=str(tiny_video_path),
            points=[[10, 10]],
            labels=[1],
            output_path=str(tmp_path / "final.mp4"),
            job_dir=str(job_dir),
            checkpoint_path="unused",
            config_name="unused",
            device="cpu",
            color="not-a-real-color",
        )


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
