import inspect
import shutil

import numpy as np
import pytest

from lightsaber_fx.pipeline import job_meta
from lightsaber_fx.pipeline.job_meta import JobNotRerenderableError
from lightsaber_fx.pipeline.runner import rerender_pipeline, run_pipeline

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_default_config_name_is_the_full_relative_sam2_config_path():
    # Do-not-regress invariant: Hydra requires the full relative path, not a bare
    # filename (a bare filename fails with Hydra's MissingConfigException). This
    # only reliably runs on a machine with the SAM2 checkpoint installed
    # (tests/pipeline/test_track.py), which is skipped elsewhere -- so this fast,
    # always-on check pins the default directly against signature inspection.
    default = inspect.signature(run_pipeline).parameters["config_name"].default
    assert default == "configs/sam2.1/sam2.1_hiera_s.yaml"


def test_run_pipeline_defaults_blade_extend_on_and_voice_neutral():
    params = inspect.signature(run_pipeline).parameters
    assert params["blade_extend"].default is True
    assert params["voice"].default == "neutral"


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
    # "motion" is its own stage, run between "track" and "glow" (A2): motion
    # is a first-class artifact both the visual and audio phases need to
    # read, so it can't be produced as a side effect of rendering.
    assert {"extract", "track", "motion", "glow", "audio", "mux"} <= set(stages_seen)
    assert stages_seen.index("track") < stages_seen.index("motion") < stages_seen.index("glow")

    # A2: run_pipeline produces the enriched motion.npz contract (blade
    # geometry per frame). render_glow/synthesize_audio (Phase B1/B2) read
    # it directly now -- there is no more legacy centroid motion.npy.
    motion = np.load(job_dir / "motion.npz")
    for key in ("centroid", "tip", "hilt", "axis", "length", "width", "angle"):
        assert key in motion.files
    n_frames = len(motion["length"])
    assert n_frames > 0
    assert motion["tip"].shape == (n_frames, 2)
    # The stubbed tracker writes an identical, non-empty mask for every
    # frame, so every frame should have fitted (non-NaN) geometry.
    assert not np.any(np.isnan(motion["length"]))

    # Phase C: no stray intermediates left behind after a successful run --
    # the legacy centroid path is gone entirely, and the lossless PNG
    # sequence the glow stage writes is a pure intermediate (unlike
    # frames/masks, kept only on request) that gets cleaned up unconditionally.
    assert not (job_dir / "motion.npy").exists()
    assert not (job_dir / "glow_video.mp4").exists()
    assert not (job_dir / "glow_frames").exists()


@requires_ffmpeg
def test_run_pipeline_threads_blade_extend_and_voice_through(tmp_path, monkeypatch, tiny_video_path):
    def fake_track_object(frames_dir, masks_dir, points, labels, checkpoint_path,
                           config_name, device, n_frames, progress_cb=None):
        import os
        os.makedirs(masks_dir, exist_ok=True)
        for i in range(n_frames):
            mask = np.zeros((48, 64), dtype=bool)
            mask[10:20, 10:20] = True
            np.save(os.path.join(masks_dir, f"{i:05d}.npy"), mask)

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", fake_track_object)

    from lightsaber_fx.pipeline.audio import synthesize_audio as real_synthesize_audio
    from lightsaber_fx.pipeline.glow import render_glow as real_render_glow

    captured = {}

    def spy_render_glow(*args, **kwargs):
        captured["blade_extend"] = kwargs.get("blade_extend")
        return real_render_glow(*args, **kwargs)

    def spy_synthesize_audio(*args, **kwargs):
        captured["voice"] = kwargs.get("voice")
        return real_synthesize_audio(*args, **kwargs)

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.render_glow", spy_render_glow)
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.synthesize_audio", spy_synthesize_audio)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"

    run_pipeline(
        input_video=str(tiny_video_path),
        points=[[10, 10]],
        labels=[1],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        config_name="unused",
        device="cpu",
        blade_extend=False,
        voice="sith",
    )

    assert captured["blade_extend"] is False
    assert captured["voice"] == "sith"


@requires_ffmpeg
def test_run_pipeline_writes_job_meta_so_the_job_is_later_rerenderable(tmp_path, monkeypatch, tiny_video_path):
    def fake_track_object(frames_dir, masks_dir, points, labels, checkpoint_path,
                           config_name, device, n_frames, progress_cb=None):
        import os
        os.makedirs(masks_dir, exist_ok=True)
        for i in range(n_frames):
            mask = np.zeros((48, 64), dtype=bool)
            mask[10:20, 10:20] = True
            np.save(os.path.join(masks_dir, f"{i:05d}.npy"), mask)

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", fake_track_object)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"

    run_pipeline(
        input_video=str(tiny_video_path),
        points=[[10, 10]],
        labels=[1],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        config_name="unused",
        device="cpu",
    )

    meta = job_meta.read_job_meta(str(job_dir))
    assert meta is not None
    import os
    assert meta["source_video"] == os.path.abspath(str(tiny_video_path))

    info = job_meta.describe_job(str(job_dir))
    assert info.rerenderable is True, info.reason


# ---------------------------------------------------------------------------
# rerender_pipeline
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_rerender_pipeline_reextracts_frames_and_produces_output(tmp_path, rerenderable_job_fixture):
    job_dir = rerenderable_job_fixture
    assert not (job_dir / "frames").exists()  # the whole point of the design

    output_path = tmp_path / "rerendered.mp4"
    result = rerender_pipeline(job_dir=str(job_dir), output_path=str(output_path))

    assert result == str(output_path)
    assert output_path.exists()
    assert (job_dir / "frames").is_dir()
    assert sorted(p.name for p in (job_dir / "frames").iterdir()) == [
        f"{i:05d}.jpg" for i in range(5)
    ]


@requires_ffmpeg
def test_rerender_pipeline_never_calls_track_object(monkeypatch, tmp_path, rerenderable_job_fixture):
    def fail_if_called(*a, **k):
        raise AssertionError("track_object should never run on the rerender path")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", fail_if_called)

    output_path = tmp_path / "rerendered.mp4"
    result = rerender_pipeline(job_dir=str(rerenderable_job_fixture), output_path=str(output_path))

    assert output_path.exists()
    assert result == str(output_path)


@requires_ffmpeg
def test_rerender_pipeline_never_calls_compute_motion(monkeypatch, tmp_path, rerenderable_job_fixture):
    # motion.npz depends only on the masks, none of which rerender's
    # parameters (color/intensity/voice/blade_extend) can affect -- so it
    # must be reused as-is, not recomputed.
    def fail_if_called(*a, **k):
        raise AssertionError("compute_motion should never run on the rerender path")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.compute_motion", fail_if_called)

    output_path = tmp_path / "rerendered.mp4"
    result = rerender_pipeline(job_dir=str(rerenderable_job_fixture), output_path=str(output_path))

    assert output_path.exists()
    assert result == str(output_path)


def test_rerender_pipeline_raises_clear_error_when_masks_missing(tmp_path, tiny_video_path):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "video_meta.txt").write_text("10.0\n5\n")
    job_meta.write_job_meta(str(job_dir), source_video=str(tiny_video_path))
    # No masks/, no motion.npz.

    with pytest.raises(JobNotRerenderableError) as exc_info:
        rerender_pipeline(job_dir=str(job_dir), output_path=str(tmp_path / "out.mp4"))

    assert "masks" in str(exc_info.value)
    assert not (tmp_path / "out.mp4").exists()


def test_rerender_pipeline_raises_clear_error_when_source_clip_missing(tmp_path, rerenderable_job_fixture, tiny_video_path):
    tiny_video_path.unlink()

    with pytest.raises(JobNotRerenderableError) as exc_info:
        rerender_pipeline(job_dir=str(rerenderable_job_fixture), output_path=str(tmp_path / "out.mp4"))

    assert "moved or deleted" in str(exc_info.value)
    assert not (tmp_path / "out.mp4").exists()


def test_rerender_pipeline_validates_color_before_extracting_frames(monkeypatch, tmp_path, rerenderable_job_fixture):
    def fail_if_called(*a, **k):
        raise AssertionError("extract_frames should not run before color is validated")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.extract_frames", fail_if_called)

    with pytest.raises(ValueError):
        rerender_pipeline(
            job_dir=str(rerenderable_job_fixture),
            output_path=str(tmp_path / "out.mp4"),
            color="not-a-real-color",
        )


@requires_ffmpeg
def test_rerender_pipeline_works_with_legacy_npy_masks(tmp_path, tiny_video_path):
    from lightsaber_fx.pipeline.blade import compute_motion

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    masks_dir = job_dir / "masks"
    masks_dir.mkdir()
    for i in range(5):
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:20, 5 + i * 3:9 + i * 3] = True
        np.save(str(masks_dir / f"{i:05d}.npy"), mask)
    compute_motion(str(masks_dir), str(job_dir / "motion.npz"))
    (job_dir / "video_meta.txt").write_text("10.0\n5\n")
    job_meta.write_job_meta(str(job_dir), source_video=str(tiny_video_path))

    output_path = tmp_path / "rerendered.mp4"
    result = rerender_pipeline(job_dir=str(job_dir), output_path=str(output_path))

    assert result == str(output_path)
    assert output_path.exists()


@requires_ffmpeg
def test_rerender_pipeline_threads_color_intensity_voice_blade_extend_through(
    monkeypatch, tmp_path, rerenderable_job_fixture
):
    from lightsaber_fx.pipeline.audio import synthesize_audio as real_synthesize_audio
    from lightsaber_fx.pipeline.glow import render_glow as real_render_glow

    captured = {}

    def spy_render_glow(*args, **kwargs):
        captured["color"] = kwargs.get("color")
        captured["spill_strength"] = kwargs.get("spill_strength")
        captured["blade_extend"] = kwargs.get("blade_extend")
        return real_render_glow(*args, **kwargs)

    def spy_synthesize_audio(*args, **kwargs):
        captured["voice"] = kwargs.get("voice")
        return real_synthesize_audio(*args, **kwargs)

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.render_glow", spy_render_glow)
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.synthesize_audio", spy_synthesize_audio)

    rerender_pipeline(
        job_dir=str(rerenderable_job_fixture),
        output_path=str(tmp_path / "out.mp4"),
        color="blue",
        intensity=0.7,
        blade_extend=False,
        voice="sith",
    )

    assert captured["color"] == (255, 90, 60)  # NAMED_COLORS["blue"], BGR
    assert captured["spill_strength"] == 0.7
    assert captured["blade_extend"] is False
    assert captured["voice"] == "sith"
