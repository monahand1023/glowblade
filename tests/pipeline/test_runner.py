import inspect
import os
import shutil

import numpy as np
import pytest

from lightsaber_fx.pipeline import job_meta
from lightsaber_fx.pipeline.blade import compute_motion, save_mask
from lightsaber_fx.pipeline.job_meta import JobNotRerenderableError
from lightsaber_fx.pipeline.runner import (
    rerender_pipeline,
    rerender_pipeline_multi,
    run_pipeline,
    run_pipeline_multi,
)

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
                           config_name, device, n_frames, prompt_frame=0, progress_cb=None):
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
                           config_name, device, n_frames, prompt_frame=0, progress_cb=None):
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
                           config_name, device, n_frames, prompt_frame=0, progress_cb=None):
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


def _fake_track_writing(mask_factory):
    """Build a `track_object` stub that writes whatever masks `mask_factory`
    returns for each frame index. Used by the coverage-guard tests below to
    simulate a track that found nothing, or found the object only briefly."""
    def fake_track_object(frames_dir, masks_dir, points, labels, checkpoint_path,
                          config_name, device, n_frames, prompt_frame=0, progress_cb=None):
        import os
        os.makedirs(masks_dir, exist_ok=True)
        for i in range(n_frames):
            np.save(os.path.join(masks_dir, f"{i:05d}.npy"), mask_factory(i))
    return fake_track_object


def _blank(_i):
    return np.zeros((48, 64), dtype=bool)


def _blade(_i):
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:34, 20:24] = True
    return mask


def test_run_pipeline_raises_before_glow_when_tracking_found_nothing(
    tmp_path, monkeypatch, tiny_video_path
):
    # An all-empty track used to cost the full glow stage (~160s on a 10s 720p
    # clip) and then emit a video identical to the input, with nothing saying
    # why -- which reads as a compositing bug rather than as bad click points.
    # Stubbing render_glow to fail if called proves the guard runs *first*:
    # asserting only that run_pipeline raises would pass even if the raise came
    # after glow had already burned the time.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("render_glow ran despite the track finding no blade")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", _fake_track_writing(_blank))
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.render_glow", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(RuntimeError, match="no blade in any of"):
        run_pipeline(
            input_video=str(tiny_video_path),
            points=[[10, 10]],
            labels=[1],
            output_path=str(tmp_path / "final.mp4"),
            job_dir=str(job_dir),
            checkpoint_path="unused",
            device="cpu",
        )


def test_run_pipeline_error_for_an_empty_track_names_the_click_points(
    tmp_path, monkeypatch, tiny_video_path
):
    # The message is the whole value of this guard, so it is asserted rather
    # than left to `match=`: a bare count would leave the user with no idea
    # what to change. Empty masks nearly always mean an include point that
    # missed the object or an exclude point that landed on it.
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", _fake_track_writing(_blank))

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        run_pipeline(
            input_video=str(tiny_video_path),
            points=[[10, 10]],
            labels=[1],
            output_path=str(tmp_path / "final.mp4"),
            job_dir=str(job_dir),
            checkpoint_path="unused",
            device="cpu",
        )

    message = str(excinfo.value)
    assert "click points" in message
    assert "exclude point" in message


@requires_ffmpeg
def test_run_pipeline_warns_but_renders_when_the_blade_is_found_in_few_frames(
    tmp_path, monkeypatch, tiny_video_path
):
    # Partial coverage is legitimate -- an object can leave frame and come back
    # -- so this must NOT raise. It reports through the normal progress channel
    # so both front ends surface it, and still produces a file.
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_object",
        _fake_track_writing(lambda i: _blade(i) if i == 0 else _blank(i)),
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"
    messages = []

    run_pipeline(
        input_video=str(tiny_video_path),
        points=[[10, 10]],
        labels=[1],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
        progress_cb=lambda stage, pct, message: messages.append((stage, message)),
    )

    warnings = [m for stage, m in messages if stage == "motion" and m.startswith("warning:")]
    assert len(warnings) == 1
    assert "only 1 of" in warnings[0]
    assert output_path.exists()


def test_compute_motion_reports_how_many_frames_produced_a_blade(tmp_path):
    from lightsaber_fx.pipeline.blade import compute_motion, save_mask

    masks_dir = tmp_path / "masks"
    masks_dir.mkdir()
    for i in range(4):
        save_mask(str(masks_dir), i, _blade(i) if i < 3 else _blank(i))

    n_tracked, n_with_blade = compute_motion(str(masks_dir), str(tmp_path / "motion.npz"))

    assert (n_tracked, n_with_blade) == (4, 3)


# ---------------------------------------------------------------------------
# run_pipeline_multi
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_run_pipeline_multi_end_to_end_with_stubbed_tracking(tmp_path, monkeypatch, tiny_video_path):
    def fake_track_objects(frames_dir, prompts, checkpoint_path, config_name, device, n_frames, progress_cb=None):
        for prompt in prompts:
            os.makedirs(prompt["masks_dir"], exist_ok=True)
            for i in range(n_frames):
                x = 5 + i * 3
                mask = np.zeros((48, 64), dtype=bool)
                mask[10:20, x:x + 6] = True
                save_mask(prompt["masks_dir"], i, mask)
            if progress_cb:
                progress_cb(100, "done")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_objects", fake_track_objects)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"

    result = run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert result == str(output_path)
    assert output_path.exists() and output_path.stat().st_size > 0
    meta = job_meta.read_job_meta(str(job_dir))
    assert meta["object_ids"] == [0, 1]
    assert os.path.isdir(job_dir / "masks" / "0")
    assert os.path.isdir(job_dir / "masks" / "1")
    assert (job_dir / "motion" / "0.npz").exists()
    assert (job_dir / "motion" / "1.npz").exists()


def test_run_pipeline_multi_validates_every_saber_color_before_tracking(tmp_path, monkeypatch, tiny_video_path):
    def fail_if_called(*a, **k):
        raise AssertionError("extract_frames should not run before every color is validated")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.extract_frames", fail_if_called)
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(ValueError):
        run_pipeline_multi(
            input_video=str(tiny_video_path),
            sabers=[
                {"points": [[1, 1]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
                {"points": [[1, 1]], "labels": [1], "color": "not-a-color", "intensity": 0.35, "voice": "neutral"},
            ],
            output_path=str(tmp_path / "final.mp4"),
            job_dir=str(job_dir),
            checkpoint_path="unused",
            device="cpu",
        )


# ---------------------------------------------------------------------------
# rerender_pipeline_multi
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_rerender_pipeline_multi_reuses_cached_masks_for_a_new_color(tmp_path, monkeypatch, tiny_video_path):
    job_dir = tmp_path / "job"
    for oid in (0, 1):
        masks_dir = job_dir / "masks" / str(oid)
        for i in range(5):
            mask = np.zeros((48, 64), dtype=bool)
            mask[10:20, 5 + i * 3:11 + i * 3] = True
            save_mask(str(masks_dir), i, mask)
        motion_dir = job_dir / "motion"
        motion_dir.mkdir(exist_ok=True)
        compute_motion(str(masks_dir), str(motion_dir / f"{oid}.npz"))
    (job_dir / "video_meta.txt").write_text("10.0\n5\n")
    job_meta.write_job_meta(str(job_dir), source_video=str(tiny_video_path), object_ids=[0, 1])

    output_path = tmp_path / "final.mp4"
    result = rerender_pipeline_multi(
        job_dir=str(job_dir),
        output_path=str(output_path),
        sabers=[
            {"color": "green", "intensity": 0.6, "voice": "neutral"},
            {"color": "blue", "intensity": 0.3, "voice": "jedi"},
        ],
    )

    assert result == str(output_path)
    assert output_path.exists() and output_path.stat().st_size > 0


def test_rerender_pipeline_multi_rejects_a_saber_count_mismatch(tmp_path, tiny_video_path):
    job_dir = tmp_path / "job"
    masks_dir = job_dir / "masks" / "0"
    for i in range(3):
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:20, 5:11] = True
        save_mask(str(masks_dir), i, mask)
    motion_dir = job_dir / "motion"
    motion_dir.mkdir(exist_ok=True)
    compute_motion(str(masks_dir), str(motion_dir / "0.npz"))
    (job_dir / "video_meta.txt").write_text("10.0\n3\n")
    job_meta.write_job_meta(str(job_dir), source_video=str(tiny_video_path), object_ids=[0])

    with pytest.raises(ValueError, match="1 tracked object"):
        rerender_pipeline_multi(
            job_dir=str(job_dir),
            output_path=str(tmp_path / "final.mp4"),
            sabers=[
                {"color": "red", "intensity": 0.35, "voice": "neutral"},
                {"color": "blue", "intensity": 0.35, "voice": "neutral"},
            ],
        )
