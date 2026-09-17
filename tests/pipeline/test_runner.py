import inspect
import os
import shutil

import cv2
import numpy as np
import pytest

from lightsaber_fx.pipeline import job_meta
from lightsaber_fx.pipeline.blade import compute_motion, load_motion, save_mask
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
    # run_pipeline (the CLI's single-object path) has no way of knowing
    # whether the user accepted an auto-detected proposal verbatim or
    # clicked their own override -- see cli.py's `run` command -- so it
    # must not guess "motion" or default to the misleading "manual". See
    # the vision-assisted-detection fix brief, item 6.
    assert meta["prompts"][0]["source"] == "unrecorded"

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


def test_rerender_pipeline_rejects_a_multi_object_job_with_a_clear_message(tmp_path, tiny_video_path):
    # A multi-object job keeps masks/{obj_id}/ and motion/{obj_id}.npz, which
    # require_rerenderable checks and passes -- so this function used to sail
    # through the guard and then die on a bare FileNotFoundError hunting for
    # the flat motion.npz it expects. The web app writes object_ids even for a
    # single saber, so this is every web-app job, not an exotic case.
    job_dir = tmp_path / "job"
    masks_dir = job_dir / "masks" / "0"
    for i in range(3):
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:20, 5:11] = True
        save_mask(str(masks_dir), i, mask)
    (job_dir / "motion").mkdir(parents=True, exist_ok=True)
    compute_motion(str(masks_dir), str(job_dir / "motion" / "0.npz"))
    (job_dir / "video_meta.txt").write_text("10.0\n3\n")
    job_meta.write_job_meta(str(job_dir), source_video=str(tiny_video_path), object_ids=[0])

    with pytest.raises(JobNotRerenderableError) as excinfo:
        rerender_pipeline(job_dir=str(job_dir), output_path=str(tmp_path / "out.mp4"))

    message = str(excinfo.value)
    assert "only handles single-object jobs" in message
    assert "web app" in message
    assert not (tmp_path / "out.mp4").exists()


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


def _blob(_i):
    # A compact square, not a blade: elongation ~1, well under MIN_ELONGATION
    # (6) -- the shape job 01 (a Mixkit knights-battling test clip) actually
    # produced when tracking locked onto a patch of tunic cloth instead of
    # the sword. `fit_blade` succeeds on it (it's a perfectly good shape,
    # just the wrong one), so this is a distinct case from `_blank`.
    mask = np.zeros((48, 64), dtype=bool)
    mask[10:34, 20:44] = True
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


@requires_ffmpeg
def test_run_pipeline_warns_but_renders_for_a_consistently_blob_shaped_track(
    tmp_path, monkeypatch, tiny_video_path
):
    # A track that is fully covered (every frame has *a* blade-fit shape)
    # but never actually blade-shaped -- job 01's real failure, invisible to
    # the coverage guard above since nothing here is missing or empty. Same
    # "warn and render anyway" contract as low coverage: degraded quality,
    # not doomed, and the render is still the user's fastest way to look.
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", _fake_track_writing(_blob))

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
    assert "elongation" in warnings[0]
    assert "100%" in warnings[0]
    assert output_path.exists()


@requires_ffmpeg
def test_run_pipeline_does_not_warn_about_elongation_for_a_properly_elongated_track(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_object", _fake_track_writing(_blade))

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
    assert warnings == []
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


def _fake_track_objects_writing(mask_factory_by_obj_id):
    """Build a `track_objects` stub writing each object's own masks, so a
    test can make one saber track cleanly while another finds nothing."""
    def fake_track_objects(frames_dir, prompts, checkpoint_path, config_name,
                           device, n_frames, progress_cb=None):
        for prompt in prompts:
            os.makedirs(prompt["masks_dir"], exist_ok=True)
            factory = mask_factory_by_obj_id[prompt["obj_id"]]
            for i in range(n_frames):
                save_mask(prompt["masks_dir"], i, factory(i))
    return fake_track_objects


def test_run_pipeline_multi_error_for_a_dead_saber_names_which_saber(
    tmp_path, monkeypatch, tiny_video_path
):
    # "The click points were wrong" is not actionable when four sabers were
    # tracked and the message doesn't say whose points. Saber 0 tracks fine
    # here; saber 1 finds nothing. Stubbing glow to fail if called also keeps
    # the existing "raise before burning the glow stage" guarantee honest.
    def fail_if_called(*args, **kwargs):
        raise AssertionError("render_glow_multi ran despite a track finding no blade")

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blank}),
    )
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.render_glow_multi", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(RuntimeError) as excinfo:
        run_pipeline_multi(
            input_video=str(tiny_video_path),
            sabers=[
                {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
                {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35, "voice": "neutral"},
            ],
            output_path=str(tmp_path / "final.mp4"),
            job_dir=str(job_dir),
            checkpoint_path="unused",
            device="cpu",
        )

    message = str(excinfo.value)
    assert "no blade for saber 1 in any of" in message, message
    assert "click points" in message  # the existing advice is still there


@requires_ffmpeg
def test_run_pipeline_multi_warns_for_the_specific_saber_that_is_blob_shaped(
    tmp_path, monkeypatch, tiny_video_path
):
    # Saber 0 tracks a real blade the whole time; saber 1 tracks a blob the
    # whole time. Both render_glow_multi allowed to run (not blocked, same
    # as single-object) -- what matters is that the warning names saber 1,
    # not saber 0, and there's exactly one of it.
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blob}),
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    messages = []

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35, "voice": "neutral"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
        progress_cb=lambda stage, pct, message: messages.append((stage, message)),
    )

    warnings = [m for stage, m in messages if stage == "motion" and m.startswith("warning:")]
    assert len(warnings) == 1
    assert "saber 1" in warnings[0]
    assert "elongation" in warnings[0]


def test_run_pipeline_multi_rejects_an_unsupported_saber_count(tmp_path, monkeypatch, tiny_video_path):
    # sabers=[] used to raise a bare KeyError from deep inside the per-object
    # path building instead of saying what was wrong.
    def fail_if_called(*a, **k):
        raise AssertionError("extract_frames should not run for an invalid saber count")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.extract_frames", fail_if_called)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    one = {"points": [[1, 1]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"}

    for sabers in ([], [one] * 5):
        with pytest.raises(ValueError, match="1-4 sabers"):
            run_pipeline_multi(
                input_video=str(tiny_video_path),
                sabers=sabers,
                output_path=str(tmp_path / "final.mp4"),
                job_dir=str(job_dir),
                checkpoint_path="unused",
                device="cpu",
            )


def test_run_pipeline_multi_threads_each_sabers_prompt_frame_to_the_tracker(
    tmp_path, monkeypatch, tiny_video_path
):
    # Automatic detection reports the frame an object was easiest to find --
    # usually mid-swing. Dropping it here silently applied those points to
    # frame 0 instead, against a frame the object had already left.
    captured = {}

    def fake_track_objects(frames_dir, prompts, checkpoint_path, config_name,
                           device, n_frames, progress_cb=None):
        captured["prompts"] = [dict(p) for p in prompts]
        raise RuntimeError("stop here -- only the prompts matter")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_objects", fake_track_objects)
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    with pytest.raises(RuntimeError, match="stop here"):
        run_pipeline_multi(
            input_video=str(tiny_video_path),
            sabers=[
                {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35,
                 "voice": "neutral", "prompt_frame": 3},
                {"points": [[20, 25]], "labels": [1], "color": "blue", "intensity": 0.35,
                 "voice": "neutral"},
            ],
            output_path=str(tmp_path / "final.mp4"),
            job_dir=str(job_dir),
            checkpoint_path="unused",
            device="cpu",
        )

    assert captured["prompts"][0]["prompt_frame"] == 3
    assert captured["prompts"][1]["prompt_frame"] == 0  # defaults when the client omits it


def test_run_pipeline_multi_records_each_sabers_source_in_job_meta(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35,
             "voice": "neutral", "source": "vlm"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35,
             "voice": "neutral", "source": "manual"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    meta = job_meta.read_job_meta(str(job_dir))
    assert [p["source"] for p in meta["prompts"]] == ["vlm", "manual"]


def test_run_pipeline_multi_calls_reconcile_pair_for_a_two_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    calls = []

    def fake_reconcile_pair(frames_dir, masks_dir_0, masks_dir_1, n_frames, checkpoint_path, config_name, device,
                             client=None):
        calls.append((masks_dir_0, masks_dir_1, n_frames))
        return False

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.reconcile_pair", fake_reconcile_pair)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert len(calls) == 1
    masks_dir_0, masks_dir_1, _n_frames = calls[0]
    assert masks_dir_0 == str(job_dir / "masks" / "0")
    assert masks_dir_1 == str(job_dir / "masks" / "1")


def test_run_pipeline_multi_skips_reconcile_pair_for_a_single_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("reconcile_pair should not run for a single-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.reconcile_pair", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[{"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"}],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )


def test_run_pipeline_multi_skips_reconcile_pair_for_a_four_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade, 2: _blade, 3: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("reconcile_pair should not run for a four-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.reconcile_pair", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "green", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )


def test_run_pipeline_multi_calls_retrack_overlap_runs_for_a_two_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    calls = []

    def fake_retrack_overlap_runs(frames_dir, masks_dir_0, masks_dir_1, motion_path_0, motion_path_1,
                                   n_frames, checkpoint_path, config_name, device):
        # Must run after both objects' compute_motion (it needs their
        # finished motion.npz to find overlap runs) and before
        # suppress_overlap_bleed.
        assert os.path.exists(motion_path_0)
        assert os.path.exists(motion_path_1)
        calls.append((masks_dir_0, masks_dir_1, n_frames))
        return set(), []

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.retrack_overlap_runs", fake_retrack_overlap_runs
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert len(calls) == 1
    masks_dir_0, masks_dir_1, _n_frames = calls[0]
    assert masks_dir_0 == str(job_dir / "masks" / "0")
    assert masks_dir_1 == str(job_dir / "masks" / "1")


def test_run_pipeline_multi_skips_retrack_overlap_runs_for_a_single_saber_job(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("retrack_overlap_runs should not run for a single-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.retrack_overlap_runs", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[{"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"}],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )


def test_run_pipeline_multi_skips_retrack_overlap_runs_for_a_four_saber_job(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade, 2: _blade, 3: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("retrack_overlap_runs should not run for a four-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.retrack_overlap_runs", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "green", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )


def test_run_pipeline_multi_calls_compute_hilt_overrides_for_a_two_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.retrack_overlap_runs",
        lambda *a, **k: (set(), [(3, 5)]),
    )
    calls = []

    def fake_compute_hilt_overrides(frames_dir, masks_dir_a, masks_dir_b, motion_path_a, motion_path_b,
                                     exclude_frame_ranges=()):
        # Must run after both objects' compute_motion (it reads their
        # finished motion.npz) and before suppress_overlap_bleed.
        assert os.path.exists(motion_path_a)
        assert os.path.exists(motion_path_b)
        calls.append(list(exclude_frame_ranges))
        return {7: (1.0, 2.0)}, {}

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.compute_hilt_overrides", fake_compute_hilt_overrides
    )

    overload_calls = []
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.suppress_overlap_bleed",
        lambda *a, hilt_overrides_a=None, hilt_overrides_b=None, **k: overload_calls.append(
            (hilt_overrides_a, hilt_overrides_b)
        ) or 0,
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert calls == [[(3, 5)]]  # same resolved_ranges threaded through as exclude_frame_ranges
    assert overload_calls == [({7: (1.0, 2.0)}, {})]


def test_run_pipeline_multi_skips_compute_hilt_overrides_for_a_single_saber_job(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("compute_hilt_overrides should not run for a single-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.compute_hilt_overrides", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[{"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"}],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )


def test_run_pipeline_multi_reruns_compute_motion_for_objects_retrack_overlap_runs_patches(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )

    real_compute_motion = compute_motion
    calls = []

    def counting_compute_motion(masks_dir, motion_out_path, progress_cb=None):
        calls.append(masks_dir)
        return real_compute_motion(masks_dir, motion_out_path, progress_cb=progress_cb)

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.compute_motion", counting_compute_motion)
    # Object 0 only: simulates a validated re-track that patched its masks.
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.retrack_overlap_runs", lambda *a, **k: ({0}, []))
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.suppress_overlap_bleed", lambda *a, **k: 0)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    obj0_dir = str(job_dir / "masks" / "0")
    obj1_dir = str(job_dir / "masks" / "1")
    assert calls.count(obj0_dir) == 2  # initial pass + re-run after retrack patched it
    assert calls.count(obj1_dir) == 1  # untouched, so no re-run needed


def test_run_pipeline_multi_passes_resolved_ranges_as_exclude_frame_ranges(
    tmp_path, monkeypatch, tiny_video_path
):
    # retrack_overlap_runs' second return value (runs it fully resolved)
    # must reach suppress_overlap_bleed as exclude_frame_ranges -- silently
    # dropping this wiring would let suppress_overlap_bleed "fix" a run
    # that a validated independent re-track already got right, overwriting
    # accurate masks with a worse interpolated approximation. See
    # blade.suppress_overlap_bleed's docstring for why genuine contact
    # still reads as high mask IoU.
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.retrack_overlap_runs",
        lambda *a, **k: ({0, 1}, [(12, 34)]),
    )
    calls = []
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.suppress_overlap_bleed",
        lambda *a, exclude_frame_ranges=(), **k: calls.append(exclude_frame_ranges) or 0,
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert calls == [[(12, 34)]]


def test_run_pipeline_multi_does_not_exclude_reconcile_pairs_own_output(
    tmp_path, monkeypatch, tiny_video_path
):
    # reconcile_pair's return alone must NOT shrink what
    # suppress_overlap_bleed is allowed to correct -- an earlier version
    # of this wiring treated a successful reconcile_pair's re-tracked span
    # as accurate for its entire length and excluded it, which left a
    # real, later overlap within that same span uncorrected (confirmed via
    # a full real end-to-end run: a completely missing blade for 162
    # frames). Only retrack_overlap_runs' resolved_ranges -- which carries
    # an actual accuracy check -- may exclude anything.
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    monkeypatch.setattr("lightsaber_fx.pipeline.runner.reconcile_pair", lambda *a, **k: True)
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.retrack_overlap_runs",
        lambda *a, **k: (set(), [(12, 34)]),
    )
    calls = []
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.suppress_overlap_bleed",
        lambda *a, exclude_frame_ranges=(), **k: calls.append(list(exclude_frame_ranges)) or 0,
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert calls == [[(12, 34)]]  # only retrack_overlap_runs' resolved range, nothing from reconcile_pair


def test_run_pipeline_multi_calls_suppress_overlap_bleed_for_a_two_saber_job(tmp_path, monkeypatch, tiny_video_path):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade}),
    )
    calls = []

    def fake_suppress_overlap_bleed(motion_path_a, masks_dir_a, motion_path_b, masks_dir_b,
                                     exclude_frame_ranges=(), hilt_overrides_a=None, hilt_overrides_b=None):
        # Must run after both objects' compute_motion, since it patches
        # already-written motion.npz rather than producing it.
        assert os.path.exists(motion_path_a)
        assert os.path.exists(motion_path_b)
        calls.append((motion_path_a, masks_dir_a, motion_path_b, masks_dir_b))
        return 0

    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.suppress_overlap_bleed", fake_suppress_overlap_bleed
    )

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert len(calls) == 1
    _motion_path_a, masks_dir_a, _motion_path_b, masks_dir_b = calls[0]
    assert masks_dir_a == str(job_dir / "masks" / "0")
    assert masks_dir_b == str(job_dir / "masks" / "1")


def test_run_pipeline_multi_skips_suppress_overlap_bleed_for_a_single_saber_job(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("suppress_overlap_bleed should not run for a single-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.suppress_overlap_bleed", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[{"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"}],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )


def test_run_pipeline_multi_skips_suppress_overlap_bleed_for_a_four_saber_job(
    tmp_path, monkeypatch, tiny_video_path
):
    monkeypatch.setattr(
        "lightsaber_fx.pipeline.runner.track_objects",
        _fake_track_objects_writing({0: _blade, 1: _blade, 2: _blade, 3: _blade}),
    )

    def fail_if_called(*a, **k):
        raise AssertionError("suppress_overlap_bleed should not run for a four-saber job")

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.suppress_overlap_bleed", fail_if_called)

    job_dir = tmp_path / "job"
    job_dir.mkdir()

    run_pipeline_multi(
        input_video=str(tiny_video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "green", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
        ],
        output_path=str(tmp_path / "final.mp4"),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )


def _write_longer_video(path, n_frames, width=64, height=48, fps=10.0):
    """Like the top-level `tiny_video_path` fixture's clip, but with a
    frame count this file controls -- merge detection needs 15+ sustained
    frames, more than that fixture's default 5."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for i in range(n_frames):
        frame = np.full((height, width, 3), (i * 5) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_run_pipeline_multi_recovers_from_a_simulated_crossing_end_to_end(tmp_path, monkeypatch):
    """Full pipeline, reproducing the shape of the real bug this feature
    fixes: two objects track separately, then (simulating track_objects
    losing the distinction between them) both collapse onto the same
    target for a sustained stretch, and reconciliation recovers the lost
    one. Mirrors the design doc's real-footage spike; this is the
    synthetic, fast, CI-safe equivalent driven through the full pipeline
    entry point rather than reconcile_pair directly."""
    video_path = tmp_path / "longer.mp4"
    _write_longer_video(video_path, n_frames=40)

    def fake_track_objects(frames_dir, prompts, checkpoint_path, config_name, device, n_frames, progress_cb=None):
        for prompt in prompts:
            os.makedirs(prompt["masks_dir"], exist_ok=True)
            for i in range(n_frames):
                # Object 0 tracks separately (x=10) for frames 0-4, then
                # collapses onto object 1's target (x=40) from frame 5 on.
                x = 10 if (prompt["obj_id"] == 0 and i < 5) else 40
                mask = np.zeros((48, 64), dtype=bool)
                mask[10:34, x:x + 6] = True
                save_mask(prompt["masks_dir"], i, mask)

    def fake_reacquire_pair(frames_dir, search_start_frame, checkpoint_path, config_name, device,
                             client=None, **kwargs):
        reacquire_frame = search_start_frame + 3
        return reacquire_frame, [
            {"centroid": (10.0, 22.0), "points": [[10, 20], [10, 22], [10, 24]]},
            {"centroid": (40.0, 22.0), "points": [[40, 20], [40, 22], [40, 24]]},
        ]

    def fake_track_object_for_reacquire(frames_dir, out_masks_dir, points, labels, checkpoint_path, config_name,
                                         device, n_frames, prompt_frame=0, progress_cb=None):
        for i in range(prompt_frame, n_frames):
            mask = np.zeros((48, 64), dtype=bool)
            mask[10:34, 16:22] = True  # x=16: distinguishable from the frozen reference (x=10) and object 1's target (x=40)
            save_mask(out_masks_dir, i, mask)

    monkeypatch.setattr("lightsaber_fx.pipeline.runner.track_objects", fake_track_objects)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.reacquire_pair", fake_reacquire_pair)
    monkeypatch.setattr("lightsaber_fx.pipeline.reacquire.track_object", fake_track_object_for_reacquire)

    job_dir = tmp_path / "job"
    job_dir.mkdir()
    output_path = tmp_path / "final.mp4"

    run_pipeline_multi(
        input_video=str(video_path),
        sabers=[
            {"points": [[10, 15]], "labels": [1], "color": "red", "intensity": 0.35, "voice": "neutral"},
            {"points": [[10, 15]], "labels": [1], "color": "blue", "intensity": 0.5, "voice": "sith"},
        ],
        output_path=str(output_path),
        job_dir=str(job_dir),
        checkpoint_path="unused",
        device="cpu",
    )

    assert output_path.exists() and output_path.stat().st_size > 0
    motion_0 = load_motion(str(job_dir / "motion" / "0.npz"))
    # Frame 0 (before the merge): original track, x=10.
    assert motion_0["centroid"][0][0] < 20
    # Frame 6 (within the frozen gap [merge_start=5, reacquire_frame=8)): held at the
    # clean reference frame's position (x=10), not yet recovered and not the merged x=40.
    assert motion_0["centroid"][6][0] < 20
    # Frame 39 (after re-acquisition): freshly re-tracked at x=16 (mask columns 16:22,
    # centroid 18.5) -- distinguishable from both the frozen value (x=10, centroid 12.5)
    # and object 1's target (x=40, centroid 42.5), proving patch_masks's fresh-track copy
    # loop, find_clean_reference's frame choice, and reacquire_pair's stub all ran for
    # real rather than degenerating to a no-op.
    assert 15 < motion_0["centroid"][-1][0] < 25


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
