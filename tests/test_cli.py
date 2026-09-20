from pathlib import Path

import cv2
import numpy as np
from click.testing import CliRunner

import glowblade.paths as paths_module
from glowblade.cli import main
from glowblade.pipeline.blade import compute_motion, save_mask
from glowblade.pipeline.job_meta import JobNotRerenderableError, write_job_meta


def _fake_checkpoint(appdata_dir):
    """Create a stand-in checkpoint file so cli.run's setup pre-flight check passes."""
    checkpoint = appdata_dir / "checkpoints" / "sam2.1_hiera_small.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake checkpoint")
    return checkpoint


def test_cli_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "glowblade" in result.output or "Usage" in result.output


def test_help_lists_all_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for name in ("setup", "run", "serve", "clean", "rerender", "jobs"):
        assert name in result.output


def test_setup_command_invokes_bootstrap(monkeypatch):
    calls = {}
    monkeypatch.setattr("glowblade.cli.bootstrap", lambda force: calls.__setitem__("force", force))

    runner = CliRunner()
    result = runner.invoke(main, ["setup", "--force"])

    assert result.exit_code == 0
    assert calls["force"] is True


def test_clean_command_reports_removed_count(monkeypatch):
    monkeypatch.setattr("glowblade.cli.paths.clean_jobs", lambda: 3)

    runner = CliRunner()
    result = runner.invoke(main, ["clean"])

    assert result.exit_code == 0
    assert "3" in result.output


def test_run_command_parses_options_and_calls_run_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")

    monkeypatch.setattr("glowblade.cli.extract_frame_at", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.detect_blade", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.pick_points_interactive", lambda *a, **k: ([[1, 2]], [1]))
    monkeypatch.setattr("glowblade.cli.select_device", lambda: "cpu")

    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr("glowblade.cli.run_pipeline", fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video), "--color", "blue", "--intensity", "0.5"])

    assert result.exit_code == 0
    assert captured["color"] == "blue"
    assert captured["intensity"] == 0.5
    assert captured["points"] == [[1, 2]]
    assert captured["labels"] == [1]


def test_run_command_aborts_when_no_points_selected(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")
    monkeypatch.setattr("glowblade.cli.extract_frame_at", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.detect_blade", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.pick_points_interactive", lambda *a, **k: ([], []))

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video)])

    assert result.exit_code != 0


def test_run_command_fails_fast_when_sam2_not_set_up(tmp_path, monkeypatch):
    # No checkpoint file created: cli.run's pre-flight check (A7) should raise
    # before ever touching detection, frame extraction or the picker.
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))

    def fail_if_called(*a, **k):
        raise AssertionError("should not be reached when the checkpoint is missing")

    monkeypatch.setattr("glowblade.cli.extract_frame_at", fail_if_called)
    monkeypatch.setattr("glowblade.cli.detect_blade", fail_if_called)
    monkeypatch.setattr("glowblade.cli.pick_points_interactive", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video)])

    assert result.exit_code != 0
    assert "glowblade setup" in result.output


def test_run_command_rejects_out_of_range_intensity(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video), "--intensity", "5"])

    assert result.exit_code != 0
    assert "0.0" in result.output and "1.0" in result.output


def test_serve_command_invokes_uvicorn_with_host_and_port(monkeypatch):
    captured = {}

    def fake_run(app_path, host, port):
        captured["app_path"] = app_path
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--host", "0.0.0.0", "--port", "9000"])

    assert result.exit_code == 0
    assert captured == {"app_path": "glowblade.web.server:app", "host": "0.0.0.0", "port": 9000}


def test_serve_open_browser_flag_does_not_error(monkeypatch):
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: None)
    monkeypatch.setattr("webbrowser.open", lambda *a, **k: None)

    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--open-browser"])

    assert result.exit_code == 0


def _fake_full_render(job_dir_path):
    """Simulate what a real run_pipeline leaves behind: frames/ and masks/
    both populated."""
    (job_dir_path / "frames").mkdir()
    (job_dir_path / "masks").mkdir()


def test_run_command_keeps_masks_but_removes_frames_by_default(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")

    monkeypatch.setattr("glowblade.cli.extract_frame_at", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.detect_blade", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.pick_points_interactive", lambda *a, **k: ([[1, 2]], [1]))
    monkeypatch.setattr("glowblade.cli.select_device", lambda: "cpu")

    def fake_run_pipeline(**kwargs):
        job_dir_path = Path(kwargs["job_dir"])
        _fake_full_render(job_dir_path)
        return kwargs["output_path"]

    monkeypatch.setattr("glowblade.cli.run_pipeline", fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video)])
    assert result.exit_code == 0

    jobs_dir = tmp_path / "appdata" / "jobs"
    (job_dir,) = list(jobs_dir.iterdir())
    # masks/ is small (W1) and needed for `rerender` -- always kept now.
    assert (job_dir / "masks").is_dir()
    # frames/ is still the disk hog -- stays gated behind --keep-intermediate.
    assert not (job_dir / "frames").exists()


def test_run_command_keep_intermediate_also_keeps_frames(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")

    monkeypatch.setattr("glowblade.cli.extract_frame_at", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.detect_blade", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.pick_points_interactive", lambda *a, **k: ([[1, 2]], [1]))
    monkeypatch.setattr("glowblade.cli.select_device", lambda: "cpu")

    def fake_run_pipeline(**kwargs):
        job_dir_path = Path(kwargs["job_dir"])
        _fake_full_render(job_dir_path)
        return kwargs["output_path"]

    monkeypatch.setattr("glowblade.cli.run_pipeline", fake_run_pipeline)

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video), "--keep-intermediate"])
    assert result.exit_code == 0

    jobs_dir = tmp_path / "appdata" / "jobs"
    (job_dir,) = list(jobs_dir.iterdir())
    assert (job_dir / "masks").is_dir()
    assert (job_dir / "frames").is_dir()


# ---------------------------------------------------------------------------
# rerender
# ---------------------------------------------------------------------------


def test_rerender_command_parses_options_and_calls_rerender_pipeline(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "abc123").mkdir(parents=True)
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    captured = {}

    def fake_rerender_pipeline(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr("glowblade.cli.rerender_pipeline", fake_rerender_pipeline)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["rerender", "abc123", "--color", "blue", "--intensity", "0.6", "--voice", "deep",
         "--no-blade-extend", "--output", "out.mp4"],
    )

    assert result.exit_code == 0, result.output
    assert captured["job_dir"] == str(jobs_dir / "abc123")
    assert captured["color"] == "blue"
    assert captured["intensity"] == 0.6
    assert captured["voice"] == "deep"
    assert captured["blade_extend"] is False
    assert captured["output_path"] == "out.mp4"


def test_rerender_command_rejects_out_of_range_intensity(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "abc123").mkdir(parents=True)
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["rerender", "abc123", "--intensity", "5"])

    assert result.exit_code != 0
    assert "0.0" in result.output and "1.0" in result.output


def test_rerender_command_errors_when_job_id_unknown(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["rerender", "does-not-exist"])

    assert result.exit_code != 0
    assert "No such job" in result.output


def test_rerender_command_reports_clear_error_when_job_not_rerenderable(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "abc123").mkdir(parents=True)
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    def fake_rerender_pipeline(**kwargs):
        raise JobNotRerenderableError("Job 'abc123' is not re-renderable: no masks/")

    monkeypatch.setattr("glowblade.cli.rerender_pipeline", fake_rerender_pipeline)

    runner = CliRunner()
    result = runner.invoke(main, ["rerender", "abc123"])

    assert result.exit_code != 0
    assert "no masks/" in result.output


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def test_jobs_command_lists_rerenderable_and_annotates_broken_job(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    good = jobs_dir / "good123"
    good.mkdir(parents=True)
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True
    save_mask(str(good / "masks"), 0, mask)
    compute_motion(str(good / "masks"), str(good / "motion.npz"))
    (good / "video_meta.txt").write_text("24.0\n1\n")
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    write_job_meta(str(good), source_video=str(video))

    broken = jobs_dir / "broken456"
    broken.mkdir(parents=True)

    runner = CliRunner()
    result = runner.invoke(main, ["jobs"])

    assert result.exit_code == 0
    assert "good123" in result.output
    assert "re-renderable" in result.output
    assert "broken456" in result.output
    assert "NOT re-renderable" in result.output


def test_jobs_command_shows_the_object_count_for_a_multi_object_job(tmp_path, monkeypatch):
    # `rerender` can't touch a multi-object job, so a listing that shows it
    # identically to a single-object one is lying about what it is offering.
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    single = jobs_dir / "single1"
    single.mkdir(parents=True)
    save_mask(str(single / "masks"), 0, mask)
    compute_motion(str(single / "masks"), str(single / "motion.npz"))
    (single / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(str(single), source_video=str(video))

    multi = jobs_dir / "multi2"
    multi.mkdir(parents=True)
    for oid in (0, 1):
        save_mask(str(multi / "masks" / str(oid)), 0, mask)
        (multi / "motion").mkdir(exist_ok=True)
        compute_motion(str(multi / "masks" / str(oid)), str(multi / "motion" / f"{oid}.npz"))
    (multi / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(str(multi), source_video=str(video), object_ids=[0, 1])

    result = CliRunner().invoke(main, ["jobs"])

    assert result.exit_code == 0
    lines = {line.split()[0]: line for line in result.output.splitlines() if line.strip()}
    assert "objects=2" in lines["multi2"], lines["multi2"]
    assert "objects=" not in lines["single1"], lines["single1"]


def test_jobs_command_reports_no_jobs_found(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["jobs"])

    assert result.exit_code == 0
    assert "No jobs found" in result.output


def _proposal(frame_index=135, points=None):
    """A BladeProposal like detect_blade returns, with a mask sized to match
    what the picker would be shown."""
    import numpy as np

    from glowblade.pipeline.detect import BladeProposal, MotionSeed

    points = points or [[377, 541], [483, 557], [588, 563]]
    mask = np.zeros((24, 32), dtype=bool)
    mask[10:14, 4:28] = True
    return BladeProposal(
        frame_index=frame_index, points=points, labels=[1] * len(points),
        mask=mask, elongation=10.1,
        seed=MotionSeed(frame_index=frame_index, point=points[0], speed=16.1, area=1825),
    )


def _auto_run_harness(tmp_path, monkeypatch, proposal):
    """Set up a `run` invocation with detection returning `proposal`, and
    capture what the picker was shown and what run_pipeline received."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")
    monkeypatch.setattr("glowblade.cli.select_device", lambda: "cpu")
    monkeypatch.setattr("glowblade.cli.detect_blade", lambda *a, **k: proposal)

    seen = {}
    monkeypatch.setattr(
        "glowblade.cli.extract_frame_at",
        lambda video_path, index, out_path: seen.update(extracted_index=index),
    )

    def fake_picker(frame_path, proposal=None):
        seen["picker_proposal"] = proposal
        return (proposal.points, proposal.labels) if proposal else ([[1, 2]], [1])

    monkeypatch.setattr("glowblade.cli.pick_points_interactive", fake_picker)

    def fake_run_pipeline(**kwargs):
        seen.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr("glowblade.cli.run_pipeline", fake_run_pipeline)
    return video, seen


def test_run_command_uses_the_detected_frame_as_the_prompt_frame(tmp_path, monkeypatch):
    # The single most important wiring detail. Detection reports points
    # against the frame it found them in -- frame 135 of 300 on the real 10 s
    # clip -- so the frame extracted for confirmation, the frame the overlay
    # is drawn on, and the frame the tracker prompts must all be that same
    # one. Reading those coordinates against frame 0 lands them on whatever
    # is there instead, which is a mistake with no symptom until the render
    # comes out wrong.
    proposal = _proposal(frame_index=135)
    video, seen = _auto_run_harness(tmp_path, monkeypatch, proposal)

    result = CliRunner().invoke(main, ["run", str(video)])

    assert result.exit_code == 0
    assert seen["extracted_index"] == 135
    assert seen["prompt_frame"] == 135
    assert seen["points"] == proposal.points


def test_run_command_passes_the_proposal_to_the_picker_for_confirmation(tmp_path, monkeypatch):
    # "Propose, then confirm" only holds if the proposal actually reaches the
    # picker; detecting and then rendering without showing anything would
    # pass every other assertion here.
    proposal = _proposal()
    video, seen = _auto_run_harness(tmp_path, monkeypatch, proposal)

    result = CliRunner().invoke(main, ["run", str(video)])

    assert result.exit_code == 0
    assert seen["picker_proposal"] is proposal
    assert "elongation 10.1" in result.output


def test_run_command_no_auto_skips_detection_entirely(tmp_path, monkeypatch):
    # --no-auto must not pay for detection at all, not merely ignore its
    # result -- on a long clip the scan plus segmentation is several seconds.
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")
    monkeypatch.setattr("glowblade.cli.select_device", lambda: "cpu")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("detect_blade ran despite --no-auto")

    monkeypatch.setattr("glowblade.cli.detect_blade", fail_if_called)
    monkeypatch.setattr("glowblade.cli.extract_frame_at", lambda *a, **k: None)
    monkeypatch.setattr("glowblade.cli.pick_points_interactive", lambda *a, **k: ([[1, 2]], [1]))

    captured = {}
    monkeypatch.setattr(
        "glowblade.cli.run_pipeline",
        lambda **kwargs: (captured.update(kwargs), kwargs["output_path"])[1],
    )

    result = CliRunner().invoke(main, ["run", str(video), "--no-auto"])

    assert result.exit_code == 0
    assert captured["prompt_frame"] == 0


def test_run_command_falls_back_to_clicking_when_detection_finds_nothing(tmp_path, monkeypatch):
    video, seen = _auto_run_harness(tmp_path, monkeypatch, None)

    result = CliRunner().invoke(main, ["run", str(video)])

    assert result.exit_code == 0
    assert "Couldn't find one automatically" in result.output
    assert seen["extracted_index"] == 0
    assert seen["prompt_frame"] == 0
    assert seen["points"] == [[1, 2]]


def test_inspect_command_prints_each_sabers_source(tmp_path, monkeypatch, capsys):
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr("glowblade.cli.paths.get_jobs_dir", lambda: jobs_dir)

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:5, 2:5] = True

    job_dir = jobs_dir / "job1"
    job_dir.mkdir(parents=True)
    cv2.imwrite(str(job_dir / "frame0.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    for oid in (0, 1):
        save_mask(str(job_dir / "masks" / str(oid)), 0, mask)
        (job_dir / "motion").mkdir(exist_ok=True)
        compute_motion(str(job_dir / "masks" / str(oid)), str(job_dir / "motion" / f"{oid}.npz"))
    (job_dir / "video_meta.txt").write_text("24.0\n1\n")
    write_job_meta(
        str(job_dir), source_video=str(video), object_ids=[0, 1],
        prompts=[
            {"points": [[1, 1]], "labels": [1], "prompt_frame": 0, "source": "vlm"},
            {"points": [[2, 2]], "labels": [1], "prompt_frame": 0, "source": "manual"},
        ],
    )

    result = CliRunner().invoke(main, ["inspect", "job1"])

    assert result.exit_code == 0, result.output
    assert "saber 0: source=vlm" in result.output
    assert "saber 1: source=manual" in result.output
