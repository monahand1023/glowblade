from pathlib import Path

import numpy as np
from click.testing import CliRunner

import lightsaber_fx.paths as paths_module
from lightsaber_fx.cli import main
from lightsaber_fx.pipeline.blade import compute_motion, save_mask
from lightsaber_fx.pipeline.job_meta import JobNotRerenderableError, write_job_meta


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
    assert "lightsaber-fx" in result.output or "Usage" in result.output


def test_help_lists_all_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for name in ("setup", "run", "serve", "clean", "rerender", "jobs"):
        assert name in result.output


def test_setup_command_invokes_bootstrap(monkeypatch):
    calls = {}
    monkeypatch.setattr("lightsaber_fx.cli.bootstrap", lambda force: calls.__setitem__("force", force))

    runner = CliRunner()
    result = runner.invoke(main, ["setup", "--force"])

    assert result.exit_code == 0
    assert calls["force"] is True


def test_clean_command_reports_removed_count(monkeypatch):
    monkeypatch.setattr("lightsaber_fx.cli.paths.clean_jobs", lambda: 3)

    runner = CliRunner()
    result = runner.invoke(main, ["clean"])

    assert result.exit_code == 0
    assert "3" in result.output


def test_run_command_parses_options_and_calls_run_pipeline(tmp_path, monkeypatch):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))
    _fake_checkpoint(tmp_path / "appdata")

    monkeypatch.setattr("lightsaber_fx.cli.extract_first_frame", lambda *a, **k: None)
    monkeypatch.setattr("lightsaber_fx.cli.pick_points_interactive", lambda *a, **k: ([[1, 2]], [1]))
    monkeypatch.setattr("lightsaber_fx.cli.select_device", lambda: "cpu")

    captured = {}

    def fake_run_pipeline(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr("lightsaber_fx.cli.run_pipeline", fake_run_pipeline)

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
    monkeypatch.setattr("lightsaber_fx.cli.extract_first_frame", lambda *a, **k: None)
    monkeypatch.setattr("lightsaber_fx.cli.pick_points_interactive", lambda *a, **k: ([], []))

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video)])

    assert result.exit_code != 0


def test_run_command_fails_fast_when_sam2_not_set_up(tmp_path, monkeypatch):
    # No checkpoint file created: cli.run's pre-flight check (A7) should raise
    # before ever touching extract_first_frame or the interactive picker.
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake video bytes")
    monkeypatch.setattr(paths_module.platformdirs, "user_data_dir", lambda name: str(tmp_path / "appdata"))

    def fail_if_called(*a, **k):
        raise AssertionError("should not be reached when the checkpoint is missing")

    monkeypatch.setattr("lightsaber_fx.cli.extract_first_frame", fail_if_called)
    monkeypatch.setattr("lightsaber_fx.cli.pick_points_interactive", fail_if_called)

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video)])

    assert result.exit_code != 0
    assert "lightsaber-fx setup" in result.output


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
    assert captured == {"app_path": "lightsaber_fx.web.server:app", "host": "0.0.0.0", "port": 9000}


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

    monkeypatch.setattr("lightsaber_fx.cli.extract_first_frame", lambda *a, **k: None)
    monkeypatch.setattr("lightsaber_fx.cli.pick_points_interactive", lambda *a, **k: ([[1, 2]], [1]))
    monkeypatch.setattr("lightsaber_fx.cli.select_device", lambda: "cpu")

    def fake_run_pipeline(**kwargs):
        job_dir_path = Path(kwargs["job_dir"])
        _fake_full_render(job_dir_path)
        return kwargs["output_path"]

    monkeypatch.setattr("lightsaber_fx.cli.run_pipeline", fake_run_pipeline)

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

    monkeypatch.setattr("lightsaber_fx.cli.extract_first_frame", lambda *a, **k: None)
    monkeypatch.setattr("lightsaber_fx.cli.pick_points_interactive", lambda *a, **k: ([[1, 2]], [1]))
    monkeypatch.setattr("lightsaber_fx.cli.select_device", lambda: "cpu")

    def fake_run_pipeline(**kwargs):
        job_dir_path = Path(kwargs["job_dir"])
        _fake_full_render(job_dir_path)
        return kwargs["output_path"]

    monkeypatch.setattr("lightsaber_fx.cli.run_pipeline", fake_run_pipeline)

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
    monkeypatch.setattr("lightsaber_fx.cli.paths.get_jobs_dir", lambda: jobs_dir)

    captured = {}

    def fake_rerender_pipeline(**kwargs):
        captured.update(kwargs)
        return kwargs["output_path"]

    monkeypatch.setattr("lightsaber_fx.cli.rerender_pipeline", fake_rerender_pipeline)

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["rerender", "abc123", "--color", "blue", "--intensity", "0.6", "--voice", "sith",
         "--no-blade-extend", "--output", "out.mp4"],
    )

    assert result.exit_code == 0, result.output
    assert captured["job_dir"] == str(jobs_dir / "abc123")
    assert captured["color"] == "blue"
    assert captured["intensity"] == 0.6
    assert captured["voice"] == "sith"
    assert captured["blade_extend"] is False
    assert captured["output_path"] == "out.mp4"


def test_rerender_command_rejects_out_of_range_intensity(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "abc123").mkdir(parents=True)
    monkeypatch.setattr("lightsaber_fx.cli.paths.get_jobs_dir", lambda: jobs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["rerender", "abc123", "--intensity", "5"])

    assert result.exit_code != 0
    assert "0.0" in result.output and "1.0" in result.output


def test_rerender_command_errors_when_job_id_unknown(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr("lightsaber_fx.cli.paths.get_jobs_dir", lambda: jobs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["rerender", "does-not-exist"])

    assert result.exit_code != 0
    assert "No such job" in result.output


def test_rerender_command_reports_clear_error_when_job_not_rerenderable(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    (jobs_dir / "abc123").mkdir(parents=True)
    monkeypatch.setattr("lightsaber_fx.cli.paths.get_jobs_dir", lambda: jobs_dir)

    def fake_rerender_pipeline(**kwargs):
        raise JobNotRerenderableError("Job 'abc123' is not re-renderable: no masks/")

    monkeypatch.setattr("lightsaber_fx.cli.rerender_pipeline", fake_rerender_pipeline)

    runner = CliRunner()
    result = runner.invoke(main, ["rerender", "abc123"])

    assert result.exit_code != 0
    assert "no masks/" in result.output


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------


def test_jobs_command_lists_rerenderable_and_annotates_broken_job(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr("lightsaber_fx.cli.paths.get_jobs_dir", lambda: jobs_dir)

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


def test_jobs_command_reports_no_jobs_found(tmp_path, monkeypatch):
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir()
    monkeypatch.setattr("lightsaber_fx.cli.paths.get_jobs_dir", lambda: jobs_dir)

    runner = CliRunner()
    result = runner.invoke(main, ["jobs"])

    assert result.exit_code == 0
    assert "No jobs found" in result.output
