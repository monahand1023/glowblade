from click.testing import CliRunner

from lightsaber_fx.cli import main


def test_cli_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "lightsaber-fx" in result.output or "Usage" in result.output


from pathlib import Path

import lightsaber_fx.paths as paths_module


def test_help_lists_all_subcommands():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    for name in ("setup", "run", "serve", "clean"):
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
    monkeypatch.setattr("lightsaber_fx.cli.extract_first_frame", lambda *a, **k: None)
    monkeypatch.setattr("lightsaber_fx.cli.pick_points_interactive", lambda *a, **k: ([], []))

    runner = CliRunner()
    result = runner.invoke(main, ["run", str(video)])

    assert result.exit_code != 0


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
