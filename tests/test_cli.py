from click.testing import CliRunner

from lightsaber_fx.cli import main


def test_cli_help_exits_zero():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "lightsaber-fx" in result.output or "Usage" in result.output
