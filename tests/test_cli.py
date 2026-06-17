from typer.testing import CliRunner

from worldcup2026.cli import app

runner = CliRunner()


def test_doctor_succeeds_from_repository_root() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Initial environment looks healthy" in result.stdout
