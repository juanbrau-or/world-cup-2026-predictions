from pathlib import Path

import pytest
from typer.testing import CliRunner

from worldcup2026.cli import app
from worldcup2026.config import get_settings

runner = CliRunner()


def test_doctor_succeeds_from_repository_root() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Initial environment looks healthy for Phase 0" in result.stdout


def test_doctor_does_not_print_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "super-secret-token")
    get_settings.cache_clear()

    try:
        result = runner.invoke(app, ["doctor"])
    finally:
        get_settings.cache_clear()

    assert result.exit_code == 0
    assert "Football data key configured: yes" in result.stdout
    assert "super-secret-token" not in result.stdout


def test_doctor_fails_outside_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 1
    assert "Missing directories" in result.stdout
    assert "Missing files" in result.stdout
