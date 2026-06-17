import pytest

from worldcup2026.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.football_data_api_key is None
    assert settings.api_football_key is None
    assert settings.open_meteo_base_url.startswith("https://")


def test_settings_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOOTBALL_DATA_API_KEY", "secret-token")
    monkeypatch.setenv("OPEN_METEO_BASE_URL", "https://example.test")

    settings = Settings(_env_file=None)

    assert settings.football_data_api_key == "secret-token"
    assert settings.open_meteo_base_url == "https://example.test"


def test_blank_secret_values_are_treated_as_missing() -> None:
    settings = Settings(_env_file=None, football_data_api_key="", api_football_key="")

    assert settings.football_data_api_key is None
    assert settings.api_football_key is None
