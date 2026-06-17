from worldcup2026.config import Settings


def test_settings_have_safe_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.football_data_api_key is None
    assert settings.api_football_key is None
    assert settings.open_meteo_base_url.startswith("https://")
