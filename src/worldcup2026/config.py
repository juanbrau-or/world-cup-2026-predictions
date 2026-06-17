"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secret values are never printed by the CLI."""

    football_data_api_key: str | None = None
    api_football_key: str | None = None
    open_meteo_base_url: str = "https://api.open-meteo.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return cached runtime settings."""

    return Settings()
