"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secret values are never printed by the CLI."""

    football_data_api_key: str | None = None
    api_football_key: str | None = None
    open_meteo_base_url: str = "https://api.open-meteo.com"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("football_data_api_key", "api_football_key", mode="before")
    @classmethod
    def empty_secret_to_none(cls, value: object) -> object:
        """Treat blank secret values from .env.example-style files as missing."""

        if value == "":
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    """Return cached runtime settings."""

    return Settings()
