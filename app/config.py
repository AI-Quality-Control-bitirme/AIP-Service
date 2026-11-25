"""Service configuration read from environment variables (and an optional .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["critical", "error", "warning", "info", "debug"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "info"

    @field_validator("log_level", mode="before")
    @classmethod
    def _lowercase_log_level(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
