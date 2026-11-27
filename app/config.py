"""Service configuration read from environment variables (and an optional .env file)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["critical", "error", "warning", "info", "debug"]
LLMProviderName = Literal["gemini", "fake"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: LogLevel = "info"

    # LLM client (app/llm). "fake" needs no key and makes no paid calls.
    llm_provider: LLMProviderName = "fake"
    gemini_api_key: SecretStr | None = None
    llm_model: str = "gemini-3.5-flash-lite"
    llm_timeout_s: float = Field(default=30, gt=0)
    llm_max_retries: int = Field(default=3, ge=0, le=3)

    @field_validator("log_level", "llm_provider", mode="before")
    @classmethod
    def _lowercase(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def _empty_key_is_unset(cls, value: object) -> object:
        # .env.example ships "GEMINI_API_KEY=", which must mean "no key", not an empty key.
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _gemini_needs_a_key(self) -> "Settings":
        if self.llm_provider == "gemini" and self.gemini_api_key is None:
            raise ValueError("LLM_PROVIDER=gemini requires GEMINI_API_KEY to be set")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
