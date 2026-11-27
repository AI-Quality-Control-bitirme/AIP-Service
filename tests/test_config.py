import pytest
from pydantic import ValidationError

from app.config import Settings

LLM_ENV_NAMES = ("LLM_PROVIDER", "GEMINI_API_KEY", "LLM_MODEL", "LLM_TIMEOUT_S", "LLM_MAX_RETRIES")


def make_settings(**overrides) -> Settings:
    # _env_file=None keeps a developer's local .env out of the tests.
    return Settings(_env_file=None, **overrides)


def clear_env(monkeypatch, names):
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_used_when_nothing_is_set(monkeypatch):
    clear_env(monkeypatch, ("HOST", "PORT", "LOG_LEVEL", *LLM_ENV_NAMES))

    settings = make_settings()

    assert settings.host == "127.0.0.1"
    assert settings.port == 8000
    assert settings.log_level == "info"


def test_values_are_read_from_environment(monkeypatch):
    monkeypatch.setenv("HOST", "0.0.0.0")
    monkeypatch.setenv("PORT", "9100")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = make_settings()

    assert settings.host == "0.0.0.0"
    assert settings.port == 9100
    assert settings.log_level == "debug"


@pytest.mark.parametrize("port", ["0", "70000", "abc"])
def test_invalid_port_is_rejected(monkeypatch, port):
    monkeypatch.setenv("PORT", port)

    with pytest.raises(ValidationError):
        make_settings()


def test_unknown_log_level_is_rejected(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "verbose")

    with pytest.raises(ValidationError):
        make_settings()


def test_llm_defaults(monkeypatch):
    clear_env(monkeypatch, LLM_ENV_NAMES)

    settings = make_settings()

    assert settings.llm_provider == "fake"
    assert settings.gemini_api_key is None
    assert settings.llm_model == "gemini-3.5-flash-lite"
    assert settings.llm_timeout_s == 30
    assert settings.llm_max_retries == 3


def test_gemini_without_key_is_rejected(monkeypatch):
    clear_env(monkeypatch, LLM_ENV_NAMES)
    monkeypatch.setenv("LLM_PROVIDER", "GEMINI")

    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        make_settings()


def test_gemini_with_blank_key_is_rejected(monkeypatch):
    clear_env(monkeypatch, LLM_ENV_NAMES)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "   ")

    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        make_settings()


def test_empty_key_from_env_example_counts_as_unset(monkeypatch):
    # .env.example ships "GEMINI_API_KEY=", so a copied .env must read as "no key", not as "".
    clear_env(monkeypatch, LLM_ENV_NAMES)
    monkeypatch.setenv("GEMINI_API_KEY", "")

    settings = make_settings()

    assert settings.gemini_api_key is None


def test_api_key_is_hidden_in_repr(monkeypatch):
    clear_env(monkeypatch, LLM_ENV_NAMES)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")

    settings = make_settings()

    assert settings.llm_provider == "gemini"
    assert settings.gemini_api_key.get_secret_value() == "test-key-123"
    assert "test-key-123" not in repr(settings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("LLM_MAX_RETRIES", "4"),
        ("LLM_MAX_RETRIES", "-1"),
        ("LLM_TIMEOUT_S", "0"),
        ("LLM_PROVIDER", "openai"),
    ],
)
def test_invalid_llm_settings_are_rejected(monkeypatch, name, value):
    clear_env(monkeypatch, LLM_ENV_NAMES)
    monkeypatch.setenv(name, value)

    with pytest.raises(ValidationError):
        make_settings()
