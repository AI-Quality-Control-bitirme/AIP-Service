import pytest
from pydantic import ValidationError

from app.config import Settings


def make_settings(**overrides) -> Settings:
    # _env_file=None keeps a developer's local .env out of the tests.
    return Settings(_env_file=None, **overrides)


def test_defaults_are_used_when_nothing_is_set(monkeypatch):
    for name in ("HOST", "PORT", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)

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
