from types import SimpleNamespace

from app.config import Settings
from app.llm.factory import build_llm_client
from app.llm.providers.gemini import adapter


def test_fake_provider_selected():
    settings = Settings(_env_file=None, llm_provider="fake", llm_max_retries=2, llm_timeout_s=10)

    client = build_llm_client(settings)

    assert client.provider.name == "fake"
    assert client.max_retries == 2
    assert client.timeout_s == 10


def test_gemini_provider_selected(monkeypatch):
    recorded: dict = {}

    def fake_client(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(adapter.genai, "Client", fake_client)
    settings = Settings(_env_file=None, llm_provider="gemini", gemini_api_key="test-key", llm_model="gemini-x")

    client = build_llm_client(settings)

    assert client.provider.name == "gemini"
    assert client.provider.model == "gemini-x"
    assert client.max_retries == 3
    assert recorded["api_key"] == "test-key"
