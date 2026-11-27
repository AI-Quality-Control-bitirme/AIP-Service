"""Builds the LLM client from settings. This is the only place that picks a provider."""

from app.config import Settings
from app.llm.client import LLMClient
from app.llm.provider import LLMProvider
from app.llm.providers.fake import FakeProvider


def build_llm_client(settings: Settings) -> LLMClient:
    provider: LLMProvider
    if settings.llm_provider == "gemini":
        # Imported here so "fake" mode never loads the Gemini SDK.
        from app.llm.providers.gemini import GeminiProvider

        assert settings.gemini_api_key is not None  # guaranteed by Settings validation
        provider = GeminiProvider(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.llm_model,
            timeout_s=settings.llm_timeout_s,
        )
    else:
        # Local stub: answers "{}" to every call, so the service runs without a key.
        provider = FakeProvider(default_text="{}")

    return LLMClient(provider, timeout_s=settings.llm_timeout_s, max_retries=settings.llm_max_retries)
