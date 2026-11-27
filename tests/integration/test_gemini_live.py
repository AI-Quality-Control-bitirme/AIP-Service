"""Real Gemini calls. Run only with `pytest -m integration`.

The unit tests replace the SDK with a stub that behaves the way we believe the SDK
behaves. These tests check those beliefs against the real API: a successful JSON
call, a rejected key and a network timeout. Together they cost at most a few dozen tokens.
"""

import json

import pytest

from app.config import Settings
from app.llm import LLMClient, LLMPermanentError, LLMRetriesExhaustedError, LLMTimeoutError
from app.llm.providers.gemini import GeminiProvider

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def api_key(settings: Settings) -> str:
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not set")
    return settings.gemini_api_key.get_secret_value()


class SleepRecorder:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


async def test_gemini_returns_valid_json_for_a_schema(settings, api_key):
    provider = GeminiProvider(api_key=api_key, model=settings.llm_model, timeout_s=settings.llm_timeout_s)
    client = LLMClient(provider, timeout_s=settings.llm_timeout_s)
    schema = {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}

    response = await client.generate(
        system="Answer with JSON only.", prompt='Return {"status": "ok"}.', json_schema=schema
    )

    assert json.loads(response.text)["status"] == "ok"
    assert response.provider == "gemini"
    assert response.input_tokens > 0


async def test_rejected_key_is_permanent_and_not_retried(settings):
    # No real key needed: Gemini rejects the request before running the model, so no tokens are used.
    sleep = SleepRecorder()
    provider = GeminiProvider(api_key="invalid-key-for-test", model=settings.llm_model, timeout_s=30)
    client = LLMClient(provider, timeout_s=30, sleep=sleep)

    with pytest.raises(LLMPermanentError) as caught:
        await client.generate(system="Answer briefly.", prompt="Say ok.")

    assert "Gemini 4" in str(caught.value)  # a 4xx status from the real API
    assert sleep.calls == []


async def test_sdk_timeout_becomes_llm_timeout_error(settings, api_key):
    # The adapter gets a 10 ms timeout and the client a generous one, so the SDK's own
    # timeout fires first: this checks the adapter's mapping of the real SDK error.
    provider = GeminiProvider(api_key=api_key, model=settings.llm_model, timeout_s=0.01)
    client = LLMClient(provider, timeout_s=30, max_retries=0)

    with pytest.raises(LLMRetriesExhaustedError) as caught:
        await client.generate(system="Answer briefly.", prompt="Say ok.")

    assert isinstance(caught.value.last_error, LLMTimeoutError)
    assert "Gemini request timed out" in str(caught.value.last_error)
