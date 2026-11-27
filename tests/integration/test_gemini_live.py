"""One real Gemini call (a few dozen tokens). Runs only with `pytest -m integration`
and a GEMINI_API_KEY in the environment or .env."""

import json

import pytest

from app.config import Settings
from app.llm import LLMClient
from app.llm.providers.gemini import GeminiProvider

pytestmark = [pytest.mark.integration, pytest.mark.anyio]


@pytest.fixture
def anyio_backend():
    return "asyncio"


async def test_gemini_returns_valid_json_for_a_schema():
    settings = Settings()
    if settings.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not set")

    provider = GeminiProvider(
        api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.llm_model,
        timeout_s=settings.llm_timeout_s,
    )
    client = LLMClient(provider, timeout_s=settings.llm_timeout_s)
    schema = {"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]}

    response = await client.generate(
        system="Answer with JSON only.", prompt='Return {"status": "ok"}.', json_schema=schema
    )

    assert json.loads(response.text)["status"] == "ok"
    assert response.provider == "gemini"
    assert response.input_tokens > 0
