"""Gemini adapter tests. The SDK client is replaced by a stub, so no network call is made."""

from types import SimpleNamespace

import httpx2
import pytest
from google.genai import errors, types

from app.llm import LLMPermanentError, LLMRequest, LLMTimeoutError, LLMTransientError
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.gemini import adapter

MODEL = "gemini-3.5-flash-lite"


class StubModels:
    """Replaces client.aio.models; records the call and returns or raises a preset."""

    def __init__(self, result) -> None:
        self.result = result
        self.kwargs: dict = {}

    async def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def make_provider(result) -> tuple[GeminiProvider, StubModels]:
    stub = StubModels(result)
    client = SimpleNamespace(aio=SimpleNamespace(models=stub))
    return GeminiProvider(api_key="test-key", model=MODEL, timeout_s=30, client=client), stub


def gemini_response(
    text: str | None = "ok",
    *,
    finish_reason: str = "STOP",
    model_version: str | None = None,
    usage: types.GenerateContentResponseUsageMetadata | None = None,
) -> types.GenerateContentResponse:
    parts = [types.Part(text=text)] if text is not None else []
    candidate = types.Candidate(content=types.Content(role="model", parts=parts), finish_reason=finish_reason)
    return types.GenerateContentResponse(candidates=[candidate], model_version=model_version, usage_metadata=usage)


def api_error(code: int) -> errors.APIError:
    return errors.APIError(code, {"error": {"code": code, "status": "STATUS", "message": "details"}})


def request(json_schema: dict | None = None) -> LLMRequest:
    return LLMRequest(system="SYSTEM", prompt="PROMPT", json_schema=json_schema)


pytestmark = pytest.mark.anyio


async def test_sends_prompt_system_and_schema():
    schema = {"type": "object", "properties": {"status": {"type": "string"}}}
    provider, stub = make_provider(gemini_response('{"status": "ok"}'))

    await provider.generate(request(schema))

    config = stub.kwargs["config"]
    assert stub.kwargs["model"] == MODEL
    assert stub.kwargs["contents"] == "PROMPT"
    assert config.system_instruction == "SYSTEM"
    assert config.response_mime_type == "application/json"
    assert config.response_json_schema == schema


async def test_without_schema_json_mode_is_off():
    provider, stub = make_provider(gemini_response("plain text"))

    await provider.generate(request())

    config = stub.kwargs["config"]
    assert config.response_mime_type is None
    assert config.response_json_schema is None


async def test_maps_text_usage_and_model_version():
    usage = types.GenerateContentResponseUsageMetadata(
        prompt_token_count=100, candidates_token_count=20, thoughts_token_count=5
    )
    provider, _ = make_provider(gemini_response("answer", model_version="gemini-3.5-flash-lite-001", usage=usage))

    output = await provider.generate(request())

    assert output.text == "answer"
    assert output.model == "gemini-3.5-flash-lite-001"
    assert (output.input_tokens, output.output_tokens) == (100, 25)


async def test_missing_usage_counts_zero():
    provider, _ = make_provider(gemini_response("answer"))

    output = await provider.generate(request())

    assert (output.input_tokens, output.output_tokens) == (0, 0)
    assert output.model == MODEL


@pytest.mark.parametrize("code", [408, 429, 500, 503])
async def test_temporary_http_errors(code):
    provider, _ = make_provider(api_error(code))

    with pytest.raises(LLMTransientError, match=str(code)):
        await provider.generate(request())


@pytest.mark.parametrize("code", [400, 401, 403, 404])
async def test_permanent_http_errors(code):
    provider, _ = make_provider(api_error(code))

    with pytest.raises(LLMPermanentError, match=str(code)):
        await provider.generate(request())


async def test_network_timeout_is_timeout_error():
    provider, _ = make_provider(httpx2.ReadTimeout("read timed out"))

    with pytest.raises(LLMTimeoutError):
        await provider.generate(request())


@pytest.mark.parametrize(
    "exc",
    [
        httpx2.ConnectError("connection refused"),
        httpx2.ReadError("connection reset mid-response"),
        httpx2.RemoteProtocolError("server disconnected"),
    ],
)
async def test_network_errors_are_temporary(exc):
    provider, _ = make_provider(exc)

    with pytest.raises(LLMTransientError):
        await provider.generate(request())


async def test_empty_text_is_permanent_error():
    provider, _ = make_provider(gemini_response(None, finish_reason="SAFETY"))

    with pytest.raises(LLMPermanentError, match="SAFETY"):
        await provider.generate(request())


def test_default_client_disables_sdk_retries(monkeypatch):
    recorded: dict = {}

    def fake_client(**kwargs):
        recorded.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(adapter.genai, "Client", fake_client)

    provider = GeminiProvider(api_key="test-key", model="m", timeout_s=30)

    assert provider.name == "gemini"
    assert provider.model == "m"
    assert recorded["api_key"] == "test-key"
    assert recorded["http_options"].timeout == 30000
    assert recorded["http_options"].retry_options.attempts == 1
