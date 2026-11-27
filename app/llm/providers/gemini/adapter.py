"""Gemini adapter: the only module that imports the google-genai SDK.

It makes one attempt and translates SDK types and errors into ours. Retries live in
LLMClient, so the SDK's own retry is switched off (attempts=1). Verified against
google-genai 3.6.1: with retry_options unset, the SDK still retries once on
connection errors when aiohttp is installed.
"""

from typing import Any

import httpx2
from google import genai
from google.genai import errors, types

from app.llm.errors import LLMError, LLMPermanentError, LLMTimeoutError, LLMTransientError
from app.llm.types import LLMRequest, ProviderOutput

# HTTP status codes worth retrying: request timeout, rate limit, server-side failures.
TRANSIENT_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
# Connection failures and connections dropped mid-response (NetworkError covers
# ConnectError, ReadError, WriteError; RemoteProtocolError is "server disconnected").
NETWORK_ERRORS = (httpx2.NetworkError, httpx2.RemoteProtocolError)


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str, timeout_s: float, *, client: Any | None = None) -> None:
        self.model = model
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=int(timeout_s * 1000),  # the SDK expects milliseconds
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )

    async def generate(self, request: LLMRequest) -> ProviderOutput:
        schema = request.json_schema
        config = types.GenerateContentConfig(
            system_instruction=request.system,
            response_mime_type="application/json" if schema is not None else None,
            response_json_schema=schema,
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=self.model, contents=request.prompt, config=config
            )
        except (errors.APIError, httpx2.TimeoutException, *NETWORK_ERRORS) as exc:
            raise _to_llm_error(exc) from exc

        return self._to_output(response)

    def _to_output(self, response: types.GenerateContentResponse) -> ProviderOutput:
        text = response.text
        if not text:
            raise LLMPermanentError(f"Gemini returned no text (finish_reason={_finish_reason(response)})")

        usage = response.usage_metadata
        input_tokens = (usage.prompt_token_count or 0) if usage else 0
        # Thinking tokens are billed as output, so they count toward output usage (AC14).
        output_tokens = ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else 0
        return ProviderOutput(
            text=text,
            model=response.model_version or self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


def _to_llm_error(exc: Exception) -> LLMError:
    if isinstance(exc, httpx2.TimeoutException):
        return LLMTimeoutError(f"Gemini request timed out: {type(exc).__name__}")
    if isinstance(exc, NETWORK_ERRORS):
        return LLMTransientError(f"Gemini connection failed: {type(exc).__name__}")
    assert isinstance(exc, errors.APIError)
    message = f"Gemini {exc.code} {exc.status}: {exc.message}"
    if exc.code in TRANSIENT_STATUS_CODES:
        return LLMTransientError(message)
    return LLMPermanentError(message)


def _finish_reason(response: types.GenerateContentResponse) -> str | None:
    if not response.candidates or response.candidates[0].finish_reason is None:
        return None
    reason = response.candidates[0].finish_reason
    return getattr(reason, "value", str(reason))
