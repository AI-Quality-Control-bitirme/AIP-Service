"""LLMClient: the one entry point agents use to call an LLM.

The client owns what is the same for every provider: the timeout, bounded retries
with exponential backoff (AC13), latency measurement and one log line per call (AC14).
The provider makes a single attempt and raises our own error types.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.llm.errors import LLMRetriesExhaustedError, LLMTimeoutError, LLMTransientError
from app.llm.provider import LLMProvider
from app.llm.types import LLMRequest, LLMResponse

logger = logging.getLogger("app.llm")

MAX_RETRIES_LIMIT = 3


class LLMClient:
    def __init__(
        self,
        provider: LLMProvider,
        *,
        timeout_s: float,
        max_retries: int = MAX_RETRIES_LIMIT,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if not 0 <= max_retries <= MAX_RETRIES_LIMIT:
            raise ValueError(f"max_retries must be between 0 and {MAX_RETRIES_LIMIT}, got {max_retries}")
        if timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}")
        self._provider = provider
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._sleep = sleep
        self._clock = clock

    @property
    def provider(self) -> LLMProvider:
        return self._provider

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    @property
    def max_retries(self) -> int:
        return self._max_retries

    async def generate(
        self, system: str, prompt: str, json_schema: dict[str, Any] | None = None
    ) -> LLMResponse:
        """Send one prompt. Raises LLMPermanentError at once, or LLMRetriesExhaustedError
        after the last temporary failure. Never returns a partial result."""
        request = LLMRequest(system=system, prompt=prompt, json_schema=json_schema)
        name = self._provider.name

        for attempt in range(self._max_retries + 1):
            started = self._clock()
            try:
                output = await asyncio.wait_for(self._provider.generate(request), self._timeout_s)
            except TimeoutError:
                error: LLMTransientError = LLMTimeoutError(f"no response within {self._timeout_s} s")
            except LLMTransientError as exc:
                error = exc
            else:
                latency_s = max(self._clock() - started, 0.0)
                # Never log the prompt or system text: transcripts are personal data (AC15).
                logger.info(
                    "llm call provider=%s model=%s input_tokens=%d output_tokens=%d latency_s=%.3f retries=%d",
                    name, output.model, output.input_tokens, output.output_tokens, latency_s, attempt,
                )
                return LLMResponse(
                    text=output.text,
                    provider=name,
                    model=output.model,
                    input_tokens=output.input_tokens,
                    output_tokens=output.output_tokens,
                    latency_s=latency_s,
                    retries=attempt,
                )

            if attempt == self._max_retries:
                logger.error(
                    "llm call failed provider=%s attempts=%d error=%s", name, attempt + 1, type(error).__name__
                )
                raise LLMRetriesExhaustedError(attempt + 1, error) from error

            delay_s = 2**attempt
            logger.warning(
                "llm call retry provider=%s attempt=%d error=%s delay_s=%d",
                name, attempt + 1, type(error).__name__, delay_s,
            )
            await self._sleep(delay_s)

        raise AssertionError("unreachable")  # the loop always returns or raises
