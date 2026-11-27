"""The interface every provider adapter implements."""

from typing import Protocol

from app.llm.types import LLMRequest, ProviderOutput


class LLMProvider(Protocol):
    """One attempt against one provider. Retries, timeout and logging belong to LLMClient."""

    name: str

    async def generate(self, request: LLMRequest) -> ProviderOutput: ...
