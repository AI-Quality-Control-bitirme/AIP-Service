"""Provider-neutral request and response types of the LLM client."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class LLMRequest(BaseModel):
    """One prompt sent to a provider. `json_schema` asks for structured (JSON) output."""

    model_config = ConfigDict(frozen=True)

    system: str
    prompt: str
    json_schema: dict[str, Any] | None = None


class ProviderOutput(BaseModel):
    """What a provider returns for one successful attempt."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class LLMResponse(BaseModel):
    """What the client returns to an agent: the text plus usage for logging and versioning (UC08, AC14)."""

    model_config = ConfigDict(frozen=True)

    text: str
    provider: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    latency_s: float = Field(ge=0)
    retries: int = Field(ge=0)
