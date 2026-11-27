"""A provider that returns preset responses. Shared test double for the client and the agents."""

import asyncio
from collections import deque
from collections.abc import Iterable

from app.llm.errors import LLMPermanentError
from app.llm.types import LLMRequest, ProviderOutput

FakeItem = str | ProviderOutput | BaseException


class FakeProvider:
    """Returns the queued items in order: a str becomes the response text, an exception is raised.

    When the queue is empty, `default_text` is returned if set; otherwise the call fails.
    Every request is recorded in `requests`, including ones that raise.
    """

    name = "fake"

    def __init__(
        self,
        responses: Iterable[FakeItem] = (),
        *,
        default_text: str | None = None,
        model: str = "fake-model",
        delay_s: float = 0.0,
    ) -> None:
        self._queue: deque[FakeItem] = deque(responses)
        self._default_text = default_text
        self.model = model
        self._delay_s = delay_s
        self.requests: list[LLMRequest] = []

    async def generate(self, request: LLMRequest) -> ProviderOutput:
        self.requests.append(request)
        if self._delay_s > 0:
            await asyncio.sleep(self._delay_s)

        if self._queue:
            item = self._queue.popleft()
        elif self._default_text is not None:
            item = self._default_text
        else:
            raise LLMPermanentError("FakeProvider has no responses left")

        if isinstance(item, BaseException):
            raise item
        if isinstance(item, ProviderOutput):
            return item
        return ProviderOutput(text=item, model=self.model, input_tokens=0, output_tokens=0)
