import pytest

from app.llm import LLMPermanentError, LLMRequest, LLMTransientError
from app.llm.providers.fake import FakeProvider

pytestmark = pytest.mark.anyio


def request(prompt: str = "hello", json_schema: dict | None = None) -> LLMRequest:
    return LLMRequest(system="You are a test.", prompt=prompt, json_schema=json_schema)


async def test_returns_queued_responses_in_order():
    fake = FakeProvider(["first", "second"])

    first = await fake.generate(request())
    second = await fake.generate(request())

    assert (first.text, second.text) == ("first", "second")
    assert first.model == "fake-model"
    assert fake.name == "fake"


async def test_records_every_request():
    schema = {"type": "object"}
    fake = FakeProvider(["ok"])

    await fake.generate(request(prompt="transcript", json_schema=schema))

    assert len(fake.requests) == 1
    assert fake.requests[0].system == "You are a test."
    assert fake.requests[0].prompt == "transcript"
    assert fake.requests[0].json_schema == schema


async def test_raises_queued_exception():
    fake = FakeProvider([LLMTransientError("503 UNAVAILABLE")])

    with pytest.raises(LLMTransientError, match="503"):
        await fake.generate(request())

    assert len(fake.requests) == 1


async def test_empty_queue_raises_permanent_error():
    fake = FakeProvider()

    with pytest.raises(LLMPermanentError, match="no responses left"):
        await fake.generate(request())


async def test_default_text_repeats():
    fake = FakeProvider(default_text="{}")

    texts = [(await fake.generate(request())).text for _ in range(3)]

    assert texts == ["{}", "{}", "{}"]
