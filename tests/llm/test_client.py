import pytest

from app.llm import (
    LLMClient,
    LLMPermanentError,
    LLMRetriesExhaustedError,
    LLMTimeoutError,
    LLMTransientError,
    ProviderOutput,
)
from app.llm.providers.fake import FakeProvider

pytestmark = pytest.mark.anyio


class SleepRecorder:
    """Stands in for asyncio.sleep so backoff is checked without waiting."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def make_client(fake: FakeProvider, sleep: SleepRecorder, **kwargs) -> LLMClient:
    kwargs.setdefault("timeout_s", 5)
    return LLMClient(fake, sleep=sleep, **kwargs)


async def test_returns_text_and_usage():
    schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
    output = ProviderOutput(text='{"ok": true}', model="fake-model-001", input_tokens=12, output_tokens=5)
    fake = FakeProvider([output])
    client = make_client(fake, SleepRecorder(), clock=iter([10.0, 10.25]).__next__)

    response = await client.generate(system="sys", prompt="prompt", json_schema=schema)

    assert response.text == '{"ok": true}'
    assert response.provider == "fake"
    assert response.model == "fake-model-001"
    assert (response.input_tokens, response.output_tokens) == (12, 5)
    assert response.latency_s == pytest.approx(0.25)
    assert response.retries == 0
    assert fake.requests[0].json_schema == schema


async def test_temporary_error_is_retried_then_succeeds():
    sleep = SleepRecorder()
    client = make_client(FakeProvider([LLMTransientError("503 UNAVAILABLE"), "ok"]), sleep)

    response = await client.generate(system="sys", prompt="prompt")

    assert response.text == "ok"
    assert response.retries == 1
    assert sleep.calls == [1]


async def test_retries_stop_after_max_and_name_last_cause():
    sleep = SleepRecorder()
    fake = FakeProvider([LLMTransientError("429 RESOURCE_EXHAUSTED") for _ in range(4)] + ["never reached"])
    client = make_client(fake, sleep)

    with pytest.raises(LLMRetriesExhaustedError) as caught:
        await client.generate(system="sys", prompt="prompt")

    assert caught.value.attempts == 4
    assert "429" in str(caught.value)
    assert isinstance(caught.value.last_error, LLMTransientError)
    assert len(fake.requests) == 4
    assert sleep.calls == [1, 2, 4]


async def test_permanent_error_is_not_retried():
    sleep = SleepRecorder()
    fake = FakeProvider([LLMPermanentError("400 INVALID_ARGUMENT"), "never reached"])
    client = make_client(fake, sleep)

    with pytest.raises(LLMPermanentError, match="400"):
        await client.generate(system="sys", prompt="prompt")

    assert sleep.calls == []
    assert len(fake.requests) == 1


async def test_max_retries_zero_makes_one_attempt():
    sleep = SleepRecorder()
    fake = FakeProvider([LLMTransientError("503"), "never reached"])
    client = make_client(fake, sleep, max_retries=0)

    with pytest.raises(LLMRetriesExhaustedError) as caught:
        await client.generate(system="sys", prompt="prompt")

    assert caught.value.attempts == 1
    assert sleep.calls == []
    assert len(fake.requests) == 1


async def test_timeout_counts_as_temporary():
    client = make_client(FakeProvider(["late"], delay_s=1), SleepRecorder(), timeout_s=0.01, max_retries=0)

    with pytest.raises(LLMRetriesExhaustedError) as caught:
        await client.generate(system="sys", prompt="prompt")

    assert isinstance(caught.value.last_error, LLMTimeoutError)


@pytest.mark.parametrize(("max_retries", "timeout_s"), [(4, 5), (-1, 5), (3, 0)])
def test_invalid_settings_rejected(max_retries, timeout_s):
    with pytest.raises(ValueError):
        LLMClient(FakeProvider(), timeout_s=timeout_s, max_retries=max_retries)


async def test_logs_carry_usage_but_no_prompt_text(caplog):
    caplog.set_level("DEBUG")
    client = make_client(FakeProvider([LLMTransientError("503"), "ok"]), SleepRecorder())

    await client.generate(system="SYSTEM-MARKER", prompt="TRANSCRIPT-MARKER")

    assert "provider=fake" in caplog.text
    assert "retries=1" in caplog.text
    assert "SYSTEM-MARKER" not in caplog.text
    assert "TRANSCRIPT-MARKER" not in caplog.text
