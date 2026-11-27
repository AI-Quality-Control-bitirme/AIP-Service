# SCRUM-59 LLM Provider Client Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An async, provider-neutral LLM client in AIP-Service with one Gemini adapter, bounded retries, a timeout, per-call usage and a shared fake provider.

**Architecture:** Agents call `LLMClient.generate(...)`. The client owns timeout, retry, latency and logging; it delegates one attempt to an `LLMProvider` (fake or Gemini). Providers return our own `ProviderOutput` and raise our own errors; only `app/llm/providers/gemini/` knows the SDK.

**Tech Stack:** Python 3.12, Pydantic 2, pydantic-settings, `google-genai` 3.6.1 (uses `httpx2`), pytest 9.2 with the anyio plugin (already installed through fastapi/google-genai; no new test dependency).

**Spec:** `docs/superpowers/specs/2026-11-27-llm-provider-client-design.md`

## Global Constraints
- Default model `gemini-3.5-flash-lite`.
- At most 3 retries (`LLM_MAX_RETRIES` 0–3, default 3); backoff 1 s, 2 s, 4 s (`2 ** attempt`).
- `google.genai` imported only under `app/llm/providers/gemini/`.
- Never log the API key, the system text or the prompt text. Log lines carry provider, model, tokens, latency, retries, error class name.
- No paid API call in the default `pytest` run; the integration test needs `-m integration` and a `GEMINI_API_KEY`.
- Secrets only in `.env` (git-ignored); `.env.example` has a placeholder.
- Commit messages start with `SCRUM-59:`; every git write is shown to Can and run only after approval.

## Verified SDK facts (google-genai 3.6.1, read from the installed source on 2026-11-27)
- `GenerateContentConfig` fields: `system_instruction`, `response_mime_type`, `response_json_schema`.
- `HttpOptions(timeout=<ms>, retry_options=HttpRetryOptions(attempts=1))`. With `retry_options=None` the SDK does not retry, but if `aiohttp` is installed it does one implicit retry on connection errors unless `retry_options` is set, so we set `attempts=1` explicitly.
- HTTP errors: `google.genai.errors.APIError` with `.code`, `.status`, `.message` (`ClientError` / `ServerError` subclass it). Network: `httpx2.TimeoutException`, `httpx2.ConnectError`.
- Response: `.text` (None when there is no text), `.model_version`, `.usage_metadata.prompt_token_count / candidates_token_count / thoughts_token_count` (each may be None).

## Calls made by the builder (not discussed with Can; listed for review)
- `LLM_PROVIDER` defaults to `fake`, so the service and CI run without a key and cannot spend money by accident; deployments set `gemini`.
- `output_tokens` = candidates + thoughts tokens (Gemini bills thinking tokens as output; AC14).
- `latency_s` is the duration of the successful attempt; retries are reported separately. Tokens of failed attempts are unknown and not counted.
- `LLM_TIMEOUT_S` default 30.
- The client also wraps each attempt in `asyncio.wait_for(timeout_s)`, so a provider that ignores its own timeout is still cut off.
- With `LLM_PROVIDER=fake` the factory returns a fake that answers `{}` to every call (a local stub, not a test double).

## Review Focus
1. **Gemini answers 200 with no text** (safety block, MAX_TOKENS): must raise `LLMPermanentError` naming the finish reason, never hand `""` or `None` to an agent. → Task 4 `test_empty_text_is_permanent_error`.
2. **SDK retries stacked on ours** (4 × N calls, N× cost): the default client must be built with `attempts=1`. → Task 4 `test_default_client_disables_sdk_retries`.
3. **Key leaks** through `repr(settings)`, logs or error text. → Task 1 `test_api_key_is_hidden_in_repr`, Task 3 `test_logs_carry_usage_but_no_prompt_text`.
4. **Provider hangs past its own timeout**: the client cuts it off and treats it as temporary. → Task 3 `test_timeout_counts_as_temporary`.
5. **Free-tier 429 four times in a row**: the caller gets one clear error naming the 429 after ~7 s of backoff, not a silent empty result. → Task 3 `test_retries_stop_after_max_and_name_last_cause`.

---

### Task 1: Settings, dependency and env example

**Files:**
- Modify: `app/config.py`, `requirements.txt`, `.env.example`, `tests/test_config.py`

**Interfaces:**
- Produces: `Settings.llm_provider: Literal["gemini", "fake"]` (default `"fake"`, lower-cased before validation), `Settings.gemini_api_key: SecretStr | None` (default None), `Settings.llm_model: str` (default `"gemini-3.5-flash-lite"`), `Settings.llm_timeout_s: float` (default 30, `gt=0`), `Settings.llm_max_retries: int` (default 3, `ge=0, le=3`). A `model_validator(mode="after")` raises when provider is `gemini` and the key is missing or blank, with a message containing `GEMINI_API_KEY`.

- [ ] **Step 1: Write failing tests** in `tests/test_config.py`. Extend the env names deleted in `test_defaults_are_used_when_nothing_is_set` with `LLM_PROVIDER, GEMINI_API_KEY, LLM_MODEL, LLM_TIMEOUT_S, LLM_MAX_RETRIES`, and add:

```python
def test_llm_defaults(monkeypatch):  # after deleting the LLM env names
    s = make_settings()
    assert (s.llm_provider, s.gemini_api_key, s.llm_model, s.llm_timeout_s, s.llm_max_retries) == (
        "fake", None, "gemini-3.5-flash-lite", 30, 3)

def test_gemini_without_key_is_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "GEMINI"); monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="GEMINI_API_KEY"):
        make_settings()

def test_api_key_is_hidden_in_repr(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini"); monkeypatch.setenv("GEMINI_API_KEY", "test-key-123")
    s = make_settings()
    assert s.gemini_api_key.get_secret_value() == "test-key-123"
    assert "test-key-123" not in repr(s)

@pytest.mark.parametrize(("name", "value"), [("LLM_MAX_RETRIES", "4"), ("LLM_MAX_RETRIES", "-1"),
                                             ("LLM_TIMEOUT_S", "0"), ("LLM_PROVIDER", "openai")])
def test_invalid_llm_settings_are_rejected(monkeypatch, name, value): ...  # raises ValidationError
```

- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/test_config.py -v` → new tests FAIL (unknown attributes).
- [ ] **Step 3: Implement** the fields and validator in `app/config.py`. Add `google-genai~=3.6.1` to `requirements.txt` with a comment (`# Gemini adapter (app/llm/providers/gemini only)`). Add an `# LLM` block to `.env.example`: `LLM_PROVIDER=fake`, `GEMINI_API_KEY=` (placeholder, comment: secret, never commit), `LLM_MODEL=gemini-3.5-flash-lite`, `LLM_TIMEOUT_S=30`, `LLM_MAX_RETRIES=3`.
- [ ] **Step 4: Run** the same command → all PASS (old tests included).

### Task 2: Core types, errors and fake provider

**Files:**
- Create: `app/llm/__init__.py`, `app/llm/types.py`, `app/llm/errors.py`, `app/llm/provider.py`, `app/llm/providers/__init__.py`, `app/llm/providers/fake.py`
- Test: `tests/llm/__init__.py`, `tests/llm/conftest.py`, `tests/llm/test_fake_provider.py`

**Interfaces:**
- Produces (`app/llm/types.py`, Pydantic, frozen):
  - `LLMRequest(system: str, prompt: str, json_schema: dict[str, Any] | None = None)`
  - `ProviderOutput(text: str, model: str, input_tokens: int ≥0, output_tokens: int ≥0)`
  - `LLMResponse(text, provider: str, model, input_tokens, output_tokens, latency_s: float ≥0, retries: int ≥0)`
- Produces (`app/llm/errors.py`): `LLMError(Exception)`; `LLMTransientError(LLMError)`; `LLMTimeoutError(LLMTransientError)`; `LLMPermanentError(LLMError)`; `LLMRetriesExhaustedError(LLMError)` with `__init__(self, attempts: int, last_error: LLMError)`, attributes `attempts`, `last_error`, message `f"LLM call failed after {attempts} attempts; last error: {type(last_error).__name__}: {last_error}"`.
- Produces (`app/llm/provider.py`): `class LLMProvider(Protocol): name: str; async def generate(self, request: LLMRequest) -> ProviderOutput`.
- Produces (`app/llm/providers/fake.py`): `FakeProvider(responses: Iterable[str | ProviderOutput | BaseException] = (), *, default_text: str | None = None, model: str = "fake-model", delay_s: float = 0.0)`; `name = "fake"`; `requests: list[LLMRequest]` (every call recorded, including ones that raise). Each call: `await asyncio.sleep(delay_s)` if > 0, then pop the next item: a `str` becomes `ProviderOutput(text=item, model=model, input_tokens=0, output_tokens=0)`, a `ProviderOutput` is returned as is, an exception is raised. Empty queue: return `default_text` if set, else raise `LLMPermanentError("FakeProvider has no responses left")`.
- `app/llm/__init__.py` re-exports the types, errors, `LLMProvider` and (after Task 3) `LLMClient`.
- `tests/llm/conftest.py`: `anyio_backend` fixture returning `"asyncio"`; async tests use `@pytest.mark.anyio`.

- [ ] **Step 1: Write failing tests** in `tests/llm/test_fake_provider.py`: `test_returns_queued_responses_in_order` (two strs → texts in order, `model == "fake-model"`), `test_records_every_request` (request fields equal what was sent, including `json_schema`), `test_raises_queued_exception` (queued `LLMTransientError("503")` is raised and the request is still recorded), `test_empty_queue_raises_permanent_error`, `test_default_text_repeats` (`default_text="{}"`, three calls → `"{}"` each).
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/llm -v` → FAIL (import error).
- [ ] **Step 3: Implement** the four modules above.
- [ ] **Step 4: Run** → PASS.

### Task 3: LLMClient – timeout, retry, latency, logging

**Files:**
- Create: `app/llm/client.py`
- Test: `tests/llm/test_client.py`

**Interfaces:**
- Consumes: Task 2 types, errors, `LLMProvider`, `FakeProvider`.
- Produces: `LLMClient(provider: LLMProvider, *, timeout_s: float, max_retries: int = 3, sleep: Callable[[float], Awaitable[None]] = asyncio.sleep, clock: Callable[[], float] = time.perf_counter)`; read-only properties `provider`, `timeout_s`, `max_retries`; `ValueError` if `max_retries` is outside 0–3 or `timeout_s <= 0`. `async def generate(self, system: str, prompt: str, json_schema: dict[str, Any] | None = None) -> LLMResponse`.
- Algorithm: for `attempt` in `0..max_retries`: time the attempt with `clock`; `await asyncio.wait_for(provider.generate(request), timeout_s)`; a builtin `TimeoutError` becomes `LLMTimeoutError(f"no response within {timeout_s} s")`; an `LLMTransientError` is retried; `LLMPermanentError` and non-LLM exceptions propagate unchanged. Before each retry: log a WARNING (`provider`, `attempt`, error class name) and `await sleep(2 ** attempt)`. After the last failed attempt: `raise LLMRetriesExhaustedError(attempt + 1, err) from err`. On success: log one INFO line on logger `app.llm` with `provider= model= input_tokens= output_tokens= latency_s= retries=` and return `LLMResponse(provider=provider.name, retries=attempt, latency_s=<end-start of that attempt>, **output fields)`.

- [ ] **Step 1: Write failing tests** in `tests/llm/test_client.py`. Helper: `sleeps = []; async def fake_sleep(s): sleeps.append(s)`.
  - `test_returns_text_and_usage`: `ProviderOutput(text='{"ok": true}', model="fake-model-001", input_tokens=12, output_tokens=5)`, `clock=iter([10.0, 10.25]).__next__` → `text`, `provider == "fake"`, `model == "fake-model-001"`, tokens 12/5, `latency_s == 0.25`, `retries == 0`; `fake.requests[0].json_schema == schema`.
  - `test_temporary_error_is_retried_then_succeeds`: `[LLMTransientError("503"), "ok"]` → `text == "ok"`, `retries == 1`, `sleeps == [1]`.
  - `test_retries_stop_after_max_and_name_last_cause`: four `LLMTransientError("429 RESOURCE_EXHAUSTED")` → `LLMRetriesExhaustedError`, `attempts == 4`, `"429" in str(err)`, `len(fake.requests) == 4`, `sleeps == [1, 2, 4]`.
  - `test_permanent_error_is_not_retried`: `[LLMPermanentError("400")]` → raises `LLMPermanentError`, `sleeps == []`, one request.
  - `test_max_retries_zero_makes_one_attempt`: one transient error, `max_retries=0` → `LLMRetriesExhaustedError` with `attempts == 1`, `sleeps == []`.
  - `test_timeout_counts_as_temporary`: `FakeProvider(["late"], delay_s=1)`, `timeout_s=0.01`, `max_retries=0` → `LLMRetriesExhaustedError` whose `last_error` is an `LLMTimeoutError`.
  - `test_invalid_settings_rejected`: parametrize `max_retries` 4 and -1, `timeout_s` 0 → `ValueError`.
  - `test_logs_carry_usage_but_no_prompt_text`: `caplog.set_level("DEBUG")`, a transient error then success, `system="SYSTEM-MARKER"`, `prompt="TRANSCRIPT-MARKER"` → `"provider=fake"` and `"retries=1"` in `caplog.text`; neither marker in `caplog.text`.
- [ ] **Step 2: Run** `.venv/Scripts/python.exe -m pytest tests/llm/test_client.py -v` → FAIL.
- [ ] **Step 3: Implement** `app/llm/client.py`; export `LLMClient` from `app/llm/__init__.py`.
- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest -v` → all PASS.

### Task 4: Gemini adapter and import boundary

**Files:**
- Create: `app/llm/providers/gemini/__init__.py` (exports `GeminiProvider`), `app/llm/providers/gemini/adapter.py`
- Test: `tests/llm/test_gemini_provider.py`, `tests/llm/test_import_boundary.py`

**Interfaces:**
- Consumes: Task 2 types and errors.
- Produces: `GeminiProvider(api_key: str, model: str, timeout_s: float, *, client: Any | None = None)`; `name = "gemini"`; public attribute `model`. Without `client` it builds `genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000), retry_options=types.HttpRetryOptions(attempts=1)))`. `async def generate(self, request: LLMRequest) -> ProviderOutput` calls `await client.aio.models.generate_content(model=model, contents=request.prompt, config=types.GenerateContentConfig(system_instruction=request.system, response_mime_type="application/json" if schema else None, response_json_schema=schema))`.
- Error mapping: `errors.APIError` with code in `{408, 429, 500, 502, 503, 504}` → `LLMTransientError`; any other `APIError` → `LLMPermanentError`; `httpx2.TimeoutException` → `LLMTimeoutError`; `httpx2.ConnectError` → `LLMTransientError`; messages `f"Gemini {code} {status}: {message}"` (never the request). Raise with `from exc`.
- Response mapping: empty or None `.text` → `LLMPermanentError(f"Gemini returned no text (finish_reason={reason})")` using the first candidate's `finish_reason` or `None`; `model = response.model_version or self.model`; `input_tokens = prompt_token_count or 0`; `output_tokens = (candidates_token_count or 0) + (thoughts_token_count or 0)`; missing `usage_metadata` → zeros.

- [ ] **Step 1: Write failing tests** in `tests/llm/test_gemini_provider.py`. Stub: `class StubModels` with `async def generate_content(self, **kwargs)` that records `kwargs` and returns or raises a preset; `client = SimpleNamespace(aio=SimpleNamespace(models=stub))`. Build real `types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=...)]), finish_reason="STOP")], model_version=..., usage_metadata=types.GenerateContentResponseUsageMetadata(...))`.
  - `test_sends_prompt_system_and_schema`: `kwargs["model"] == "gemini-3.5-flash-lite"`, `kwargs["contents"] == prompt`, `config.system_instruction == system`, `config.response_mime_type == "application/json"`, `config.response_json_schema == schema`.
  - `test_without_schema_json_mode_is_off`: `config.response_mime_type is None` and `config.response_json_schema is None`.
  - `test_maps_text_usage_and_model_version`: usage 100 / 20 / thoughts 5, `model_version="gemini-3.5-flash-lite-001"` → `ProviderOutput(text, "gemini-3.5-flash-lite-001", 100, 25)`.
  - `test_missing_usage_counts_zero`: no `usage_metadata`, no `model_version` → tokens 0/0, `model == "gemini-3.5-flash-lite"`.
  - `test_temporary_http_errors`: parametrize 408, 429, 500, 503 → `errors.APIError(code, {"error": {"code": code, "status": "S", "message": "m"}})` maps to `LLMTransientError` (not `LLMPermanentError`).
  - `test_permanent_http_errors`: parametrize 400, 401, 403, 404 → `LLMPermanentError`.
  - `test_network_timeout_and_connect_error`: `httpx2.ReadTimeout("t")` → `LLMTimeoutError`; `httpx2.ConnectError("c")` → `LLMTransientError`.
  - `test_empty_text_is_permanent_error`: candidate with no parts, `finish_reason="SAFETY"` → `LLMPermanentError` with `"SAFETY"` in the message.
  - `test_default_client_disables_sdk_retries`: monkeypatch `app.llm.providers.gemini.adapter.genai.Client` with a recorder; `GeminiProvider(api_key="test-key", model="m", timeout_s=30)` → recorded `http_options.retry_options.attempts == 1`, `http_options.timeout == 30000`, `api_key == "test-key"`.
- [ ] **Step 2: Write** `tests/llm/test_import_boundary.py`: `test_genai_imported_only_in_gemini_adapter` walks `app/**/*.py` with `ast`, flags `import google.genai…`, `from google.genai… import …` and `from google import genai` outside `app/llm/providers/gemini/`, asserts the offender list is `[]`; `test_boundary_check_is_not_vacuous` asserts the same scan finds the import inside `app/llm/providers/gemini/adapter.py`.
- [ ] **Step 3: Run** `.venv/Scripts/python.exe -m pytest tests/llm -v` → new tests FAIL.
- [ ] **Step 4: Implement** the adapter.
- [ ] **Step 5: Run** `.venv/Scripts/python.exe -m pytest -v` → all PASS.

### Task 5: Factory, integration test, docs

**Files:**
- Create: `app/llm/factory.py`, `tests/llm/test_factory.py`, `tests/integration/__init__.py`, `tests/integration/test_gemini_live.py`
- Modify: `pyproject.toml`, `README.md`

**Interfaces:**
- Consumes: `Settings` (Task 1), `LLMClient` (Task 3), `FakeProvider` (Task 2), `GeminiProvider` (Task 4).
- Produces: `build_llm_client(settings: Settings) -> LLMClient`. `gemini` → `GeminiProvider(api_key=settings.gemini_api_key.get_secret_value(), model=settings.llm_model, timeout_s=settings.llm_timeout_s)`, imported inside the function so `fake` mode never loads the SDK; `fake` → `FakeProvider(default_text="{}")`. Both get `timeout_s=settings.llm_timeout_s, max_retries=settings.llm_max_retries`.

- [ ] **Step 1: Write failing tests** in `tests/llm/test_factory.py`: `test_fake_provider_selected` (`Settings(_env_file=None, llm_provider="fake", llm_max_retries=2)` → `client.provider.name == "fake"`, `client.max_retries == 2`); `test_gemini_provider_selected` (key `"test-key"`, model `"gemini-x"` → `client.provider.name == "gemini"`, `client.provider.model == "gemini-x"`; monkeypatch `genai.Client` as in Task 4 so no network client is built).
- [ ] **Step 2: Run** → FAIL; **implement** `app/llm/factory.py`; **run** → PASS.
- [ ] **Step 3: Integration test.** `pyproject.toml` `[tool.pytest.ini_options]`: `markers = ["integration: makes a real, paid LLM call; run with -m integration"]`, `addopts = "-m 'not integration'"`. `tests/integration/test_gemini_live.py`, marked `integration` and `anyio`: `settings = Settings()` (reads `.env`); `pytest.skip` if `settings.gemini_api_key` is None; `LLMClient(GeminiProvider(...), timeout_s=settings.llm_timeout_s)`; `generate(system="Answer with JSON only.", prompt='Return {"status": "ok"}.', json_schema={"type": "object", "properties": {"status": {"type": "string"}}, "required": ["status"]})` → `json.loads(resp.text)["status"] == "ok"`, `resp.provider == "gemini"`, `resp.input_tokens > 0`.
- [ ] **Step 4: Run** `.venv/Scripts/python.exe -m pytest -v` → all unit tests PASS, the integration test shows as deselected.
- [ ] **Step 5: Ask Can before the paid call**, then run `.venv/Scripts/python.exe -m pytest -m integration -v` → PASS (or SKIP if no key in `.env`; report which).
- [ ] **Step 6: README** – a short "LLM client" section: the five env variables, `LLM_PROVIDER=fake` default and why, how to run the integration test, and the import-boundary rule.

### Task 6: Verification and commit

- [ ] **Step 1:** `.venv/Scripts/python.exe -m pytest -v` → all PASS; show the output.
- [ ] **Step 2:** `git status` and `git diff --stat`; confirm `.env` is not staged and no key appears in the diff (`git diff | grep -i "AIza"` returns nothing).
- [ ] **Step 3:** Show Can the commit commands, the message `SCRUM-59: add provider-neutral LLM client with Gemini adapter`, and the author (`git config user.name` / `user.email`); run only after approval. Push and PR are separate approvals.
