# SCRUM-59: LLM provider client – design

Date: 2026-11-27 · Card: SCRUM-59 · Branch: `task/SCRUM-59`

## Intent
The AI Evaluation Service needs one place that talks to an LLM. The precheck (SCRUM-76) and the specialist agents (SCRUM-60) send prompts through it. The SoW asks for a replaceable provider (§14), bounded retries (AC13), per-call usage (AC14) and the model identifier stored with each result (UC08); these belong to this layer.

## Goals
- An async, provider-neutral interface that agents depend on: our own request/response types and errors.
- One adapter: Gemini (`gemini-3.5-flash-lite` by default), through the official `google-genai` SDK.
- Structured output: a request can carry a JSON schema; the client returns the response text. Validation of the content is done by the agents (SCRUM-60).
- Timeout per request; temporary failures retried at most 3 times, then a clear error.
- Every response carries provider, model, input/output tokens, latency and the number of retries.
- A fake provider in the package, reused by SCRUM-60 and SCRUM-76 tests.
- API key and settings from environment variables (`app/config.py`), placeholder in `.env.example`.

## Non-goals
Tool calling (SCRUM-76), storing usage or computing cost (later), an OpenAI or Claude adapter, streaming, request-rate throttling (a 429 is handled by the retry).

## Anti-goals
- Gemini SDK types or imports outside the Gemini adapter.
- Paid API calls in unit tests.
- The API key or prompt/transcript text in logs (AC15; transcripts are personal data).

## Approaches
- **A. Official SDK behind our interface (chosen).** Little code; the SDK handles the Gemini wire format. We own the interface, retry, usage reporting and fake provider. Cost: a heavy dependency, and SDK errors must be mapped to ours.
- **B. Direct REST calls with httpx.** Lightest and fully ours, but we would maintain the request/response format ourselves; too much for this card.
- **C. A multi-provider library (e.g. LiteLLM).** Rejected: it would own the adapter design we want to build ourselves, and it is oversized for one provider.

Chosen by Can on 2026-11-27, with two additions: simple schemas (below) and an import-boundary test.

## Decisions
- **Layout:** `app/llm/` holds the interface, client, errors and settings glue; `app/llm/providers/fake.py`; `app/llm/providers/gemini/` holds everything Gemini.
- **Import boundary:** a test parses every file under `app/` and fails if `google.genai` (or `from google import genai`) is imported outside `app/llm/providers/gemini/`.
- **Schemas:** agent schemas stay simple (flat Pydantic models, basic types, no `oneOf` or conditional rules). The provider schema enforces the shape; our Pydantic validation enforces the rules (ranges, conditions, segment IDs). Each adapter translates the schema to its provider's dialect.
- **Errors:** `LLMError` base; temporary errors (timeout, 429, 5xx) are retried; permanent errors (bad request, auth, invalid schema) are raised at once; after 3 retries the client raises a "retries exhausted" error that names the last cause.
- **Retry timing:** exponential backoff (1 s, 2 s, 4 s); the sleep function is injectable so tests run instantly.
- **Settings:** `LLM_PROVIDER` (`gemini` or `fake`), `GEMINI_API_KEY` (secret), `LLM_MODEL`, `LLM_TIMEOUT_S`, `LLM_MAX_RETRIES` (0–3, default 3). Choosing `gemini` without a key fails at startup with a clear message.
- **Logging:** one line per call with provider, model, tokens, latency and retries; never the key or the prompt text.
- **Tests:** unit tests use the fake provider and a mocked SDK client; the integration test makes one real Gemini call, is marked `integration`, and runs only when `GEMINI_API_KEY` is set and integration tests are selected explicitly.

## Left to the builder
- The exact `google-genai` version (SCRUM-28 used 2.34.0) and the SDK parameter for a JSON-schema response: verify against the current SDK docs.
- How the SDK reports timeouts and HTTP status codes, for the error mapping.
- Names of classes and functions.
