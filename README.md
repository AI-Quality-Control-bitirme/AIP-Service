# AI Evaluation Service (AIP-Service)

The AI Evaluation Service (Epic E2 in the Statement of Work) evaluates recorded Turkish hotel reservation calls after the transcript is ready. It will contain the precheck agent, the tool execution layer for the mock hotel API, the five specialist agents, JSON schema validation and scoring. The agents are modules inside this service, not separate services.

This repository is currently a **walking skeleton** (Sprint 4, SCRUM-58): a FastAPI application with environment configuration and a health check, plus the LLM provider client (SCRUM-59) the agents will use. The evaluation features are added by later cards.

## Requirements

- Python 3.12
- No external services are needed to run the tests or the service. A Gemini API key is needed only for `LLM_PROVIDER=gemini` and the integration test.

## Setup

Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
Copy-Item .env.example .env
```

Linux / macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
```

`.env` is git-ignored. Never commit secrets; add new variables to `.env.example` with a safe default or an empty value.

## Run

```bash
python -m app
```

The service reads `HOST`, `PORT` and `LOG_LEVEL` from the environment or `.env`. For development with auto-reload:

```bash
uvicorn app.main:app --reload
```

Then:

- `GET http://127.0.0.1:8000/health` → `200 {"status": "ok", "service": "ai-evaluation", "version": "0.1.0"}`
- Interactive API docs: `http://127.0.0.1:8000/docs`

## Test

```bash
pytest
```

The default run makes no paid API calls. The integration tests check the Gemini adapter against the real API (a successful JSON call, a rejected key and a network timeout; a few dozen tokens in total). They run only when selected; the tests that need a real key are skipped when `GEMINI_API_KEY` is not set in the environment or `.env`:

```bash
pytest -m integration
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` inside a container. |
| `PORT` | `8000` | Port to listen on (1–65535). |
| `LOG_LEVEL` | `info` | `critical`, `error`, `warning`, `info` or `debug`. |
| `LLM_PROVIDER` | `fake` | `fake` or `gemini`. `fake` needs no key and never calls a paid API, so a fresh checkout and CI cannot spend money by accident. Deployments set `gemini`. |
| `GEMINI_API_KEY` | – | Secret. Required when `LLM_PROVIDER=gemini`; keep it only in `.env`. |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Model identifier sent to the provider. |
| `LLM_TIMEOUT_S` | `30` | Seconds to wait for one response. |
| `LLM_MAX_RETRIES` | `3` | Retries for temporary failures (timeouts, 429, 5xx), 0–3 (AC13). |

Invalid values stop the service at start-up with a validation error.

## LLM client

`app/llm` is the single place the service talks to an LLM. Agents build a client with `build_llm_client(get_settings())` and call `await client.generate(system=..., prompt=..., json_schema=...)`. The response carries the text plus `provider`, `model`, `input_tokens`, `output_tokens`, `latency_s` and `retries`, so callers can store the model version (UC08) and log usage (AC14). Validating the content of the text is the agent's job.

- **Errors:** temporary failures (timeouts, 429, 5xx) are retried with 1 s, 2 s, 4 s backoff; after the last one the client raises `LLMRetriesExhaustedError` naming the cause. Permanent failures (bad request, auth, empty answer) raise `LLMPermanentError` at once. All errors derive from `LLMError`.
- **Logging:** one line per call with provider, model, tokens, latency and retries. Prompt text, system text and the API key are never logged (transcripts are personal data, AC15).
- **Tests:** `app.llm.providers.fake.FakeProvider` returns preset texts or raises preset errors; use it in agent tests instead of a real provider.
- **Import rule:** only `app/llm/providers/gemini/` may import the `google-genai` SDK. `tests/llm/test_import_boundary.py` enforces this, so another provider can be added as a new adapter without changing agent code (SoW §14).

## Health check

`GET /health` is a liveness check: it reports that the process is up and which version is running. It does not check external dependencies, because the service has none yet.

## Project layout

```
app/
  __init__.py      service version
  __main__.py      `python -m app` entry point
  config.py        settings from environment variables (pydantic-settings)
  main.py          FastAPI application factory
  api/health.py    GET /health
  llm/             provider-neutral LLM client, errors and fake provider
  llm/providers/gemini/  Gemini adapter (the only user of google-genai)
tests/             pytest tests
```

## Not in this card

Tool calling, storing usage and cost, the precheck and the mock hotel API, specialist agents, versioned OpenAPI contracts, correlation IDs, database, job queue and the container image are planned in other cards.
