# AI Evaluation Service (AIP-Service)

The AI Evaluation Service (Epic E2 in the Statement of Work) evaluates recorded Turkish hotel reservation calls after the transcript is ready. It will contain the precheck agent, the tool execution layer for the mock hotel API, the five specialist agents, JSON schema validation and scoring. The agents are modules inside this service, not separate services.

This repository is currently a **walking skeleton** (Sprint 4, SCRUM-58): a FastAPI application with environment configuration and a health check. The evaluation features are added by later cards.

## Requirements

- Python 3.12
- No external services are needed yet (no database, queue or LLM provider).

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

## Configuration

| Variable | Default | Description |
|---|---|---|
| `HOST` | `127.0.0.1` | Address to listen on. Use `0.0.0.0` inside a container. |
| `PORT` | `8000` | Port to listen on (1–65535). |
| `LOG_LEVEL` | `info` | `critical`, `error`, `warning`, `info` or `debug`. |

Invalid values stop the service at start-up with a validation error.

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
tests/             pytest tests
```

## Not in this card

Precheck and the mock hotel API, specialist agents, versioned OpenAPI contracts, correlation IDs, database, job queue and the container image are planned in other cards.
