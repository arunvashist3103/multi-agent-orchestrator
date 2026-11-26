# LangGraph Research and Writing Service

A FastAPI service that researches a topic with Tavily, pauses for human review,
and uses a Groq-hosted writer to produce a short paragraph. LangGraph coordinates
the workflow and saves checkpoints in PostgreSQL so a later HTTP request can
resume the same thread, including after a service restart.

## Workflow

```mermaid
flowchart LR
    A[POST /research] --> B[Researcher and web search]
    B --> C[Pause for human review]
    C --> D[POST /review/thread_id]
    D -->|Approve or edit| E[Writer]
    D -->|Reject within redo budget| B
    E --> F[Final answer]
```

Research and writing run during their respective HTTP requests. The request
ends when human input is needed; there is no background job waiting for approval.
An edit replaces the notes supplied to the writer. Rejection sends the current
draft and reviewer feedback into another research pass.

The default `MAX_REDOS=2` permits two additional research passes after the first.
At the limit, the API refuses another rejection with HTTP 400 and requires
approval or editing. Direct graph callers bypass that endpoint guard: the graph
has a fallback that proceeds to writing when the limit is exceeded.

## Local setup

Requirements: Python 3.11+, a running PostgreSQL server, and Groq and Tavily API
keys. Run these commands from the directory containing `main.py`:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install fastapi uvicorn python-dotenv langgraph \
  langgraph-checkpoint-postgres 'psycopg[binary]' psycopg-pool \
  langchain langchain-groq langchain-tavily pydantic-settings
cp .env.example .env
```

Set `GROQ_API_KEY`, `TAVILY_API_KEY`, and `DATABASE_URL` in `.env`. Create the
database named in the URL before starting the service. For an existing local
PostgreSQL server using the default connection settings:

```bash
createdb -h localhost -p 5432 -U postgres checkpoints
```

The configured database role needs permission to create checkpoint tables.
Startup calls `PostgresSaver.setup()` to initialize them; it does not create the
database itself.

Start the development server:

```bash
uvicorn main:app --reload
```

Interactive API documentation is available at `http://localhost:8000/docs`.
Use `uvicorn main:app` without reload when automatic development reloads are not
needed. This repository does not include a frontend or deployment configuration.

## Configuration

Settings are validated in `config.py`. Environment variables override `.env`;
unknown entries are ignored. Settings are cached per process, so restart the
service after changing them.

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | Required | Groq authentication. |
| `TAVILY_API_KEY` | Required | Tavily authentication. |
| `DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/checkpoints` | Checkpoint database connection. |
| `APP_TITLE` | `Research and Writing Agent Service` | FastAPI application title. |
| `ENVIRONMENT` | `development` | `development`, `staging`, or `production`. |
| `LOG_LEVEL` | `INFO` | Python logging level. |
| `DB_POOL_MIN_SIZE` | `2` | Minimum connections per application process. |
| `DB_POOL_MAX_SIZE` | `10` | Maximum connections per application process. |
| `RESEARCHER_MODEL_NAME` | `qwen/qwen3.8-27b` | Groq model used for research. |
| `WRITER_MODEL_NAME` | `openai/gpt-oss-120b` | Groq model used for writing. |
| `LLM_TEMPERATURE` | `0.2` | Model sampling temperature. |
| `LLM_TIMEOUT` | `30.0` | Provider request timeout in seconds. |
| `LLM_MAX_RETRIES` | `2` | Groq SDK retries. |
| `TAVILY_MAX_RESULTS` | `3` | Maximum results per search. |
| `MAX_REDOS` | `2` | Reviewer-requested additional research passes. |

Keep the maximum pool size at least as large as the minimum. The configured
model names must be available to your Groq account.

Researcher and writer nodes separately allow up to three attempts for
`BadRequestError`, `APITimeoutError`, and `APIConnectionError`. These node attempts
are in addition to SDK-level retries and do not consume the human-review redo
budget.

## API examples

Start research:

```bash
curl -i http://localhost:8000/research \
  -H 'Content-Type: application/json' \
  -d '{"topic":"Recent advances in grid-scale battery storage"}'
```

The response has this shape (notes and thread ID vary):

```json
{
  "thread_id": "<returned-thread-id>",
  "status": "pending_review",
  "bullet_points": "- First finding\n- Second finding",
  "redo_count_so_far": 0,
  "redos_remaining": 2
}
```

Approve the notes using the returned thread ID:

```bash
curl -i 'http://localhost:8000/review/<returned-thread-id>' \
  -H 'Content-Type: application/json' \
  -d '{"type":"approve"}'
```

Replace `<returned-thread-id>` before running the command. Alternatively, submit
one of these JSON bodies to the same review endpoint:

```json
{"type":"edit","edited_content":"- Replace the draft with these reviewed research notes."}
```

```json
{"type":"reject","feedback":"Include more recent examples and supporting evidence."}
```

Approval or editing returns:

```json
{
  "thread_id": "<returned-thread-id>",
  "status": "complete",
  "final_answer": "The generated paragraph appears here."
}
```

Rejection within the budget returns another `pending_review` response with an
incremented redo count. Topic length must be 3-500 characters, edited content
20-5,000 characters, and rejection feedback 10-1,000 characters. Surrounding
whitespace is removed before validation.

| Endpoint | Purpose |
|---|---|
| `POST /research` | Create a new thread and research a topic. |
| `POST /review/{thread_id}` | Resume that thread with a review decision. |
| `GET /health` | Liveness only; no external services are contacted. |
| `GET /ready` | Check graph initialization and execute a database query. |

Readiness does not test Groq or Tavily availability.

| HTTP status | Meaning |
|---|---|
| `400` | Another rejection would exceed the redo budget. |
| `404` | No pending work was found for the review thread. |
| `422` | Request fields or decision-specific requirements are invalid. |
| `429` | Groq returned a token/rate-limit response (413 or 429). |
| `502` | Groq returned a bad request or another handled API status error. |
| `503` | Database operational error or failed readiness check. |
| `504` | Groq request timed out after applicable retries. |
| `500` | An otherwise unhandled application/provider failure. |

## Code map

| File | Responsibility |
|---|---|
| `main.py` | HTTP routes, application lifecycle, database pool, logging, error translation. |
| `graph.py` | Agent construction, search wrapper, workflow state, review routing, node retries. |
| `schemas.py` | Request validation and pending/completed response contracts. |
| `config.py` | Environment configuration and cached validated settings. |
| `prompts.py` | Researcher and writer instructions. |
| `.env.example` | API-key placeholders for local configuration. |
| `requirements.txt` | Dependency declarations. |

`build_graph` accepts a checkpointer, optional settings, model, and search tool.
This permits local verification with an in-memory saver and fake providers.
The HTTP service supplies a PostgreSQL saver. The repository currently has no
committed automated test suite.

## Application logging

Set `LOG_LEVEL` in `.env` or the environment (default: `INFO`). Application
logs use the `research_service` and `research_service.graph` logger names and
include timestamps, severity, and event details.

- `INFO`: startup/shutdown, request status and duration, research and writing
  attempts, search result counts, review decisions, and workflow completion.
- `WARNING`: refused reviews, retryable agent failures, and unexpected search
  response formats.
- `ERROR`: startup, database, provider, search, and unhandled request failures.
- `DEBUG`: checkpoint initialization, search start, review entry, and successful
  health probes.

HTTP responses handled by the request logger include an `X-Request-ID` header.
Use `request_id` to find a request and `thread_id` to follow a workflow across
research and review requests. Durations are recorded in milliseconds.
Application events omit topics, prompts, generated content, credentials, and
raw exception bodies; failures record the exception type instead. Third-party
libraries control their own log content, especially when enabling `DEBUG`.

`GET /health` checks liveness; `GET /ready` checks workflow initialization and
database connectivity. Successful probes use `DEBUG` to avoid routine log noise.

Timing measures elapsed time until a response is produced, not completion of
streaming a response body. Unhandled exceptions are re-raised by the middleware;
the resulting server-generated 500 response does not receive its `X-Request-ID`
header. The failure log still records the request ID.

## Credentials and persisted data

Keep credentials in environment variables or `.env`, which is excluded from
Git. PostgreSQL checkpoints contain topics, research notes, feedback, and
messages even though application logs omit that content. The current API has
no authentication or per-user thread ownership checks; a thread ID identifies
a workflow but does not establish permission to access it.
