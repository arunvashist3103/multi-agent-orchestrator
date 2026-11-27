"""HTTP API tests for FastAPI routes, probes, error mapping, and log privacy."""

import logging
from contextlib import contextmanager
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from groq import APIStatusError, APITimeoutError, BadRequestError
from psycopg import OperationalError

import main


@pytest.fixture
def client(make_graph, monkeypatch):
    """TestClient bypassing real Postgres lifespan and injecting mocked pool + graph."""
    monkeypatch.setattr(main, "settings", main.settings.model_copy(update={"max_redos": 2}))

    def _configure(responses: list[str]):
        graph, fake_llm = make_graph(responses=responses)
        mock_conn = MagicMock()
        mock_pool = MagicMock()

        @contextmanager
        def _fake_connection():
            yield mock_conn

        mock_pool.connection = _fake_connection
        main._state["pool"] = mock_pool
        main._state["graph"] = graph
        return TestClient(main.app, raise_server_exceptions=False), fake_llm, mock_conn

    yield _configure
    main._state.clear()


# --- 1. Health & Readiness Probe Tests ---


def test_health_and_ready_probes_return_200_when_healthy(client):
    http, _, _ = client(responses=[])

    health = http.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert "X-Request-ID" in health.headers

    ready = http.get("/ready")
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "database": "connected"}


def test_ready_probe_returns_503_when_uninitialized_or_db_fails(client):
    # Uninitialized state
    raw_client = TestClient(main.app, raise_server_exceptions=False)
    main._state.clear()
    assert raw_client.get("/ready").status_code == 503

    # Initialized pool, but SELECT 1 raises OperationalError
    http, _, mock_conn = client(responses=[])
    mock_conn.execute.side_effect = OperationalError("db down")
    assert http.get("/ready").status_code == 503


# --- 2. Workflow Endpoints & Retry Quota Contract Tests ---


def test_research_and_approve_flow_returns_expected_contract(client):
    http, _, _ = client(
        responses=[
            "- Bullet 1\n- Bullet 2",
            "Final synthesized answer paragraph.",
        ]
    )

    start_res = http.post("/research", json={"topic": "Grid-scale battery storage"})
    assert start_res.status_code == 200
    body = start_res.json()
    assert body["status"] == "pending_review"
    assert body["bullet_points"] == "- Bullet 1\n- Bullet 2"
    assert body["redo_count_so_far"] == 0
    assert body["redos_remaining"] == 2

    review_res = http.post(f"/review/{body['thread_id']}", json={"type": "approve"})
    assert review_res.status_code == 200
    done = review_res.json()
    assert done["status"] == "complete"
    assert done["final_answer"] == "Final synthesized answer paragraph."


def test_review_reject_countdown_and_blocks_at_max_redos(client):
    http, _, _ = client(
        responses=[
            "- Draft 0",
            "- Draft 1",
            "- Draft 2",
        ]
    )

    start = http.post("/research", json={"topic": "Sodium-ion grid storage"}).json()
    tid = start["thread_id"]
    assert start["redos_remaining"] == 2

    # Reject #1 -> redos_remaining drops to 1
    r1 = http.post(
        f"/review/{tid}",
        json={"type": "reject", "feedback": "Add 2025 deployment numbers."},
    )
    assert r1.status_code == 200
    assert r1.json()["redos_remaining"] == 1

    # Reject #2 -> redos_remaining drops to 0
    r2 = http.post(
        f"/review/{tid}",
        json={"type": "reject", "feedback": "Add grid connection bottlenecks."},
    )
    assert r2.status_code == 200
    assert r2.json()["redos_remaining"] == 0

    # Reject #3 -> blocked at API boundary with 400 Bad Request
    r3 = http.post(
        f"/review/{tid}",
        json={"type": "reject", "feedback": "This should be blocked with 400."},
    )
    assert r3.status_code == 400
    assert "Maximum retries" in r3.json()["detail"]


def test_review_unknown_thread_returns_404(client):
    http, _, _ = client(responses=[])
    res = http.post("/review/non-existent-thread-id", json={"type": "approve"})
    assert res.status_code == 404


# --- 3. Upstream Exception Handlers & Log Privacy Tests ---


@pytest.mark.parametrize(
    ("exc_factory", "expected_status"),
    [
        (
            lambda: BadRequestError(
                message="tool_use_failed",
                response=httpx.Response(400, request=httpx.Request("POST", "https://groq.com")),
                body=None,
            ),
            502,
        ),
        (
            lambda: APIStatusError(
                message="rate limit",
                response=httpx.Response(429, request=httpx.Request("POST", "https://groq.com")),
                body=None,
            ),
            429,
        ),
        (
            lambda: APITimeoutError(request=httpx.Request("POST", "https://groq.com")),
            504,
        ),
        (
            lambda: OperationalError("connection refused"),
            503,
        ),
    ],
)
def test_upstream_errors_map_to_clean_http_statuses(client, exc_factory, expected_status):
    http, _, _ = client(responses=[])
    failing_graph = MagicMock()
    failing_graph.invoke.side_effect = exc_factory()
    main._state["graph"] = failing_graph

    res = http.post("/research", json={"topic": "Test error mapping"})
    assert res.status_code == expected_status
    assert "detail" in res.json()


def test_application_logs_redact_user_topics_and_notes(client, caplog):
    caplog.set_level(logging.INFO, logger="research_service")
    secret_topic = "CONFIDENTIAL_PROJECT_ORION_2026"
    secret_bullet = "- SECRET_BULLET_DATA_998877"

    http, _, _ = client(responses=[secret_bullet, "Final public summary."])
    res = http.post("/research", json={"topic": secret_topic})
    assert res.status_code == 200

    all_logs = "\n".join(record.getMessage() for record in caplog.records)
    assert secret_topic not in all_logs
    assert secret_bullet not in all_logs
    assert "thread_id=" in all_logs
