"""FastAPI service for the checkpointed research-and-writing workflow."""

import logging
from contextlib import asynccontextmanager 
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from groq import APIStatusError, APITimeoutError, BadRequestError
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from psycopg_pool import ConnectionPool
from psycopg import OperationalError

from graph import build_graph
from schemas import ResearchRequest, ReviewDecision, PendingReview, FinalAnswer
from config import get_settings

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("research_service")

# Process-local resources; workflow state is persisted in PostgreSQL.
_state = {}

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize checkpoints and the graph; release service resources on exit."""
    started = perf_counter()
    initialized = False
    logger.info(
        "Service starting environment=%s pool_min_size=%s pool_max_size=%s",
        settings.environment, settings.db_pool_min_size, settings.db_pool_max_size,
    )
    try:
        with ConnectionPool(
            conninfo=str(settings.database_url),
            min_size=settings.db_pool_min_size,
            max_size=settings.db_pool_max_size,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        ) as pool:
            logger.debug("Initializing PostgreSQL checkpoint tables")
            checkpointer = PostgresSaver(pool)
            checkpointer.setup()

            _state["pool"] = pool
            _state["graph"] = build_graph(checkpointer=checkpointer, settings=settings)
            initialized = True
            logger.info("Service ready duration_ms=%.2f", (perf_counter() - started) * 1000)
            try:
                yield
            finally:
                logger.info("Service shutting down; closing database pool")
    except Exception as exc:
        # Provider/DB exception messages can contain credentials or request content.
        logger.error(
            "Service lifecycle failed phase=%s error_type=%s",
            "shutdown" if initialized else "startup", type(exc).__name__,
        )
        raise
    finally:
        _state.clear()
    logger.info("Service shutdown complete")
    

app = FastAPI(title=get_settings().app_title, lifespan=lifespan)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request IDs, status, and time to response; re-raise unhandled errors."""
    request.state.request_id = str(uuid4())
    started = perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(
            "Request failed request_id=%s method=%s route=%s thread_id=%s error_type=%s duration_ms=%.2f",
            request.state.request_id, request.method,
            getattr(request.scope.get("route"), "path", "unmatched"),
            getattr(request.state, "thread_id", "-"), type(exc).__name__,
            (perf_counter() - started) * 1000,
        )
        raise
    # Route templates avoid logging arbitrary URL paths or query strings.
    route = getattr(request.scope.get("route"), "path", "unmatched")
    level = logging.ERROR if response.status_code >= 500 else (
        logging.WARNING if response.status_code >= 400 else logging.INFO
    )
    if route in ("/health", "/ready") and response.status_code < 400:
        level = logging.DEBUG
    logger.log(
        level,
        "Request completed request_id=%s method=%s route=%s thread_id=%s status=%s duration_ms=%.2f",
        request.state.request_id, request.method, route,
        getattr(request.state, "thread_id", "-"), response.status_code,
        (perf_counter() - started) * 1000,
    )
    response.headers["X-Request-ID"] = request.state.request_id
    return response

@app.exception_handler(BadRequestError)
async def llm_bad_request_handler(request: Request, exc: BadRequestError):
    """Translate a provider bad request into 502 without exposing its body."""
    logger.error("Upstream LLM bad request request_id=%s", request.state.request_id)
    return JSONResponse(
        status_code = status.HTTP_502_BAD_GATEWAY,
        content={"detail": "Upstream LLM failed to format a valid tool call. Please retry."},
    )

@app.exception_handler(APIStatusError)
async def llm_api_error_handler(request: Request, exc: APIStatusError):
    """Map provider token/rate limits to 429 and other status errors to 502."""
    logger.error("Upstream LLM API error request_id=%s status=%s", request.state.request_id, exc.status_code)
    if exc.status_code in (413, 429):
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Upstream LLM rate or token limit exceeded. Please retry shortly."},
        )
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "Upstream LLM service returned an error."},
    )

@app.exception_handler(APITimeoutError)
async def llm_timeout_handler(request: Request, exc: APITimeoutError):
    """Return 504 when the provider times out after applicable retries."""
    logger.error("Upstream LLM timeout request_id=%s", request.state.request_id)
    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content={"detail": "Upstream LLM request timed out. Please try again."},
    )

@app.exception_handler(OperationalError)
async def database_error_handler(request: Request, exc: OperationalError):
    """Return 503 for database operational errors without logging a DSN."""
    logger.error("PostgreSQL connection error request_id=%s error_type=%s", request.state.request_id, type(exc).__name__)
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": "Checkpoint database is temporarily unavailable."},
    )

@app.get("/health", tags=["Operations"])
def health_check():
    """Confirm the API responds; this does not contact the database or models."""
    return {"status": "ok"}

@app.get("/ready", tags=["Operations"])
def rediness_check():
    """Check graph and database readiness; external providers are not probed."""
    pool: ConnectionPool | None = _state.get("pool")
    if pool is None or "graph" not in _state:
        logger.warning("Readiness check failed reason=service_not_initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not initialized",
        )
    try:
        with pool.connection() as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        logger.error("Readiness check failed error_type=%s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection failed",
        ) from exc
    return {"status": "ready", "database": "connected"}


def _format_response(thread_id: str, result: dict):
    """Convert a single review interrupt or completed run into an API response."""

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        logger.info(
            "Workflow paused for review thread_id=%s redo_count=%s",
            thread_id, payload["redo_count_so_far"],
        )
    
        return PendingReview(
            thread_id = thread_id,
            status = "pending_review",
            bullet_points = payload["bullet_points"],
            redo_count_so_far = payload["redo_count_so_far"],
            redos_remaining = max(0, settings.max_redos - payload["redo_count_so_far"])
        )

    logger.info("Workflow completed thread_id=%s", thread_id)
    return FinalAnswer(
            thread_id =  thread_id,
            status = "complete",
            final_answer = result["messages"][-1].content,
    )   

@app.post("/research" , response_model= PendingReview | FinalAnswer)
def start_research(request: ResearchRequest, http_request: Request):
    """Run initial research and return notes with a thread ID for human review."""
    thread_id = str(uuid4())
    http_request.state.thread_id = thread_id
    logger.info(
        "Research requested request_id=%s thread_id=%s",
        http_request.state.request_id, thread_id,
    )
    config = {"configurable": {"thread_id": thread_id}}


    result  = _state["graph"].invoke(
        {
            "topic": request.topic,
            "messages": [HumanMessage(content = request.topic)],
            "research_notes": "",
            "review_action": "",
            "redo_count": 0,
        },
        config = config
    )

    return _format_response(thread_id, result)

@app.post("/review/{thread_id}", response_model= PendingReview | FinalAnswer)
def submit_review(thread_id : str, decision: ReviewDecision, http_request: Request):
    """Resume a review and return revised notes or the final answer.

    Return 404 if no work is pending, or 400 if rejection exceeds the redo budget.
    """
    http_request.state.thread_id = thread_id
    logger.info(
        "Review submitted request_id=%s thread_id=%s decision=%s",
        http_request.state.request_id, thread_id, decision.type,
    )
    config = {"configurable" : {"thread_id" : thread_id}}
    graph = _state["graph"]

    state = graph.get_state(config)

    if not state.next:
        logger.warning("Review refused thread_id=%s reason=no_pending_review", thread_id)
        raise HTTPException(status_code=404, detail="No pending review for this thread_id")

    current_redos = state.values.get("redo_count", 0)
    if decision.type == "reject" and current_redos >= settings.max_redos:
        logger.warning(
            "Review refused thread_id=%s reason=retry_limit redo_count=%s max_redos=%s",
            thread_id, current_redos, settings.max_redos,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Maximum retries ({settings.max_redos}) reached. Please approve or edit the research notes.",
        )

    # The same thread ID resumes its checkpoint; Command supplies the review.
    result = graph.invoke(Command(resume=decision.model_dump()), config = config)

    return _format_response(thread_id, result)

