import os
from contextlib import asynccontextmanager 
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command
from langchain_core.messages import HumanMessage
from psycopg_pool import ConnectionPool

from graph import build_graph
from schemas import ResearchRequest, ReviewDecision, PendingReview, FinalAnswer
from config import get_settings

settings = get_settings()
_state = {}

@asynccontextmanager
async def lifespan(_: FastAPI):
    with ConnectionPool(
        conninfo=str(settings.database_url),
        min_size=settings.db_pool_min_size,
        max_size=settings.db_pool_max_size,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()

        _state["pool"] = pool
        _state["graph"] = build_graph(checkpointer=checkpointer, settings=settings)
        yield
    

app = FastAPI(title=get_settings().app_title, lifespan=lifespan)

def _format_response(thread_id: str, result: dict):
    """Shared by both endpoints : a run either pauses again (another review) or finishes. This is the one place that shape gets decided"""

    if "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
    
        return PendingReview(
            thread_id = thread_id,
            status = "pending_review",
            bullet_points = payload["bullet_points"],
            redo_count_so_far = payload["redo_count_so_far"],
            redos_remaining = max(0, settings.max_redos - payload["redo_count_so_far"])
        )

    return FinalAnswer(
            thread_id =  thread_id,
            status = "complete",
            final_answer = result["messages"][-1].content,
    )   

@app.post("/research" , response_model= PendingReview | FinalAnswer)
def start_research(request: ResearchRequest):
    thread_id = str(uuid4())
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

    # Function returns here - the request is done. No process is left waiting.
    return _format_response(thread_id, result)

@app.post("/review/{thread_id}", response_model= PendingReview | FinalAnswer)
def submit_review(thread_id : str, decision: ReviewDecision):
    config = {"configurable" : {"thread_id" : thread_id}}
    graph = _state["graph"]

    state = graph.get_state(config)

    if not state.next:
        raise HTTPException(status_code=404, detail="No pending review for this thread_id")

    current_redos = state.values.get("redo_count", 0)
    if decision.type == "reject" and current_redos >= settings.max_redos:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum retries ({settings.max_redos}) reached. Please approve or edit the research notes.",
        )

    result = graph.invoke(Command(resume=decision.model_dump()), config = config)

    return _format_response(thread_id, result)

