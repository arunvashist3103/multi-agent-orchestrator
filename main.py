"""
Multi-Agent Research & Writing Pipeline
=======================================

This module implements a Langgraph workflow that combines two LLM agents
(researcher + writer) with a human-in-the-loop review step.

Workflow
--------
1. **START** → **researcher** — the researcher agent uses web search to gather
   factual bullet points on the user's request.
2. **researcher** → **human_review** — the human reviewer is prompted (via
   ``interrupt``) to approve, edit, or reject the research output.
3. **human_review** → **writer** (approved / edited) — the writer agent turns the
   bullet points into a single polished paragraph.
4. **writer** → **END** — the final paragraph is returned.

If the reviewer rejects the research, the workflow loops back to the
researcher (up to ``MAX_REDOS`` times) with feedback before proceeding
to the writer anyway.
"""

import os
from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv

# LangGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

# LangChain
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

# Load environment variables from the .env file (GROQ_API_KEY, TAVILY_API_KEY)
load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Maximum number of times the research step can be retried before the workflow
# proceeds to the writer agent regardless of the reviewer's decision.
MAX_REDOS = 2

# --------------------------------------------------------------------------- #
# Shared LLM + Tools
# --------------------------------------------------------------------------- #
# Groq-hosted model used by both agents (OpenAI-compatible endpoint).
model = ChatGroq(model="openai/gpt-oss-120b")

# Web-search tool injected into the researcher agent.
websearch_tool = TavilySearch(max_results=5)

# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #

research_agent = create_agent(
    model=model,
    tools=[websearch_tool],
    system_prompt=(
        "You are a Researcher agent. Your ONLY job is to gather factual bullet "
        "points on the topic you're given, using the available search tool. Do "
        "NOT write prose or a final answer; return only concise bullet points."
    ),
    name="researcher_agent",
)

# Writer Agent
writer_agent = create_agent(
    model=model,
    tools=[],
    system_prompt=(
        "You are a Writer agent. Take the research bullet points already gathered "
        "in the conversation and turn them into one polished, engaging paragraph "
        "(3-5 sentences). Do not add facts that are not in the bullet points."
    ),
    name="writer_agent",
)

# --------------------------------------------------------------------------------------------------------------
# Graph State

class GraphState(TypedDict):
    """State carried between research, review, and writing nodes."""

    messages: Annotated[list, add_messages]
    review_action: Literal["", "writer", "researcher"]
    redo_count: int


# Nodes

def research_node(state: GraphState) -> dict:
    """Run web research using the conversation collected so far."""
    result = research_agent.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def human_review_node(state: GraphState) -> dict:
    """Pause for human approval and route to writing or another research pass."""
    bullet_points = state["messages"][-1].content

    decision = interrupt({
        "action": "review_search",
        "bullet_points": bullet_points,
        "redo_count_so_far": state["redo_count"],
    })

    decision_type = decision.get("type")

    if decision_type == "approve":
        return {"review_action": "writer"}

    if decision_type == "edit":
        return {
            "review_action": "writer",
            "messages": [
                HumanMessage(
                    content=f"[Human-edited research notes]\n"
                    f"{decision['edited_content']}"
                )
            ],
        }

    # Keep rejected research in the conversation so the next pass can address
    # the reviewer's feedback rather than repeating the same search blindly.
    new_redo_count = state["redo_count"] + 1
    if new_redo_count > MAX_REDOS:
        return {
            "review_action": "writer",
            "redo_count": new_redo_count,
            "messages": [
                HumanMessage(
                    content=(
                        f"Research was rejected {MAX_REDOS} times. Proceeding to "
                        "writing with the best available research as-is."
                    )
                )
            ],
        }

    return {
        "review_action": "researcher",
        "redo_count": new_redo_count,
        "messages": [
            HumanMessage(
                content=(
                    "Reviewer feedback; please redo the research: "
                    f"{decision.get('feedback', 'Not detailed enough')}"
                )
            )
        ],
    }


def writer_node(state: GraphState) -> dict:
    """Turn the user's request and reviewed research into a final paragraph."""
    msgs = state["messages"]
    clean = []
    for m in msgs:
        if getattr(m, "tool_calls", None):
            continue
        if getattr(m, "type", "") == "tool":
            continue
        clean.append(m)

    # Tool-call messages are implementation details and can confuse the writer.
    # Retain the original request and latest research even for a sparse state.
    if not clean:
        clean = [msgs[0], msgs[-1]]

    result = writer_agent.invoke({"messages": clean})
    return {"messages": result["messages"]}


def route_after_review(state: GraphState) -> Literal["writer", "researcher"]:
    """Return the graph edge selected by the human review node."""
    return state["review_action"]


# Assemble the graph.
graph = StateGraph(GraphState)

graph.add_node("researcher", research_node)
graph.add_node("human_review", human_review_node)
graph.add_node("writer", writer_node)

graph.add_edge(START, "researcher")
graph.add_edge("researcher", "human_review")
graph.add_conditional_edges(
    "human_review",
    route_after_review,
    {"writer": "writer", "researcher": "researcher"},
)
graph.add_edge("writer", END)

app = graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    """Run the workflow with an interactive terminal review step."""
    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit("Set GROQ_API_KEY before running this demo")

    if not os.environ.get("TAVILY_API_KEY"):
        raise SystemExit("Set TAVILY_API_KEY before running this demo")

    config = {"configurable": {"thread_id": "demo_run_1"}}

    result = app.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a short paragraph explaining what LangGraph is "
                        "based on the current information on the web."
                    ),
                }
            ],
            "review_action": "",
            "redo_count": 0,
        },
        config=config,
    )

    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        print("\n=== HUMAN APPROVAL REQUIRED ===")
        print("Payload of bullet points :", payload["bullet_points"])
        print(f"\n(redo attempts so far: {payload['redo_count_so_far']}/{MAX_REDOS})")


        choice = input("\n[a]pprove / [e]dit / [r]eject? ").strip().lower()

        if choice == "a":
            resume_value = {"type": "approve"}

        elif choice == "e":
            edited = input("Enter your corrected bullet points: \n")
            resume_value = {"type": "edit", "edited_content": edited}

        else:
            feedback = input("What's wrong with the bullet points? \n")
            resume_value = {"type": "reject", "feedback": feedback}

        # Resuming the same thread re-enters the paused node with its state intact.
        result = app.invoke(Command(resume=resume_value), config=config)

    print("\n=== FULL MESSAGE TRACE (who said what) ===")
    for m in result["messages"]:
        role = getattr(m, "name", None) or m.type
        print(f"\n[{role}]\n{m.content}")

    print("\n=== FINAL ANSWER ===")
    print(result["messages"][-1].content)