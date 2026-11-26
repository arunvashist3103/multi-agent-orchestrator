"""Checkpointed research, human review, and writing with injectable providers."""

import logging
from time import perf_counter
from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import RetryPolicy, interrupt
from groq import APIConnectionError, APITimeoutError, BadRequestError

from config import Settings, get_settings
from prompts import RESEARCHER_SYSTEM_PROMPT, WRITER_SYSTEM_PROMPT

logger = logging.getLogger("research_service.graph")


def _thread_id(config: RunnableConfig) -> str:
    """Read the workflow correlation key, with a fallback for standalone calls."""
    return str(config.get("configurable", {}).get("thread_id", "-"))


def _invoke_agent(agent, prompt: str, node: str, config: RunnableConfig):
    """Log one agent attempt without content; propagate results and failures.

    retryable classifies the error; the node policy decides whether to retry.
    """
    started = perf_counter()
    thread_id = _thread_id(config)
    logger.info("Agent started thread_id=%s node=%s", thread_id, node)
    try:
        result = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
    except Exception as exc:
        retryable = isinstance(exc, (BadRequestError, APITimeoutError, APIConnectionError))
        logger.log(
            logging.WARNING if retryable else logging.ERROR,
            "Agent attempt failed thread_id=%s node=%s error_type=%s retryable=%s duration_ms=%.2f",
            thread_id, node, type(exc).__name__, retryable,
            (perf_counter() - started) * 1000,
        )
        raise
    logger.info(
        "Agent completed thread_id=%s node=%s duration_ms=%.2f",
        thread_id, node, (perf_counter() - started) * 1000,
    )
    return result


class GraphState(TypedDict):
    """Shared state: research_notes is the current draft or human replacement.

    messages merges by ID; redo_count counts human-requested research repeats.
    """

    topic: str
    messages: Annotated[list, add_messages]
    research_notes: str
    review_action: Literal["", "writer", "researcher"]
    redo_count: int


def _create_web_search_tool(tavily_client: TavilySearch) -> BaseTool:
    """Wrap Tavily with compact results and a query-only model-facing schema.

    LangChain injects RunnableConfig; the model does not supply it.
    """

    @tool("web_search")
    def web_search(query: str, config: RunnableConfig) -> str:
        """Search the web for factual, up-to-date information on a given query."""
        # The tool docstring is sent to the model; changes affect its behavior.
        started = perf_counter()
        thread_id = _thread_id(config)
        logger.debug("Web search started thread_id=%s", thread_id)
        try:
            raw = tavily_client.invoke({"query": query}, config=config)
        except Exception as exc:
            logger.error(
                "Web search failed thread_id=%s error_type=%s duration_ms=%.2f",
                thread_id, type(exc).__name__, (perf_counter() - started) * 1000,
            )
            raise
        results = raw.get("results", []) if isinstance(raw, dict) else raw
        if not isinstance(results, list):
            logger.warning(
                "Web search returned unexpected format thread_id=%s result_type=%s duration_ms=%.2f",
                thread_id, type(results).__name__, (perf_counter() - started) * 1000,
            )
            return str(raw)[:1500]

        logger.info(
            "Web search completed thread_id=%s result_count=%s duration_ms=%.2f",
            thread_id, len(results), (perf_counter() - started) * 1000,
        )
        compact = []
        for item in results:
            title = item.get("title", "")
            content = (item.get("content") or "")[:400]
            compact.append(f"- {title}: {content}")
        return "\n".join(compact) if compact else "No relevant results found."

    return web_search

def route_after_review(state: GraphState) -> Literal["writer", "researcher"]:
        """Route to writer or loop back to researcher."""
        return state["review_action"]


def build_graph(
    checkpointer: BaseCheckpointSaver,
    settings: Settings | None = None,
    llm: BaseChatModel | None = None,
    search_tool: BaseTool | None = None,
):
    """Compile the workflow with a caller-owned, initialized checkpointer.

    An injected llm serves both agents; otherwise use the configured providers.
    Invoke with configurable.thread_id and resume with Command(resume=decision).
    Direct callers must validate review payloads before resuming.
    """
    cfg = settings or get_settings()
    logger.info(
        "Building workflow researcher_model=%s writer_model=%s max_redos=%s injected_llm=%s injected_search=%s",
        cfg.researcher_model_name, cfg.writer_model_name, cfg.max_redos,
        llm is not None, search_tool is not None,
    )

    researcher_llm = llm or ChatGroq(
        model=cfg.researcher_model_name,
        temperature=cfg.llm_temperature,
        timeout=cfg.llm_timeout,
        max_retries=cfg.llm_max_retries,
        api_key=cfg.groq_api_key.get_secret_value(),
    )

    writer_llm = llm or ChatGroq(
        model=cfg.writer_model_name,
        temperature=cfg.llm_temperature,
        timeout=cfg.llm_timeout,
        max_retries=cfg.llm_max_retries,
        api_key=cfg.groq_api_key.get_secret_value(),
    )

    
    if search_tool is not None:
        tool_to_use = search_tool
    else:
        raw_tavily = TavilySearch(
            max_results=cfg.tavily_max_results,
            tavily_api_key=cfg.tavily_api_key.get_secret_value(),
        )
        tool_to_use = _create_web_search_tool(raw_tavily)

    research_agent = create_agent(
        model=researcher_llm,
        tools=[tool_to_use],
        system_prompt=RESEARCHER_SYSTEM_PROMPT,
        name="researcher_agent",
    )

    writer_agent = create_agent(
        model=writer_llm,
        tools=[],
        system_prompt=WRITER_SYSTEM_PROMPT,
        name="writer_agent",
    )

    
    def research_node(state: GraphState, config: RunnableConfig) -> dict:
        """Research or revise notes; merge only the final message into outer state."""
        logger.info(
            "Research pass started thread_id=%s redo_count=%s",
            _thread_id(config), state["redo_count"],
        )
        if state["redo_count"] == 0:
            prompt = state["topic"]
        else:
            latest_feedback = state["messages"][-1].content
            prompt = (
                f"Topic: {state['topic']}\n\n"
                f"Previous Draft:\n{state['research_notes']}\n\n"
                f"{latest_feedback}"
            )

        result = _invoke_agent(research_agent, prompt, "researcher", config)
        latest_ai_msg = result["messages"][-1]
        return {
            "messages": [latest_ai_msg],
            "research_notes": latest_ai_msg.content,
        }


    def human_review_node(state: GraphState, config: RunnableConfig) -> dict:
        """Pause for a review, then update notes and select the next node."""
        thread_id = _thread_id(config)
        # LangGraph replays this node on resume; avoid side effects before interrupt.
        # Log the actual pause at the API boundary to avoid duplicate pause events.
        logger.debug("Entering human review thread_id=%s", thread_id)
        decision = interrupt({
            "bullet_points": state["research_notes"],
            "redo_count_so_far": state["redo_count"],
        })

        decision_type = decision.get("type")

        if decision_type == "approve":
            logger.info("Research approved thread_id=%s next_node=writer", thread_id)
            return {"review_action": "writer"}

        if decision_type == "edit":
            logger.info("Research edited thread_id=%s next_node=writer", thread_id)
            return {
                "review_action": "writer",
                "research_notes": decision["edited_content"].strip(),
            }

        if decision_type == "reject":
            new_redo_count = state["redo_count"] + 1
            # Direct-call fallback; the API refuses rejection beyond this limit.
            if new_redo_count > cfg.max_redos:
                logger.warning(
                    "Research retry limit exceeded thread_id=%s redo_count=%s next_node=writer",
                    thread_id, new_redo_count,
                )
                return {
                    "review_action": "writer",
                    "redo_count": new_redo_count,
                }

            logger.info(
                "Research rejected thread_id=%s redo_count=%s next_node=researcher",
                thread_id, new_redo_count,
            )
            return {
                "review_action": "researcher",
                "redo_count": new_redo_count,
                "messages": [
                    HumanMessage(
                        content=f"Reviewer feedback; please redo the research: {decision['feedback'].strip()}"
                    )
                ],
            }

        logger.error("Unsupported review decision thread_id=%s", thread_id)
        raise ValueError(f"Unsupported review decision type: {decision_type}")
    

    def writer_node(state: GraphState, config: RunnableConfig) -> dict:
        """Write from current reviewed notes, excluding old drafts and tool history."""
        prompt = (
            f"Topic: {state['topic']}\n\n"
            f"Approved Research Bullet Points:\n{state['research_notes']}"
        )
        result = _invoke_agent(writer_agent, prompt, "writer", config)
        return {"messages": [result["messages"][-1]]}


    # Three attempts include the initial call, with SDK retries inside each.
    # These do not consume the reviewer's research redo budget.
    node_retry = RetryPolicy(
        max_attempts=3,
        initial_interval=1.0,
        backoff_factor=2.0,
        retry_on=(BadRequestError, APITimeoutError, APIConnectionError),
    )


    graph = StateGraph(GraphState)
    graph.add_node("researcher", research_node, retry_policy=node_retry)
    graph.add_node("human_review", human_review_node)
    graph.add_node("writer", writer_node, retry_policy=node_retry)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "human_review")
    graph.add_conditional_edges(
        "human_review",
        route_after_review,
        {"writer": "writer", "researcher": "researcher"},
    )
    graph.add_edge("writer", END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info("Workflow compiled node_max_attempts=%s", node_retry.max_attempts)
    return compiled
