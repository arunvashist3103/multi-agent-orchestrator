"""Multi-Agent Research & Writing LangGraph Workflow."""

from typing import Annotated, Literal, TypedDict

from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
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


class GraphState(TypedDict):
    """Explicit state contract passed between workflow nodes."""

    topic: str
    messages: Annotated[list, add_messages]
    research_notes: str
    review_action: Literal["", "writer", "researcher"]
    redo_count: int


def _create_web_search_tool(tavily_client: TavilySearch) -> BaseTool:
    """Wrap TavilySearch in a minimal single-argument schema to prevent LLM argument hallucination."""

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web for factual, up-to-date information on a given query."""
        return tavily_client.invoke({"query": query})

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
    """Compile the research-and-writing state machine with injected dependencies."""
    cfg = settings or get_settings()

    # model = llm or ChatGroq(
    #     model=cfg.llm_model_name,
    #     temperature=cfg.llm_temperature,
    #     api_key=cfg.groq_api_key.get_secret_value(),
    # )

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

    def research_node(state: GraphState) -> dict:
        """Run web research and commit only the synthesized bullet points."""
        result = research_agent.invoke({"messages": state["messages"]})
        latest_ai_msg = result["messages"][-1]
        return {
            "messages": [latest_ai_msg],
            "research_notes": latest_ai_msg.content,
        }

    def human_review_node(state: GraphState) -> dict:
        """Suspend execution for human review and route based on decision."""
        decision = interrupt({
            "bullet_points": state["research_notes"],
            "redo_count_so_far": state["redo_count"],
        })

        decision_type = decision.get("type")

        if decision_type == "approve":
            return {"review_action": "writer"}

        if decision_type == "edit":
            return {
                "review_action": "writer",
                "research_notes": decision["edited_content"].strip(),
            }

        if decision_type == "reject":
            new_redo_count = state["redo_count"] + 1
            if new_redo_count > cfg.max_redos:
                return {
                    "review_action": "writer",
                    "redo_count": new_redo_count,
                }

            return {
                "review_action": "researcher",
                "redo_count": new_redo_count,
                "messages": [
                    HumanMessage(
                        content=f"Reviewer feedback; please redo the research: {decision['feedback'].strip()}"
                    )
                ],
            }

        raise ValueError(f"Unsupported review decision type: {decision_type}")
    

    def writer_node(state: GraphState) -> dict:
        """Synthesize the original topic and single approved research_notes into prose."""
        prompt = (
            f"Topic: {state['topic']}\n\n"
            f"Approved Research Bullet Points:\n{state['research_notes']}"
        )
        result = writer_agent.invoke({"messages": [HumanMessage(content=prompt)]})
        return {"messages": [result["messages"][-1]]}


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

    return graph.compile(checkpointer=checkpointer)
