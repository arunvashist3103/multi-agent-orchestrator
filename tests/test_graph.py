"""Workflow-level tests for the LangGraph state machine using MemorySaver and scripted LLMs."""

import pytest
from groq import APITimeoutError
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from config import Settings
from graph import build_graph
from tests.conftest import ScriptedChatModel


def _initial_input(topic: str) -> dict:
    return {
        "topic": topic,
        "messages": [HumanMessage(content=topic)],
        "research_notes": "",
        "review_action": "",
        "redo_count": 0,
    }


def test_approve_path_pauses_then_completes(make_graph):
    graph, fake_llm = make_graph(
        responses=[
            "- Initial research bullet 1\n- Initial research bullet 2",
            "Final synthesized paragraph based on approved research.",
        ]
    )
    config = {"configurable": {"thread_id": "graph-approve-1"}}

    paused = graph.invoke(_initial_input("Grid battery storage"), config=config)

    assert "__interrupt__" in paused
    assert paused["__interrupt__"][0].value == {
        "bullet_points": "- Initial research bullet 1\n- Initial research bullet 2",
        "redo_count_so_far": 0,
    }

    completed = graph.invoke(Command(resume={"type": "approve"}), config=config)

    assert "__interrupt__" not in completed
    assert completed["messages"][-1].content == (
        "Final synthesized paragraph based on approved research."
    )
    assert len(fake_llm.recorded_prompts) == 2


def test_edit_path_replaces_research_notes_without_draft_leakage(make_graph):
    graph, fake_llm = make_graph(
        responses=[
            "- Rejected initial draft bullet that should never reach writer",
            "Final paragraph written strictly from human-edited notes.",
        ]
    )
    config = {"configurable": {"thread_id": "graph-edit-1"}}

    graph.invoke(_initial_input("Sodium-ion batteries"), config=config)

    edited_notes = "- Human-edited bullet 1\n- Human-edited bullet 2"
    completed = graph.invoke(
        Command(resume={"type": "edit", "edited_content": edited_notes}),
        config=config,
    )

    assert completed["research_notes"] == edited_notes
    writer_prompt = fake_llm.recorded_prompts[-1]
    assert edited_notes in writer_prompt
    assert "Rejected initial draft bullet" not in writer_prompt


def test_reject_loop_passes_compact_feedback_and_caps_at_max_redos(make_graph):
    graph, fake_llm = make_graph(
        responses=[
            "- Draft 0 bullets",
            "- Draft 1 bullets with 2025 figures",
            "- Draft 2 bullets with grid bottlenecks",
            "Final synthesis using Draft 2 after max_redos reached.",
        ]
    )
    config = {"configurable": {"thread_id": "graph-reject-1"}}

    # Initial run -> redo_count = 0
    step0 = graph.invoke(_initial_input("Utility-scale storage"), config=config)
    assert step0["__interrupt__"][0].value["redo_count_so_far"] == 0

    # Reject #1 -> redo_count = 1
    step1 = graph.invoke(
        Command(resume={"type": "reject", "feedback": "Add 2025 deployment figures."}),
        config=config,
    )
    assert step1["__interrupt__"][0].value == {
        "bullet_points": "- Draft 1 bullets with 2025 figures",
        "redo_count_so_far": 1,
    }
    assert "Previous Draft:\n- Draft 0 bullets" in fake_llm.recorded_prompts[1]
    assert "Add 2025 deployment figures." in fake_llm.recorded_prompts[1]

    # Reject #2 -> redo_count = 2 (max_redos = 2)
    step2 = graph.invoke(
        Command(resume={"type": "reject", "feedback": "Include grid bottlenecks."}),
        config=config,
    )
    assert step2["__interrupt__"][0].value == {
        "bullet_points": "- Draft 2 bullets with grid bottlenecks",
        "redo_count_so_far": 2,
    }

    # Reject #3 (beyond max_redos) -> routes straight to writer without a 4th research pass
    step3 = graph.invoke(
        Command(resume={"type": "reject", "feedback": "Extra feedback past limit."}),
        config=config,
    )
    assert "__interrupt__" not in step3
    assert step3["messages"][-1].content == (
        "Final synthesis using Draft 2 after max_redos reached."
    )
    # 3 researcher calls + 1 writer call = 4 total LLM invocations
    assert len(fake_llm.recorded_prompts) == 4


def test_researcher_node_retries_transient_llm_timeout(test_settings: Settings, fake_search_tool):
    class FlakyChatModel(ScriptedChatModel):
        attempts: int = 0

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise APITimeoutError(request=None)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    flaky_llm = FlakyChatModel(
        responses=["- Recovered bullet points on attempt 3"],
        recorded_prompts=[],
    )
    graph = build_graph(
        checkpointer=MemorySaver(),
        settings=test_settings,
        llm=flaky_llm,
        search_tool=fake_search_tool,
    )

    result = graph.invoke(
        _initial_input("Test transient timeout recovery"),
        config={"configurable": {"thread_id": "graph-retry-1"}},
    )

    assert flaky_llm.attempts == 3
    assert result["__interrupt__"][0].value["bullet_points"] == (
        "- Recovered bullet points on attempt 3"
    )
