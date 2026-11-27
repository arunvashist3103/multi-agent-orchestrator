"""Fast unit tests for Pydantic schemas, pure routing, sanitization, and tool formatting."""

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from graph import _create_web_search_tool, route_after_review
from main import _format_response
from schemas import ResearchRequest, ReviewDecision


# --- 1. Request & Review Schema Validation Tests ---


def test_research_request_strips_whitespace_and_enforces_length():
    req = ResearchRequest(topic="   Quantum computing breakthroughs   ")
    assert req.topic == "Quantum computing breakthroughs"

    with pytest.raises(ValidationError):
        ResearchRequest(topic="ai")  # Below min_length=3


def test_review_decision_approve_allows_empty_optional_fields():
    decision = ReviewDecision(type="approve", edited_content="", feedback="   ")
    assert decision.type == "approve"
    assert decision.edited_content is None
    assert decision.feedback is None


def test_review_decision_edit_requires_valid_edited_content():
    valid = ReviewDecision(
        type="edit",
        edited_content="  - Updated bullet point exceeding twenty characters.  ",
    )
    assert valid.edited_content == "- Updated bullet point exceeding twenty characters."

    with pytest.raises(ValidationError):
        ReviewDecision(type="edit", edited_content="")

    with pytest.raises(ValidationError):
        ReviewDecision(type="edit", edited_content="Too short")


def test_review_decision_reject_requires_valid_feedback():
    valid = ReviewDecision(type="reject", feedback="  Focus more on 2025 deployment data.  ")
    assert valid.feedback == "Focus more on 2025 deployment data."

    with pytest.raises(ValidationError):
        ReviewDecision(type="reject", feedback="   ")

    with pytest.raises(ValidationError):
        ReviewDecision(type="reject", feedback="short")


# --- 2. Pure Routing & Output Sanitization Tests ---


@pytest.mark.parametrize(
    ("review_action", "expected_next"),
    [
        ("writer", "writer"),
        ("researcher", "researcher"),
    ],
)
def test_route_after_review_returns_expected_node(review_action, expected_next):
    state = {
        "topic": "Test",
        "messages": [],
        "research_notes": "- Fact",
        "review_action": review_action,
        "redo_count": 0,
    }
    assert route_after_review(state) == expected_next


def test_format_response_maps_interrupt_and_final_answer():
    from langchain_core.messages import AIMessage
    from langgraph.types import Interrupt

    interrupt_result = {
        "__interrupt__": [
            Interrupt(value={"bullet_points": "- Fact 1\n- Fact 2", "redo_count_so_far": 1})
        ]
    }
    pending = _format_response("thread-123", interrupt_result)
    assert pending.thread_id == "thread-123"
    assert pending.status == "pending_review"
    assert pending.bullet_points == "- Fact 1\n- Fact 2"
    assert pending.redo_count_so_far == 1
    assert pending.redos_remaining == 1

    complete_result = {"messages": [AIMessage(content="Final synthesized summary.")]}
    done = _format_response("thread-123", complete_result)
    assert done.thread_id == "thread-123"
    assert done.status == "complete"
    assert done.final_answer == "Final synthesized summary."


# --- 3. Compact Web Search Tool Wrapper Tests ---


def test_web_search_tool_formats_and_truncates_tavily_results():
    mock_tavily = MagicMock()
    mock_tavily.invoke.return_value = {
        "results": [
            {"title": "Source A", "content": "X" * 500},
            {"title": "Source B", "content": "Short summary"},
        ]
    }

    web_search = _create_web_search_tool(mock_tavily)
    output = web_search.invoke("grid batteries")

    lines = output.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("- Source A: ")
    # Title prefix (12 chars) + truncated content (400 chars) = 362 chars max
    assert len(lines[0]) <= 415
    assert lines[1] == "- Source B: Short summary"
