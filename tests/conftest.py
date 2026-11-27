"""Shared pytest fixtures and deterministic test doubles."""

from typing import Any

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import BaseTool, tool
from langgraph.checkpoint.memory import MemorySaver

from config import Settings
from graph import build_graph


class ScriptedChatModel(FakeListChatModel):
    """Deterministic fake LLM supporting tool binding and call inspection."""

    recorded_prompts: list[str] = []

    def bind_tools(self, tools: Any, *, tool_choice: Any = None, **kwargs: Any):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if messages:
            self.recorded_prompts.append(str(messages[-1].content))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


@pytest.fixture
def test_settings() -> Settings:
    """Isolated Settings instance that never reads a local .env file."""
    return Settings(
        _env_file=None,
        groq_api_key="test-groq-key",
        tavily_api_key="test-tavily-key",
        database_url="postgresql://user:pass@localhost:5432/test_db",
        max_redos=2,
        log_level="DEBUG",
    )


@pytest.fixture
def fake_search_tool() -> BaseTool:
    """Deterministic web_search tool returning predictable results."""

    @tool("web_search")
    def web_search(query: str) -> str:
        """Search the web for factual, up-to-date information on a given query."""
        return f"- Result for {query}: Battery costs dropped 35% in 2025."

    return web_search


@pytest.fixture
def make_graph(test_settings: Settings, fake_search_tool: BaseTool):
    """Factory fixture to compile an in-memory LangGraph workflow with scripted LLM responses."""

    def _factory(responses: list[str]):
        fake_llm = ScriptedChatModel(responses=responses, recorded_prompts=[])
        graph = build_graph(
            checkpointer=MemorySaver(),
            settings=test_settings,
            llm=fake_llm,
            search_tool=fake_search_tool,
        )
        return graph, fake_llm

    return _factory
