"""Tests for LangGraph-based orchestrator."""

import pytest
import json
from kai.core.agent import KaiAgent


class TestLangGraphOrchestrator:
    """Test suite for LangGraph orchestrator functionality."""

    @pytest.fixture
    def agent_with_capture(self, capsys):
        """Create agent with captured output for VSCode message monitoring."""
        agent = KaiAgent(llm_provider='ollama', model="qwen3:0.6b")

        agent.session_metadata.update({
            'active': True,
            'session_id': 'test_langgraph',
            'session_timestamp': '2025-01-01_12-00-00',
            'notebook_uri': 'file:///test.ipynb',
            'iteration_counter': 1,
            'iteration_timestamp': '12-01-00'
        })

        return agent, capsys

    def parse_vscode_messages(self, capsys):
        """Parse JSON messages from captured stdout."""
        captured = capsys.readouterr()
        output = captured.out
        messages = []
        for line in output.strip().split('\n'):
            if line.strip():
                try:
                    message = json.loads(line)
                    messages.append(message)
                except json.JSONDecodeError:
                    continue
        return messages

    @pytest.mark.asyncio
    async def test_orchestrator_instantiation(self, agent_with_capture):
        """Test that LangGraph orchestrator is created correctly."""
        agent, _ = agent_with_capture

        assert agent.orchestrator is not None
        assert hasattr(agent.orchestrator, 'autonomous_graph')
        assert hasattr(agent.orchestrator, 'regular_graph')
        assert hasattr(agent.orchestrator, 'tools')

    @pytest.mark.asyncio
    async def test_tool_registry(self, agent_with_capture):
        """Test that all consolidated tools are registered."""
        agent, _ = agent_with_capture

        expected_tools = [
            "search_code_snippets",
            "search_workflows",
            "workflow_refinement",
            "generate_code",
            "update_code",
            "execute_cell",
            "restart_and_rerun",
            "plan_tasks",
            "manage_progress",
            "handle_error",
            "backtrack",
            "classify_intent",
            "answer_question",
            "review_code",
            "respond_with_reasoning",
            "notebook_operations",
        ]

        for tool_name in expected_tools:
            assert tool_name in agent.orchestrator.tools, f"Missing tool: {tool_name}"

    @pytest.mark.asyncio
    async def test_regular_mode(self, agent_with_capture):
        """Test regular (non-autonomous) mode - simple question."""
        agent, capsys = agent_with_capture

        # Minimal context for simple question
        context = {
            "session_metadata": agent.session_metadata,
            "current_cell": "import pandas",
            "current_cell_index": 0,
            "notebook_structure": {"totalCells": 1, "allCells": ["import pandas"]},
            "notebook_cells": [],
            "execution_history": [],
            "conversation_history": [],
            "last_execution_failed": False,
            "request_id": "test_regular",
            "rag_enabled": False,  # Disable RAG for simple test
            "autonomous_mode": False,
            "autonomous_mode_continue": False,
            "error_message": "",
            "task_list": {},
            "excluded_workflows": []
        }

        # Very simple query for tiny LLM
        user_query = "How do I read CSV?"

        await agent.orchestrator.process_request(user_query, context)

        messages = self.parse_vscode_messages(capsys)
        # Should produce at least one message (answer or code)
        assert len(messages) > 0

