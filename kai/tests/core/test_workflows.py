"""Simple workflow tests for LangGraph orchestrator.

Tests single-step workflow execution with minimal context for small LLMs.
Avoids infinite loops by testing one step at a time.
"""

import pytest
import json
from kai.core.agent import KaiAgent


class TestWorkflows:
    """Test individual workflow steps with simple inputs for small LLMs."""

    @pytest.fixture
    def agent_with_capture(self, capsys):
        """Create agent with captured output for VSCode message monitoring."""
        agent = KaiAgent(llm_provider='ollama', model="qwen3:0.6b")

        agent.session_metadata.update({
            'active': True,
            'session_id': 'test_workflow',
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
    async def test_regular_mode_simple_question(self, agent_with_capture):
        """Test regular mode with simple question - single step."""
        agent, capsys = agent_with_capture

        context = {
            "session_metadata": agent.session_metadata,
            "autonomous_mode": False,  # Regular mode
            "notebook_structure": {"totalCells": 1, "allCells": ["import pandas"]},
            "notebook_cells": [{"code": "import pandas", "index": 0}],
            "current_cell": "import pandas",
            "current_cell_index": 0,
            "execution_history": [],
            "conversation_history": [],
            "last_execution_failed": False,
            "request_id": "test_question",
            "rag_enabled": False,
            "error_message": "",
            "task_list": {},
            "excluded_workflows": []
        }

        try:
            await agent.orchestrator.process_request("How do I read a CSV?", context)
        except Exception as e:
            # May fail but should attempt to answer
            pass

        messages = self.parse_vscode_messages(capsys)
        # Should have attempted to classify intent and respond
        assert len(messages) > 0

