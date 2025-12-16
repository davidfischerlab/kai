"""Tests for LangGraph orchestrator routing logic.

These tests verify that the orchestrator routes to the correct tools
based on state conditions. This catches production bugs where the orchestrator
does nothing or gets stuck in loops.
"""

import pytest
from kai.core.agent import KaiAgent


class TestOrchestratorRouting:
    """Test orchestrator routing decisions."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        return KaiAgent(llm_provider='ollama', model="qwen3:0.6b")

    @pytest.mark.asyncio
    async def test_autonomous_mode_continue_false_allows_planning(self, agent, capsys):
        """Test that autonomous_mode_continue=False still allows initial planning.

        Production bug: Router might check autonomous_mode_continue BEFORE
        checking if tasks exist, preventing initial planning.

        Expected: autonomous_mode_continue=False should stop AFTER planning,
        not BEFORE.
        """
        context = {
            "session_metadata": {
                "session_id": "test_continue_false",
                "session_timestamp": "2025-01-01_12-00-00",
                "iteration_timestamp": "12-01-00",
                "iteration_counter": 1,
                "notebook_uri": "file:///test.ipynb",
                "active": True,
                "request_id": "test_continue"
            },
            "autonomous_mode": True,
            "autonomous_mode_continue": False,  # Should still plan!
            "notebook_structure": {"totalCells": 0, "allCells": []},
            "notebook_cells": [],
            "current_cell": "",
            "current_cell_index": 0,
            "execution_history": [],
            "conversation_history": [],
            "last_execution_failed": False,
            "rag_enabled": False,
            "error_message": "",
            "task_list": {},  # No tasks yet
            "excluded_workflows": []
        }

        try:
            await agent.orchestrator.process_request("Simple task", context)
        except Exception:
            pass

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should plan despite autonomous_mode_continue=False
        assert "task_list_display" in output, \
            "Should plan even when autonomous_mode_continue=False (first iteration)"
        assert ("tasks" in output or '\\"tasks\\"' in output), \
            "Task list should contain tasks after planning"

        print("✅ autonomous_mode_continue=False allows initial planning")

    @pytest.mark.asyncio
    async def test_routing_with_existing_tasks(self, agent, capsys):
        """Test that routing stops when autonomous_mode_continue=False AND tasks exist."""
        context = {
            "session_metadata": {
                "session_id": "test_with_tasks",
                "session_timestamp": "2025-01-01_12-00-00",
                "iteration_timestamp": "12-01-00",
                "iteration_counter": 1,
                "notebook_uri": "file:///test.ipynb",
                "active": True,
                "request_id": "test_tasks"
            },
            "autonomous_mode": True,
            "autonomous_mode_continue": False,
            "notebook_structure": {"totalCells": 0, "allCells": []},
            "notebook_cells": [],
            "current_cell": "",
            "current_cell_index": 0,
            "execution_history": [],
            "conversation_history": [],
            "last_execution_failed": False,
            "rag_enabled": False,
            "error_message": "",
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "Load data", "status": "pending"}
                ]
            },
            "excluded_workflows": []
        }

        try:
            await agent.orchestrator.process_request("Continue", context)
        except Exception:
            pass

        captured = capsys.readouterr()
        output = captured.out + captured.err

        # Should complete because tasks exist and autonomous_mode_continue=False (first iteration)
        # Verify it activated the first task and exited
        assert "task_list_display" in output, \
            "Should display task list"
        assert "LOOP_INCOMPLETE" in output, \
            "Should exit with LOOP_INCOMPLETE when first iteration complete"

        print("✅ Routing correctly stops when tasks exist and autonomous_mode_continue=False")
