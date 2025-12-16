"""Fast, comprehensive router state transition tests.

Tests every single router decision point with minimal setup.
This catches routing bugs (#7, #8, #12) before they hit production.

Each test runs in <1ms - entire suite should complete in <100ms.
"""

import pytest
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator
from unittest.mock import MagicMock


class TestRouterStateTransitions:
    """Fast unit tests for every router transition."""

    @pytest.fixture
    def router(self):
        """Get router function without needing full orchestrator setup."""
        # Create minimal orchestrator just to get router
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        # Use smart routing for these tests (they were written for smart router)
        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm, use_deterministic_routing=False)
        return orch._route_autonomous_action

    def test_no_tasks_no_queries_routes_to_plan(self, router):
        """Empty state → plan_tasks"""
        state = {
            "task_list": {"tasks": []},
            "autonomous_mode_continue": False
        }
        assert router(state) == "plan_tasks"

    def test_no_tasks_with_queries_routes_to_plan_tasks(self, router):
        """
        No tasks but has retrieval queries → plan_tasks.

        The workflow search happens WITHIN the planning phase, controlled by the
        planning phase router. The main router just routes to plan_tasks when no tasks exist.
        """
        state = {
            "task_list": {"tasks": []},
            "retrieval_queries": ["query1"],
            "autonomous_mode_continue": False
        }
        assert router(state) == "plan_tasks"

    def test_first_iteration_all_pending_routes_to_manage_progress(self, router):
        """Bug #7: After planning, all pending → manage_progress (activate first task)"""
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "Task 1", "status": "pending"},
                    {"id": 2, "task": "Task 2", "status": "pending"}
                ]
            },
            "autonomous_mode_continue": False  # First iteration
        }
        result = router(state)
        assert result == "manage_progress", \
            "Bug #7: After planning (all pending), should activate first task, not exit"

    def test_first_iteration_has_active_routes_to_complete(self, router):
        """Bug #8: After activation, has active task → complete (show task list to user)"""
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "Task 1", "status": "active"},
                    {"id": 2, "task": "Task 2", "status": "pending"}
                ]
            },
            "autonomous_mode_continue": False  # First iteration
        }
        result = router(state)
        assert result == "complete", \
            "Bug #8: After activating first task, should exit to show task list to user"

    def test_first_iteration_in_progress_routes_to_complete(self, router):
        """User manually stopped with tasks in progress → complete"""
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "Task 1", "status": "completed"},
                    {"id": 2, "task": "Task 2", "status": "active"}
                ]
            },
            "autonomous_mode_continue": False
        }
        assert router(state) == "complete"

    def test_all_complete_routes_to_complete(self, router):
        """All tasks done → complete"""
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "Task 1", "status": "completed"},
                    {"id": 2, "task": "Task 2", "status": "completed"}
                ]
            },
            "autonomous_mode_continue": True
        }
        assert router(state) == "complete"

    def test_error_context_routes_to_handle_error(self, router):
        """Has error_context → handle_error"""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "error_context": {"error": "Something failed"},
            "last_execution_failed": True
        }
        assert router(state) == "handle_error"

    def test_should_backtrack_routes_to_backtrack(self, router):
        """Has backtrack strategy → backtrack"""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "error_context": {"recovery_strategy": "BACKTRACK"},
            "last_execution_failed": True
        }
        assert router(state) == "backtrack"

    def test_generated_code_routes_to_execute_cell(self, router):
        """Bug #12: Has generated_code → execute_cell"""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "generated_code": "import pandas as pd",
            "target_cell": 0
        }
        result = router(state)
        assert result == "execute_cell", \
            "Bug #12: When code is generated, should route to execute_cell"

    def test_no_active_task_routes_to_manage_progress(self, router):
        """No active task → manage_progress"""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "pending"}]
            },
            "autonomous_mode_continue": True
        }
        assert router(state) == "manage_progress"

    def test_reasoning_task_routes_to_reasoning(self, router):
        """Active task starts with 'reason' → respond_with_reasoning"""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Reason about the approach", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "active_task": {"id": 1, "task": "Reason about the approach", "status": "active"}
        }
        assert router(state) == "respond_with_reasoning"

    def test_normal_active_task_routes_to_generate_code(self, router):
        """Has active task, no code generated → generate_code"""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Load CSV data", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "active_task": {"id": 1, "task": "Load CSV data", "status": "active"}
        }
        assert router(state) == "generate_code"


class TestRouterPriorityOrder:
    """Test that router checks conditions in the correct order."""

    @pytest.fixture
    def router(self):
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()
        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm, use_deterministic_routing=False)
        return orch._route_autonomous_action

    def test_generated_code_takes_priority_over_active_task(self, router):
        """If code is generated, execute it before generating more."""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "generated_code": "import pandas as pd",  # Code ready to execute
            "active_task": {"id": 1, "task": "Task 1", "status": "active"}
        }
        result = router(state)
        assert result == "execute_cell", \
            "Should execute generated code before generating more"


class TestRouterEdgeCases:
    """Test edge cases and malformed states."""

    @pytest.fixture
    def router(self):
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()
        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm, use_deterministic_routing=False)
        return orch._route_autonomous_action

    def test_missing_task_list_defaults_to_empty(self, router):
        """Missing task_list → treat as empty → plan_tasks"""
        state = {
            "autonomous_mode_continue": False
        }
        assert router(state) == "plan_tasks"

    def test_task_list_none_defaults_to_empty(self, router):
        """task_list=None → treat as empty → plan_tasks"""
        state = {
            "task_list": None,
            "autonomous_mode_continue": False
        }
        assert router(state) == "plan_tasks"

    def test_empty_tasks_array_routes_to_plan(self, router):
        """task_list with empty tasks array → plan_tasks"""
        state = {
            "task_list": {"tasks": []},
            "autonomous_mode_continue": True
        }
        assert router(state) == "plan_tasks"

    def test_active_task_none_routes_to_manage_progress(self, router):
        """Bug #11: active_task=None should route to manage_progress, not crash"""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "active_task": None,  # Not set yet
            "just_executed": False
        }
        result = router(state)
        assert result == "manage_progress", \
            "Bug #11: When active_task not set, should activate it via manage_progress"

    def test_generated_code_empty_string_routes_to_generate(self, router):
        """generated_code="" (empty string) should not route to execute."""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "generated_code": "",  # Empty string (falsy)
            "active_task": {"id": 1, "task": "Task 1", "status": "active"},
            "just_executed": False
        }
        result = router(state)
        assert result == "generate_code", \
            "Empty generated_code should be treated as not generated"

    def test_generated_code_whitespace_routes_to_execute(self, router):
        """generated_code with only whitespace is truthy → execute."""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "generated_code": "   \n  ",  # Whitespace (truthy)
            "just_executed": False
        }
        result = router(state)
        # This is actually a bug - whitespace-only code shouldn't execute
        # But documenting current behavior
        assert result == "execute_cell", \
            "Current behavior: whitespace-only code still routes to execute (potential bug)"


class TestRouterStateTypes:
    """Test router handles both dict and Pydantic state."""

    @pytest.fixture
    def router(self):
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()
        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm, use_deterministic_routing=False)
        return orch._route_autonomous_action

    def test_dict_state(self, router):
        """Router works with dict state (LangGraph standard)."""
        state = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "active_task": {"id": 1, "task": "Task 1", "status": "active"},
            "just_executed": False
        }
        result = router(state)
        assert result == "generate_code"

    def test_pydantic_state(self, router):
        """Router works with Pydantic model state (if used)."""
        from kai.core.state import KaiState
        from typing import cast

        # Create a mock object that acts like Pydantic model
        class MockPydanticState:
            task_list = {
                "tasks": [{"id": 1, "task": "Task 1", "status": "active"}]
            }
            autonomous_mode_continue = True
            active_task = {"id": 1, "task": "Task 1", "status": "active"}
            just_executed = False

        state = MockPydanticState()
        result = router(state)
        assert result == "generate_code"


if __name__ == "__main__":
    # Run with: pytest test_router_state_transitions.py -v
    pytest.main([__file__, "-v"])
