"""Test task list backup and reversion in LangGraph orchestrator."""

import pytest
from unittest.mock import MagicMock
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator


class TestTaskListReversion:
    """Test that task list reversion works when critique fails."""

    def test_router_backs_up_before_update(self):
        """Router should route to backup_task_list before autonomous_update_tasks."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)
        router = orch._route_autonomous_action

        # State: ready to update tasks, no backup yet
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "status": "completed"},
                    {"id": 2, "status": "pending"},
                ]
            },
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": False,
            "tasks_updated": False,
            "update_approved": False,
            "task_list_backup": None,  # No backup yet
        }

        result = router(state)

        # Should route to backup first
        assert result == "backup_task_list", \
            f"Expected backup_task_list, got {result}"

    def test_router_routes_to_update_after_backup(self):
        """After backup, router should route to autonomous_update_tasks."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)
        router = orch._route_autonomous_action

        # State: ready to update tasks, backup exists
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "status": "completed"},
                    {"id": 2, "status": "pending"},
                ]
            },
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": False,
            "tasks_updated": False,
            "update_approved": False,
            "task_list_backup": {"tasks": [{"id": 1, "status": "completed"}]},  # Backup exists
        }

        result = router(state)

        # Should route to update now
        assert result == "autonomous_update_tasks", \
            f"Expected autonomous_update_tasks, got {result}"

    def test_router_reverts_after_max_critique_iterations(self):
        """Router should route to revert_task_list after 3 critique iterations."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)
        router = orch._route_autonomous_action

        # State: critique failed after max iterations
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "status": "completed"},
                    {"id": 2, "status": "pending"},
                    {"id": 3, "status": "pending"},  # BAD UPDATE
                ]
            },
            "task_list_backup": {
                "tasks": [
                    {"id": 1, "status": "completed"},
                    {"id": 2, "status": "pending"},
                ]
            },
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": False,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "critique_iteration": 3,  # Max iterations reached
            "autonomous_update_approval": "REJECTED",
        }

        result = router(state)

        # Should route to revert
        assert result == "revert_task_list", \
            f"Expected revert_task_list, got {result}"

    @pytest.mark.asyncio
    async def test_backup_node_creates_deep_copy(self):
        """Backup node should create a deep copy of task list."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        original_task_list = {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "completed"},
                {"id": 2, "task": "Task 2", "status": "pending"},
            ]
        }

        state = {"task_list": original_task_list}

        result = await orch._backup_task_list_node(state)

        # Should return backup
        assert "task_list_backup" in result
        backup = result["task_list_backup"]
        assert len(backup["tasks"]) == 2

        # Verify deep copy - modifying original shouldn't affect backup
        original_task_list["tasks"][0]["status"] = "MODIFIED"
        assert backup["tasks"][0]["status"] == "completed", \
            "Backup was not a deep copy!"

    @pytest.mark.asyncio
    async def test_revert_node_restores_backup(self):
        """Revert node should restore task list from backup."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        bad_task_list = {
            "tasks": [
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "pending"},
                {"id": 3, "status": "pending"},  # BAD TASK
            ]
        }

        backup_task_list = {
            "tasks": [
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "pending"},
            ]
        }

        state = {
            "task_list": bad_task_list,
            "task_list_backup": backup_task_list
        }

        result = await orch._revert_task_list_node(state)

        # Should restore backup
        assert "task_list" in result
        assert len(result["task_list"]["tasks"]) == 2
        assert result["task_list"]["tasks"][0]["id"] == 1
        assert result["task_list"]["tasks"][1]["id"] == 2

        # Should clear backup
        assert result["task_list_backup"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
