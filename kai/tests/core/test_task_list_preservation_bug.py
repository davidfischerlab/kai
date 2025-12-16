"""Test for task list preservation bug in autonomous_update_tasks.

Bug: When LLM returns partial task list (e.g., only completed tasks),
the tool was dropping all pending tasks, causing premature termination.

Note: These tests directly test the _process_structured_result method
which handles the task list merge logic, avoiding the need for LLM mocks.
"""

import pytest
from unittest.mock import MagicMock
from kai.core.orchestration.prompt_tools import AutonomousUpdateTasksTool
from kai.tests.core.test_helpers import create_full_execution_context


class TestTaskListPreservationBug:
    """Test that autonomous_update_tasks preserves all tasks."""

    def test_process_structured_result_preserves_pending_tasks(self):
        """Bug: LLM returns only completed tasks, should preserve pending tasks.

        Scenario from breastcancer run:
        - Original: 17 tasks (4 completed, 13 pending)
        - LLM returns: 4 tasks (only the completed ones)
        - Bug behavior: Final list has 4 tasks (13 pending tasks LOST!)
        - Fixed behavior: Final list has 17 tasks (13 pending tasks preserved)

        Tests the _process_structured_result method directly.
        """
        # Create mock structured result - LLM returns only completed tasks
        mock_result = MagicMock()
        mock_result.update_rule = "UPDATE"
        mock_result.update_rationale = "No changes needed"
        mock_result.retrieval_queries = []
        mock_result.tasks = [
            MagicMock(model_dump=lambda: {"id": 1, "task": "Task 1", "status": "completed"}),
            MagicMock(model_dump=lambda: {"id": 2, "task": "Task 2", "status": "completed"}),
            MagicMock(model_dump=lambda: {"id": 3, "task": "Task 3", "status": "completed"}),
            MagicMock(model_dump=lambda: {"id": 4, "task": "Task 4", "status": "completed"}),
        ]

        # Create execution context with 17 tasks (4 completed, 13 pending)
        original_task_list = {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "completed"},
                {"id": 2, "task": "Task 2", "status": "completed"},
                {"id": 3, "task": "Task 3", "status": "completed"},
                {"id": 4, "task": "Task 4", "status": "completed"},
                {"id": 5, "task": "Task 5", "status": "pending"},
                {"id": 6, "task": "Task 6", "status": "pending"},
                {"id": 7, "task": "Task 7", "status": "pending"},
                {"id": 8, "task": "Task 8", "status": "pending"},
                {"id": 9, "task": "Task 9", "status": "pending"},
                {"id": 10, "task": "Task 10", "status": "pending"},
                {"id": 11, "task": "Task 11", "status": "pending"},
                {"id": 12, "task": "Task 12", "status": "pending"},
                {"id": 13, "task": "Task 13", "status": "pending"},
                {"id": 14, "task": "Task 14", "status": "pending"},
                {"id": 15, "task": "Task 15", "status": "pending"},
                {"id": 16, "task": "Task 16", "status": "pending"},
                {"id": 17, "task": "Task 17", "status": "pending"},
            ]
        }

        exec_context = create_full_execution_context(
            user_query="Continue analysis",
            task_list=original_task_list,
            rag_enabled=True,
            last_execution_failed=False,
            active_task_objective="Test task",
        )

        # Create tool with mock LLM (won't be called)
        mock_llm = MagicMock()
        tool = AutonomousUpdateTasksTool(mock_llm)

        # Directly test the merge logic
        result = tool._process_structured_result(mock_result, exec_context)

        # Verify result
        assert "task_list" in result.output_workflow
        updated_task_list = result.output_workflow["task_list"]

        # CRITICAL: All 17 tasks must be preserved!
        assert len(updated_task_list["tasks"]) == 17, \
            f"BUG: Only {len(updated_task_list['tasks'])}/17 tasks preserved!"

        # Verify completed tasks are still completed
        for i in range(4):
            assert updated_task_list["tasks"][i]["status"] == "completed"

        # Verify pending tasks are preserved
        for i in range(4, 17):
            assert updated_task_list["tasks"][i]["status"] == "pending", \
                f"Task {i+1} should be pending"

    def test_process_structured_result_applies_llm_updates(self):
        """LLM can update existing tasks while preserving others."""
        # Mock result - LLM updates task 2 and 3
        mock_result = MagicMock()
        mock_result.update_rule = "UPDATE"
        mock_result.update_rationale = "Updated tasks 2 and 3"
        mock_result.retrieval_queries = []
        mock_result.tasks = [
            MagicMock(model_dump=lambda: {"id": 2, "task": "Task 2 UPDATED", "status": "pending"}),
            MagicMock(model_dump=lambda: {"id": 3, "task": "Task 3 UPDATED", "status": "active"}),
        ]

        original_task_list = {
            "tasks": [
                {"id": 1, "task": "Task 1", "status": "completed"},
                {"id": 2, "task": "Task 2", "status": "pending"},
                {"id": 3, "task": "Task 3", "status": "pending"},
                {"id": 4, "task": "Task 4", "status": "pending"},
                {"id": 5, "task": "Task 5", "status": "pending"},
            ]
        }

        exec_context = create_full_execution_context(
            user_query="Update tasks",
            task_list=original_task_list,
            rag_enabled=False,
            last_execution_failed=False,
            active_task_objective="Test task",
        )

        mock_llm = MagicMock()
        tool = AutonomousUpdateTasksTool(mock_llm)
        result = tool._process_structured_result(mock_result, exec_context)

        updated_tasks = result.output_workflow["task_list"]["tasks"]

        # All 5 tasks preserved
        assert len(updated_tasks) == 5

        # Task 1 unchanged
        assert updated_tasks[0]["task"] == "Task 1"
        assert updated_tasks[0]["status"] == "completed"

        # Task 2 updated
        assert updated_tasks[1]["task"] == "Task 2 UPDATED"
        assert updated_tasks[1]["status"] == "pending"

        # Task 3 updated
        assert updated_tasks[2]["task"] == "Task 3 UPDATED"
        assert updated_tasks[2]["status"] == "active"

        # Tasks 4-5 preserved
        assert updated_tasks[3]["task"] == "Task 4"
        assert updated_tasks[4]["task"] == "Task 5"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
