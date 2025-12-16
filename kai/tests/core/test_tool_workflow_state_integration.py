"""Test that tools correctly set workflow output for state management.

These tests ensure tools return the right fields in output_workflow so the
router can make correct decisions. Catches Bugs #11, #13.

Fast tests - entire suite runs in <500ms for unit tests.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from kai.core.orchestration.base_tool import ToolResult, ToolOutputType
from kai.tests.core.test_helpers import create_full_execution_context, create_positioning_info

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


class TestTaskManagementToolsWorkflowOutput:
    """Test that task management tools set active_task in workflow output.

    Catches Bug #11.
    """

    @pytest.fixture
    def exec_context(self):
        """Create execution context for task management."""
        from kai.tests.core.test_helpers import create_task_list

        return create_full_execution_context(
            user_query="",
            task_list=create_task_list(num_tasks=2, active_index=-1)  # Both pending
        )

    @pytest.mark.asyncio
    async def test_mark_next_task_active_sets_active_task_dict(self, exec_context):
        """Bug #11: MarkNextTaskActiveTool must return active_task dict, not just string."""
        from kai.core.orchestration.deterministic_tools import MarkNextTaskActiveTool

        tool = MarkNextTaskActiveTool()
        result = await tool.execute(exec_context)

        # Must have output_workflow
        assert result.output_workflow is not None, \
            "MarkNextTaskActiveTool must return output_workflow"

        # Bug #11: Must have active_task DICT (not just active_task_objective string)
        assert "active_task" in result.output_workflow, \
            "Bug #11: output_workflow must have 'active_task' dict for router"

        # Should also have active_task_objective for legacy compatibility
        assert "active_task_objective" in result.output_workflow, \
            "Should have active_task_objective string for legacy code"

        # active_task should be a dict with proper fields
        active_task = result.output_workflow["active_task"]
        assert isinstance(active_task, dict), \
            "active_task should be a dict, not a string"
        assert "id" in active_task, "active_task dict should have 'id'"
        assert "task" in active_task, "active_task dict should have 'task'"
        assert "status" in active_task, "active_task dict should have 'status'"
        assert active_task["status"] == "active", "Task should be marked active"

        # active_task_objective should be a string
        assert isinstance(result.output_workflow["active_task_objective"], str), \
            "active_task_objective should be a string"


class TestToolWorkflowOutputTypes:
    """Test that tool output_workflow has correct types for all fields."""

    def test_generated_code_is_string(self):
        """generated_code field should always be a string."""
        result = ToolResult(
            output_workflow={"generated_code": "import pandas"},
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT
        )
        assert isinstance(result.output_workflow["generated_code"], str)

    def test_target_cell_is_int(self):
        """target_cell field should always be an int."""
        result = ToolResult(
            output_workflow={"target_cell": 5},
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT
        )
        assert isinstance(result.output_workflow["target_cell"], int)

    def test_active_task_is_dict(self):
        """active_task field should always be a dict."""
        result = ToolResult(
            output_workflow={
                "active_task": {"id": 1, "task": "Test", "status": "active"}
            },
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT
        )
        assert isinstance(result.output_workflow["active_task"], dict)

    def test_just_executed_is_bool(self):
        """just_executed field should always be a bool."""
        result = ToolResult(
            output_workflow={"just_executed": True},
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT
        )
        assert isinstance(result.output_workflow["just_executed"], bool)


class TestToolResultUIOutput:
    """Test that tools send correct UI output.

    Catches Bug #9 (tools not sending results to UI).
    """

    @pytest.mark.asyncio
    async def test_task_management_sends_task_list_display(self):
        """Task management tools should send TASK_LIST_DISPLAY to UI."""
        from kai.core.orchestration.deterministic_tools import MarkNextTaskActiveTool
        from kai.tests.core.test_helpers import create_task_list

        exec_context = create_full_execution_context(
            user_query="",
            task_list=create_task_list(num_tasks=1, active_index=-1)
        )

        tool = MarkNextTaskActiveTool()
        result = await tool.execute(exec_context)

        # Should have UI output
        assert result.output_ui is not None, \
            "Task management tool should send UI output"

        # Should be TASK_LIST_DISPLAY type
        assert result.output_type == ToolOutputType.TASK_LIST_DISPLAY, \
            "Task management should use TASK_LIST_DISPLAY output type"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
