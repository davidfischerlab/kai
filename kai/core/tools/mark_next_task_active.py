"""Task activation tool (deterministic).

This module contains:
- MarkNextTaskActiveTool: Deterministic tool for marking the next pending task as active
"""

import json
from typing import TYPE_CHECKING

from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState

logger = setup_logger(__name__)


# =============================================================================
# Tool
# =============================================================================

class MarkNextTaskActiveTool(BaseTool):
    """Deterministic tool for marking the next pending task as active and sending UI update.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows updated task list with new active task

    **Workflow Returns:**
    - `task_list`: Updated task list with next task marked as active
    - `active_task_objective`: Description of the newly active task

    **Used by workflows:** Autonomous execution workflows to advance to the next task
    """

    def __init__(self):
        super().__init__("mark_next_task_active")

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Mark the next pending task as active and send updated task list to UI."""
        if not state["task_list"]:
            return ToolResult(
                output_ui="No task list available for task marking",
                output_type=ToolOutputType.RESPONSE,
                output_workflow={}
            )

        # Find the first non-completed task and mark it as active
        # Ignore cases in which all are completed / active
        active_task_description = ""
        active_task_dict = None

        i = 0
        for i, task in enumerate(state["task_list"]['tasks']):
            if task.get('status') == 'active':
                # There is already an active task
                active_task_description = task.get('task', '')
                active_task_dict = task
                break
            if task.get('status') == 'pending':
                # Update the task status
                task['status'] = 'active'
                active_task_description = task.get('task', '')
                active_task_dict = task
                logger.info(f"Marking task {task.get('id')} as active: {active_task_description[:75]}")
                break
        if i < len(state["task_list"]['tasks']) - 1:
            next_pending_task_objective = state["task_list"]['tasks'][i + 1]["task"]
        else:
            next_pending_task_objective = ""
        is_reasoning_task = "[reasoning]" in active_task_description

        # Create task list display
        import json
        task_list_json = json.dumps(state["task_list"])
        vscode_response = {
            "text": task_list_json,
        }

        # Prepare workflow output
        workflow_output = {
            "task_list": state["task_list"],
            "active_task": active_task_dict,  # Full task dict (for router)
            "active_task_objective": active_task_description,  # String description for routing
            "is_reasoning_task": is_reasoning_task,
            "next_pending_task_objective": next_pending_task_objective,
            "next_task_activated": True  # For deterministic router phase tracking
        }

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=workflow_output
        )
