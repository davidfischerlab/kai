"""Task completion marking tool and schema.

This module contains:
- AutonomousMarkCompletion: Pydantic schema for marking task completion with backtracking support
- AutonomousMarkCompletionTool: LLM-based tool for analyzing task completion status
"""

import json
from typing import List, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict, model_validator

from kai.core.tools.prompt_base import StructuredPromptTool, validate_task_list_format
from kai.core.tools.base import ToolResult, ToolOutputType
from kai.core.tools.common_schemas import TaskStatusUpdate
from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class AutonomousMarkCompletion(BaseModel):
    """Schema for marking task completion status with backtracking support.

    Updates status of ALL tasks. Can backtrace by setting earlier tasks back to pending.
    Must maintain logical order: no completed tasks after pending tasks.
    Backtracking is detected when recovery_objective is provided.
    """
    model_config = ConfigDict(extra='forbid')

    status_updates: List[TaskStatusUpdate] = Field(
        description="Status updates for ALL tasks by ID. Must maintain logical order - no completed after pending."
    )
    retry_objective: Optional[str] = Field(
        default=None,
        description="Optional: if task was not addressed sufficiently, explanation of what needs to change in the next attempt."
    )
    recovery_objective: Optional[str] = Field(
        default=None,
        description="Optional: if backtracking, explanation of what needs to change in earlier tasks based on later analysis."
    )

    @model_validator(mode='after')
    def validate_logical_order(self):
        """Ensure logical task ordering: completed -> active -> pending."""
        if not self.status_updates:
            return self

        # Sort by ID to check order
        sorted_updates = sorted(self.status_updates, key=lambda x: x.id)

        # Define status priority for ordering validation (0 = earliest valid, 2 = latest valid)
        status_priority = {"completed": 0, "active": 1, "pending": 2}

        # Check that status priorities are non-decreasing (completed -> active -> pending)
        for i in range(len(sorted_updates) - 1):
            current_priority = status_priority[sorted_updates[i].status]
            next_priority = status_priority[sorted_updates[i + 1].status]

            if current_priority > next_priority:
                raise ValueError(
                    f"Invalid task order: Task {sorted_updates[i+1].id} is {sorted_updates[i+1].status} "
                    f"but comes after {sorted_updates[i].status} task {sorted_updates[i].id}. "
                    f"Tasks must be ordered: completed -> active -> pending."
                )

        # Ensure only one task is active at a time
        active_count = sum(1 for update in sorted_updates if update.status == "active")
        if active_count > 1:
            active_tasks = [str(update.id) for update in sorted_updates if update.status == "active"]
            raise ValueError(
                f"Only one task can be active at a time. Found active tasks: {', '.join(active_tasks)}"
            )

        return self

    @property
    def backtrack_detected(self) -> bool:
        """Detect if backtrack was initiated - derived from recovery_objective being provided."""
        return self.recovery_objective is not None

    @property
    def retry_detected(self) -> bool:
        """Detect if retry was initiated - derived from recovery_objective being provided."""
        return self.retry_objective is not None

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "status_updates": [
        {"id": 1, "status": "completed"},
        {"id": 2, "status": "completed"},
        {"id": 3, "status": "pending"},
        {"id": 4, "status": "pending"}
    ],
    "retry_objective": "Optional: if task was not addressed sufficiently, explanation of what needs to change in the next attempt.",
    "recovery_objective": "Optional: if backtracking, explanation of what needs to change in earlier tasks based on later analysis."
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class AutonomousMarkCompletionTool(StructuredPromptTool):
    """Tool for analyzing task completion status in autonomous mode - handles success, error, and backtracking cases.

    **UI Returns:**
    - `output_ui`: Dict with "text" field containing JSON task list for VSCode display
    - `output_type`: TASK_LIST_DISPLAY - always shows updated task list in chat

    **Workflow Returns:**
    - `recovery_objective`: Description of recovery needed (if backtracking detected)
    - `backtrack_to_task`: Task object to backtrack to (if backtracking detected)

    **Used by workflows:** Autonomous continuation workflow to update task statuses and detect backtracking

    **Special behavior:** Detects when backtracking is needed and provides recovery context.
    Always sends task list updates to chat to keep UI synchronized.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autonomous_mark_completion", PromptScenario.AUTONOMOUS_MARK_COMPLETION, llm_interface)

    def _process_structured_result(self, structured_result, state: Union[None, 'KaiState'] = None) -> ToolResult:
        """Process the structured TaskCompletionUpdate response with status updates and backtracking support."""
        # Get the original task list structure
        original_task_list = state["task_list"]

        # Extract tasks from original structure
        original_tasks = original_task_list['tasks'].copy()

        # Apply status updates to existing tasks
        status_updates = {update.id: update.status for update in structured_result.status_updates}

        updated_tasks = []
        for task in original_tasks:
            assert isinstance(task, dict), task
            # Check that updates do not affect pending tasks:
            if task['status'] == "pending" and task['id'] in status_updates.keys() and status_updates[task['id']] != "pending":
                raise ValueError(f"Tried to set task {task['id']} from pending to {status_updates[task['id']]}.")
            task_copy = task.copy()
            if task.get('id') in status_updates:
                task_copy['status'] = status_updates[task['id']]
            updated_tasks.append(task_copy)
        updated_task_list = {'tasks': updated_tasks}

        # Validate task list format
        validate_task_list_format(updated_task_list["tasks"], "AutonomousMarkCompletionTool")

        # Always provide task_list in output_workflow for state propagation
        output_workflow = {
            "task_list": updated_task_list,
            "task_completion_analyzed": True,  # For deterministic router phase tracking
            "generated_code": None,  # Clear for next iteration
            "reasoning_response": None,  # Clear for next iteration
            "reasoning_grade": None,  # Clear reasoning state for next task
            "reasoning_evaluation_iteration": 0,  # Reset reasoning evaluation counter
            "task_update_evaluation_iteration": 0,  # Reset task update evaluation counter
        }

        # Handle backtracking if detected - add backtracking context
        if structured_result.backtrack_detected:
            first_pending_task = None
            reset_tasks = []

            # Find tasks that were reset from completed/active to pending
            for task, orig_task in zip(updated_tasks, original_tasks):
                if task['status'] == 'pending':
                    if first_pending_task is None:
                        first_pending_task = task
                    # Track tasks that were completed/active but are now pending (reset)
                    if orig_task['status'] in ('completed', 'active'):
                        reset_tasks.append(task)

            # Add backtracking info for workflow orchestration
            output_workflow["recovery_objective"] = structured_result.recovery_objective
            output_workflow["backtrack_to_task"] = first_pending_task or {}
            output_workflow["reset_tasks"] = reset_tasks  # Tasks being reset for backtrack_recovery

        # Handle retry if detected - pass through to orchestrator
        # Note: No retry limit - LLM decides when to backtrack vs retry
        if structured_result.retry_objective:
            output_workflow["retry_objective"] = structured_result.retry_objective

        # Check if ALL tasks are now complete - signal LOOP_COMPLETE to stop the UI loop
        all_complete = all(
            t.get("status") == "completed" for t in updated_task_list.get("tasks", [])
        )
        if all_complete:
            output_workflow["auto_loop_update"] = "LOOP_COMPLETE"
            logger.info(
                "[AUTONOMOUS_MARK_COMPLETION] All tasks complete - signaling LOOP_COMPLETE"
            )

        # Always send task list update to UI
        import json
        updated_task_json = json.dumps(updated_task_list)
        vscode_response = {"text": updated_task_json}
        if structured_result.retry_objective:
            vscode_response["agent_notification"] = structured_result.retry_objective
        if all_complete:
            vscode_response["agent_notification"] = "All tasks completed! 🎉"
        self._log_task_list_updates(vscode_response, state)

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=output_workflow
        )
