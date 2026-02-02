"""Task update tool and schema.

This module contains:
- AutonomousTaskUpdate: Pydantic schema for autonomous task updates (no decision-making)
- AutonomousUpdateTasksTool: LLM-based tool for updating tasks in autonomous mode
"""

import json
from typing import List, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

from kai.core.tools.prompt_base import StructuredPromptTool, validate_task_list_format
from kai.core.tools.base import ToolResult, ToolOutputType
from kai.core.tools.common_schemas import TaskItem
from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState, BacktrackingContext
    from kai.core.llm_interface import LLMInterface

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class AutonomousTaskUpdate(BaseModel):
    """Schema for autonomous task updates (no decision-making)."""
    model_config = ConfigDict(extra='forbid')

    tasks: List[TaskItem] = Field(description="Updated list of non-completed tasks - this will be appended to the completed tasks.")
    retrieval_queries: List[str] = Field(description="Query to retrieve snippets of API documentation and workflow examples to guide code generation for current task.")
    update_rationale: str = Field(description="Reasoning for performing the update.")
    update_rule: Literal["KEEP", "UPDATE"] = Field(description="Whether to keep the current task list or to apply updates.")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "tasks": [
        {"id": 2, "task": "Process data", "status": "pending"},
    ],
    "retrieval_queries": ["query 1", "query 2"],
    "update_rationale": "Reasoning for performing the update.",
    "update_rule": "UPDATE",
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class AutonomousUpdateTasksTool(StructuredPromptTool):
    """Tool for updating tasks in autonomous mode (no decision-making).

    **UI Returns:**
    - `output_ui`: Dict with "text" field containing JSON updated task list for VSCode display
    - `output_type`: TASK_LIST_DISPLAY - shows formatted updated task list in chat

    **Workflow Returns:**
    - `task_list`: Complete updated task list structure

    **Used by workflows:** Feedback continuation workflow when user requests task modifications

    **Special behavior:** Updates task list based on user feedback without making autonomous decisions
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autonomous_update_tasks", PromptScenario.AUTONOMOUS_UPDATE_TASKS, llm_interface)

    def _modify_user_query(self, state: 'KaiState') -> None:
        """Create PromptContext from tool inputs, including current task list."""
        from kai.core.orchestration.state import BacktrackingContext
        # Include user query and task list
        user_query = state["user_query"]
        backtracking_context = BacktrackingContext.from_state(state)

        query_parts = []
        if backtracking_context and backtracking_context.is_active:
            query_parts.append(f"recovery_objective: {backtracking_context.recovery_objective}")

        # Only modify user query if we have backtracking context
        if query_parts:
            user_query = "\n\n".join(query_parts)
            # Update the user query in state
            state["user_query"] = user_query

    def _process_structured_result(self, structured_result, state: 'KaiState') -> ToolResult:
        """Process the structured AutonomousTaskUpdate response."""
        original_tasks = state["task_list"]["tasks"].copy()
        # Check if update was requested:
        if structured_result.update_rule == "UPDATE":
            # Get the original task list structure
            # Use the structured result directly (includes analysis_type and tasks)
            updated_tasks = [task.model_dump() for task in structured_result.tasks]

            # Build map of updated tasks by ID
            updated_task_map = {task["id"]: task for task in updated_tasks}

            # Preserve ALL original tasks, applying updates where provided
            # This ensures we don't lose pending tasks when LLM returns partial list
            # CRITICAL: Preserve "completed" status - autonomous_mark_completion sets this
            # and autonomous_update_tasks should NOT revert completed tasks to active/pending
            new_tasks = []
            for task in original_tasks:
                task_id = task["id"]
                if task_id in updated_task_map:
                    # Use updated version, but preserve completed status
                    updated_task = updated_task_map[task_id].copy()
                    if task.get("status") == "completed":
                        # Never revert a completed task - preserve the status
                        updated_task["status"] = "completed"
                    new_tasks.append(updated_task)
                else:
                    # Preserve original task (not modified by LLM)
                    new_tasks.append(task)

            # Add any NEW tasks from LLM that weren't in original list
            original_task_ids = {task["id"] for task in original_tasks}
            for task in updated_tasks:
                if task["id"] not in original_task_ids:
                    new_tasks.append(task)
        else:
            assert structured_result.update_rule == "KEEP"
            new_tasks = original_tasks
        # Create JSON-embedded text format for consistent frontend parsing
        updated_task_list = {"tasks": new_tasks}

        # Validate task list format
        validate_task_list_format(updated_task_list["tasks"], "AutonomousTaskUpdateTool")

        updated_task_json = json.dumps(updated_task_list)
        output_workflow = {
            "task_list": updated_task_list,
            "task_list_update_rule": structured_result.update_rule,
            "task_list_update_rationale": structured_result.update_rationale,
            "tasks_updated": True,  # For deterministic router phase tracking
            # CRITICAL: Clear evaluation state to prevent infinite loop when regenerating
            # after evaluation failure. Without this, the router sees old grade="REJECTED"
            # and keeps routing back to autonomous_update_tasks indefinitely.
            "task_update_grade": None,
            "task_update_feedback": None,
        }

        # Extract optional rag queries
        rag_enabled = state["rag_enabled"]
        has_error = state["last_execution_failed"]
        if (rag_enabled and not has_error) and structured_result.retrieval_queries:
            output_workflow["snippet_retrieval_query"] = structured_result.retrieval_queries
            logger.info(f"[RAG] Task list update requested {len(structured_result.retrieval_queries)} retrieval queries for next iteration")

        # Create display response with updated task list - only include fields VSCode uses
        # Use update_rationale string for display
        agent_notification = structured_result.update_rationale or ""

        vscode_response = {
            "text": updated_task_json,
            "agent_notification": agent_notification,
        }
        self._log_task_list_updates(vscode_response, state)

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=output_workflow
        )
