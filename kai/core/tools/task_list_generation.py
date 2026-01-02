"""Task list generation tool and schema.

This module contains:
- TaskListGeneration: Pydantic schema for task list generation structured output
- TaskListGenerationTool: LLM-based tool for generating task lists in autonomous mode
"""

import json
from typing import List, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

from kai.core.tools.prompt_base import StructuredPromptTool, validate_task_list_format
from kai.core.tools.base import ToolResult, ToolOutputType
from kai.core.tools.common_schemas import TaskItem
from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class TaskListGeneration(BaseModel):
    """Schema for task list generation responses."""
    model_config = ConfigDict(extra='forbid')

    tasks: List[TaskItem] = Field(description="List of analysis steps as tasks")
    retrieval_queries: List[str] = Field(description="List of queries for further retrieval  of reference workflows.", default_factory=list)

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "tasks": [
        {"id": 1, "task": "Description of analysis step", "status": "pending"},
        {"id": 2, "task": "Another analysis step", "status": "pending"}
    ],
    "retrieval_queries": ["List of string queries for further retrieval of reference workflows."]
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class TaskListGenerationTool(StructuredPromptTool):
    """Tool for generating task lists in autonomous mode.

    **UI Returns:**
    - `output_ui`: Dict with "text" field containing JSON task list for VSCode display
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_list`: Complete structured task list with tasks array

    **Used by workflows:** Autonomous initiation workflow to create initial task breakdown
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("task_list_generation", PromptScenario.TASK_LIST_GENERATION, llm_interface)

    def _process_structured_result(self, structured_result, state: 'KaiState') -> ToolResult:
        """Process structured task list result for VSCode."""
        # Create JSON-embedded text format for consistent frontend parsing
        task_list = {"tasks": [task.model_dump() for task in structured_result.tasks]}
        if len(task_list["tasks"]) == 0:
            raise Exception("Generated task list did not include any tasks.")

        # Log task generation summary with iteration counter
        current_iteration = state.get("task_planning_iteration", 0)
        logger.info(f"Generated task list with {len(task_list['tasks'])} tasks")

        # Validate task list format
        validate_task_list_format(task_list["tasks"], "TaskListGenerationTool")

        # Include formatted reference workflows in JSON for VSCode display
        # Use reference_workflow_percentages to format with 📚 and percentages
        # (same formatting as ReferenceWorkflowCellSelectionTool)
        reference_workflow_percentages = state.get("reference_workflow_percentages", {})
        if reference_workflow_percentages:
            # Format each workflow with 📚 emoji and percentage, sorted by ID
            formatted_lines = [
                f"📚 {full_id} (considering {percentage:.0f}% of file)"
                for full_id, percentage in sorted(reference_workflow_percentages.items())
            ]
            task_list["reference_workflow_ids"] = "\n".join(formatted_lines)

        json_text = json.dumps(task_list)

        # Get current iteration for logging
        current_iteration = state.get("task_planning_iteration", 0)

        output_workflow = {
            "task_list": task_list,
            "retrieval_queries": structured_result.retrieval_queries,
            "planning_phase": "task_planning",  # Set phase for router
            # NOTE: Router will increment task_planning_iteration when routing back for next iteration
            "task_list_approval": None,  # Clear previous approval status for next iteration
        }

        # Create VSCode-ready response for task list display - only include fields VSCode uses
        vscode_response = {
            "text": json_text,
        }
        if structured_result.retrieval_queries and len(structured_result.retrieval_queries) > 0:
            vscode_response["agent_notification"] = "\n".join(["Reading up on:"] + structured_result.retrieval_queries)
        self._log_task_list_updates(vscode_response, state)

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=output_workflow
        )
