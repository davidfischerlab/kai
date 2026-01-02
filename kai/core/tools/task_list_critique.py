"""Task list critique tool and schema.

This module contains:
- TaskListCritique: Pydantic schema for task list critique structured output
- TaskListCritiqueTool: LLM-based tool for critiquing task lists in autonomous mode
"""

from typing import Literal, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict

from kai.core.tools.prompt_base import StructuredPromptTool
from kai.core.tools.base import ToolResult, ToolOutputType
from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class TaskListCritique(BaseModel):
    """Schema for task list critique responses."""
    model_config = ConfigDict(extra='forbid')

    approval: Literal["APPROVED", "MODIFY"] = Field(
        description="Whether the task list is sufficient or needs to be modified."
    )
    critique: str = Field(description="If applicable, any suggestions for improving the task list.", default="")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "approval": "APPROVED" or "MODIFY",
    "critique": "If applicable, any suggestions for improving the task list."
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class TaskListCritiqueTool(StructuredPromptTool):
    """Tool for critiquing task lists in autonomous mode.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_list`: Complete structured task list with tasks array

    **Used by workflows:** Autonomous initiation workflow to create initial task breakdown
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("task_list_critique", PromptScenario.TASK_LIST_CRITIQUE, llm_interface)

    def _process_structured_result(self, structured_result, state: 'KaiState') -> ToolResult:
        approval = structured_result.approval.strip()
        critique = structured_result.critique.strip()

        # Validate logic of output:
        if approval == "MODIFY" and not critique:
            # If MODIFY but no critique, force a retry with better instruction
            raise ValueError("MODIFY approval requires a critique explaining what needs to be changed")

        # Log critique result
        if approval == "APPROVED":
            logger.info(f"Task list critique: APPROVED")
        else:
            # Show the critique reason
            critique_msg = critique if critique else "No specific feedback provided"
            logger.info(f"Task list critique: REJECTED - {critique_msg}")

        # Send critique to VSCode for display if we have one
        vscode_response = {}
        if critique:
            vscode_response = {"critique": critique}
        self._log_task_list_updates(vscode_response, state)

        # Prepare output workflow
        output_workflow = {
            "task_list_approval": approval,
            "task_list_critique": critique,
            "planning_phase": "task_list_critique",  # Signal to router that critique was run
        }

        # Set task_text_old for next iteration if critique rejected
        # This helps guide the next generation with the previous version
        if approval != "APPROVED":
            from kai.core.utils import format_task_list
            task_list_old = format_task_list(state["task_list"])
            output_workflow["task_text_old"] = task_list_old

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow=output_workflow,
            output_type=ToolOutputType.TASK_LIST_DISPLAY if critique else ToolOutputType.NO_OUTPUT,
        )

        return result
