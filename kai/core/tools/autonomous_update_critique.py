"""Task update critique tool and schema.

This module contains:
- AutonomousUpdateCritique: Pydantic schema for task list update critique structured output
- AutonomousUpdateCritiqueTool: LLM-based tool for critiquing task list updates in autonomous mode
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

class AutonomousUpdateCritique(BaseModel):
    """Schema for task list update critique responses."""
    model_config = ConfigDict(extra='forbid')

    approval: Literal["APPROVED", "MODIFY"] = Field(
        description="Whether the task list update is sufficient or needs to be modified."
    )
    critique: str = Field(description="If applicable, any suggestions for improving the task list update.", default="")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "approval": "APPROVED" or "MODIFY",
    "critique": "If applicable, any suggestions for improving the task list update."
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class AutonomousUpdateCritiqueTool(StructuredPromptTool):
    """Tool for critiquing task list updates in autonomous mode.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_list`: Complete structured task list with tasks array

    **Used by workflows:** Autonomous planning workflow to create initial task breakdown
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autonomous_update_critique", PromptScenario.AUTONOMOUS_UPDATE_CRITIQUE, llm_interface)

    def _process_structured_result(self, structured_result, state: 'KaiState') -> ToolResult:
        approval = structured_result.approval.strip()
        critique = structured_result.critique.strip()

        # Validate logic of output:
        if approval == "MODIFY" and not critique:
            # If MODIFY but no critique, force a retry with better instruction
            raise ValueError("MODIFY approval requires a critique explaining what needs to be changed")

        # Send critique to VSCode for display if we have one
        vscode_response = {}
        if critique:
            vscode_response = {"critique": critique}
        self._log_task_list_updates(vscode_response, state)

        # Increment autonomous update critique iteration counter for router tracking
        # Uses separate counter from reasoning_critique to avoid collision
        current_iteration = state.get("autonomous_update_critique_iteration", 0)

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow={
                "autonomous_update_approval": approval,
                "autonomous_update_critique": critique,
                "autonomous_update_critique_iteration": current_iteration + 1,  # Increment for next iteration
                "update_approved": (approval == "APPROVED"),  # Set flag if approved
            },
            output_type=ToolOutputType.TASK_LIST_DISPLAY if critique else ToolOutputType.NO_OUTPUT,
        )

        return result
