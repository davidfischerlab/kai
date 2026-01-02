"""Reasoning critique tool and schema.

This module provides schema and tool for critiquing reasoning responses
in autonomous mode, ensuring quality and accuracy before proceeding.
"""

from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

from kai.core.prompt_manager import PromptScenario
from kai.core.tools.base import ToolResult, ToolOutputType
from kai.core.tools.prompt_base import StructuredPromptTool
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class ReasoningCritique(BaseModel):
    """Schema for reasoning critique responses."""
    model_config = ConfigDict(extra='forbid')

    approval: Literal["APPROVED", "MODIFY"] = Field(
        description="Whether the reasoning is valid or needs to be modified."
    )
    critique: str = Field(description="If applicable, any suggestions for improving the reasoning.", default="")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "approval": "APPROVED" or "MODIFY",
    "critique": "If applicable, any suggestions for improving the reasoning."
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class ReasoningCritiqueTool(StructuredPromptTool):
    """Tool for critiquing reasoning responses in autonomous mode.

    Reviews generated reasoning/explanation text for accuracy, completeness,
    and clarity. Can request regeneration with specific feedback if issues found.

    **UI Returns:**
    - Dict with "critique" field if critique text exists
    - `output_type`: TASK_LIST_DISPLAY if critique exists, NO_OUTPUT otherwise

    **Workflow Returns:**
    - `reasoning_approval`: "APPROVED" or "MODIFY"
    - `reasoning_critique`: Critique text (if any)
    - `critique_iteration`: Incremented counter for tracking critique loops
    - `reasoning_response`: Set to None if rejected (triggers regeneration)

    **Used by workflows:** Autonomous execution workflows after reasoning generation

    **Special behavior:**
    - Clears reasoning_response on rejection to trigger regeneration
    - Increments critique_iteration counter to prevent infinite loops
    - Validates that MODIFY approval includes critique text
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("reasoning_critique", PromptScenario.REASONING_CRITIQUE, llm_interface)

    def _process_structured_result(self, structured_result: ReasoningCritique, state: 'KaiState') -> ToolResult:
        approval = structured_result.approval.strip()
        critique = structured_result.critique.strip()

        # Validate logic of output:
        if approval == "MODIFY" and not critique:
            # If MODIFY but no critique, force a retry with better instruction
            raise ValueError("MODIFY approval requires a critique explaining what needs to be changed")

        # Log critique result (without verbose critique text)
        if approval == "APPROVED":
            logger.info(f"✅ Reasoning critique: APPROVED")
        else:
            logger.info(f"❌ Reasoning critique: REJECTED")

        # Send critique to VSCode for display if we have one
        vscode_response = {}
        if critique:
            vscode_response = {"critique": critique}
        self._log_task_list_updates(vscode_response, state)

        # Increment reasoning critique iteration counter for router tracking
        # Uses separate counter from autonomous_update_critique to avoid collision
        current_iteration = state.get("reasoning_critique_iteration", 0)

        # Note: We do NOT clear reasoning_response on rejection.
        # The original reasoning is needed by the regeneration prompt to show
        # what needs to be improved. The router uses reasoning_approval to decide
        # whether to regenerate (not the presence of reasoning_response).
        output_workflow = {
            "reasoning_approval": approval,
            "reasoning_critique": critique,
            "reasoning_critique_iteration": current_iteration + 1,  # Increment for next iteration
        }

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow=output_workflow,
            output_type=ToolOutputType.TASK_LIST_DISPLAY if critique else ToolOutputType.NO_OUTPUT,
        )

        return result
