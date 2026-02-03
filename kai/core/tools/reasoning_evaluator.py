"""Reasoning evaluator tool and schema.

This module provides schema and tool for evaluating reasoning responses
in autonomous mode, ensuring quality and accuracy before proceeding.

This follows the evaluator-optimizer pattern recommended by LangGraph.
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

class ReasoningEvaluation(BaseModel):
    """Schema for reasoning evaluation responses."""
    model_config = ConfigDict(extra='forbid')

    grade: Literal["APPROVED", "REJECTED"] = Field(
        description="Whether the reasoning meets quality standards or needs improvement."
    )
    feedback: str = Field(
        description="If rejected, specific feedback on what needs to be improved.",
        default=""
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "grade": "APPROVED" or "REJECTED",
    "feedback": "If rejected, specific feedback on what needs to be improved."
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class ReasoningEvaluatorTool(StructuredPromptTool):
    """Tool for evaluating reasoning responses in autonomous mode.

    This is the EVALUATOR in the evaluator-optimizer pattern.
    Reviews generated reasoning/explanation text for accuracy, completeness,
    and clarity. Can request regeneration with specific feedback if issues found.

    **UI Returns:**
    - Dict with "feedback" field if feedback text exists
    - `output_type`: TASK_LIST_DISPLAY if feedback exists, NO_OUTPUT otherwise

    **Workflow Returns:**
    - `reasoning_grade`: "APPROVED" or "REJECTED"
    - `reasoning_feedback`: Feedback text (if any)
    - `reasoning_evaluation_iteration`: Incremented counter

    **Used by workflows:** Execution phase evaluator-optimizer loop

    **Special behavior:**
    - Does NOT clear reasoning_response on rejection (keeps for regeneration context)
    - Increments evaluation iteration counter to prevent infinite loops
    - Validates that REJECTED grade includes feedback text
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__(
            "reasoning_evaluator",
            PromptScenario.REASONING_EVALUATION,
            llm_interface
        )

    def _process_structured_result(
        self, structured_result: ReasoningEvaluation, state: 'KaiState'
    ) -> ToolResult:
        grade = structured_result.grade.strip()
        feedback = structured_result.feedback.strip()

        # Validate logic of output:
        if grade == "REJECTED" and not feedback:
            # If REJECTED but no feedback, force a retry with better instruction
            raise ValueError("REJECTED grade requires feedback explaining what needs to be improved")

        # Log evaluation result (without verbose feedback text)
        if grade == "APPROVED":
            logger.info(f"Reasoning evaluation: APPROVED")
        else:
            logger.info(f"Reasoning evaluation: REJECTED")

        # Send feedback to VSCode for display if we have one
        vscode_response = {}
        if feedback:
            vscode_response = {"feedback": feedback}
        self._log_task_list_updates(vscode_response, state)

        # Increment reasoning evaluation iteration counter
        # Uses separate counter from task update evaluation to avoid collision
        current_iteration = state.get("reasoning_evaluation_iteration", 0)

        # Note: We do NOT clear reasoning_response on rejection.
        # The original reasoning is needed by the regeneration prompt to show
        # what needs to be improved. The router uses reasoning_grade to decide
        # whether to regenerate (not the presence of reasoning_response).
        output_workflow = {
            "reasoning_grade": grade,
            "reasoning_feedback": feedback,
            "reasoning_evaluation_iteration": current_iteration + 1,
        }

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow=output_workflow,
            output_type=ToolOutputType.TASK_LIST_DISPLAY if feedback else ToolOutputType.NO_OUTPUT,
        )

        return result


