"""Task update evaluator tool and schema.

This module contains:
- TaskUpdateEvaluation: Pydantic schema for task update evaluation structured output
- TaskUpdateEvaluatorTool: LLM-based tool for evaluating task list updates in autonomous mode

This follows the evaluator-optimizer pattern recommended by LangGraph.
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

class TaskUpdateEvaluation(BaseModel):
    """Schema for task update evaluation responses."""
    model_config = ConfigDict(extra='forbid')

    grade: Literal["APPROVED", "REJECTED"] = Field(
        description="Whether the task list update meets quality standards or needs improvement."
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

class TaskUpdateEvaluatorTool(StructuredPromptTool):
    """Tool for evaluating task list updates in autonomous mode.

    This is the EVALUATOR in the evaluator-optimizer pattern.
    It assesses task update quality and provides feedback for improvement.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_update_grade`: "APPROVED" or "REJECTED"
    - `task_update_feedback`: Feedback text for improvement
    - `task_update_evaluation_iteration`: Incremented counter
    - `update_approved`: Boolean flag (True if approved)

    **Used by workflows:** Execution phase evaluator-optimizer loop
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__(
            "task_update_evaluator",
            PromptScenario.TASK_UPDATE_EVALUATION,
            llm_interface
        )

    def _process_structured_result(self, structured_result, state: 'KaiState') -> ToolResult:
        grade = structured_result.grade.strip()
        feedback = structured_result.feedback.strip()

        # Validate logic of output:
        if grade == "REJECTED" and not feedback:
            # If REJECTED but no feedback, force a retry with better instruction
            raise ValueError("REJECTED grade requires feedback explaining what needs to be improved")

        # Send feedback to VSCode for display if we have one
        vscode_response = {}
        if feedback:
            vscode_response = {"feedback": feedback}
        self._log_task_list_updates(vscode_response, state)

        # Increment evaluation iteration counter
        # Uses separate counter from reasoning evaluation to avoid collision
        current_iteration = state.get("task_update_evaluation_iteration", 0)

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow={
                "task_update_grade": grade,
                "task_update_feedback": feedback,
                "task_update_evaluation_iteration": current_iteration + 1,
                "update_approved": (grade == "APPROVED"),
            },
            output_type=ToolOutputType.TASK_LIST_DISPLAY if feedback else ToolOutputType.NO_OUTPUT,
        )

        return result


