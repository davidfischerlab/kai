"""Task list evaluator tool and schema.

This module contains:
- TaskListEvaluation: Pydantic schema for task list evaluation structured output
- TaskListEvaluatorTool: LLM-based tool for evaluating task lists in autonomous mode

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

class TaskListEvaluation(BaseModel):
    """Schema for task list evaluation responses."""
    model_config = ConfigDict(extra='forbid')

    grade: Literal["APPROVED", "REJECTED"] = Field(
        description="Whether the task list meets quality standards or needs improvement."
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

class TaskListEvaluatorTool(StructuredPromptTool):
    """Tool for evaluating task lists in autonomous mode.

    This is the EVALUATOR in the evaluator-optimizer pattern.
    It assesses task list quality and provides feedback for improvement.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_list_grade`: "APPROVED" or "REJECTED"
    - `task_list_feedback`: Feedback text for improvement
    - `task_list_evaluation_iteration`: Incremented counter

    **Used by workflows:** Planning phase evaluator-optimizer loop
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__(
            "task_list_evaluator",
            PromptScenario.TASK_LIST_EVALUATION,
            llm_interface
        )

    def _process_structured_result(self, structured_result, state: 'KaiState') -> ToolResult:
        grade = structured_result.grade.strip()
        feedback = structured_result.feedback.strip()

        # Validate logic of output:
        if grade == "REJECTED" and not feedback:
            # If REJECTED but no feedback, force a retry with better instruction
            raise ValueError("REJECTED grade requires feedback explaining what needs to be improved")

        # Log evaluation result
        if grade == "APPROVED":
            logger.info(f"Task list evaluation: APPROVED")
        else:
            # Show the feedback reason
            feedback_msg = feedback if feedback else "No specific feedback provided"
            logger.info(f"Task list evaluation: REJECTED - {feedback_msg}")

        # Send feedback to VSCode for display if we have one
        vscode_response = {}
        if feedback:
            vscode_response = {"feedback": feedback}
        self._log_task_list_updates(vscode_response, state)

        # Increment evaluation iteration counter
        current_iteration = state.get("task_list_evaluation_iteration", 0)

        # Prepare output workflow
        output_workflow = {
            "task_list_grade": grade,
            "task_list_feedback": feedback,
            "task_list_evaluation_iteration": current_iteration + 1,
            "planning_phase": "task_list_evaluation",  # Signal to router that evaluation was run
        }

        # Set task_text_old for next iteration if rejected
        # This helps guide the next generation with the previous version
        if grade != "APPROVED":
            from kai.core.utils import format_task_list
            task_list_old = format_task_list(state["task_list"])
            output_workflow["task_text_old"] = task_list_old

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow=output_workflow,
            output_type=ToolOutputType.TASK_LIST_DISPLAY if feedback else ToolOutputType.NO_OUTPUT,
        )

        return result


