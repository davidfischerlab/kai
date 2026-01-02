"""Autoloop intent classification tool for classifying user feedback during autonomous mode.

This module contains:
- AutoLoopIntentClassification schema for structured output
- AutoLoopIntentClassificationTool for classifying feedback type
"""

from typing import TYPE_CHECKING, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict

from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger
from .base import ToolResult, ToolOutputType
from .prompt_base import StructuredPromptTool

if TYPE_CHECKING:
    from kai.core.llm_interface import LLMInterface
    from kai.core.orchestration.state import KaiState

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class AutoLoopIntentClassification(BaseModel):
    """Schema for classifying user feedback during autonomous mode.

    Determines whether the user wants to modify the task list or change code implementation.
    """
    model_config = ConfigDict(extra='forbid')

    intent: Literal["TASK_LIST_MODIFICATION", "CODE_IMPLEMENTATION_FEEDBACK", "APPROVAL"] = Field(
        description="Type of feedback - task list changes or implementation changes"
    )
    modification_description: str = Field(
        description="What the user wants to change"
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "intent": "TASK_LIST_MODIFICATION", "CODE_IMPLEMENTATION_FEEDBACK", or "APPROVAL"
    "modification_description": "User wants to change the differential expression method from t-test to wilcoxon"
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class AutoLoopIntentClassificationTool(StructuredPromptTool):
    """Tool for classifying user input during autonomous mode.

    **UI Returns:**
    - `output_ui`: Dict with "intent", "target_tasks", "modification_description" fields
    - `output_type`: RESPONSE - internal classification result

    **Workflow Returns:**
    - None - UI output used directly by workflow orchestration for routing decisions

    **Used by workflows:** Feedback continuation workflow to classify feedback type and route appropriately

    **Possible intents:** TASK_LIST_MODIFICATION, CODE_IMPLEMENTATION_FEEDBACK, CONTINUE_WITH_FEEDBACK
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autoloop_intent_classification", PromptScenario.AUTOLOOP_INTENT_CLASSIFICATION, llm_interface)

    def _process_structured_result(self, result, state: Optional['KaiState'] = None) -> ToolResult:
        """Process feedback intent classification result."""
        return ToolResult(
            output_ui={},
            output_workflow=result.model_dump(),
            output_type=ToolOutputType.NO_OUTPUT
        )
