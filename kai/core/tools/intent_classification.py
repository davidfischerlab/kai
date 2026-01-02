"""Intent classification tool for classifying user intents.

This module contains:
- IntentClassification schema for structured output
- IntentClassificationTool for determining how to handle user requests
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

class IntentClassification(BaseModel):
    """Schema for user intent classification.

    Matches the intent classification system from prompt_tools.py:
    - question_about_code: User is asking about existing code, methods, or concepts
    - generate_code: User wants to generate new code (will create new cells)
    - generate_code_in_place: User wants to modify/fix existing code (will replace current cell)
    - remove_code: User wants to remove code
    """
    model_config = ConfigDict(extra='forbid')

    intent: Literal[
        "question_about_code",
        "generate_code",
        "generate_code_in_place",
        "remove_code"
    ] = Field(description="Classified user intent")
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for the classification choice"
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "intent": "generate_code",
    "reasoning": "Reasoning for the classification choice"
}

Valid intent values: "question_about_code", "generate_code", "generate_code_in_place", "remove_code"
Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class IntentClassificationTool(StructuredPromptTool):
    """Tool for classifying user intents using structured output.

    **UI Returns:**
    - `output_ui`: Dict with "intent" field containing classification result
    - `output_type`: RESPONSE - internal classification result

    **Workflow Returns:**
    - None - UI output used directly by workflow orchestration for routing decisions

    **Used by workflows:** Regular request workflow to determine how to handle user requests

    **Possible intents:** generate_code, question_about_code, generate_code_in_place, remove_code
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("intent_classification", PromptScenario.INTENT_CLASSIFICATION, llm_interface)

    def _process_structured_result(self, result, state: Optional['KaiState'] = None) -> ToolResult:
        """Process structured intent classification result."""
        # Return schema output directly - orchestration handles categorical classification
        return ToolResult(
            output_ui={},
            output_workflow=result.model_dump(),
            output_type=ToolOutputType.NO_OUTPUT
        )
