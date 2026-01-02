"""Error recovery tool for analyzing errors and determining recovery strategy.

This module contains:
- ErrorRecoveryStrategy schema for structured output
- ErrorRecoveryTool for analyzing errors and determining fixing approach
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

class ErrorRecoveryStrategy(BaseModel):
    """Schema for error recovery decisions."""
    model_config = ConfigDict(extra='forbid')

    intent: Literal["REPLACE_AND_RETRY", "REPLACE_AND_RESTART"] = Field(
        description="Error recovery strategy to apply"
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "intent": "REPLACE_AND_RETRY"
}

Valid intent values: "REPLACE_AND_RETRY", "REPLACE_AND_RESTART"
Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class ErrorRecoveryTool(StructuredPromptTool):
    """Tool for analyzing errors and determining recovery strategy.

    **UI Returns:**
    - `output_ui`: String intent value ("code_fixing", "replace_and_restart", etc.)
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `error_recovery_strategy`: Recovery strategy intent for CodeFixingTool to use

    **Used by workflows:** Error recovery workflow to analyze errors and determine fixing approach

    **Special behavior:** Modifies user_query with structured error context and failed code
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("error_recovery", PromptScenario.ERROR_RECOVERY, llm_interface)

    def _process_structured_result(self, result, state: Optional['KaiState'] = None) -> ToolResult:
        """Process structured error recovery result."""
        return ToolResult(
            output_ui=result.intent,
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "error_recovery_strategy": result.intent
            }
        )
