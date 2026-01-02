"""Execution monitor tool for monitoring long-running cell execution.

This module contains:
- ExecutionMonitor schema for structured output
- ExecutionMonitorTool for deciding whether to continue or terminate stuck cells
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

class ExecutionMonitor(BaseModel):
    """Schema for execution progress monitoring decisions."""
    model_config = ConfigDict(extra='forbid')

    action: Literal["continue", "terminate"] = Field(
        description="Whether to continue execution or terminate the stuck cell."
    )
    feedback: str = Field(
        description="If a cell was terminated - instructions for updating the cell."
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "action": "continue" or "terminate",
    "feedback": "If a cell was terminated - instructions for updating the cell"
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class ExecutionMonitorTool(StructuredPromptTool):
    """Tool for monitoring long-running cell execution and deciding whether to continue or terminate.

    **UI Returns:**
    - `output_ui`: Action decision ("continue" or "terminate")
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `action`: "continue" or "terminate" - decision for execution control
    - `feedback`: String detailing suggeseted changes

    **Used by workflows:** Execution progress check workflow to analyze stuck cells

    **Special behavior:** Analyzes cell code, elapsed time, and partial outputs to detect stuck execution
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("execution_monitor", PromptScenario.EXECUTION_MONITOR, llm_interface)

    def _process_structured_result(self, result, state: Optional['KaiState'] = None) -> ToolResult:
        """Process structured execution monitor result."""
        return ToolResult(
            output_ui=result.action,
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "action": result.action,
                "feedback": result.feedback
            }
        )
