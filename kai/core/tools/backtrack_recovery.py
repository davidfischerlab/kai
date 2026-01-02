"""Backtrack recovery tool for determining if notebook restart is needed.

This module contains:
- BacktrackRecoveryStrategy schema for structured output
- BacktrackRecoveryTool for determining if restart is needed for backtracking
"""

from typing import TYPE_CHECKING, Optional

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

class BacktrackRecoveryStrategy(BaseModel):
    """Schema for backtracking recovery decisions."""
    model_config = ConfigDict(extra='forbid')

    restart_required: bool = Field(
        description="Whether notebook restart is required before continuing with backtracking recovery"
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "restart_required": true
}

Valid restart_required values: true, false
Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class BacktrackRecoveryTool(StructuredPromptTool):
    """Tool for determining if notebook restart is needed for backtracking recovery.

    **UI Returns:**
    - `output_ui`: Boolean indicating if kernel restart is required
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `restart_required`: Boolean flag for workflow orchestration decisions

    **Used by workflows:** Backtracking recovery workflow to determine if restart is needed

    **Special behavior:** Analyzes deleted tasks and error context to make restart decision
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("backtrack_recovery", PromptScenario.BACKTRACK_RECOVERY, llm_interface)

    def _modify_user_query(self, state: 'KaiState') -> None:
        """Create PromptContext with recovery objective and error details."""
        from kai.core.orchestration.state import BacktrackingContext

        # Build context for backtrack recovery decision
        context_parts = []

        backtracking_context = BacktrackingContext.from_state(state)
        if backtracking_context and backtracking_context.is_active:
            context_parts.append(f"## Recovery Objective:\n{backtracking_context.recovery_objective}")
            context_parts.append("")

        error_details = state.get("error_details", "")
        if error_details:
            context_parts.append(f"## Observed Errors:\n{error_details}")
            context_parts.append("")

        reset_tasks = state["reset_tasks"]
        if reset_tasks:
            context_parts.append("## Tasks Being Reset (corresponding to deleted code):")
            for task in reset_tasks:
                task_desc = task["task"]
                context_parts.append(f"- **Task {task['id']}**: {task_desc}")
            context_parts.append("")

        user_query = "\n".join(context_parts)

        # Update the user query in state
        state["user_query"] = user_query

    def _process_structured_result(self, result, state: Optional['KaiState'] = None) -> ToolResult:
        """Process structured backtrack recovery result."""
        return ToolResult(
            output_ui=result.restart_required,
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "restart_required": result.restart_required,
                "backtrack_recovery_done": True,  # For deterministic router phase tracking
            }
        )
