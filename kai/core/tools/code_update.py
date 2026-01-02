"""Code update tool for generating updated code after error analysis or feedback.

This module contains the CodeUpdateTool, an UnstructuredPromptTool that generates
updated code based on error analysis, user feedback, or retry objectives.
"""

from typing import TYPE_CHECKING, Optional

from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger
from .base import ToolResult, ToolOutputType
from .prompt_base import UnstructuredPromptTool, extract_code_from_response

if TYPE_CHECKING:
    from kai.core.llm_interface import LLMInterface
    from kai.core.orchestration.state import KaiState

logger = setup_logger(__name__)


class CodeUpdateTool(UnstructuredPromptTool):
    """Tool for generating updated code after error analysis or feedback.

    **UI Returns:**
    - Autonomous mode: Dict with "code", "should_replace", "error_recovery_strategy", "positioning_info" for VSCode
    - Manual mode: Raw LLM response string for chat display
    - `output_type`: EXECUTE_ONLY (autonomous) or RESPONSE (manual)

    **Workflow Returns:**
    - `error_recovery_strategy`: Strategy used for fixing (from ErrorRecoveryTool output)

    **Used by workflows:** Error recovery workflow after ErrorRecoveryTool determines strategy

    **Special behavior:** Dynamically switches between CODE_FIXING, CODE_FIXING_WITH_GUIDANCE, CODE_UPDATE_WITH_GUIDANCE scenarios
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("code_update", PromptScenario.CODE_FIXING, llm_interface)

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Execute code fixing with active task guidance if available."""
        # Choose scenario based on whether we have active task information
        has_error = state["last_execution_failed"]
        active_task_objective = state.get("active_task_objective")
        retry_objective = state.get("retry_objective")
        if has_error:
            if active_task_objective:
                self.scenario = PromptScenario.CODE_FIXING_WITH_GUIDANCE
            else:
                # Use regular code fixing scenario
                self.scenario = PromptScenario.CODE_FIXING
        elif retry_objective is not None:
            # Use feedback-centric code update
            self.scenario = PromptScenario.CODE_UPDATE_WITH_GUIDANCE
        else:
            raise ValueError((has_error, active_task_objective, retry_objective))
        result = await super().execute(state, **kwargs)
        # Reset default scenario:
        self.scenario = PromptScenario.CODE_FIXING
        return result

    async def _process_response(self, response: str, state: 'KaiState') -> ToolResult:
        """Process code fixing response and format for VSCode execution."""
        autonomous_mode = state["autonomous_mode"]
        output_type = ToolOutputType.EXECUTE_ONLY if autonomous_mode else ToolOutputType.RESPONSE

        # Extract context for code fixing first
        positioning_info = state["positioning_info"]
        error_recovery_strategy = state.get("error_recovery_strategy")

        # Extract clean code from response
        extracted_code = extract_code_from_response(response)

        # If no code was extracted, raise an error to trigger retry in orchestration loop
        if extracted_code is None:
            raise ValueError(f"CodeUpdateTool could not extract code from response. Response: {response}")

        # Create VSCode-ready response - only include fields VSCode uses
        if autonomous_mode:
            vscode_response = {
                "code": extracted_code,
                "should_replace": True,
                "error_recovery_strategy": error_recovery_strategy,
                "positioning_info": positioning_info,
                "cell_type": "code"
            }
            output_type = ToolOutputType.EXECUTE_ONLY

            # Create workflow output for LangGraph state (router needs these fields)
            workflow_output = {
                "generated_code": extracted_code,
                "target_cell": positioning_info.get("target_cell_index", 0),
                # Clear backtracking/retry state to prevent it from persisting to next iteration
                "cells_to_delete": None,
                "cells_deleted": None,
                "backtrack_recovery_done": None,
                "recovery_objective": None,  # This triggers is_backtracking check in router
                "retry_objective": None,  # This triggers is_standard_retry check in router
                # Note: Do NOT clear error_recovery_strategy here - it's needed by router
                # to determine that we're in standard_retry branch until we exit to "complete"
            }
        else:
            vscode_response = response  # Manual mode: return full response as string
            output_type = ToolOutputType.RESPONSE
            workflow_output = {}

        return ToolResult(
            output_ui=vscode_response,
            output_type=output_type,
            output_workflow=workflow_output
        )
