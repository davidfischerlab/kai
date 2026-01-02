"""Reasoning response tool for autonomous mode.

This tool generates reasoning/explanation responses as markdown cells
before executing code changes in autonomous mode.
"""

from typing import TYPE_CHECKING

from kai.core.prompt_manager import PromptScenario
from kai.core.tools.base import ToolResult, ToolOutputType
from kai.core.tools.prompt_base import UnstructuredPromptTool
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface

logger = setup_logger(__name__)


class ReasoningResponseWithGuidanceTool(UnstructuredPromptTool):
    """Tool for generating reasoning/explanation responses in autonomous mode.

    Generates markdown cells explaining what the agent is about to do before
    actually executing code changes. This provides transparency and allows
    users to understand the agent's decision-making process.

    **UI Returns:**
    - Dict with "code" (markdown content), "positioning_info", "should_replace" (bool), "cell_type"
    - `output_type`: EXECUTE_ONLY - creates markdown cell without chat display

    **Workflow Returns:**
    - `reasoning_response`: The generated reasoning text
    - Clears backtracking/retry state fields (cells_to_delete, recovery_objective, retry_objective)

    **Used by workflows:** Autonomous execution workflows when reasoning task is active

    **Special behavior:**
    - Determines replace vs insert based on reasoning_critique or retry_objective presence
    - Clears backtracking state to prevent it from persisting to next iteration
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("reasoning_response_with_guidance", PromptScenario.REASONING_RESPONSE_WITH_GUIDANCE, llm_interface)

    async def _process_response(self, response: str, state: 'KaiState') -> ToolResult:
        """Process code generation response and format for VSCode (always autonomous mode)."""
        positioning_info = state["positioning_info"]

        # Check if this is a re-generation (after critique) - if so, replace
        # the previous reasoning cell:
        # 1) replace if critique iteration (reasoning_critique has value)
        # 2) replace if retry of reasoning task (retry_objective has value)
        # IMPORTANT: Check value is not None, not just key presence - LangGraph
        # state has keys with None values, and `"key" in dict` returns True
        # even when value is None.
        should_replace = (
            state.get("reasoning_critique") is not None or
            state.get("retry_objective") is not None
        )

        # Log reasoning generation action
        response_preview = response[:150].replace('\n', ' ') if len(response) > 150 else response.replace('\n', ' ')
        target_cell = positioning_info.get("target_cell", positioning_info.get("target_cell_index", -1))
        action = "Updated" if should_replace else "Generated"
        logger.info(f"{action} reasoning for cell {target_cell + 1}: {response_preview}...")

        # Create VSCode-ready response - only include fields VSCode uses
        vscode_response = {
            "code": response,
            "positioning_info": positioning_info,
            "should_replace": should_replace,  # Boolean: True if critique rejected or retry flagged incomplete
            "cell_type": "markdown"
        }
        # Make reasoning available for potential critiques:
        output_workflow = {
            "reasoning_response": response,
            # CRITICAL: Clear reasoning_approval after regeneration so router routes to critique
            # Without this, the router keeps seeing approval="MODIFY" and regenerates infinitely
            "reasoning_approval": None,
            # Clear backtracking/retry state to prevent it from persisting to next iteration
            "cells_to_delete": None,
            "cells_deleted": None,
            "backtrack_recovery_done": None,
            "recovery_objective": None,  # This triggers is_backtracking check in router
            "retry_objective": None,  # This triggers is_standard_retry check in router
        }

        # CRITICAL: When INSERTING (not replacing) a new reasoning cell, update positioning
        # to point to the NEW cell position. The UI inserts AFTER target_cell, so the new
        # cell is at target_cell + 1. Without this update, if critique rejects and triggers
        # regeneration, should_replace=True would replace the wrong cell (original code
        # cell at target_cell instead of the reasoning cell at target_cell + 1).
        if not should_replace:
            new_target = target_cell + 1
            output_workflow["positioning_info"] = {"target_cell": new_target}
            logger.info(
                f"[REASONING] Inserted new cell - updated positioning from {target_cell} to {new_target}"
            )

        return ToolResult(
            output_type=ToolOutputType.EXECUTE_ONLY,
            output_ui=vscode_response,
            output_workflow=output_workflow,
        )
