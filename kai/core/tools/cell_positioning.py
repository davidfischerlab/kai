"""Cell positioning tool for determining cell placement using LLM.

This module contains:
- CellPositioning schema for structured output
- CellPositioningTool for determining where to insert/replace code cells
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List

from pydantic import BaseModel, Field, ConfigDict

from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger
from .base import ToolResult, ToolOutputType
from .prompt_base import StructuredPromptTool

if TYPE_CHECKING:
    from kai.core.llm_interface import LLMInterface
    from kai.core.orchestration.state import KaiState, BacktrackingContext

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class CellPositioning(BaseModel):
    """Schema for cell positioning decisions."""
    model_config = ConfigDict(extra='forbid')

    target_cell: int = Field(
        description="Target cell number for insertion/replacement"
    )
    reasoning: str = Field(
        description="Reasoning for the positioning choice"
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "target_cell": 2,
    "reasoning": "Reasoning for the positioning choice"
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class CellPositioningTool(StructuredPromptTool):
    """Tool for determining cell positioning using LLM with addition/replacement logic.

    **UI Returns:**
    - `output_ui`: Dict with "target_cell" field containing selected cell index
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `positioning_info.target_cell`: Cell index for code generation tools to use

    **Used by workflows:** Multiple workflows before code generation to determine cell placement

    **Special behavior:** Dynamically switches scenarios (ADDITION vs REPLACEMENT) based on context
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        # Start with ADDITION scenario, will be determined dynamically
        super().__init__("cell_positioning", PromptScenario.CELL_SELECTION_ADDITION, llm_interface)

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Execute cell positioning with proper scenario selection."""
        from kai.core.orchestration.state import BacktrackingContext
        # Check for backtracking context (from autonomous workflow)
        backtracking_context = BacktrackingContext.from_state(state)

        # Check if this is error recovery context
        error_recovery = state.get("error_recovery", False)

        if backtracking_context and backtracking_context.is_active:
            self.scenario = PromptScenario.CELL_SELECTION_REPLACEMENT
        elif error_recovery:
            self.scenario = PromptScenario.CELL_SELECTION_REPLACEMENT
        else:
            self.scenario = PromptScenario.CELL_SELECTION_ADDITION

        return await super().execute(state, **kwargs)

    def _modify_user_query(self, state: 'KaiState') -> None:
        """Create PromptContext with backtracking information if available."""
        from kai.core.orchestration.state import BacktrackingContext
        # Check for backtracking context (autonomous continue workflow)
        backtracking_context = BacktrackingContext.from_state(state)

        if backtracking_context and backtracking_context.is_active:
            # In backtracking mode - build structured context
            context_parts = []

            # Recovery objective
            recovery_objective = backtracking_context.recovery_objective
            if recovery_objective:
                context_parts.append(f"## Recovery Objective:\n{recovery_objective}")
                context_parts.append(
                    "You are selecting a position in a notebook at which to add new code to start a recovery of a failed analysis attempt. "
                    "This failed atttempt involved deletion of parts to the analysis. "
                    "You are given the positions of cell deletions of failed tasks " \
                    "and a description of the last valid completed task as a reference point. "
                    "Use both to determine where to position the new code to be added as part of the recovery.")
                context_parts.append("")

            # Cell deletion info - convert to cleaned indices
            deleted_cells = backtracking_context.deleted_cells
            index_translation = backtracking_context.index_translation

            if deleted_cells and index_translation:
                # Find the gaps in the current notebook where cells were deleted
                current_notebook_gaps = self._find_deletion_gaps(deleted_cells, index_translation)
                if current_notebook_gaps:
                    context_parts.append("## Cells Removed:")
                    context_parts.append(f"Cells were deleted. The content now at indices {current_notebook_gaps} immediately preceded deleted cells.")
                    context_parts.append("")

            user_query = "\n".join(context_parts)

        else:
            user_query = state["user_query"]

        # Update the user query in state
        state["user_query"] = user_query

    def _process_structured_result(self, result, state: Optional['KaiState'] = None) -> ToolResult:
        """Convert LLM positioning result to format expected by code generation tools."""
        positioning_info = {
            "target_cell": result.target_cell
        }

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "positioning_info": positioning_info
            }
        )

    def _find_deletion_gaps(self, deleted_cells: List[int], index_translation: Dict[int, int]) -> List[int]:
        """Find cells that preceded deletions - these are safe insertion points.

        Args:
            deleted_cells: Original cell indices that were deleted (e.g., [5, 6, 7])
            index_translation: Mapping from original -> current indices

        Returns:
            List of current notebook indices that preceded the deleted sections
        """
        if not deleted_cells or not index_translation:
            return []

        preceding_cells = []
        deleted_set = set(deleted_cells)

        # Find cells that came right before the deleted cells
        for original_idx in sorted(index_translation.keys()):
            # Check if this cell came right before a deletion
            if original_idx + 1 in deleted_set:
                current_idx = index_translation[original_idx]
                preceding_cells.append(current_idx)

        return sorted(list(set(preceding_cells)))
