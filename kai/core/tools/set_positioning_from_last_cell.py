"""Cell positioning utility tools.

This module provides deterministic tools for managing cell positioning
in autonomous mode workflows.
"""

from typing import TYPE_CHECKING

from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState

logger = setup_logger(__name__)


class SetPositioningFromLastCellTool(BaseTool):
    """Set positioning_info from last_cell_modified_in_auto_mode.

    In standard continuation and error recovery, positioning is determined by
    the last modified cell, NOT the LLM. This ensures we add/replace at the
    correct position after cells have been inserted.

    Use cases:
    - Standard continue (success): Position at last modified cell to add after it
    - Standard retry (error): Position at last modified cell to replace it
    - NOT used for: First execution (no last_cell yet) or backtracking (indices changed)

    **UI Returns:**
    - `output_type`: NO_OUTPUT - internal positioning tool

    **Workflow Returns:**
    - `positioning_info`: Dict with target_cell from last_cell_modified_in_auto_mode
    """

    def __init__(self):
        super().__init__("set_positioning_from_last_cell")

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Set positioning from last cell modified in auto mode."""
        last_cell = state.get("last_cell_modified_in_auto_mode")

        if last_cell is None:
            # Fallback to error cell if available (for retry scenarios)
            error_cell = state.get("error_cell_index", -1)
            if error_cell >= 0:
                last_cell = error_cell
                logger.info(f"[SET_POSITIONING] Using error_cell_index as fallback: {error_cell}")
            else:
                # This shouldn't happen in normal flow - log warning
                logger.warning("[SET_POSITIONING] No last_cell_modified_in_auto_mode or error_cell_index found")
                # Ultimate fallback - use last cell in notebook
                notebook_structure = state.get("notebook_structure", {})
                total_cells = notebook_structure.get("totalCells", 0)
                last_cell = max(0, total_cells - 1)
                logger.info(f"[SET_POSITIONING] Using notebook last cell as fallback: {last_cell}")

        positioning_info = {"target_cell": last_cell}
        logger.info(f"[SET_POSITIONING] Set positioning to cell {last_cell}")

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={"positioning_info": positioning_info}
        )


class IncrementPositioningTool(BaseTool):
    """Increment positioning_info target_cell by 1.

    Used after adding a new cell (e.g., reasoning cell) so that subsequent
    operations (like critique regeneration) target the NEW cell, not the original.

    **UI Returns:**
    - `output_type`: NO_OUTPUT - internal positioning tool

    **Workflow Returns:**
    - `positioning_info`: Dict with target_cell incremented by 1
    """

    def __init__(self):
        super().__init__("increment_positioning")

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Increment positioning target_cell by 1."""
        positioning_info = state.get("positioning_info", {})
        current_target = positioning_info.get("target_cell", 0)
        new_target = current_target + 1

        new_positioning = {"target_cell": new_target}
        logger.info(
            f"[INCREMENT_POSITIONING] Incremented positioning "
            f"from {current_target} to {new_target}"
        )

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "positioning_info": new_positioning,
                "reasoning_positioning_incremented": True  # Prevent double-increment
            }
        )
