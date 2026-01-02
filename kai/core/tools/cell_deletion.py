"""Cell deletion execution tool.

This module provides a deterministic tool for executing cell deletions
and calculating index translations after backtracking.
"""

from typing import Dict, List, TYPE_CHECKING

from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState

logger = setup_logger(__name__)


class CellDeletionTool(BaseTool):
    """Tool for actually deleting selected cells and translating indices.

    **UI Returns:**
    - `output_ui`: Dict with "text", "vscode_commands", "deleted_cells", "index_translation" for VSCode
    - `output_type`: EXECUTE_ONLY - executes cell deletions without chat display

    **Workflow Returns:**
    - `deleted_cells`: List of deleted cell indices for backtracking context
    - `index_translation`: Mapping of original->new indices after deletion

    **Used by workflows:** Backtracking workflow after CellSelectionDeletionTool selects cells

    **Special behavior:** Creates VSCode deletion commands and calculates index translation mapping
    """

    def __init__(self):
        super().__init__("cell_deletion")

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Execute cell deletion and index translation."""
        # Get cells to delete from previous tool
        cells_to_delete = state["cells_to_delete"]

        if not cells_to_delete:
            return ToolResult(
                output_ui="No cells selected for deletion",
                output_type=ToolOutputType.NO_OUTPUT,
            )

        # Sort in descending order to delete from end to beginning (preserves indices)
        cells_to_delete = sorted(cells_to_delete, reverse=True)

        # Create VSCode commands for cell deletion)
        vscode_commands = []
        for cell_num in cells_to_delete:
            vscode_commands.append({
                "command": "deleteCell",
                "cellIndex": cell_num
            })

        # Calculate index translation mapping for remaining cells
        index_translation = self._calculate_index_translation(cells_to_delete)

        # Log cell deletion
        deleted_list = sorted(cells_to_delete)
        cells_str = ", ".join(str(c) for c in deleted_list)
        logger.info(f"Deleted {len(deleted_list)} cells: {cells_str}")

        # Create output dict with all necessary data for VSCode
        output_data = {
            "text": f"Deleted cells: {sorted(cells_to_delete)}",
            "vscode_commands": vscode_commands,
            "deleted_cells": sorted(cells_to_delete),
            "index_translation": index_translation
        }

        return ToolResult(
            output_ui=output_data,
            output_type=ToolOutputType.EXECUTE_ONLY,
            output_workflow={
                "deleted_cells": sorted(cells_to_delete),
                "index_translation": index_translation,
                "cells_deleted": True,  # For deterministic router phase tracking
            }
        )

    def _calculate_index_translation(self, deleted_cells: List[int]) -> Dict[int, int]:
        """Calculate how original cell indices map to new indices after deletion.

        Args:
            deleted_cells: List of cell indices that were deleted (sorted)

        Returns:
            Dict mapping original_index -> new_index for remaining cells
        """
        if not deleted_cells:
            return {}

        deleted_set = set(deleted_cells)
        translation = {}
        new_index = 0

        # Assume we have cells up to max deleted cell + some buffer
        max_cell = max(deleted_cells) + 20  # Buffer for cells after deletions

        for original_index in range(max_cell):
            if original_index not in deleted_set:
                translation[original_index] = new_index
                new_index += 1

        return translation
