"""Cell selection deletion tool for selecting cells to delete during backtracking.

This module contains:
- CellDeletionSelection schema for structured output
- CellSelectionDeletionTool for intelligently selecting cells for deletion
"""

from typing import TYPE_CHECKING, Optional, List, Union

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

class CellDeletionSelection(BaseModel):
    """Schema for cell deletion selection responses."""
    model_config = ConfigDict(extra='forbid')
    cells_to_delete: List[int] = Field(
        description="List of cell numbers to delete (0-indexed)"
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "cells_to_delete": [3, 5, 7],
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class CellSelectionDeletionTool(StructuredPromptTool):
    """Tool for selecting cells to delete during backtracking.

    **UI Returns:**
    - `output_ui`: String describing selected cells and reasoning
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - None - cells_to_delete passed via context to CellDeletionTool

    **Used by workflows:** Backtracking workflow to intelligently select cells for deletion

    **Special behavior:** Modifies user_query with reset tasks and recovery objective context
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("cell_selection_deletion", PromptScenario.CELL_SELECTION_DELETION_FOR_BACKTRACKING, llm_interface)

    def _process_structured_result(self, structured_result, state: Union[None, 'KaiState'] = None) -> ToolResult:
        """Process structured CellDeletionSelection response."""
        cells_to_delete = structured_result.cells_to_delete

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "cells_to_delete": cells_to_delete
            }
        )
