"""Section code review tool for reviewing and fixing code sections with errors.

This module contains:
- SectionCodeReview schema for structured output
- SectionCodeReviewTool for analyzing section code and recommending fixes
"""

from typing import TYPE_CHECKING, Optional, Dict, Any, List, Union, Literal

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

class SectionCodeReview(BaseModel):
    """Schema for section code review and recovery decisions."""
    model_config = ConfigDict(extra='forbid')

    operation: Literal["delete", "replace", "insert"] = Field(
        description="Type of operation to perform: delete, replace, or insert cells"
    )
    position: Union[int, List[int]] = Field(
        description="For insert: integer index where new code should be placed. For delete/replace: list of cell indices to modify (0-based relative to section)"
    )
    intent: str = Field(
        description="User query/intent describing what the code should accomplish - used for code generation"
    )
    reasoning: str = Field(
        description="Reasoning for why this fix addresses the error"
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "operation": "replace",
    "position": [3, 4],
    "intent": "User query describing what the code should accomplish",
    "reasoning": "Reasoning for the fix"
}

Valid operation values: "delete", "replace", "insert"
For insert: position should be an integer
For delete/replace: position should be a list of integers
Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class SectionCodeReviewTool(StructuredPromptTool):
    """Tool for reviewing and fixing code sections that encounter errors during execution.

    **UI Returns:**
    - `output_ui`: Structured result with "operation" and "target_cells" fields
    - `output_type`: NO_OUTPUT - internal tool used by RunSectionSubgraph

    **Workflow Returns:**
    - `operation`: Fix operation to perform ("fix", "skip", "restart")
    - `target_cells`: List of cell indices that need attention

    **Used by workflows:** RunSectionSubgraph for intelligent section execution with error recovery

    **Special behavior:** Analyzes section code, errors, and previous fix attempts to recommend actions
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("section_code_review", PromptScenario.SECTION_CODE_REVIEW, llm_interface)

    def _modify_user_query(self, state: 'KaiState') -> None:
        """Create PromptContext with section code, error details, and fix history."""
        context_parts = []

        # Add section code being executed
        section_code = state["section_code"]
        if section_code:
            context_parts.append("## Code Section Being Executed:")
            for i, cell_code in enumerate(section_code):
                context_parts.append(f"**Cell {i}:**")
                context_parts.append(f"```python\n{cell_code}\n```")
            context_parts.append("")

        # Add error details
        last_execution_failed = state["last_execution_failed"]
        error_cell = state["error_cell"]
        error_message = state["error_message"]
        if last_execution_failed:
            context_parts.append("## Error Encountered:")
            if error_cell is not None:
                context_parts.append(f"**Error in Cell {error_cell}:**")
            context_parts.append(f"```\n{error_message}\n```")
            context_parts.append("")

        # Add previous fix attempts if any
        fix_attempts = state["fix_attempts"]
        if fix_attempts:
            context_parts.append("## Previous Fix Attempts:")
            for i, attempt in enumerate(fix_attempts):
                context_parts.append(f"**Attempt {i+1}:** {attempt['operation']} on cells {attempt['target_cells']}")
                context_parts.append(f"Reasoning: {attempt['reasoning']}")
            context_parts.append("")

        # Add current objective
        context_parts.append("## Objective:")
        context_parts.append("Fix the error with minimal changes to make this code section run successfully.")

        user_query = "\n".join(context_parts)

        # Update the user query in state
        state["user_query"] = user_query

    def _process_structured_result(self, result, state: Optional['KaiState'] = None) -> ToolResult:
        """Process structured section code review result."""
        return ToolResult(
            output_ui=result.model_dump(),
            output_type=ToolOutputType.NO_OUTPUT  # Internal only - used by RunSectionWorkflow
        )
