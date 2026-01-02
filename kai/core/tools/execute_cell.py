"""Execution tools for running cells and managing kernel."""

from typing import TYPE_CHECKING
from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface


class ExecuteCellTool(BaseTool):
    """
    Execute a cell in the notebook.

    Note: Actual execution is handled by VSCode/Jupyter interface.
    This tool prepares the execution context.
    """

    def __init__(self):
        super().__init__("execute_cell")

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        # Clear generated_code and target_cell after execution to prevent infinite loop
        return ToolResult(
            output_ui={},
            output_workflow={
                "generated_code": None,  # Clear to prevent re-execution
                "target_cell": None,
            },
            output_type=ToolOutputType.NO_OUTPUT
        )
