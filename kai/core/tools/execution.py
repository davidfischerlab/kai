"""Execution tools for running cells and managing kernel."""

from typing import TYPE_CHECKING
from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType
from kai.core.orchestration.prompt_tools import RestartAndRerunTool

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext
    from kai.core.llm_interface import LLMInterface


class ExecuteCellTool(BaseTool):
    """
    Execute a cell in the notebook.

    Note: Actual execution is handled by VSCode/Jupyter interface.
    This tool prepares the execution context.
    """

    def __init__(self):
        super().__init__("execute_cell")

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return ToolResult(
            output_ui={},
            output_workflow={},
            output_type=ToolOutputType.NO_OUTPUT
        )


class RestartAndRerunTool(BaseTool):
    """
    Restart kernel and re-execute cells.

    Used for clean recovery from errors and state corruption.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("restart_and_rerun")
        from kai.core.orchestration.prompt_tools import RestartAndRerunTool as BaseRestart
        self.restart_tool = BaseRestart(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return await self.restart_tool.execute(exec_context, **kwargs)
