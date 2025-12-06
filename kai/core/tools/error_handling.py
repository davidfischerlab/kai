"""Error handling, recovery, and backtracking tools."""

from typing import TYPE_CHECKING
from kai.core.orchestration.base_tool import BaseTool, ToolResult
from kai.core.orchestration.prompt_tools import (
    ErrorRecoveryTool as BaseErrorRecovery,
    BacktrackRecoveryTool as BaseBacktrackRecovery,
    CellSelectionDeletionTool,
    ExecutionMonitorTool as BaseExecutionMonitor
)
from kai.core.orchestration.deterministic_tools import CellDeletionTool
from kai.core.orchestration.prompt_tools import CellPositioningTool

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext
    from kai.core.llm_interface import LLMInterface


class HandleErrorTool(BaseTool):
    """
    Analyze and recover from execution errors.

    Multi-step process:
    1. Analyze error (LLM)
    2. Determine recovery strategy (REPLACE_AND_RETRY vs REPLACE_AND_RESTART)
    3. Set up context for code update
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("handle_error")
        self.error_recovery_tool = BaseErrorRecovery(llm_interface)
        self.execution_monitor_tool = BaseExecutionMonitor(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        result = await self.error_recovery_tool.execute(exec_context, **kwargs)
        return result


class BacktrackTool(BaseTool):
    """
    Backtrack to earlier state by deleting cells and resetting tasks.

    Multi-step process:
    1. Analyze failure (LLM)
    2. Select cells to delete (LLM)
    3. Delete cells (deterministic)
    4. Reset task statuses (deterministic)
    5. Determine positioning for recovery (deterministic)
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("backtrack")
        self.backtrack_recovery_tool = BaseBacktrackRecovery(llm_interface)
        self.cell_selection_tool = CellSelectionDeletionTool(llm_interface)
        self.cell_deletion_tool = CellDeletionTool()
        self.positioning_tool = CellPositioningTool(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        recovery_result = await self.backtrack_recovery_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(recovery_result.output_workflow or {})

        selection_result = await self.cell_selection_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(selection_result.output_workflow or {})

        deletion_result = await self.cell_deletion_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(deletion_result.output_workflow or {})

        positioning_result = await self.positioning_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(positioning_result.output_workflow or {})

        return ToolResult(
            output_workflow=exec_context.inputs.context,
            output_ui=recovery_result.output_ui,
            output_type=recovery_result.output_type
        )


class ExecutionMonitorTool(BaseTool):
    """
    Monitor long-running cell execution and decide whether to continue or terminate.

    Used for timeout management and runaway process detection.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("execution_monitor")
        self.monitor_tool = BaseExecutionMonitor(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return await self.monitor_tool.execute(exec_context, **kwargs)
