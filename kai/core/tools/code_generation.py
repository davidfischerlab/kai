"""Code generation and update tools."""

from typing import TYPE_CHECKING
from kai.core.orchestration.base_tool import BaseTool, ToolResult
from kai.core.orchestration.prompt_tools import (
    CellPositioningTool,
    CodeGenerationTool,
    CodeGenerationWithGuidanceTool,
    CodeUpdateTool
)
from kai.core.prompt_manager import PromptScenario

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext
    from kai.core.llm_interface import LLMInterface


class GenerateCodeTool(BaseTool):
    """
    Generate code with automatic positioning.

    Combines:
    - Cell positioning (if needed)
    - Code generation
    - Internal critique loop (with guidance variant)
    """

    def __init__(self, llm_interface: 'LLMInterface', with_guidance: bool = True):
        super().__init__("generate_code")
        self.positioning_tool = CellPositioningTool(llm_interface)

        if with_guidance:
            self.generation_tool = CodeGenerationWithGuidanceTool(llm_interface)
        else:
            self.generation_tool = CodeGenerationTool(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        if "positioning_info" not in exec_context.inputs.context:
            pos_result = await self.positioning_tool.execute(exec_context, **kwargs)
            exec_context.inputs.context.update(pos_result.output_workflow or {})

        gen_result = await self.generation_tool.execute(exec_context, **kwargs)

        return gen_result


class UpdateCodeTool(BaseTool):
    """
    Update existing code in a cell.

    Used for error recovery and code refinement.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("update_code")
        self.update_tool = CodeUpdateTool(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return await self.update_tool.execute(exec_context, **kwargs)
