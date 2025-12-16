"""Task management and progress tracking tools."""

from typing import TYPE_CHECKING
from kai.core.orchestration.base_tool import BaseTool, ToolResult
from kai.core.orchestration.prompt_tools import (
    AutonomousMarkCompletionTool,
    AutonomousUpdateTasksTool,
    AutonomousUpdateCritiqueTool,
    TaskListGenerationTool,
    TaskListCritiqueTool
)
from kai.core.orchestration.deterministic_tools import MarkNextTaskActiveTool
from kai.core.utils import format_task_list

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext
    from kai.core.llm_interface import LLMInterface


class ManageProgressTool(BaseTool):
    """
    Assess progress and update task list.

    Multi-step process:
    1. Assess what was accomplished (LLM)
    2. Update tasks if needed (LLM)
    3. Critique updates (LLM, optional)
    4. Advance to next task (deterministic)
    """

    def __init__(self, llm_interface: 'LLMInterface', use_critique: bool = True):
        super().__init__("manage_progress")
        self.assess_tool = AutonomousMarkCompletionTool(llm_interface)
        self.update_tool = AutonomousUpdateTasksTool(llm_interface)
        self.critique_tool = AutonomousUpdateCritiqueTool(llm_interface)
        self.advance_tool = MarkNextTaskActiveTool()
        self.use_critique = use_critique

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        assess_result = await self.assess_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(assess_result.output_workflow or {})

        is_standard_retry = ("retry_objective" in exec_context.inputs.context and
                              "recovery_objective" not in exec_context.inputs.context)

        if not is_standard_retry:
            needs_update = exec_context.inputs.context.get("task_list_update_rule") != "NO_UPDATE"

            if needs_update:
                exec_context.inputs.context["task_text_old"] = format_task_list(exec_context.inputs.task_list)
                original_task_list = exec_context.inputs.task_list.copy()

                update_result = await self.update_tool.execute(exec_context, **kwargs)
                exec_context.inputs.context.update(update_result.output_workflow or {})

                if self.use_critique:
                    max_iterations = 3
                    for iteration in range(max_iterations):
                        critique_result = await self.critique_tool.execute(exec_context, **kwargs)
                        exec_context.inputs.context.update(critique_result.output_workflow or {})

                        if exec_context.inputs.context.get("autonomous_update_approval") == "APPROVED":
                            break
                        elif iteration == max_iterations - 1:
                            exec_context.inputs.task_list = original_task_list
                            exec_context.inputs.context["task_list"] = original_task_list
                            break

                        update_result = await self.update_tool.execute(exec_context, **kwargs)
                        exec_context.inputs.context.update(update_result.output_workflow or {})

        advance_result = await self.advance_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(advance_result.output_workflow or {})

        return ToolResult(
            output_workflow=exec_context.inputs.context,
            output_ui=advance_result.output_ui,  # Use advance_result (has activated task)
            output_type=advance_result.output_type  # TASK_LIST_DISPLAY with updated statuses
        )


class PlanTasksTool(BaseTool):
    """
    Generate initial task list from user request.

    Multi-step process:
    1. Generate task list (LLM)
    2. Critique task list (LLM)
    3. Iterate until approved or max iterations
    """

    def __init__(self, llm_interface: 'LLMInterface', use_critique: bool = True):
        super().__init__("plan_tasks")
        self.generation_tool = TaskListGenerationTool(llm_interface)
        self.critique_tool = TaskListCritiqueTool(llm_interface)
        self.use_critique = use_critique

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        max_iterations = 10 if self.use_critique else 1

        for iteration in range(max_iterations):
            gen_result = await self.generation_tool.execute(exec_context, **kwargs)
            exec_context.inputs.context.update(gen_result.output_workflow or {})

            retrieval_queries = exec_context.inputs.context.get("retrieval_queries", [])
            has_retrieval_queries = retrieval_queries and len(retrieval_queries) > 0

            if has_retrieval_queries:
                continue

            if self.use_critique:
                critique_result = await self.critique_tool.execute(exec_context, **kwargs)
                exec_context.inputs.context.update(critique_result.output_workflow or {})

                if exec_context.inputs.context.get("task_list_approval") == "APPROVED":
                    break

                if iteration < max_iterations - 1:
                    task_list_old = format_task_list(exec_context.inputs.task_list)
                    exec_context.inputs.context["task_text_old"] = task_list_old
            else:
                break

        return ToolResult(
            output_workflow=exec_context.inputs.context,
            output_ui=gen_result.output_ui,
            output_type=gen_result.output_type
        )
