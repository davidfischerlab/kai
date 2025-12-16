"""Workflow search and retrieval tool."""

from typing import TYPE_CHECKING
from kai.utils.logger import get_logger
from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType

logger = get_logger(__name__)
from kai.core.orchestration.prompt_tools import (
    ReferenceWorkflowSelectionTool,
    ReferenceWorkflowSelectionOnlyTool,
    ReferenceWorkflowCellSelectionTool
)
from kai.core.orchestration.deterministic_tools import (
    ReferenceWorkflowQueryPreparationTool,
    FilterUnusedReferenceWorkflowsTool
)
from kai.core.prompt_manager import PromptScenario

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext
    from kai.core.llm_interface import LLMInterface
    from kai.retrieval import ChromaDbManager


class SearchWorkflowsTool(BaseTool):
    """
    Find and extract relevant cells from reference analysis notebooks.

    Multi-step pipeline:
    1. Query preparation (deterministic search)
    2. Workflow selection (LLM)
    3. Cell extraction (LLM)
    4. Filter unused workflows (deterministic)
    """

    def __init__(
        self,
        llm_interface: 'LLMInterface',
        knowledge_base: 'ChromaDbManager',
        mode: str = "full"
    ):
        super().__init__("search_workflows")
        self.mode = mode

        from kai.config.settings import Settings
        from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
        from kai.retrieval.workflow_summaries.notebook_selector import NotebookSelector
        from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag

        settings = Settings.from_env()

        storage = NotebookStorage(settings.NOTEBOOK_SUMMARIES_PATH)
        selector = NotebookSelector(storage)
        summary_search = WorkflowSummaryRag(settings.NOTEBOOK_SUMMARIES_PATH)

        self.query_prep_tool = ReferenceWorkflowQueryPreparationTool(
            summary_search=summary_search
        )

        if mode == "full":
            self.selection_tool = ReferenceWorkflowSelectionTool(
                scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION,
                llm_interface=llm_interface,
                notebook_selector=selector
            )
        else:
            self.selection_tool = ReferenceWorkflowSelectionOnlyTool(
                scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
                llm_interface=llm_interface,
                notebook_selector=selector
            )

        self.cell_selection_tool = ReferenceWorkflowCellSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION,
            llm_interface=llm_interface,
            notebook_selector=selector
        )

        self.filter_tool = FilterUnusedReferenceWorkflowsTool()

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """Execute workflow search pipeline and manage planning phase state."""

        # Get current iteration counter
        current_iteration = exec_context.inputs.context.get("workflow_retrieval_iteration", 0)
        retrieval_queries = exec_context.inputs.context.get("retrieval_queries", [])
        had_retrieval_queries = bool(retrieval_queries and len(retrieval_queries) > 0)

        # Log planning iteration header for initial workflow retrieval
        logger.info(f"[PLANNING ITERATION {current_iteration + 1}/12]")
        if had_retrieval_queries:
            queries_str = ", ".join(f'"{q}"' for q in retrieval_queries)
            logger.info(f"  Searching {len(retrieval_queries)} new queries: {queries_str}")

        # Execute workflow search pipeline
        result = await self.query_prep_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(result.output_workflow or {})

        result = await self.selection_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(result.output_workflow or {})

        result = await self.cell_selection_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(result.output_workflow or {})

        if self.mode == "full":
            result = await self.filter_tool.execute(exec_context, **kwargs)
            # Filter tool may return empty output_workflow when no filtering needed
            # In this case, preserve reference_workflow_content from context
            exec_context.inputs.context.update(result.output_workflow or {})

        # Update planning phase state for explicit control flow
        # - Set phase to "workflow_retrieval" to signal we're in retrieval loop
        # - Increment iteration counter
        # - Keep retrieval_queries from selection tool (router will check if empty to exit loop)
        #
        # IMPORTANT: Preserve PERSISTENT fields from execution context, not just last tool's output.
        # The filter tool may return empty output_workflow when no filtering is needed (all workflows
        # mentioned in tasks), which would lose reference_workflow_content set by cell_selection_tool.
        # Since all sub-tools update exec_context.inputs.context, we read from there for PERSISTENT fields.
        result_with_phase_tracking = ToolResult(
            output_ui=result.output_ui,
            output_type=result.output_type,
            output_workflow={
                # Preserve PERSISTENT fields from execution context
                "reference_workflow_content": exec_context.inputs.context.get("reference_workflow_content", {}),
                "reference_workflow_percentages": exec_context.inputs.context.get("reference_workflow_percentages", {}),
                "excluded_workflows": exec_context.inputs.context.get("excluded_workflows", []),
                "retrieval_queries": exec_context.inputs.context.get("retrieval_queries", []),
                # Add phase tracking fields
                "planning_phase": "workflow_retrieval",  # Explicit phase tracking
                "workflow_retrieval_iteration": current_iteration + 1,  # Increment counter
                # Router will check: empty queries OR iteration >= 2 → exit to task_planning
            }
        )

        return result_with_phase_tracking


class WorkflowRefinementTool(BaseTool):
    """
    Refine reference workflows based on retrieval queries from task generation.

    This matches lines 285-293 in kai_dev/core/orchestration/workflow_orchestrator.py:
    - Uses retrieval_queries from plan_tasks tool (NOT from LLM selection)
    - Runs: query_prep → selection_only → cell_selection
    - Clears retrieval_queries after execution (line 293)
    - Increments task_planning_iteration counter

    Key difference from SearchWorkflowsTool:
    - Uses ReferenceWorkflowSelectionOnlyTool (doesn't generate new queries)
    - Manages task_planning_iteration (not workflow_retrieval_iteration)
    - Clears queries after to prevent infinite loop
    """

    def __init__(self, llm_interface: 'LLMInterface', knowledge_base: 'ChromaDbManager'):
        super().__init__("workflow_refinement")

        from kai.config.settings import Settings
        from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
        from kai.retrieval.workflow_summaries.notebook_selector import NotebookSelector
        from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag

        settings = Settings.from_env()

        storage = NotebookStorage(settings.NOTEBOOK_SUMMARIES_PATH)
        selector = NotebookSelector(storage)
        summary_search = WorkflowSummaryRag(settings.NOTEBOOK_SUMMARIES_PATH)

        self.query_prep_tool = ReferenceWorkflowQueryPreparationTool(
            summary_search=summary_search
        )

        # Use selection_only tool (doesn't generate new retrieval queries)
        self.selection_tool = ReferenceWorkflowSelectionOnlyTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
            llm_interface=llm_interface,
            notebook_selector=selector
        )

        self.cell_selection_tool = ReferenceWorkflowCellSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION,
            llm_interface=llm_interface,
            notebook_selector=selector
        )

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """
        Execute workflow refinement pipeline.

        Matches old orchestrator lines 285-293:
        1. Query prep: Search knowledge base with queries from plan_tasks
        2. Selection only: LLM selects workflows (no new queries generated)
        3. Cell selection: Extract relevant cells
        4. Clear retrieval_queries (line 293)
        5. Increment task_planning_iteration
        """
        # Get current iteration counter and check if there were queries BEFORE refinement
        current_iteration = exec_context.inputs.context.get("task_planning_iteration", -1)
        workflow_iteration = exec_context.inputs.context.get("workflow_retrieval_iteration", 0)
        retrieval_queries = exec_context.inputs.context.get("retrieval_queries", [])
        had_retrieval_queries = bool(retrieval_queries and len(retrieval_queries) > 0)

        # Log workflow retrieval as planning iterations
        # Initial workflow retrieval (before task generation) counts as planning iterations 1-2
        # Task generation iterations will be 3-12 (max 10 task generation iterations)
        if current_iteration == -1:
            # Initial workflow retrieval phase - count as planning iterations 1-2
            logger.info(f"[PLANNING ITERATION {workflow_iteration + 1}/12]")
            if had_retrieval_queries:
                queries_str = ", ".join(f'"{q}"' for q in retrieval_queries)
                logger.info(f"  Searching {len(retrieval_queries)} new queries: {queries_str}")
            else:
                logger.info(f"  Re-evaluating existing workflows (protecting cited, removing uncited)")
        elif had_retrieval_queries:
            # Workflow refinement during task generation iterations (indented under planning iteration)
            queries_str = ", ".join(f'"{q}"' for q in retrieval_queries)
            logger.info(f"  Searching {len(retrieval_queries)} new queries: {queries_str}")
        else:
            logger.info(f"  Re-evaluating existing workflows (protecting cited, removing uncited)")

        # Execute workflow refinement pipeline
        result = await self.query_prep_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(result.output_workflow or {})

        result = await self.selection_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(result.output_workflow or {})

        result = await self.cell_selection_tool.execute(exec_context, **kwargs)
        exec_context.inputs.context.update(result.output_workflow or {})

        # Update state to match old orchestrator behavior
        # IMPORTANT: Preserve PERSISTENT fields from execution context (same issue as SearchWorkflowsTool)
        result_with_state = ToolResult(
            output_ui=result.output_ui,
            output_type=result.output_type,
            output_workflow={
                # Preserve PERSISTENT fields from execution context
                "reference_workflow_content": exec_context.inputs.context.get("reference_workflow_content", {}),
                "reference_workflow_percentages": exec_context.inputs.context.get("reference_workflow_percentages", {}),
                "excluded_workflows": exec_context.inputs.context.get("excluded_workflows", []),
                # Clear retrieval queries after refinement (kai_dev line 293)
                "retrieval_queries": [],
                # Track whether there were queries BEFORE clearing (kai_dev line 296 checks this)
                "had_retrieval_queries_before_refinement": had_retrieval_queries,
                "planning_phase": "workflow_refinement",  # Signal that workflow refinement is complete
                # Router will check this and route back to task_list_generation (kai_dev line 296-297)
                # NOTE: Router will increment task_planning_iteration when routing back to task_list_generation
            }
        )

        return result_with_state
