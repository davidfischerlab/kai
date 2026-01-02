"""Workflow search and retrieval tool."""

from typing import TYPE_CHECKING
from kai.utils.logger import get_logger
from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType
from kai.core.tools.reference_workflow_selection import (
    ReferenceWorkflowSelectionTool,
    ReferenceWorkflowSelectionOnlyTool,
)
from kai.core.tools.reference_workflow_cell_selection import ReferenceWorkflowCellSelectionTool
from kai.core.tools.reference_workflow_query_preparation import (
    ReferenceWorkflowQueryPreparationTool,
    FilterUnusedReferenceWorkflowsTool,
)
from kai.core.prompt_manager import PromptScenario

logger = get_logger(__name__)

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
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

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Execute workflow search pipeline and manage planning phase state."""

        # Get current iteration counter and max iterations from config
        current_iteration = state.get("workflow_retrieval_iteration", 0)
        max_retrieval_iterations = state.get("max_workflow_retrieval_iterations", 2)
        retrieval_queries = state.get("retrieval_queries", [])
        searched_queries = state.get("searched_retrieval_queries", [])

        # Calculate actually NEW queries (not yet searched)
        new_queries = [q for q in retrieval_queries if q not in searched_queries]
        had_new_queries = bool(new_queries and len(new_queries) > 0)

        # Log planning iteration header for initial workflow retrieval
        logger.info(f"[READING ITERATION {current_iteration + 1}/{max_retrieval_iterations}]")
        if had_new_queries:
            queries_str = ", ".join(f'"{q}"' for q in new_queries)
            logger.info(f"  Searching {len(new_queries)} new queries: {queries_str}")

        # Execute workflow search pipeline
        result = await self.query_prep_tool.execute(state, **kwargs)
        state.update(result.output_workflow or {})

        result = await self.selection_tool.execute(state, **kwargs)
        state.update(result.output_workflow or {})

        result = await self.cell_selection_tool.execute(state, **kwargs)
        state.update(result.output_workflow or {})

        if self.mode == "full":
            result = await self.filter_tool.execute(state, **kwargs)
            # Filter tool may return empty output_workflow when no filtering needed
            # In this case, preserve reference_workflow_content from state
            state.update(result.output_workflow or {})

        # Update planning phase state for explicit control flow
        # - Set phase to "workflow_retrieval" to signal we're in retrieval loop
        # - Increment iteration counter
        # - Keep retrieval_queries from selection tool (router will check if empty to exit loop)
        #
        # IMPORTANT: Preserve PERSISTENT fields from state, not just last tool's output.
        # The filter tool may return empty output_workflow when no filtering is needed (all workflows
        # mentioned in tasks), which would lose reference_workflow_content set by cell_selection_tool.
        # Since all sub-tools update state, we read from there for PERSISTENT fields.
        # Mark all current queries as searched (for accurate "new queries" logging)
        all_searched = list(set(searched_queries + retrieval_queries))

        result_with_phase_tracking = ToolResult(
            output_ui=result.output_ui,
            output_type=result.output_type,
            output_workflow={
                # Preserve PERSISTENT fields from state
                "reference_workflow_content": state.get("reference_workflow_content", {}),
                "reference_workflow_percentages": state.get("reference_workflow_percentages", {}),
                "reference_workflow_ids": state.get("reference_workflow_ids", ""),  # For VSCode display
                "excluded_workflows": state.get("excluded_workflows", []),
                "retrieval_queries": state.get("retrieval_queries", []),
                # Track which queries have been searched
                "searched_retrieval_queries": all_searched,
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

    Workflow refinement pipeline:
    - Uses retrieval_queries from plan_tasks tool (NOT from LLM selection)
    - Runs: query_prep → selection_only → cell_selection
    - Clears retrieval_queries after execution
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

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
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
        current_iteration = state.get("task_planning_iteration", -1)
        workflow_iteration = state.get("workflow_retrieval_iteration", 0)
        max_retrieval_iterations = state.get("max_workflow_retrieval_iterations", 2)
        retrieval_queries = state.get("retrieval_queries", [])
        searched_queries = state.get("searched_retrieval_queries", [])

        # Calculate actually NEW queries (not yet searched)
        new_queries = [q for q in retrieval_queries if q not in searched_queries]
        had_new_queries = bool(new_queries and len(new_queries) > 0)

        # Log workflow retrieval as planning iterations
        # Initial workflow retrieval (before task generation) counts as reading iterations (max configurable)
        # Task generation iterations are logged separately with their own max
        if current_iteration == -1:
            # Initial workflow retrieval phase - count as reading iterations
            logger.info(f"[READING ITERATION {workflow_iteration + 1}/{max_retrieval_iterations}]")
            if had_new_queries:
                queries_str = ", ".join(f'"{q}"' for q in new_queries)
                logger.info(f"  Searching {len(new_queries)} new queries: {queries_str}")
            else:
                logger.info(f"  Re-evaluating existing workflows (protecting cited, removing uncited)")
        elif had_new_queries:
            # Workflow refinement during task generation iterations (indented under planning iteration)
            queries_str = ", ".join(f'"{q}"' for q in new_queries)
            logger.info(f"  Searching {len(new_queries)} new queries: {queries_str}")
        else:
            logger.info(f"  Re-evaluating existing workflows (protecting cited, removing uncited)")

        # Execute workflow refinement pipeline
        result = await self.query_prep_tool.execute(state, **kwargs)
        state.update(result.output_workflow or {})

        result = await self.selection_tool.execute(state, **kwargs)
        state.update(result.output_workflow or {})

        result = await self.cell_selection_tool.execute(state, **kwargs)
        state.update(result.output_workflow or {})

        # Mark all current queries as searched (for accurate "new queries" logging)
        all_searched = list(set(searched_queries + retrieval_queries))

        # Update state to match old orchestrator behavior
        # IMPORTANT: Preserve PERSISTENT fields from state (same issue as SearchWorkflowsTool)
        result_with_state = ToolResult(
            output_ui=result.output_ui,
            output_type=result.output_type,
            output_workflow={
                # Preserve PERSISTENT fields from state
                "reference_workflow_content": state.get("reference_workflow_content", {}),
                "reference_workflow_percentages": state.get("reference_workflow_percentages", {}),
                "reference_workflow_ids": state.get("reference_workflow_ids", ""),  # For VSCode display
                "excluded_workflows": state.get("excluded_workflows", []),
                # Clear retrieval queries after refinement
                "retrieval_queries": [],
                # Track which queries have been searched
                "searched_retrieval_queries": all_searched,
                # Track whether there were queries BEFORE clearing (router checks this)
                "had_retrieval_queries_before_refinement": had_new_queries,
                "planning_phase": "workflow_refinement",  # Signal that workflow refinement is complete
                # Router will check this and route back to task_list_generation
                # NOTE: Router will increment task_planning_iteration when routing back to task_list_generation
            }
        )

        return result_with_state
