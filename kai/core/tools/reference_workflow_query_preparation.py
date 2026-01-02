"""Workflow utility tools.

This module provides deterministic tools for managing reference workflows,
including query preparation and filtering.
"""

from typing import TYPE_CHECKING, Optional

from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag

logger = setup_logger(__name__)


class ReferenceWorkflowQueryPreparationTool(BaseTool):
    """Deterministic tool for preparing reference workflow queries.

    Extracts queries from execution context and performs semantic search
    to populate putative_reference_workflow_summaries for the LLM prompt.
    This replaces the _modify_user_query functionality in ReferenceWorkflowSelectionTool.

    Note: retrieval_queries can be reset each iteration - cumulalation of selected reference
    candidates happens here at the level of putative_reference_workflow_summaries.

    **UI Returns:**
    - `output_type`: NO_OUTPUT - internal query preparation tool

    **Workflow Returns:**
    - `putative_reference_workflow_summaries`: Concatenated summaries for LLM prompt
    """

    def __init__(self, summary_search: Optional['WorkflowSummaryRag'] = None):
        super().__init__("reference_workflow_query_preparation")
        self.summary_search = summary_search

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Prepare semantic search results for reference workflow selection."""
        # Extract queries from context
        has_specific_retrieval_query = len(state.get("retrieval_queries", [])) > 0

        if has_specific_retrieval_query:
            queries = state["retrieval_queries"]
            # If semantic search is available, get top candidates
            candidates = []
            if self.summary_search:
                for q in queries:
                    # Top 10 hits for each query:
                    candidates.extend(
                        self.summary_search.search_summaries(q, n_results=10)
                    )

            existing_summaries = state.get(
                "putative_reference_workflow_summaries", ""
            )
            # Build enhanced prompt with semantic candidates
            summary_list = []
            for result in candidates:
                notebook_id = result["notebook_id"]
                # Check that this notebook ID is not already in existing summaries to avoid duplicates:
                if notebook_id not in existing_summaries:
                    summary = result["summary"]
                    metadata = result["metadata"]
                    score = result["similarity_score"]
                    # Use full notebook ID if available, otherwise fall back to internal ID
                    display_id = metadata.get('full_notebook_id', notebook_id)

                    summary_list.append(
                        f"> Notebook ID: '{display_id}' (similarity: {score:.2f})\n"
                        f"Repository: {metadata.get('source_repository', 'unknown')}\n"
                        f">> Summary:\n{summary}\n\n"
                    )
            summary_str = "".join(summary_list)
            # Update execution context with preselection results
            output_workflow = {
                "putative_reference_workflow_summaries": existing_summaries + summary_str
            }
        else:
            output_workflow = {}

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow=output_workflow
        )


class FilterUnusedReferenceWorkflowsTool(BaseTool):
    """Filters reference workflows to only keep those mentioned in the task list.

    Removes any selected reference workflow IDs and their content that are not
    explicitly cited in task descriptions via hard ID matching.
    Should run at the end of initial planning to clean up unused references.

    **UI Returns:**
    - Dict with "text" showing filtered workflows with percentages
    - `output_type`: REFERENCE_WORKFLOWS

    **Workflow Returns:**
    - `reference_workflow_ids`: Comma-separated filtered full IDs
    - `reference_workflow_content`: Dict with only cited workflows
    """

    def __init__(self):
        super().__init__("filter_unused_reference_workflows")

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Filter reference workflows to only those mentioned in task list."""
        # Get current reference workflow IDs and content
        reference_workflow_ids = state.get("reference_workflow_ids", "")
        reference_workflow_content = state.get("reference_workflow_content", {})
        percentages = state.get("reference_workflow_percentages", {})

        # Get task list
        task_list = state["task_list"]

        if not reference_workflow_ids or not task_list:
            # Nothing to filter
            return ToolResult(
                output_ui={},
                output_type=ToolOutputType.NO_OUTPUT,
                output_workflow={}
            )

        # Build mapping: internal_id -> full_id from content dict
        internal_to_full = {}
        for internal_id, content in reference_workflow_content.items():
            # Extract full ID from first line: "> Notebook ID: full_id"
            first_line = content.split('\n')[0] if content else ""
            if first_line.startswith("> Notebook ID:"):
                full_id = first_line.replace("> Notebook ID:", "").strip()
                internal_to_full[internal_id] = full_id

        # Parse full notebook IDs from reference_workflow_ids string
        full_notebook_ids = [id.strip() for id in reference_workflow_ids.split(",")]

        # Extract all notebook IDs mentioned in task descriptions
        mentioned_full_ids = set()
        for task in task_list.get("tasks", []):
            task_desc = task.get("task", "")
            # Look for citations like: [adapted from: 'notebook_id', cells: ...]
            for full_id in full_notebook_ids:
                if full_id in task_desc:
                    mentioned_full_ids.add(full_id)

        # If all notebooks are mentioned, no filtering needed
        if len(mentioned_full_ids) == len(full_notebook_ids):
            return ToolResult(
                output_ui={},
                output_type=ToolOutputType.NO_OUTPUT,
                output_workflow={}
            )

        # Build reverse mapping
        full_to_internal = {v: k for k, v in internal_to_full.items()}

        # Filter content dict - keep only mentioned workflows
        filtered_content_dict = {}
        filtered_full_ids = []
        for full_id in mentioned_full_ids:
            internal_id = full_to_internal.get(full_id)
            if internal_id and internal_id in reference_workflow_content:
                filtered_content_dict[internal_id] = reference_workflow_content[internal_id]
                filtered_full_ids.append(full_id)

        # Sort for consistent ordering
        filtered_full_ids.sort()

        # Rebuild reference_workflow_ids string
        filtered_ids_str = ", ".join(filtered_full_ids)

        # Build display message with filtered notebooks
        bullet_list_items = []
        for full_id in filtered_full_ids:
            # Get percentage from stored dict
            percentage = percentages.get(full_id, 0)
            if percentage > 0:
                bullet_list_items.append(f"📚 {full_id} (considering {percentage:.0f}% of file)")
            else:
                bullet_list_items.append(f"📚 {full_id}")

        # Format display message
        bullet_list = "\n".join(bullet_list_items)

        # Return filtered data with updated display
        return ToolResult(
            output_ui={"text": bullet_list},
            output_workflow={
                "reference_workflow_ids": filtered_ids_str,
                "reference_workflow_content": filtered_content_dict
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        )
