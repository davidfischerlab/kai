"""Reference workflow cell selection tool and schema.

This module provides schema and tool for selecting relevant cells from
reference workflows, optimized to reuse filtered content for unchanged workflows.
"""

from typing import List, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict

from kai.core.prompt_manager import PromptScenario
from kai.core.tools.base import ToolResult, ToolOutputType
from kai.core.tools.prompt_base import StructuredPromptTool
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface
    from kai.retrieval.workflow_summaries.notebook_selector import NotebookSelector

logger = setup_logger(__name__)


# =============================================================================
# Schema
# =============================================================================

class ReferenceWorkflowCellSelection(BaseModel):
    """Schema for selecting relevant cells from a single reference workflow."""
    model_config = ConfigDict(extra='forbid')
    selected_cells: List[int] = Field(description="List of cell indices to include from the notebook")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "selected_cells": [0, 2, 5, 10, 15]
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tool
# =============================================================================

class ReferenceWorkflowCellSelectionTool(StructuredPromptTool):
    """Tool for selecting relevant cells from reference workflows.

    Optimized to reuse filtered content for unchanged workflow IDs, running LLM only on new ones.

    Storage Format:
        - reference_workflow_content: Dict {internal_id: filtered_content_string}
        - Selection tools set unfiltered content, this tool filters cells
        - Kept IDs: reuse previous filtered content from dict
        - New IDs: run LLM cell selection, add to dict

    Change Detection:
        - Derives previous IDs from context["reference_workflow_percentages"]
        - Calculates: kept_ids (reuse content), new_ids (run LLM)
        - If no new IDs: returns UI message with existing percentages (replaces loading)

    Processing Flow:
        1. Get current IDs from context
        2. Derive previous IDs from percentages
        3. Calculate kept vs new
        4. If no new: return UI with existing percentages
        5. If new: run LLM on new IDs only
        6. Merge: kept content + new filtered content

    Performance:
        - O(new_workflows) LLM calls instead of O(total_workflows)
        - Example: 5 existing + 1 new = 1 LLM call instead of 6 (83% reduction)
        - Dict lookup O(1) for reusing kept content

    Context Updates:
        - reference_workflow_content: {internal_id: filtered_string} for all current IDs
        - reference_workflow_percentages: {full_id: percentage} for all workflows
        - excluded_workflows: Internal IDs of workflows with 0 cells selected

    UI Output:
        - Always sends message to replace loading state
        - Format: "📚 org/repo/file.ipynb (50% of file)" for each workflow
    """

    def __init__(self, scenario: PromptScenario, llm_interface: 'LLMInterface',
                 notebook_selector: 'NotebookSelector'):
        super().__init__(
            name="reference_workflow_cell_selection",
            scenario=scenario,
            llm_interface=llm_interface
        )
        self.selector = notebook_selector

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Execute cell selection for each selected notebook.

        Reuses filtered content for kept workflow IDs, runs LLM only on new IDs.
        """
        # Get current content dict
        current_content_dict = state.get("reference_workflow_content", {})
        if not current_content_dict:
            return ToolResult(
                output_ui={},
                output_workflow={},
                output_type=ToolOutputType.NO_OUTPUT
            )

        # Derive current internal IDs from dict keys
        current_ids = set(current_content_dict.keys())
        previous_percentages = state.get("reference_workflow_percentages", {})

        # Build mapping from internal ID to full ID for current workflows
        all_notebooks = self.selector.get_selected_notebook_content(list(current_ids))
        internal_to_full = {}
        for internal_id, notebook_data in all_notebooks.items():
            metadata = notebook_data.get("metadata", {})
            full_id = f"{metadata.get('source_repository')}/{metadata.get('workflow_filename')}"
            internal_to_full[internal_id] = full_id

        # Derive previous IDs from percentages
        full_to_internal = {v: k for k, v in internal_to_full.items()}
        previous_ids = set()
        for full_id in previous_percentages.keys():
            internal_id = full_to_internal.get(full_id)
            if internal_id:
                previous_ids.add(internal_id)

        # Calculate changes - preserve order from current_content_dict
        kept_ids = current_ids & previous_ids  # Workflows to reuse
        # Preserve order by filtering list instead of set difference
        new_ids = [nid for nid in current_content_dict.keys() if nid not in previous_ids]

        # If no new workflows, just send UI to replace loading state
        if not new_ids:
            results = []
            for full_id, percentage in previous_percentages.items():
                results.append((full_id, percentage))

            if results:
                results.sort(key=lambda x: x[0])
                bullet_list = "\n".join([f"📚 {full_id} (considering {percentage:.0f}% of file)" for full_id, percentage in results])
                return ToolResult(
                    output_ui={"text": bullet_list},
                    output_workflow={},  # No updates
                    output_type=ToolOutputType.REFERENCE_WORKFLOWS
                )
            else:
                return ToolResult(
                    output_ui={},
                    output_workflow={},
                    output_type=ToolOutputType.NO_OUTPUT
                )

        # Run LLM cell selection ONLY on NEW workflows
        selected_ranges = {}
        new_notebooks = {nid: all_notebooks[nid] for nid in new_ids if nid in all_notebooks}

        for notebook_id, notebook_data in new_notebooks.items():
            # Temporarily store this single notebook in state for the prompt
            state["current_notebook_for_cell_selection"] = {
                "notebook_id": notebook_id,
                "notebook_data": notebook_data
            }

            # Use the parent's LLM call mechanism
            structured_result = await self._call_llm_structured(state, **kwargs)

            # Validate selected cells
            actual_cell_indices = {cell.get("order") for cell in notebook_data.get("cells", [])}
            valid_cells = [idx for idx in structured_result.selected_cells if idx in actual_cell_indices]
            selected_ranges[notebook_id] = sorted(set(valid_cells))

        # Clean up temporary state
        state.pop("current_notebook_for_cell_selection", None)

        # Format NEW workflows with selected cell ranges
        new_content_dict = {}
        if new_notebooks:
            selection_data = {
                "selected_notebooks": list(new_notebooks.keys()),
                "notebook_content": new_notebooks,
            }
            new_content_dict = self.selector.format_notebook_context_dict(selection_data, selected_ranges=selected_ranges)

        # Merge: kept content from previous + new content from LLM
        merged_content_dict = {}
        for internal_id in kept_ids:
            if internal_id in current_content_dict:
                merged_content_dict[internal_id] = current_content_dict[internal_id]
        merged_content_dict.update(new_content_dict)

        # Build percentages dict combining kept and new
        percentages_dict = {}
        results = []

        # Add kept workflows with their previous percentages
        for internal_id in kept_ids:
            full_id = internal_to_full.get(internal_id)
            if full_id and full_id in previous_percentages:
                percentage = previous_percentages[full_id]
                percentages_dict[full_id] = percentage
                results.append((full_id, percentage))

        # Add new workflows with calculated percentages
        excluded_workflows = []
        for internal_id in new_ids:
            if internal_id in all_notebooks:
                full_id = internal_to_full.get(internal_id)
                notebook_data = all_notebooks[internal_id]
                total_cells = len(notebook_data.get("cells", []))
                selected_cells = len(selected_ranges.get(internal_id, []))

                if total_cells > 0:
                    percentage = min((selected_cells / total_cells * 100), 100)
                else:
                    percentage = 0

                percentages_dict[full_id] = percentage
                results.append((full_id, percentage))

                # Track empty selections
                if selected_cells == 0:
                    excluded_workflows.append(internal_id)

        # Sort and format UI message
        results.sort(key=lambda x: x[0])
        bullet_list = "\n".join([f"📚 {full_id} (considering {percentage:.0f}% of file)" for full_id, percentage in results])

        # Log cell selection results for production visibility
        num_new = len(new_content_dict)
        num_kept = len(kept_ids)
        if num_new > 0 and num_kept > 0:
            logger.info(f"  Cell selection: {num_new} new + {num_kept} existing = {len(results)} total workflows:")
        elif num_new > 0:
            logger.info(f"  Cell selection results for {len(results)} workflows:")
        else:
            logger.info(f"  Kept {len(results)} existing workflows:")

        for full_id, percentage in results:
            logger.info(f"     {full_id}: {percentage:.0f}% of cells selected")
        # Don't log excluded workflows - it's normal to exclude workflows with 0 cells
        if excluded_workflows:
            logger.debug(f"Excluded {len(excluded_workflows)} workflows with 0 cells selected: {excluded_workflows}")

        return ToolResult(
            output_ui={"text": bullet_list},
            output_workflow={
                "reference_workflow_content": merged_content_dict,  # Dict format
                "reference_workflow_percentages": percentages_dict,
                "excluded_workflows": excluded_workflows,
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        )

    async def _call_llm_structured(self, state: 'KaiState', **kwargs) -> ReferenceWorkflowCellSelection:
        """Call LLM and parse structured output with logging."""
        # Get notebook ID for logging context
        notebook_info = state.get("current_notebook_for_cell_selection", {})
        notebook_id = notebook_info.get("notebook_id", "unknown")

        # Build prompt using parent's mechanism
        use_json_prompting = not self.llm_provider.use_structured_output
        system_prompt, user_prompt = self.prompt_manager.generate_prompt(
            state,
            self.scenario,
            structured_output=not use_json_prompting
        )

        # Log prompt (once per notebook)
        log_filename = self._log_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            state=state
        )

        # Call LLM with structured output
        response = await self.llm_provider.generate_structured(
            prompt=user_prompt,
            schema=ReferenceWorkflowCellSelection,
            system_prompt=system_prompt,
            task_type=self._get_task_type(),
            tool_name=f"{self.name}_{notebook_id}",
            **kwargs
        )

        # Log result (once per notebook)
        if log_filename:
            # Create a minimal ToolResult for logging
            temp_result = ToolResult(
                output_ui={"notebook_id": notebook_id},
                output_workflow={
                    "selected_cells": response.selected_cells,
                    "cell_count": len(response.selected_cells)
                },
                output_type=ToolOutputType.NO_OUTPUT
            )
            self._log_result(temp_result, log_filename)

        return response

    def _process_structured_result(self, structured_result: ReferenceWorkflowCellSelection,
                                   state: 'KaiState') -> ToolResult:
        """Required by BasePromptTool but not used here.

        This tool overrides execute() directly because it needs to make
        multiple LLM calls (one per notebook) rather than a single call.
        """
        raise NotImplementedError("Use execute() instead")
