"""Reference workflow selection tools and schemas.

This module provides schemas and tools for selecting reference workflows
from a collection of candidate notebooks, with support for both initial
selection and iterative refinement.
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
# Schemas
# =============================================================================

class ReferenceWorkflowSelection(BaseModel):
    """Schema for reference workflow retrieval output."""
    model_config = ConfigDict(extra='forbid')
    selected_notebooks: List[str] = Field(description="List of selected notebook IDs")
    retrieval_queries: List[str] = Field(description="List of queries for further retrieval", default_factory=list)

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "selected_notebooks": ["scverse/scanpy-tutorials/pbmc3k_tutorial.ipynb", "scverse/decoupler-tutorials/rna_sc.ipynb"],
    "retrieval_queries": ["classify cell types"],
}

Ensure all JSON is valid and complete."""


class ReferenceWorkflowSelectionOnly(BaseModel):
    """Schema for reference workflow retrieval output."""
    model_config = ConfigDict(extra='forbid')
    selected_notebooks: List[str] = Field(description="List of selected notebook IDs")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "selected_notebooks": ["scverse/scanpy-tutorials/pbmc3k_tutorial.ipynb", "scverse/decoupler-tutorials/rna_sc.ipynb"],
}

Ensure all JSON is valid and complete."""


# =============================================================================
# Tools
# =============================================================================

class ReferenceWorkflowSelectionTool(StructuredPromptTool):
    """Selects reference workflows from putative summaries and generates new retrieval queries.

    This tool is used in initial planning to select workflows and optionally generate queries
    for iterative refinement. It processes summaries from ReferenceWorkflowQueryPreparationTool.

    ID Handling:
        - LLM sees full IDs in summaries: "scverse/scanpy-tutorials/pbmc3k.ipynb"
        - LLM returns full IDs in selected_notebooks
        - Tool converts full IDs → internal IDs for storage operations

    Context Inputs:
        - putative_reference_workflow_summaries: Candidate workflows from semantic search

    Context Updates:
        - reference_workflow_ids: Comma-separated full IDs for UI display
        - reference_workflow_internal_ids: List of internal IDs for storage
        - reference_workflow_content: Formatted notebook content (not cell-filtered yet)
        - retrieval_queries: New queries for next iteration (if iterative mode)

    UI Output:
        - Comma-separated list of selected workflow full IDs
    """

    def __init__(self, scenario: PromptScenario, llm_interface: 'LLMInterface',
                 notebook_selector: 'NotebookSelector'):
        """Initialize with standard prompt tool parameters and notebook components.

        Args:
            scenario: Prompt scenario for this tool
            llm_interface: LLM interface for prompt execution
            notebook_selector: Notebook selection system
        """
        super().__init__(
            name="reference_workflow_selection",
            scenario=scenario,
            llm_interface=llm_interface
        )
        self.selector = notebook_selector

    def _process_structured_result(self, structured_result: ReferenceWorkflowSelection, state: 'KaiState') -> ToolResult:
        # Convert LLM's full IDs to internal IDs
        # LLM sees: "scverse/scanpy-tutorials/pbmc3k.ipynb"
        # Storage needs: "scverse_scanpy_tutorials_pbmc3k"
        internal_ids = []
        for notebook_id in structured_result.selected_notebooks:
            # Check if already internal format (no slashes/dots) or needs conversion
            if "/" in notebook_id or ".ipynb" in notebook_id:
                # Convert full path to internal ID
                internal_id = notebook_id.replace("/", "_").replace("-", "_").replace(".ipynb", "")
                internal_ids.append(internal_id)
            else:
                # Already internal format
                internal_ids.append(notebook_id)

        # Log selected workflows for debugging (internal IDs only, full paths logged below)
        logger.debug(f"📓 Selected {len(structured_result.selected_notebooks)} reference workflows (internal IDs): {internal_ids}")

        # Convert structured result to expected format
        selection_data = {"selected_notebooks": internal_ids}

        # Add notebook content to selection data
        selected_notebooks = self.selector.get_selected_notebook_content(
            internal_ids
        )
        selection_data["notebook_content"] = selected_notebooks

        # Format as dict {internal_id: content_string}
        rag_summary_dict = self.selector.format_notebook_context_dict(selection_data)

        # Get selected notebook IDs as full paths (org/repo/filename.ipynb)
        full_ids = []
        for notebook_id, notebook_data in selected_notebooks.items():
            metadata = notebook_data.get("metadata", {})
            full_id = f"{metadata.get('source_repository', 'unknown')}/{metadata.get('workflow_filename', notebook_id)}"
            full_ids.append(full_id)
        selected_notebook_ids = ", ".join(full_ids)

        # Don't log here - cell selection tool will log the final results with percentages
        logger.debug(f"📚 Reference workflows (full paths): {full_ids}")

        vscode_response = {"text": selected_notebook_ids}
        if structured_result.retrieval_queries and len(structured_result.retrieval_queries) > 0:
            vscode_response["agent_notification"] = "\n".join(
                ["Reading up on:"] + structured_result.retrieval_queries
            )

        return ToolResult(
            output_ui=vscode_response,
            output_workflow={
                "reference_workflow_ids": selected_notebook_ids,
                "reference_workflow_content": rag_summary_dict,
                "retrieval_queries": structured_result.retrieval_queries,
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        )


class ReferenceWorkflowSelectionOnlyTool(StructuredPromptTool):
    """Updates workflow selection without generating new retrieval queries.

    Used in task iteration loops where retrieval queries come from task generation.
    Updates the workflow selection based on new candidates while preserving workflows
    cited in the task list.

    ID Handling:
        - Same as ReferenceWorkflowSelectionTool
        - Automatic conversion: full IDs ↔ internal IDs

    Context Inputs:
        - putative_reference_workflow_summaries: Candidates from query preparation
        - task_list: Current tasks (workflows cited here are protected from removal)
        - reference_workflow_content: Current workflows (shown to LLM for context)

    Context Updates:
        - reference_workflow_ids: Updated comma-separated full IDs
        - reference_workflow_internal_ids: Updated list of internal IDs
        - reference_workflow_content: Formatted content for new selection (not cell-filtered)

    UI Output:
        - None (cell selection will show final list with percentages)
    """

    def __init__(self, scenario: PromptScenario, llm_interface: 'LLMInterface',
                 notebook_selector: 'NotebookSelector'):
        super().__init__(
            name="reference_workflow_selection_only",
            scenario=scenario,
            llm_interface=llm_interface
        )
        self.selector = notebook_selector

    def _extract_cited_workflows(self, state: 'KaiState') -> set:
        """Extract internal IDs of workflows cited in task list.

        Parses task descriptions looking for citations in format:
        [adapted from: 'org/repo/file.ipynb', cells: X-Y]

        Returns set of internal IDs (e.g., 'org_repo_file')
        """
        import re

        cited_workflows = set()
        task_list = state["task_list"]

        if not task_list or "tasks" not in task_list:
            return cited_workflows

        # Pattern matches: [adapted from: 'path/to/file.ipynb', cells: ...]
        # or [custom from: 'path/to/file.ipynb']
        citation_pattern = r"\[(?:adapted|custom) from: ['\"]([^'\"]+\.ipynb)['\"]"

        for task in task_list["tasks"]:
            task_text = task.get("task", "")
            matches = re.findall(citation_pattern, task_text)

            for full_path in matches:
                # Convert FULL ID to INTERNAL ID
                # "org/repo-name/file.ipynb" -> "org_repo_name_file"
                internal_id = full_path.replace("/", "_").replace("-", "_").replace(".ipynb", "")
                cited_workflows.add(internal_id)

        return cited_workflows

    def _process_structured_result(self, structured_result: ReferenceWorkflowSelectionOnly, state: 'KaiState') -> ToolResult:
        # Convert LLM's full IDs to internal IDs
        # LLM sees: "scverse/scanpy-tutorials/pbmc3k.ipynb"
        # Storage needs: "scverse_scanpy_tutorials_pbmc3k"
        internal_ids = set()
        for notebook_id in structured_result.selected_notebooks:
            # Check if already internal format (no slashes/dots) or needs conversion
            if "/" in notebook_id or ".ipynb" in notebook_id:
                # Convert full path to internal ID
                internal_id = notebook_id.replace("/", "_").replace("-", "_").replace(".ipynb", "")
                internal_ids.add(internal_id)
            else:
                # Already internal format
                internal_ids.add(notebook_id)

        # Defensive filtering: ensure workflows cited in task list are never removed
        # Extract cited workflows from task list (already in internal format)
        cited_workflows = self._extract_cited_workflows(state)

        # Merge LLM selection with cited workflows (cited workflows take priority)
        missing_cited = cited_workflows - internal_ids
        if missing_cited:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"LLM removed {len(missing_cited)} cited workflow(s) from selection. Re-adding: {missing_cited}")
        internal_ids.update(cited_workflows)  # Add any cited workflows LLM forgot

        # Convert structured result to expected format
        selection_data = {"selected_notebooks": list(internal_ids)}

        # Add notebook content to selection data
        selected_notebooks = self.selector.get_selected_notebook_content(
            list(internal_ids)
        )
        selection_data["notebook_content"] = selected_notebooks

        # Format as dict {internal_id: content_string}
        rag_summary_dict = self.selector.format_notebook_context_dict(selection_data)

        # Get selected notebook IDs as full paths (org/repo/filename.ipynb)
        full_ids = []
        for notebook_id, notebook_data in selected_notebooks.items():
            metadata = notebook_data.get("metadata", {})
            full_id = f"{metadata.get('source_repository', 'unknown')}/{metadata.get('workflow_filename', notebook_id)}"
            full_ids.append(full_id)
        selected_notebook_ids = ", ".join(full_ids)

        # No UI output - cell selection will show the final list with percentages
        return ToolResult(
            output_ui={},
            output_workflow={
                "reference_workflow_ids": selected_notebook_ids,
                "reference_workflow_content": rag_summary_dict,  # Dict format
            },
            output_type=ToolOutputType.NO_OUTPUT
        )
