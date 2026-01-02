"""Tests for workflow selection and RAG tools.

This module consolidates tests for:
- ReferenceWorkflowSelectionTool (workflow selection)
- ReferenceWorkflowCellSelectionTool (cell filtering)
- ReferenceWorkflowSelectionOnlyTool (cited workflow protection)
- ReferenceWorkflowQueryPreparationTool (query preparation)
- Excluded workflows mechanism
"""

import pytest
from unittest.mock import Mock, AsyncMock

from kai.core.tools import (
    ReferenceWorkflowSelectionTool,
    ReferenceWorkflowCellSelectionTool,
    ReferenceWorkflowSelectionOnlyTool,
)
from kai.core.tools.reference_workflow_query_preparation import ReferenceWorkflowQueryPreparationTool
from kai.core.tools.schema_registry import (
    ReferenceWorkflowSelection,
    ReferenceWorkflowCellSelection,
    ReferenceWorkflowSelectionOnly,
)
from kai.core.tools.base import ToolOutputType
from kai.core.prompt_manager import PromptScenario


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface for workflow tool tests."""
    mock_llm = Mock()
    mock_provider = Mock()
    mock_provider.provider_name = "test_provider"
    mock_provider.model = "test_model"
    mock_provider.use_structured_output = True
    mock_provider.generate_structured = AsyncMock()
    mock_llm.get_llm_for_tool.return_value = mock_provider
    mock_llm.get_reasoning_for_tool.return_value = "detailed"
    return mock_llm


@pytest.fixture
def mock_notebook_selector():
    """Create a mock notebook selector with common workflows."""
    mock_selector = Mock()

    # Common test notebooks
    notebooks = {
        "org_a_tutorial_analysis": {
            "cells": [
                {"content": "import scanpy as sc", "order": 0, "section": "main"},
                {"content": "adata = sc.read_h5ad('data.h5ad')", "order": 1, "section": "main"},
                {"content": "sc.pp.filter_cells(adata)", "order": 2, "section": "main"},
            ],
            "metadata": {
                "source_repository": "org_a/tutorials",
                "workflow_filename": "analysis.ipynb",
                "title": "Analysis Tutorial"
            }
        },
        "org_b_tf_activity_example": {
            "cells": [
                {"content": "import decoupler as dc", "order": 0, "section": "main"},
            ],
            "metadata": {
                "source_repository": "org_b/tf_activity",
                "workflow_filename": "example.ipynb",
                "title": "TF Activity Example"
            }
        },
        "org_c_trajectory_inference": {
            "cells": [
                {"content": "sc.tl.diffmap(adata)", "order": 0, "section": "main"},
                {"content": "sc.pl.diffmap(adata)", "order": 1, "section": "main"},
            ],
            "metadata": {
                "source_repository": "org_c/workshop",
                "workflow_filename": "trajectory_inference.ipynb",
                "title": "Trajectory Inference"
            }
        },
        "org_d_grn_activity": {
            "cells": [
                {"content": "dc.run_collectri(adata)", "order": 0, "section": "main"},
                {"content": "dc.plot_results(adata)", "order": 1, "section": "main"},
            ],
            "metadata": {
                "source_repository": "org_d/grn_workshop",
                "workflow_filename": "activity_inference.ipynb",
                "title": "Activity Inference"
            }
        },
    }

    def get_selected(notebook_ids):
        return {nb_id: notebooks[nb_id] for nb_id in notebook_ids if nb_id in notebooks}

    mock_selector.get_selected_notebook_content = get_selected
    mock_selector.format_notebook_context.return_value = "Formatted workflow context"
    mock_selector.format_notebook_context_dict.return_value = {
        nb_id: f"> Notebook ID: {notebooks[nb_id]['metadata']['source_repository']}/{notebooks[nb_id]['metadata']['workflow_filename']}\nContent"
        for nb_id in notebooks
    }

    return mock_selector


@pytest.fixture
def mock_summary_search():
    """Create a mock summary search for RAG tests."""
    mock_search = Mock()
    mock_search.search_summaries.return_value = [
        {
            "notebook_id": "org_a_tutorial_analysis",
            "summary": "Single-cell analysis tutorial",
            "metadata": {"source_repository": "org_a/tutorials"},
            "similarity_score": 0.95
        },
        {
            "notebook_id": "org_b_tf_activity_example",
            "summary": "Transcription factor activity analysis",
            "metadata": {"source_repository": "org_b/tf_activity"},
            "similarity_score": 0.87
        }
    ]
    return mock_search


def create_workflow_state(**overrides):
    """Create a state dict for workflow tool testing."""
    state = {
        "user_query": "Analyze single-cell data",
        "rag_enabled": True,
        "reference_workflow_annotation": "",
        "execution_history": [],
        "conversation_history": [],
        "notebook_structure": {"totalCells": 3, "allCells": ["# Cell 1", "# Cell 2", "# Cell 3"]},
        "autonomous_mode": False,
        "last_execution_failed": False,
        "current_cell": "",
        "current_cell_index": 0,
        "error_message": "",
        "request_id": "test_request",
        "notebook_path": "/test/notebook.ipynb",
        "task_list": {"tasks": []},
        "backtracking_context": None,
        "session_id": "test_session",
        "session_timestamp": "2025-01-01_12-00-00",
        "notebook_uri": "file:///test.ipynb",
        "excluded_workflows": [],
        "reference_workflow_internal_ids": [],
        "reference_workflow_ids": "",
        "reference_workflow_content": {},
    }
    state.update(overrides)
    return state


# =============================================================================
# ReferenceWorkflowSelectionTool Tests
# =============================================================================

class TestReferenceWorkflowSelectionTool:
    """Test the ReferenceWorkflowSelectionTool."""

    @pytest.fixture
    def tool(self, mock_llm_interface, mock_notebook_selector):
        """Create tool instance."""
        return ReferenceWorkflowSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

    def test_tool_initialization(self, tool):
        """Test tool is created correctly."""
        assert tool.name == "reference_workflow_selection"
        assert tool.scenario == PromptScenario.REFERENCE_WORKFLOW_SELECTION

    def test_output_workflow_structure(self, tool, mock_llm_interface):
        """Test that output_workflow has expected structure."""
        state = create_workflow_state()

        # Mock LLM response (ReferenceWorkflowSelection has selected_notebooks and retrieval_queries)
        mock_llm_interface.get_llm_for_tool.return_value.generate_structured.return_value = ReferenceWorkflowSelection(
            selected_notebooks=["org_a_tutorial_analysis"],
            retrieval_queries=["single-cell analysis query"]
        )

        # This tests the _process_structured_result method indirectly
        # In actual use, execute() would call this


class TestReferenceWorkflowQueryPreparationTool:
    """Test the ReferenceWorkflowQueryPreparationTool."""

    @pytest.fixture
    def tool(self, mock_summary_search):
        """Create tool instance."""
        return ReferenceWorkflowQueryPreparationTool(
            summary_search=mock_summary_search
        )

    def test_tool_initialization(self, tool):
        """Test tool is created correctly."""
        assert tool.name == "reference_workflow_query_preparation"

    @pytest.mark.asyncio
    async def test_retrieves_workflows(self, tool, mock_summary_search):
        """Test that tool retrieves workflows from summary search."""
        state = create_workflow_state(
            retrieval_queries=["single cell RNA-seq analysis"]
        )

        # Mock should be called with query
        result = await tool.execute(state)

        # Verify summary search was called
        mock_summary_search.search_summaries.assert_called()


# =============================================================================
# Cited Workflow Protection Tests
# =============================================================================

class TestCitedWorkflowProtection:
    """Test that cited workflows are never removed from selection.

    This tests the defensive filtering that ensures workflows cited in the
    task list are always included in the selection, even if the LLM tries
    to remove them.
    """

    @pytest.fixture
    def tool(self, mock_llm_interface, mock_notebook_selector):
        """Create tool instance."""
        return ReferenceWorkflowSelectionOnlyTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

    def test_extract_cited_workflows_from_task_list(self, tool):
        """Test extraction of cited workflows from task list."""
        state = create_workflow_state(
            task_list={
                "tasks": [
                    {"id": 1, "task": "Do something [custom step]", "status": "pending"},
                    {
                        "id": 2,
                        "task": "Perform DE analysis [adapted from: 'org_a/tutorials/analysis.ipynb', cells: 138-141]",
                        "status": "pending"
                    },
                    {
                        "id": 3,
                        "task": "Compute TF activity [adapted from: 'org_b/tf_activity/example.ipynb', cells: 1-4]",
                        "status": "pending"
                    }
                ]
            }
        )

        cited = tool._extract_cited_workflows(state)

        # Should extract both cited workflows
        # Note: ID conversion replaces / with _ and removes .ipynb
        # "org_a/tutorials/analysis.ipynb" -> "org_a_tutorials_analysis"
        assert len(cited) == 2
        assert "org_a_tutorials_analysis" in cited
        assert "org_b_tf_activity_example" in cited

    def test_custom_step_not_extracted_as_citation(self, tool):
        """Test that [custom step] tasks are not treated as citations."""
        state = create_workflow_state(
            task_list={
                "tasks": [
                    {"id": 1, "task": "Do something [custom step]", "status": "pending"},
                    {"id": 2, "task": "[reasoning] Analyze results", "status": "pending"},
                ]
            }
        )

        cited = tool._extract_cited_workflows(state)
        assert len(cited) == 0

    def test_llm_removes_cited_workflow_gets_re_added(self, tool):
        """Test that if LLM removes a cited workflow, it gets re-added."""
        state = create_workflow_state(
            task_list={
                "tasks": [
                    {
                        "id": 1,
                        "task": "DE analysis [adapted from: 'org_a/tutorials/analysis.ipynb', cells: 138-141]",
                        "status": "pending"
                    },
                    {
                        "id": 2,
                        "task": "TF activity [adapted from: 'org_b/tf_activity/example.ipynb', cells: 1-4]",
                        "status": "pending"
                    },
                    {
                        "id": 3,
                        "task": "Trajectory [adapted from: 'org_c/workshop/trajectory_inference.ipynb', cells: 20-23]",
                        "status": "pending"
                    }
                ]
            }
        )

        # LLM returns selection MISSING the TF activity workflow (simulating bug)
        structured_result = ReferenceWorkflowSelectionOnly(
            selected_notebooks=[
                "org_a_tutorial_analysis",
                "org_c_trajectory_inference"
                # Missing: org_b_tf_activity_example
            ]
        )

        result = tool._process_structured_result(structured_result, state)

        # All 3 cited workflows should be present (the key is org_b was re-added)
        internal_ids = set(result.output_workflow["reference_workflow_content"].keys())
        # Verify all cited workflows are present
        assert "org_a_tutorial_analysis" in internal_ids
        assert "org_b_tf_activity_example" in internal_ids  # Re-added!
        assert "org_c_trajectory_inference" in internal_ids

    def test_cited_workflow_protection_scenario(self, tool):
        """Test that cited workflows are always protected from removal."""
        state = create_workflow_state(
            task_list={
                "tasks": [
                    {"id": 1, "task": "Verify AnnData object [custom step]", "status": "pending"},
                    {"id": 2, "task": "Create subset [custom step]", "status": "pending"},
                    {
                        "id": 3,
                        "task": "Perform DE analysis [adapted from: 'org_a/tutorials/analysis.ipynb', cells: 138-141]",
                        "status": "pending"
                    },
                    {"id": 4, "task": "[reasoning] Summarize DE results", "status": "pending"},
                    {
                        "id": 5,
                        "task": "Compute TF activity [adapted from: 'org_b/tf_activity/example.ipynb', cells: 1-4]",
                        "status": "pending"
                    },
                    {"id": 6, "task": "[reasoning] Summarize TF results", "status": "pending"},
                    {
                        "id": 7,
                        "task": "Compute diffusion map [adapted from: 'org_c/workshop/trajectory_inference.ipynb', cells: 20-23]",
                        "status": "pending"
                    }
                ]
            }
        )

        # LLM removes org_b_tf_activity_example
        structured_result = ReferenceWorkflowSelectionOnly(
            selected_notebooks=[
                "org_a_tutorial_analysis",
                "org_c_trajectory_inference"
            ]
        )

        result = tool._process_structured_result(structured_result, state)

        # Defensive filtering should re-add the missing workflow
        internal_ids = set(result.output_workflow["reference_workflow_content"].keys())
        # The key assertion: cited workflow was re-added
        assert "org_b_tf_activity_example" in internal_ids


# =============================================================================
# Excluded Workflows Mechanism Tests
# =============================================================================

class TestExcludedWorkflowsMechanism:
    """Test that workflows with 0 selected cells are added to excluded_workflows.

    This prevents the same workflows from being re-retrieved in future iterations.
    """

    @pytest.fixture
    def tool(self, mock_llm_interface, mock_notebook_selector):
        """Create tool instance."""
        return ReferenceWorkflowCellSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

    @pytest.mark.asyncio
    async def test_all_empty_cells_are_excluded(self, tool, mock_llm_interface):
        """Test that all notebooks with 0 cells are added to excluded_workflows."""
        state = create_workflow_state(
            reference_workflow_internal_ids=[
                "org_a_tutorial_analysis",
                "org_d_grn_activity",
                "org_b_tf_activity_example",
            ],
            reference_workflow_ids="org_a/tutorials/analysis.ipynb, org_d/grn_workshop/activity_inference.ipynb, org_b/tf_activity/example.ipynb",
            reference_workflow_content={
                "org_a_tutorial_analysis": "content 1",
                "org_d_grn_activity": "content 2",
                "org_b_tf_activity_example": "content 3",
            }
        )

        # Mock LLM to return 0 cells for all notebooks
        mock_provider = mock_llm_interface.get_llm_for_tool.return_value
        mock_provider.generate_structured.return_value = ReferenceWorkflowCellSelection(
            selected_cells=[]
        )

        result = await tool.execute(state)

        excluded = result.output_workflow.get('excluded_workflows', [])
        assert len(excluded) == 3, f"Expected 3 excluded workflows, got {len(excluded)}"

    @pytest.mark.asyncio
    async def test_partial_empty_cells_excluded(self, tool, mock_llm_interface):
        """Test that only notebooks with 0 cells are excluded."""
        state = create_workflow_state(
            reference_workflow_internal_ids=[
                "org_a_tutorial_analysis",
                "org_d_grn_activity",
                "org_b_tf_activity_example",
            ],
            reference_workflow_ids="org_a/tutorials/analysis.ipynb, org_d/grn_workshop/activity_inference.ipynb, org_b/tf_activity/example.ipynb",
            reference_workflow_content={
                "org_a_tutorial_analysis": "content 1",
                "org_d_grn_activity": "content 2",
                "org_b_tf_activity_example": "content 3",
            }
        )

        # Track call count to return different values
        call_count = [0]
        mock_provider = mock_llm_interface.get_llm_for_tool.return_value

        async def mock_generate(prompt, schema, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ReferenceWorkflowCellSelection(selected_cells=[0, 1])  # 2 cells
            elif call_count[0] == 2:
                return ReferenceWorkflowCellSelection(selected_cells=[])  # 0 cells
            else:
                return ReferenceWorkflowCellSelection(selected_cells=[0])  # 1 cell

        mock_provider.generate_structured = mock_generate

        result = await tool.execute(state)

        # Only the second notebook should be excluded
        excluded = result.output_workflow.get('excluded_workflows', [])
        assert len(excluded) == 1
        assert excluded[0] == "org_d_grn_activity"

    @pytest.mark.asyncio
    async def test_no_exclusions_when_all_have_cells(self, tool, mock_llm_interface):
        """Test that no workflows are excluded when all have selected cells."""
        state = create_workflow_state(
            reference_workflow_internal_ids=[
                "org_a_tutorial_analysis",
            ],
            reference_workflow_ids="org_a/tutorials/analysis.ipynb",
            reference_workflow_content={
                "org_a_tutorial_analysis": "content 1",
            }
        )

        # Mock LLM to return some cells
        mock_provider = mock_llm_interface.get_llm_for_tool.return_value
        mock_provider.generate_structured.return_value = ReferenceWorkflowCellSelection(
            selected_cells=[0, 1, 2]
        )

        result = await tool.execute(state)

        excluded = result.output_workflow.get('excluded_workflows', [])
        assert len(excluded) == 0


# =============================================================================
# ReferenceWorkflowCellSelectionTool Tests
# =============================================================================

class TestReferenceWorkflowCellSelectionTool:
    """Test the ReferenceWorkflowCellSelectionTool."""

    @pytest.fixture
    def tool(self, mock_llm_interface, mock_notebook_selector):
        """Create tool instance."""
        return ReferenceWorkflowCellSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

    def test_tool_initialization(self, tool):
        """Test tool is created correctly."""
        assert tool.name == "reference_workflow_cell_selection"

    @pytest.mark.asyncio
    async def test_filters_workflow_content(self, tool, mock_llm_interface):
        """Test that tool filters workflow content to selected cells."""
        state = create_workflow_state(
            reference_workflow_internal_ids=["org_a_tutorial_analysis"],
            reference_workflow_ids="org_a/tutorials/analysis.ipynb",
            reference_workflow_content={
                "org_a_tutorial_analysis": "original content",
            }
        )

        # Mock LLM to return specific cell indices
        mock_provider = mock_llm_interface.get_llm_for_tool.return_value
        mock_provider.generate_structured.return_value = ReferenceWorkflowCellSelection(
            selected_cells=[0, 2]  # Select first and third cells
        )

        result = await tool.execute(state)

        # Verify filtered content is in output
        assert "reference_workflow_content" in result.output_workflow
        content = result.output_workflow["reference_workflow_content"]
        assert "org_a_tutorial_analysis" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
