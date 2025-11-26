"""Unit tests for excluded workflows mechanism."""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from kai.core.orchestration.prompt_tools import ReferenceWorkflowCellSelectionTool
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
from kai.core.orchestration.schemas import ReferenceWorkflowCellSelection
from kai.core.prompt_manager import PromptScenario


class TestExcludedWorkflowsMechanism:
    """Test that workflows with 0 cells are correctly added to excluded_workflows."""

    @pytest.fixture
    def mock_llm_interface(self):
        """Create a mock LLM interface."""
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
    def mock_notebook_selector(self):
        """Create a mock notebook selector with 5 notebooks."""
        mock_selector = Mock()

        # Return 5 notebooks with cell data
        mock_selector.get_selected_notebook_content.return_value = {
            "scverse_scanpy_tutorials_day1_01_solutions": {
                "cells": [
                    {"content": "cell content 1", "order": 0, "section": "main"},
                    {"content": "cell content 2", "order": 1, "section": "main"},
                    {"content": "cell content 3", "order": 2, "section": "main"}
                ],
                "metadata": {"source_repository": "scverse/scanpy-tutorials", "workflow_filename": "day1_01_solutions.ipynb", "title": "Day 1 Solutions"}
            },
            "saezlab_2024_ebi_grn_3_tf_activity_inference": {
                "cells": [
                    {"content": "cell content 1", "order": 0, "section": "main"},
                    {"content": "cell content 2", "order": 1, "section": "main"}
                ],
                "metadata": {"source_repository": "saezlab/2024_EBI_GRN", "workflow_filename": "3_TF_activity_inference.ipynb", "title": "TF Activity Inference"}
            },
            "theislab_transcription_factor_activity_example": {
                "cells": [
                    {"content": "cell content 1", "order": 0, "section": "main"}
                ],
                "metadata": {"source_repository": "theislab/transcription_factor_activity", "workflow_filename": "example.ipynb", "title": "TF Activity Example"}
            },
            "theislab_regvelo_reproducibility_1_data_preparation": {
                "cells": [
                    {"content": "cell content 1", "order": 0, "section": "main"},
                    {"content": "cell content 2", "order": 1, "section": "main"},
                    {"content": "cell content 3", "order": 2, "section": "main"},
                    {"content": "cell content 4", "order": 3, "section": "main"}
                ],
                "metadata": {"source_repository": "theislab/regvelo_reproducibility", "workflow_filename": "1_data_preparation.ipynb", "title": "Data Preparation"}
            },
            "theislab_hrca_reproducibility_03_3_1_prepare_custom_harmony_input": {
                "cells": [
                    {"content": "cell content 1", "order": 0, "section": "main"},
                    {"content": "cell content 2", "order": 1, "section": "main"}
                ],
                "metadata": {"source_repository": "theislab/HRCA-reproducibility", "workflow_filename": "03_3_1_prepare_custom_harmony_input.ipynb", "title": "Prepare Custom Harmony Input"}
            }
        }

        mock_selector.format_notebook_context.return_value = "Formatted notebook context"
        return mock_selector

    @pytest.mark.asyncio
    async def test_all_empty_cells_are_excluded(self, mock_llm_interface, mock_notebook_selector):
        """Test that all 5 notebooks with 0 cells are added to excluded_workflows."""

        # Create tool
        tool = ReferenceWorkflowCellSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

        # Setup context with 5 notebooks
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="test query",
                context={
                    "reference_workflow_internal_ids": [
                        "scverse_scanpy_tutorials_day1_01_solutions",
                        "saezlab_2024_ebi_grn_3_tf_activity_inference",
                        "theislab_transcription_factor_activity_example",
                        "theislab_regvelo_reproducibility_1_data_preparation",
                        "theislab_hrca_reproducibility_03_3_1_prepare_custom_harmony_input"
                    ],
                    "reference_workflow_ids": "scverse/scanpy-tutorials/day1_01_solutions.ipynb, saezlab/2024_EBI_GRN/3_TF_activity_inference.ipynb, theislab/transcription_factor_activity/example.ipynb, theislab/regvelo_reproducibility/1_data_preparation.ipynb, theislab/HRCA-reproducibility/03_3_1_prepare_custom_harmony_input.ipynb",
                    "execution_history": [],
                    "conversation_history": [],
                    "notebook_structure": {"cells": [], "totalCells": 0, "allCells": []},
                    "autonomous_mode": False,
                    "last_execution_failed": False,
                    "current_cell": "",
                    "error_message": ""
                },
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        # Mock LLM to return 0 cells for all notebooks
        mock_llm_provider = mock_llm_interface.get_llm_for_tool.return_value
        mock_llm_provider.generate_structured.return_value = ReferenceWorkflowCellSelection(
            selected_cells=[]  # 0 cells
        )

        # Execute
        result = await tool.execute(exec_context)

        # Print debug info
        print("\n=== TEST DEBUG INFO ===")
        print(f"internal_ids: {exec_context.inputs.context['reference_workflow_internal_ids']}")
        print(f"reference_workflow_ids: {exec_context.inputs.context['reference_workflow_ids']}")
        print(f"Result output_workflow: {result.output_workflow}")
        print(f"excluded_workflows in result: {result.output_workflow.get('excluded_workflows', [])}")

        # Assert all 5 notebooks are in excluded_workflows
        excluded = result.output_workflow.get('excluded_workflows', [])
        assert len(excluded) == 5, f"Expected 5 excluded workflows, got {len(excluded)}: {excluded}"

        # Assert they're in INTERNAL ID format
        expected_internal_ids = {
            "scverse_scanpy_tutorials_day1_01_solutions",
            "saezlab_2024_ebi_grn_3_tf_activity_inference",
            "theislab_transcription_factor_activity_example",
            "theislab_regvelo_reproducibility_1_data_preparation",
            "theislab_hrca_reproducibility_03_3_1_prepare_custom_harmony_input"
        }
        assert set(excluded) == expected_internal_ids, f"Excluded workflows don't match. Got: {excluded}"

    @pytest.mark.asyncio
    async def test_partial_empty_cells(self, mock_llm_interface, mock_notebook_selector):
        """Test that only notebooks with 0 cells are excluded."""

        tool = ReferenceWorkflowCellSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="test query",
                context={
                    "reference_workflow_internal_ids": [
                        "scverse_scanpy_tutorials_day1_01_solutions",
                        "saezlab_2024_ebi_grn_3_tf_activity_inference",
                        "theislab_transcription_factor_activity_example"
                    ],
                    "reference_workflow_ids": "scverse/scanpy-tutorials/day1_01_solutions.ipynb, saezlab/2024_EBI_GRN/3_TF_activity_inference.ipynb, theislab/transcription_factor_activity/example.ipynb",
                    "execution_history": [],
                    "conversation_history": [],
                    "notebook_structure": {"cells": [], "totalCells": 0, "allCells": []},
                    "autonomous_mode": False,
                    "last_execution_failed": False,
                    "current_cell": "",
                    "error_message": ""
                },
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        # Mock LLM to return different cell counts
        mock_llm_provider = mock_llm_interface.get_llm_for_tool.return_value

        # Track call count to return different values
        call_count = [0]

        async def mock_generate(prompt, schema, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ReferenceWorkflowCellSelection(selected_cells=[0, 1])  # 2 cells
            elif call_count[0] == 2:
                return ReferenceWorkflowCellSelection(selected_cells=[])  # 0 cells
            else:
                return ReferenceWorkflowCellSelection(selected_cells=[5])  # 1 cell

        mock_llm_provider.generate_structured = mock_generate

        # Execute
        result = await tool.execute(exec_context)

        # Print debug info
        print("\n=== TEST DEBUG INFO (partial) ===")
        print(f"excluded_workflows: {result.output_workflow.get('excluded_workflows', [])}")

        # Only the second notebook should be excluded
        excluded = result.output_workflow.get('excluded_workflows', [])
        assert len(excluded) == 1, f"Expected 1 excluded workflow, got {len(excluded)}: {excluded}"
        assert excluded[0] == "saezlab_2024_ebi_grn_3_tf_activity_inference"
