"""Test that cited workflows are never removed from selection."""

import pytest
from unittest.mock import Mock
from kai.core.orchestration.prompt_tools import ReferenceWorkflowSelectionOnlyTool
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
from kai.core.orchestration.schemas import ReferenceWorkflowSelectionOnly
from kai.core.prompt_manager import PromptScenario


class TestCitedWorkflowProtection:
    """Test defensive filtering that prevents LLM from removing cited workflows."""

    @pytest.fixture
    def mock_llm_interface(self):
        """Create a mock LLM interface."""
        mock_llm = Mock()
        mock_provider = Mock()
        mock_provider.provider_name = "test_provider"
        mock_provider.model = "test_model"
        mock_provider.use_structured_output = True
        mock_llm.get_llm_for_tool.return_value = mock_provider
        mock_llm.get_reasoning_for_tool.return_value = "detailed"
        return mock_llm

    @pytest.fixture
    def mock_notebook_selector(self):
        """Create a mock notebook selector."""
        mock_selector = Mock()

        # Create a mapping of all available notebooks
        all_notebooks = {
            "scverse_scanpy_tutorials_day1_01_solutions": {
                "cells": [],
                "metadata": {
                    "source_repository": "scverse/scanpy-tutorials",
                    "workflow_filename": "day1_01_solutions.ipynb"
                }
            },
            "theislab_transcription_factor_activity_example": {
                "cells": [],
                "metadata": {
                    "source_repository": "theislab/transcription_factor_activity",
                    "workflow_filename": "example.ipynb"
                }
            },
            "scverse_202504_workshop_gscn_nb5_trajectory_inference": {
                "cells": [],
                "metadata": {
                    "source_repository": "scverse/202504_workshop_GSCN",
                    "workflow_filename": "nb5_trajectory_inference.ipynb"
                }
            }
        }

        # Make get_selected_notebook_content return only requested notebooks
        def get_selected(notebook_ids):
            return {nb_id: all_notebooks[nb_id] for nb_id in notebook_ids if nb_id in all_notebooks}

        mock_selector.get_selected_notebook_content = get_selected
        mock_selector.format_notebook_context.return_value = "Formatted workflow context"
        return mock_selector

    def test_extract_cited_workflows(self, mock_llm_interface, mock_notebook_selector):
        """Test extraction of cited workflows from task list."""
        tool = ReferenceWorkflowSelectionOnlyTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

        # Create context with task list containing citations
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="test query",
                context={},
                task_list={
                    "tasks": [
                        {"id": 1, "task": "Do something [custom step]", "status": "pending"},
                        {
                            "id": 2,
                            "task": "Perform DE analysis [adapted from: 'scverse/scanpy-tutorials/day1_01_solutions.ipynb', cells: 138-141]",
                            "status": "pending"
                        },
                        {
                            "id": 3,
                            "task": "Compute TF activity [adapted from: 'theislab/transcription_factor_activity/example.ipynb', cells: 1-4]",
                            "status": "pending"
                        }
                    ]
                },
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        cited = tool._extract_cited_workflows(exec_context)

        # Should extract both cited workflows
        assert len(cited) == 2
        assert "scverse_scanpy_tutorials_day1_01_solutions" in cited
        assert "theislab_transcription_factor_activity_example" in cited

    def test_llm_removes_cited_workflow_gets_re_added(self, mock_llm_interface, mock_notebook_selector):
        """Test that if LLM removes a cited workflow, it gets re-added."""
        tool = ReferenceWorkflowSelectionOnlyTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

        # Task list cites 2 workflows
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="test query",
                context={},
                task_list={
                    "tasks": [
                        {
                            "id": 1,
                            "task": "DE analysis [adapted from: 'scverse/scanpy-tutorials/day1_01_solutions.ipynb', cells: 138-141]",
                            "status": "pending"
                        },
                        {
                            "id": 2,
                            "task": "TF activity [adapted from: 'theislab/transcription_factor_activity/example.ipynb', cells: 1-4]",
                            "status": "pending"
                        },
                        {
                            "id": 3,
                            "task": "Trajectory [adapted from: 'scverse/202504_workshop_GSCN/nb5_trajectory_inference.ipynb', cells: 20-23]",
                            "status": "pending"
                        }
                    ]
                },
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        # LLM returns selection that's MISSING the TF activity workflow (simulating the bug)
        structured_result = ReferenceWorkflowSelectionOnly(
            selected_notebooks=[
                "scverse_scanpy_tutorials_day1_01_solutions",
                "scverse_202504_workshop_gscn_nb5_trajectory_inference"
                # Missing: theislab_transcription_factor_activity_example
            ]
        )

        # Process the result - should re-add the missing cited workflow
        result = tool._process_structured_result(structured_result, exec_context)

        # Verify all 3 cited workflows are in the final result
        internal_ids = result.output_workflow["reference_workflow_internal_ids"]
        assert len(internal_ids) == 3
        assert "scverse_scanpy_tutorials_day1_01_solutions" in internal_ids
        assert "theislab_transcription_factor_activity_example" in internal_ids  # Re-added!
        assert "scverse_202504_workshop_gscn_nb5_trajectory_inference" in internal_ids

        # Verify the full IDs are also correct
        full_ids = result.output_workflow["reference_workflow_ids"]
        assert "scverse/scanpy-tutorials/day1_01_solutions.ipynb" in full_ids
        assert "theislab/transcription_factor_activity/example.ipynb" in full_ids
        assert "scverse/202504_workshop_GSCN/nb5_trajectory_inference.ipynb" in full_ids

    def test_llm_selection_without_citations(self, mock_llm_interface, mock_notebook_selector):
        """Test normal case where LLM correctly keeps cited workflows."""
        tool = ReferenceWorkflowSelectionOnlyTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="test query",
                context={},
                task_list={
                    "tasks": [
                        {
                            "id": 1,
                            "task": "DE analysis [adapted from: 'scverse/scanpy-tutorials/day1_01_solutions.ipynb', cells: 138-141]",
                            "status": "pending"
                        }
                    ]
                },
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        # LLM correctly includes the cited workflow
        structured_result = ReferenceWorkflowSelectionOnly(
            selected_notebooks=["scverse_scanpy_tutorials_day1_01_solutions"]
        )

        result = tool._process_structured_result(structured_result, exec_context)

        # Should have exactly the 1 cited workflow
        internal_ids = result.output_workflow["reference_workflow_internal_ids"]
        assert len(internal_ids) == 1
        assert "scverse_scanpy_tutorials_day1_01_solutions" in internal_ids

    def test_real_world_bug_scenario(self, mock_llm_interface, mock_notebook_selector):
        """Test the exact scenario from the bug report where TF activity workflow was removed."""
        tool = ReferenceWorkflowSelectionOnlyTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
            llm_interface=mock_llm_interface,
            notebook_selector=mock_notebook_selector
        )

        # Real task list from 2025-10-05_20-08-54_task_list_generation_result.txt
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="test query",
                context={},
                task_list={
                    "tasks": [
                        {"id": 1, "task": "Verify AnnData object [custom step]", "status": "pending"},
                        {"id": 2, "task": "Create subset [custom step]", "status": "pending"},
                        {
                            "id": 3,
                            "task": "Perform DE analysis [adapted from: 'scverse/scanpy-tutorials/day1_01_solutions.ipynb', cells: 138-141]",
                            "status": "pending"
                        },
                        {"id": 4, "task": "[reasoning] Summarize DE results", "status": "pending"},
                        {
                            "id": 5,
                            "task": "Compute TF activity [adapted from: 'theislab/transcription_factor_activity/example.ipynb', cells: 1-4]",
                            "status": "pending"
                        },
                        {"id": 6, "task": "[reasoning] Summarize TF results", "status": "pending"},
                        {
                            "id": 7,
                            "task": "Compute diffusion map [adapted from: 'scverse/202504_workshop_GSCN/nb5_trajectory_inference.ipynb', cells: 20-23]",
                            "status": "pending"
                        }
                    ]
                },
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        # Real LLM output from 2025-10-05_20-09-08_reference_workflow_selection_only_result.txt
        # LLM REMOVED theislab_transcription_factor_activity_example even though it's cited!
        structured_result = ReferenceWorkflowSelectionOnly(
            selected_notebooks=[
                "scverse_scanpy_tutorials_day1_01_solutions",
                "scverse_202504_workshop_gscn_nb5_trajectory_inference"
                # Missing: theislab_transcription_factor_activity_example (cited in task 5!)
            ]
        )

        # Process - defensive filtering should re-add the missing workflow
        result = tool._process_structured_result(structured_result, exec_context)

        # Verify all 3 cited workflows are present
        internal_ids = result.output_workflow["reference_workflow_internal_ids"]
        assert len(internal_ids) == 3, f"Expected 3 workflows, got {len(internal_ids)}: {internal_ids}"
        assert "scverse_scanpy_tutorials_day1_01_solutions" in internal_ids
        assert "theislab_transcription_factor_activity_example" in internal_ids, "TF activity workflow should be re-added!"
        assert "scverse_202504_workshop_gscn_nb5_trajectory_inference" in internal_ids
