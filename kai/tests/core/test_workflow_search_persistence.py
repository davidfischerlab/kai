"""Test that workflow search tools preserve reference_workflow_content in output_workflow.

This tests the fix for the bug where reference_workflow_content was lost when the filter
tool returned an empty output_workflow dict.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from kai.core.tools.workflow_search import SearchWorkflowsTool, WorkflowRefinementTool
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
from kai.core.orchestration.base_tool import ToolResult, ToolOutputType


@pytest.fixture
def mock_llm_interface():
    """Create mock LLM interface."""
    return MagicMock()


@pytest.fixture
def mock_knowledge_base():
    """Create mock knowledge base."""
    return MagicMock()


@pytest.fixture
def exec_context_with_workflows():
    """Create execution context with reference workflows already set."""
    context = {
        "reference_workflow_content": {
            "workflow_1": "Cell 1\nCell 2\nCell 3",
            "workflow_2": "Cell 1\nCell 5\nCell 8"
        },
        "reference_workflow_percentages": {
            "org/repo/file1.ipynb": 50.0,
            "org/repo/file2.ipynb": 30.0
        },
        "excluded_workflows": [],
        "retrieval_queries": ["query1", "query2"],
        "workflow_retrieval_iteration": 0,
        "task_planning_iteration": 0,
        "user_query": "Test query",
        "task_list": {}
    }

    inputs = ExecutionInputs(
        context=context,
        task_list={},
        user_query="Test query",
        backtracking_context=None,
        excluded_workflows=[]
    )

    return ExecutionContext(inputs=inputs, session_metadata={})


class TestSearchWorkflowsPersistence:
    """Test SearchWorkflowsTool preserves reference_workflow_content in output_workflow."""

    @pytest.mark.asyncio
    async def test_search_workflows_preserves_content_when_filter_returns_empty(
        self, mock_llm_interface, mock_knowledge_base, exec_context_with_workflows
    ):
        """Test that reference_workflow_content is preserved even when filter returns empty dict.

        This is the main bug fix: when filter_tool returns output_workflow={} (no filtering needed),
        the final ToolResult must still include reference_workflow_content from exec_context.
        """
        tool = SearchWorkflowsTool(mock_llm_interface, mock_knowledge_base, mode="full")

        # Mock the sub-tools
        # Query prep: returns empty
        tool.query_prep_tool.execute = AsyncMock(return_value=ToolResult(
            output_ui={},
            output_workflow={},
            output_type=ToolOutputType.NO_OUTPUT
        ))

        # Selection tool: adds some workflows
        tool.selection_tool.execute = AsyncMock(return_value=ToolResult(
            output_ui={},
            output_workflow={
                "reference_workflow_content": {
                    "workflow_1": "Unfiltered content",
                    "workflow_2": "Unfiltered content"
                }
            },
            output_type=ToolOutputType.NO_OUTPUT
        ))

        # Cell selection tool: filters the content
        tool.cell_selection_tool.execute = AsyncMock(return_value=ToolResult(
            output_ui={"text": "Selected cells"},
            output_workflow={
                "reference_workflow_content": {
                    "workflow_1": "Cell 1\nCell 2",
                    "workflow_2": "Cell 5"
                },
                "reference_workflow_percentages": {
                    "org/repo/file1.ipynb": 40.0,
                    "org/repo/file2.ipynb": 20.0
                },
                "excluded_workflows": []
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        ))

        # Filter tool: returns EMPTY dict (no tasks yet, so no filtering)
        tool.filter_tool.execute = AsyncMock(return_value=ToolResult(
            output_ui={},
            output_workflow={},  # EMPTY - this was causing the bug!
            output_type=ToolOutputType.NO_OUTPUT
        ))

        # Execute the search_workflows tool
        result = await tool.execute(exec_context_with_workflows)

        # Verify that reference_workflow_content is PRESERVED in output_workflow
        assert "reference_workflow_content" in result.output_workflow
        assert result.output_workflow["reference_workflow_content"] == {
            "workflow_1": "Cell 1\nCell 2",
            "workflow_2": "Cell 5"
        }

        # Verify other PERSISTENT fields are also preserved
        assert "reference_workflow_percentages" in result.output_workflow
        assert result.output_workflow["reference_workflow_percentages"] == {
            "org/repo/file1.ipynb": 40.0,
            "org/repo/file2.ipynb": 20.0
        }

        assert "excluded_workflows" in result.output_workflow
        assert result.output_workflow["excluded_workflows"] == []

        # Verify phase tracking fields are set
        assert result.output_workflow["planning_phase"] == "workflow_retrieval"
        assert result.output_workflow["workflow_retrieval_iteration"] == 1


class TestWorkflowRefinementPersistence:
    """Test WorkflowRefinementTool preserves reference_workflow_content in output_workflow."""

    @pytest.mark.asyncio
    async def test_workflow_refinement_preserves_content(
        self, mock_llm_interface, mock_knowledge_base, exec_context_with_workflows
    ):
        """Test that WorkflowRefinementTool preserves reference_workflow_content in output_workflow.

        Same issue as SearchWorkflowsTool: must preserve PERSISTENT fields from exec_context.
        """
        tool = WorkflowRefinementTool(mock_llm_interface, mock_knowledge_base)

        # Mock the sub-tools
        tool.query_prep_tool.execute = AsyncMock(return_value=ToolResult(
            output_ui={},
            output_workflow={},
            output_type=ToolOutputType.NO_OUTPUT
        ))

        tool.selection_tool.execute = AsyncMock(return_value=ToolResult(
            output_ui={},
            output_workflow={
                "reference_workflow_content": {
                    "workflow_3": "New workflow content"
                }
            },
            output_type=ToolOutputType.NO_OUTPUT
        ))

        tool.cell_selection_tool.execute = AsyncMock(return_value=ToolResult(
            output_ui={"text": "Refined cells"},
            output_workflow={
                "reference_workflow_content": {
                    "workflow_1": "Refined Cell 1\nCell 3",
                    "workflow_3": "Refined New Cell"
                },
                "reference_workflow_percentages": {
                    "org/repo/file1.ipynb": 35.0,
                    "org/repo/file3.ipynb": 25.0
                },
                "excluded_workflows": ["workflow_2"]
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        ))

        # Execute the workflow_refinement tool
        result = await tool.execute(exec_context_with_workflows)

        # Verify that reference_workflow_content is PRESERVED in output_workflow
        assert "reference_workflow_content" in result.output_workflow
        assert result.output_workflow["reference_workflow_content"] == {
            "workflow_1": "Refined Cell 1\nCell 3",
            "workflow_3": "Refined New Cell"
        }

        # Verify other PERSISTENT fields are also preserved
        assert "reference_workflow_percentages" in result.output_workflow
        assert result.output_workflow["reference_workflow_percentages"] == {
            "org/repo/file1.ipynb": 35.0,
            "org/repo/file3.ipynb": 25.0
        }

        assert "excluded_workflows" in result.output_workflow
        assert result.output_workflow["excluded_workflows"] == ["workflow_2"]

        # Verify retrieval_queries is CLEARED
        assert result.output_workflow["retrieval_queries"] == []

        # Verify phase tracking fields are set
        # WorkflowRefinementTool sets planning_phase to workflow_refinement
        assert result.output_workflow["planning_phase"] == "workflow_refinement"
