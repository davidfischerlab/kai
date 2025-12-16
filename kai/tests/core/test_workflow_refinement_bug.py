"""Test to debug WorkflowRefinementTool.execute() not being called."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator


@pytest.mark.asyncio
async def test_workflow_refinement_tool_execute_is_called():
    """Test that WorkflowRefinementTool.execute() is actually called during planning.

    This test verifies the critical bug where WorkflowRefinementTool.execute()
    was not being invoked even though the router routed to 'workflow_refinement'.
    """
    mock_llm = MagicMock()
    mock_kb = MagicMock()
    mock_comm = MagicMock()
    mock_comm.send_workflow_result = AsyncMock()
    mock_comm.send_tool_result = AsyncMock()

    orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

    # Patch the WorkflowRefinementTool.execute method to track if it's called
    workflow_refinement_tool = orch.tools.get("workflow_refinement")
    assert workflow_refinement_tool is not None, "workflow_refinement tool must exist"

    original_execute = workflow_refinement_tool.execute
    execute_called = {"count": 0, "queries": None}

    async def mock_execute(exec_context, **kwargs):
        execute_called["count"] += 1
        execute_called["queries"] = exec_context.inputs.context.get("retrieval_queries", [])
        # Call original to get proper return value
        return await original_execute(exec_context, **kwargs)

    with patch.object(workflow_refinement_tool, 'execute', side_effect=mock_execute):
        # Create initial state with RAG enabled and retrieval queries (triggers workflow refinement)
        initial_state = {
            "user_input": "Test input",
            "session_metadata": {"session_id": "test_session"},
            "autonomous_mode": True,
            "rag_enabled": True,
            "use_critique": False,
            "retrieval_queries": ["query1", "query2", "query3"],
            "planning_phase": "task_planning",
            "task_planning_iteration": 0,
            "notebook_structure": {"totalCells": 0, "allCells": []},
            "current_cell": "",
            "current_cell_index": 0,
        }

        # Run planning graph
        results = []
        try:
            async for output in orch.planning_graph.astream(initial_state, {"recursion_limit": 25}):
                results.append(output)
                # Stop after workflow_refinement node executes
                if "workflow_refinement" in output:
                    break
        except Exception as e:
            print(f"Graph execution error: {e}")
            # Continue to check if execute was called

        # Verify execute was called
        assert execute_called["count"] > 0, \
            f"WorkflowRefinementTool.execute() was called {execute_called['count']} times, expected > 0"

        assert execute_called["queries"] is not None, \
            "Execute was called but queries were not captured"

        print(f"✅ WorkflowRefinementTool.execute() was called {execute_called['count']} times")
        print(f"✅ Queries passed to execute: {execute_called['queries']}")


@pytest.mark.asyncio
async def test_workflow_refinement_sets_had_queries_flag():
    """Test that WorkflowRefinementTool properly sets had_retrieval_queries_before_refinement.

    This is the core of the bug - the flag wasn't being set, causing incorrect routing.
    """
    from kai.core.tools.workflow_search import WorkflowRefinementTool
    from kai.core.orchestration.execution_context import ExecutionContext

    mock_llm = MagicMock()
    mock_kb = MagicMock()

    tool = WorkflowRefinementTool(mock_llm, mock_kb)

    # Create execution context with retrieval queries
    context = {
        "retrieval_queries": ["query1", "query2"],
        "task_planning_iteration": 0,
        "reference_workflow_content": {},
        "reference_workflow_percentages": {},
        "excluded_workflows": [],
    }

    exec_context = MagicMock()
    exec_context.inputs.context = context

    # Mock the sub-tools to return empty results
    tool.query_prep_tool.execute = AsyncMock(return_value=MagicMock(
        output_ui={},
        output_type="NO_OUTPUT",
        output_workflow={}
    ))
    tool.selection_tool.execute = AsyncMock(return_value=MagicMock(
        output_ui={},
        output_type="NO_OUTPUT",
        output_workflow={}
    ))
    tool.cell_selection_tool.execute = AsyncMock(return_value=MagicMock(
        output_ui={"workflows": []},
        output_type="REFERENCE_WORKFLOWS",
        output_workflow={}
    ))

    # Execute the tool
    result = await tool.execute(exec_context)

    # Verify the flag was set correctly
    assert "had_retrieval_queries_before_refinement" in result.output_workflow, \
        "had_retrieval_queries_before_refinement should be in output_workflow"

    assert result.output_workflow["had_retrieval_queries_before_refinement"] is True, \
        "had_retrieval_queries_before_refinement should be True when queries exist"

    assert result.output_workflow["retrieval_queries"] == [], \
        "retrieval_queries should be cleared after refinement"

    assert result.output_workflow["planning_phase"] == "workflow_refinement", \
        "planning_phase should be set to workflow_refinement"

    print("✅ WorkflowRefinementTool correctly sets had_retrieval_queries_before_refinement=True")


@pytest.mark.asyncio
async def test_workflow_refinement_flag_false_when_no_queries():
    """Test that had_retrieval_queries_before_refinement is False when no queries."""
    from kai.core.tools.workflow_search import WorkflowRefinementTool

    mock_llm = MagicMock()
    mock_kb = MagicMock()

    tool = WorkflowRefinementTool(mock_llm, mock_kb)

    # Create execution context WITHOUT retrieval queries
    context = {
        "retrieval_queries": [],  # Empty!
        "task_planning_iteration": 0,
        "reference_workflow_content": {},
        "reference_workflow_percentages": {},
        "excluded_workflows": [],
    }

    exec_context = MagicMock()
    exec_context.inputs.context = context

    # Mock the sub-tools
    tool.query_prep_tool.execute = AsyncMock(return_value=MagicMock(
        output_ui={},
        output_type="NO_OUTPUT",
        output_workflow={}
    ))
    tool.selection_tool.execute = AsyncMock(return_value=MagicMock(
        output_ui={},
        output_type="NO_OUTPUT",
        output_workflow={}
    ))
    tool.cell_selection_tool.execute = AsyncMock(return_value=MagicMock(
        output_ui={"workflows": []},
        output_type="REFERENCE_WORKFLOWS",
        output_workflow={}
    ))

    # Execute the tool
    result = await tool.execute(exec_context)

    # Verify the flag is False when no queries
    assert result.output_workflow["had_retrieval_queries_before_refinement"] is False, \
        "had_retrieval_queries_before_refinement should be False when no queries"

    print("✅ WorkflowRefinementTool correctly sets had_retrieval_queries_before_refinement=False when no queries")
