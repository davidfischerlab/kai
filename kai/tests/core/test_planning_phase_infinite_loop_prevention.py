"""Test that planning phase prevents infinite loops.

This test simulates the exact production bug where search_workflows
kept looping infinitely because:
1. Router used retrieval_queries for BOTH data AND routing
2. Tools set new retrieval_queries → router re-routes to search_workflows
3. INFINITE LOOP

The fix: Explicit phase tracking with iteration counter (max 2).
"""

import pytest
from unittest.mock import Mock, AsyncMock
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator
from kai.core.tools.workflow_search import SearchWorkflowsTool
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs


class TestPlanningPhaseInfiniteLoopPrevention:
    """Test that planning phase cannot loop infinitely."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_old_bug_retrieval_queries_causes_loop(self, orchestrator):
        """
        OLD BUG: If router only checks retrieval_queries presence,
        setting new queries causes infinite loop.

        This test verifies the NEW behavior prevents this.
        """
        # Iteration 1: Router routes to search_workflows
        state_iter1 = {
            "task_list": {},
            "retrieval_queries": ["Initial query"],
            "rag_enabled": True,
            "planning_phase": None,
            "workflow_retrieval_iteration": 0,
        }

        route1 = orchestrator._route_planning_phase(state_iter1)
        assert route1 == "search_workflows", "First iteration should route to search_workflows"

        # Simulate search_workflows tool execution
        # Tool sets: planning_phase="workflow_retrieval", iteration=1, NEW retrieval_queries
        state_after_tool1 = {
            "task_list": {},
            "retrieval_queries": ["Refined query from LLM"],  # NEW queries set by tool
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",  # Set by tool
            "workflow_retrieval_iteration": 1,  # Incremented by tool
        }

        # Iteration 2: Router should route to search_workflows AGAIN (iteration 1 < 2)
        route2 = orchestrator._route_planning_phase(state_after_tool1)
        assert route2 == "search_workflows", "Iteration 1 with queries should continue"

        # Simulate search_workflows tool execution AGAIN
        state_after_tool2 = {
            "task_list": {},
            "retrieval_queries": ["Yet another refined query"],  # Tool sets NEW queries again
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 2,  # Now at max
        }

        # Iteration 3: Router should EXIT (max iterations reached)
        # Goes to increment_task_planning_iteration to transition to phase 2
        route3 = orchestrator._route_planning_phase(state_after_tool2)
        assert route3 == "increment_task_planning_iteration", \
            "Should exit after 2 iterations (max reached)"

        print("✅ Infinite loop prevented: Max 2 iterations enforced")

    def test_workflow_transition_sequence_max_iterations(self, orchestrator):
        """
        Test the EXACT workflow transition sequence that caused infinite loop:

        Production logs showed:
        11:03:17 - [ROUTER] No tasks + retrieval queries → search_workflows
        11:03:37 - [ROUTER] No tasks + retrieval queries → search_workflows
        11:03:52 - [ROUTER] No tasks + retrieval queries → search_workflows
        [...repeated 30+ times...]

        This test verifies the sequence now terminates after 2 iterations.
        """
        transitions = []

        # Start: No tasks, RAG enabled, initial query
        state = {
            "task_list": {},
            "retrieval_queries": ["User query"],
            "rag_enabled": True,
            "planning_phase": None,
            "workflow_retrieval_iteration": 0,
        }

        # Simulate the workflow loop
        for i in range(5):  # Try 5 iterations (should stop at 2)
            route = orchestrator._route_planning_phase(state)
            transitions.append(f"iter_{i}: {route}")

            if route == "increment_task_planning_iteration":
                # Exit the loop - this is the expected termination
                break

            # Simulate search_workflows tool updating state
            state = {
                "task_list": {},
                "retrieval_queries": [f"Query iteration {i+1}"],
                "rag_enabled": True,
                "planning_phase": "workflow_retrieval",
                "workflow_retrieval_iteration": i + 1,
            }

        # Verify transitions
        assert transitions == [
            "iter_0: search_workflows",  # First iteration
            "iter_1: search_workflows",  # Second iteration
            "iter_2: increment_task_planning_iteration",  # MAX REACHED → exit
        ], f"Got transitions: {transitions}"

        print("✅ Workflow transitions terminate after 2 iterations")
        print(f"   Transitions: {' → '.join(transitions)}")

    def test_empty_queries_exits_immediately(self, orchestrator):
        """
        Even if iteration < 2, empty queries should exit.
        """
        state = {
            "task_list": {},
            "retrieval_queries": [],  # Empty
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 1,  # Only iteration 1, but queries empty
        }

        route = orchestrator._route_planning_phase(state)
        assert route == "increment_task_planning_iteration", \
            "Empty queries should exit even before max iterations"

        print("✅ Empty queries exit immediately")

    def test_tool_cannot_override_max_iterations(self, orchestrator):
        """
        Verify that even if tool sets queries, max iterations is respected.
        """
        # Iteration 2 (max reached)
        state = {
            "task_list": {},
            "retrieval_queries": ["Tool set new queries"],  # Tool tried to continue
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 2,  # At max
        }

        route = orchestrator._route_planning_phase(state)
        assert route == "increment_task_planning_iteration", \
            "Max iterations overrides presence of queries"

        print("✅ Max iterations cannot be bypassed")

    def test_comparison_old_vs_new_behavior(self, orchestrator):
        """
        Compare OLD (broken) vs NEW (fixed) behavior side-by-side.
        """
        # State after first search_workflows execution
        state = {
            "task_list": {},
            "retrieval_queries": ["New queries from tool"],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 3,  # Hypothetically at iteration 3
        }

        # OLD BEHAVIOR (if we only checked retrieval_queries):
        # if retrieval_queries: return "search_workflows"  # Would loop forever!

        # NEW BEHAVIOR (explicit phase tracking):
        route = orchestrator._route_planning_phase(state)
        assert route == "increment_task_planning_iteration", \
            "NEW: Iteration counter prevents infinite loop"

        print("✅ OLD: Would loop forever if queries present")
        print("   NEW: Iteration counter terminates loop at 2")


class TestSearchWorkflowsToolPhaseManagement:
    """Test that SearchWorkflowsTool correctly manages phase state."""

    @pytest.mark.asyncio
    async def test_tool_increments_iteration_counter(self):
        """Verify SearchWorkflowsTool increments workflow_retrieval_iteration."""
        from unittest.mock import AsyncMock, Mock
        from kai.core.orchestration.prompt_tools import ReferenceWorkflowSelectionTool
        from kai.core.orchestration.deterministic_tools import (
            ReferenceWorkflowQueryPreparationTool,
            FilterUnusedReferenceWorkflowsTool
        )

        # Mock all the sub-tools
        mock_query_prep = Mock()
        mock_query_prep.execute = AsyncMock(return_value=Mock(
            output_workflow={"retrieval_queries": ["query1"]},
            output_ui=None,
            output_type=None
        ))

        mock_selection = Mock()
        mock_selection.execute = AsyncMock(return_value=Mock(
            output_workflow={"reference_workflow_ids": ["wf1"]},
            output_ui=None,
            output_type=None
        ))

        mock_cell_selection = Mock()
        mock_cell_selection.execute = AsyncMock(return_value=Mock(
            output_workflow={"reference_workflow_content": {"cells": []}},
            output_ui=None,
            output_type=None
        ))

        mock_filter = Mock()
        mock_filter.execute = AsyncMock(return_value=Mock(
            output_workflow={},
            output_ui=None,
            output_type=None
        ))

        # Create tool and inject mocks
        llm = Mock()
        kb = Mock()
        tool = SearchWorkflowsTool(llm, kb, mode="full")
        tool.query_prep_tool = mock_query_prep
        tool.selection_tool = mock_selection
        tool.cell_selection_tool = mock_cell_selection
        tool.filter_tool = mock_filter

        # Create execution context
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="Test query",
                context={
                    "workflow_retrieval_iteration": 0,  # Start at 0
                },
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        # Execute tool
        result = await tool.execute(exec_context)

        # Verify iteration was incremented
        assert result.output_workflow is not None
        assert result.output_workflow["workflow_retrieval_iteration"] == 1, \
            "Tool should increment from 0 to 1"
        assert result.output_workflow["planning_phase"] == "workflow_retrieval", \
            "Tool should set planning_phase"

        print("✅ SearchWorkflowsTool correctly increments iteration counter")

    @pytest.mark.asyncio
    async def test_tool_increments_from_iteration_1(self):
        """Verify tool increments correctly from iteration 1 to 2."""
        from unittest.mock import AsyncMock, Mock

        # Mock sub-tools
        mock_query_prep = Mock()
        mock_query_prep.execute = AsyncMock(return_value=Mock(
            output_workflow={}, output_ui=None, output_type=None
        ))
        mock_selection = Mock()
        mock_selection.execute = AsyncMock(return_value=Mock(
            output_workflow={}, output_ui=None, output_type=None
        ))
        mock_cell_selection = Mock()
        mock_cell_selection.execute = AsyncMock(return_value=Mock(
            output_workflow={}, output_ui=None, output_type=None
        ))
        mock_filter = Mock()
        mock_filter.execute = AsyncMock(return_value=Mock(
            output_workflow={}, output_ui=None, output_type=None
        ))

        llm = Mock()
        kb = Mock()
        tool = SearchWorkflowsTool(llm, kb, mode="full")
        tool.query_prep_tool = mock_query_prep
        tool.selection_tool = mock_selection
        tool.cell_selection_tool = mock_cell_selection
        tool.filter_tool = mock_filter

        # Start at iteration 1
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="Test",
                context={"workflow_retrieval_iteration": 1},
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        assert result.output_workflow["workflow_retrieval_iteration"] == 2, \
            "Should increment from 1 to 2"

        print("✅ Tool increments correctly: 1 → 2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
