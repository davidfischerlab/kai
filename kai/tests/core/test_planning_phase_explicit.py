"""Test explicit planning phase management."""

import pytest
from unittest.mock import Mock
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator


@pytest.fixture
def orchestrator():
    """Create orchestrator with mocked dependencies."""
    llm = Mock()
    kb = Mock()
    return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)


class TestPlanningPhaseRouting:
    """Test planning phase router logic."""

    def test_initial_planning_with_rag_goes_to_workflow_retrieval(self, orchestrator):
        """First iteration with RAG enabled → search_workflows."""
        state = {
            "task_list": {},  # No tasks
            "planning_phase": None,  # First time
            "rag_enabled": True,
            "retrieval_queries": ["Analyze single-cell RNA-seq data"],
            "workflow_retrieval_iteration": 0,
        }

        next_node = orchestrator._route_planning_phase(state)

        assert next_node == "search_workflows", "Should route to workflow retrieval with RAG enabled"

    def test_initial_planning_with_rag_but_no_queries_goes_to_plan_tasks(self, orchestrator):
        """First iteration with RAG but empty queries → plan_tasks (edge case)."""
        state = {
            "task_list": {},
            "planning_phase": None,
            "rag_enabled": True,
            "retrieval_queries": [],  # Empty even though RAG enabled
            "workflow_retrieval_iteration": 0,
        }

        next_node = orchestrator._route_planning_phase(state)

        # Matches kai_dev: initial planning (even with RAG but no queries) → increment counter → task_list_generation
        assert next_node == "increment_task_planning_iteration", "Should increment counter first, then route to task_list_generation"

    def test_initial_planning_without_rag_goes_to_task_planning(self, orchestrator):
        """First iteration without RAG → increment_task_planning_iteration."""
        state = {
            "task_list": {},
            "planning_phase": None,
            "rag_enabled": False,
            "retrieval_queries": [],
            "workflow_retrieval_iteration": 0,
        }

        next_node = orchestrator._route_planning_phase(state)

        # Matches kai_dev: initial planning → increment counter → task_list_generation
        assert next_node == "increment_task_planning_iteration", "Should increment counter first, then route to task_list_generation"

    def test_workflow_retrieval_exits_after_max_iterations(self, orchestrator):
        """Workflow retrieval exits after 2 iterations even with queries."""
        state = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": ["More queries"],  # Still has queries
            "workflow_retrieval_iteration": 2,  # Max reached
        }

        next_node = orchestrator._route_planning_phase(state)

        # Matches kai_dev: workflow retrieval max iterations → increment counter → task_list_generation
        assert next_node == "increment_task_planning_iteration", "Should increment counter first, then route to task_list_generation"

    def test_workflow_retrieval_exits_when_queries_empty(self, orchestrator):
        """Workflow retrieval exits when no more queries."""
        state = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": [],  # Empty
            "workflow_retrieval_iteration": 1,  # Less than max
        }

        next_node = orchestrator._route_planning_phase(state)

        # Matches kai_dev: workflow retrieval with no queries → increment counter → task_list_generation
        assert next_node == "increment_task_planning_iteration", "Should increment counter first, then route to task_list_generation"

    def test_workflow_retrieval_continues_iteration_1(self, orchestrator):
        """Workflow retrieval continues on iteration 1 with queries."""
        state = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": ["Refined query from iteration 0"],
            "workflow_retrieval_iteration": 1,  # Iteration 1 of 2
        }

        next_node = orchestrator._route_planning_phase(state)

        assert next_node == "search_workflows", "Should continue retrieval on iteration 1"

    def test_task_planning_phase_goes_to_plan_tasks(self, orchestrator):
        """
        Once in task_planning phase with retrieval_queries, route to workflow_refinement.
        With no queries → complete.
        """
        # With queries → workflow_refinement
        state_with_queries = {
            "task_list": {"tasks": [{"task": "Test"}]},
            "planning_phase": "task_planning",
            "rag_enabled": True,
            "retrieval_queries": ["query"],
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 0,
        }

        next_node = orchestrator._route_planning_phase(state_with_queries)
        assert next_node == "workflow_refinement", \
            "Should route to workflow_refinement with queries"

        # Without queries → still goes to workflow_refinement (kai_dev ALWAYS does workflow selection when rag_enabled)
        state_no_queries = {
            "task_list": {"tasks": [{"task": "Test"}]},
            "planning_phase": "task_planning",
            "rag_enabled": True,
            "retrieval_queries": [],
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 1,
        }

        next_node = orchestrator._route_planning_phase(state_no_queries)
        # Matches kai_dev lines 271-297: ALWAYS run workflow refinement when rag_enabled
        assert next_node == "workflow_refinement", \
            "Should ALWAYS run workflow_refinement when rag_enabled (even without new queries)"

    def test_workflow_retrieval_with_none_queries_exits(self, orchestrator):
        """Workflow retrieval with None (not list) → plan_tasks."""
        state = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": None,  # None instead of list
            "workflow_retrieval_iteration": 0,
        }

        next_node = orchestrator._route_planning_phase(state)

        # Matches kai_dev: None queries treated as empty → increment counter → task_list_generation
        assert next_node == "increment_task_planning_iteration", "Should handle None queries gracefully by routing to task_list_generation"

    def test_workflow_retrieval_iteration_0_with_queries_continues(self, orchestrator):
        """Workflow retrieval at iteration 0 with queries → search_workflows."""
        state = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": ["First query"],
            "workflow_retrieval_iteration": 0,  # Just started
        }

        next_node = orchestrator._route_planning_phase(state)

        assert next_node == "search_workflows", "Should continue at iteration 0"


class TestSearchWorkflowsToolPhaseTracking:
    """Test that SearchWorkflowsTool properly manages phase state."""

    @pytest.mark.asyncio
    async def test_search_workflows_increments_iteration(self):
        """SearchWorkflowsTool increments workflow_retrieval_iteration."""
        from kai.core.tools.workflow_search import SearchWorkflowsTool
        from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
        from unittest.mock import AsyncMock, MagicMock

        # Create tool with mocked dependencies
        llm = Mock()
        kb = Mock()
        tool = SearchWorkflowsTool(llm, kb, mode="full")

        # Mock all sub-tools to return empty results
        empty_result = Mock()
        empty_result.output_ui = {}
        empty_result.output_type = "NO_OUTPUT"
        empty_result.output_workflow = {}

        tool.query_prep_tool.execute = AsyncMock(return_value=empty_result)
        tool.selection_tool.execute = AsyncMock(return_value=empty_result)
        tool.cell_selection_tool.execute = AsyncMock(return_value=empty_result)
        tool.filter_tool.execute = AsyncMock(return_value=empty_result)

        # Create execution context with iteration 0
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="Test query",
                context={"workflow_retrieval_iteration": 0},
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        # Execute tool
        result = await tool.execute(exec_context)

        # Verify iteration was incremented
        assert result.output_workflow["workflow_retrieval_iteration"] == 1, "Should increment iteration"
        assert result.output_workflow["planning_phase"] == "workflow_retrieval", "Should set phase"

    @pytest.mark.asyncio
    async def test_search_workflows_increments_from_iteration_1(self):
        """SearchWorkflowsTool increments from iteration 1 to 2."""
        from kai.core.tools.workflow_search import SearchWorkflowsTool
        from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
        from unittest.mock import AsyncMock

        llm = Mock()
        kb = Mock()
        tool = SearchWorkflowsTool(llm, kb, mode="full")

        empty_result = Mock()
        empty_result.output_ui = {}
        empty_result.output_type = "NO_OUTPUT"
        empty_result.output_workflow = {}

        tool.query_prep_tool.execute = AsyncMock(return_value=empty_result)
        tool.selection_tool.execute = AsyncMock(return_value=empty_result)
        tool.cell_selection_tool.execute = AsyncMock(return_value=empty_result)
        tool.filter_tool.execute = AsyncMock(return_value=empty_result)

        # Start at iteration 1
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="Test query",
                context={"workflow_retrieval_iteration": 1},
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        # Should go to iteration 2
        assert result.output_workflow["workflow_retrieval_iteration"] == 2


class TestFullPlanningFlow:
    """Integration test of full planning flow."""

    def test_planning_flow_without_rag(self, orchestrator):
        """Test flow: initial → plan_tasks (no RAG)."""
        # State 1: Initial
        state = {
            "task_list": {},
            "planning_phase": None,
            "rag_enabled": False,
            "retrieval_queries": [],
            "workflow_retrieval_iteration": 0,
        }

        step1 = orchestrator._route_planning_phase(state)
        # Matches kai_dev: initial planning → increment counter → task_list_generation
        assert step1 == "increment_task_planning_iteration", "Step 1: Should increment counter first"

    def test_planning_flow_with_rag_one_iteration(self, orchestrator):
        """Test flow: initial → search_workflows → plan_tasks (1 iteration, queries cleared)."""

        # State 1: Initial with RAG
        state1 = {
            "task_list": {},
            "planning_phase": None,
            "rag_enabled": True,
            "retrieval_queries": ["Initial query"],
            "workflow_retrieval_iteration": 0,
        }

        step1 = orchestrator._route_planning_phase(state1)
        assert step1 == "search_workflows", "Step 1: Should start workflow retrieval"

        # State 2: After search_workflows (iteration 1, queries cleared by selection tool)
        state2 = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": [],  # Selection tool cleared queries
            "workflow_retrieval_iteration": 1,
        }

        step2 = orchestrator._route_planning_phase(state2)
        # Matches kai_dev: workflow retrieval complete → increment counter → task_list_generation
        assert step2 == "increment_task_planning_iteration", "Step 2: Should increment counter after workflow retrieval"

    def test_planning_flow_with_rag_two_iterations(self, orchestrator):
        """Test flow: initial → search_workflows → search_workflows → plan_tasks (max iterations)."""

        # State 1: Initial
        state1 = {
            "task_list": {},
            "planning_phase": None,
            "rag_enabled": True,
            "retrieval_queries": ["Initial query"],
            "workflow_retrieval_iteration": 0,
        }

        step1 = orchestrator._route_planning_phase(state1)
        assert step1 == "search_workflows", "Step 1: Start retrieval"

        # State 2: After iteration 1, selection tool provided new queries
        state2 = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": ["Refined query from iteration 1"],
            "workflow_retrieval_iteration": 1,
        }

        step2 = orchestrator._route_planning_phase(state2)
        assert step2 == "search_workflows", "Step 2: Continue to iteration 2"

        # State 3: After iteration 2 (max reached)
        state3 = {
            "task_list": {},
            "planning_phase": "workflow_retrieval",
            "rag_enabled": True,
            "retrieval_queries": ["Still more queries"],  # Even with queries
            "workflow_retrieval_iteration": 2,  # Max reached
        }

        step3 = orchestrator._route_planning_phase(state3)
        # Matches kai_dev: workflow retrieval max iterations → increment counter → task_list_generation
        assert step3 == "increment_task_planning_iteration", "Step 3: Should increment counter after max iterations"


if __name__ == "__main__":
    import sys
    import asyncio

    # Run synchronous tests
    print("=== Testing Planning Phase Routing ===\n")
    llm = Mock()
    kb = Mock()
    orch = LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    test_routing = TestPlanningPhaseRouting()
    test_routing.test_initial_planning_with_rag_goes_to_workflow_retrieval(orch)
    print("✅ Initial planning with RAG → search_workflows")

    test_routing.test_initial_planning_without_rag_goes_to_task_planning(orch)
    print("✅ Initial planning without RAG → plan_tasks")

    test_routing.test_workflow_retrieval_exits_after_max_iterations(orch)
    print("✅ Workflow retrieval exits after 2 iterations")

    test_routing.test_workflow_retrieval_exits_when_queries_empty(orch)
    print("✅ Workflow retrieval exits when queries empty")

    test_routing.test_workflow_retrieval_continues_iteration_1(orch)
    print("✅ Workflow retrieval continues on iteration 1")

    test_routing.test_task_planning_phase_goes_to_plan_tasks(orch)
    print("✅ Task planning phase → plan_tasks")

    print("\n=== Testing Full Planning Flow ===\n")
    test_flow = TestFullPlanningFlow()
    test_flow.test_planning_flow_without_rag(orch)
    print("✅ Planning flow without RAG works")

    test_flow.test_planning_flow_with_rag_one_iteration(orch)
    print("✅ Planning flow with RAG (1 iteration) works")

    test_flow.test_planning_flow_with_rag_two_iterations(orch)
    print("✅ Planning flow with RAG (2 iterations) works")

    print("\n=== All Tests Passed! ===\n")
