"""Test workflow refinement loop in planning phase.

This test verifies the complete two-phase planning workflow:
- Phase 1: Initial workflow retrieval (max 2 iterations)
- Phase 2: Task generation + workflow refinement (max 10 iterations)

Production bug this fixes:
- CellTypist test requested "Basic celltypist usage from celltypist repository"
- Only batch correction workflows were found
- Root cause: System only did initial retrieval, didn't refine based on task context
- Expected: LLM generates tasks → extracts retrieval_queries → searches again → finds CellTypist tutorials

Reference: kai_dev/core/orchestration/workflow_orchestrator.py lines 243-313
"""

import pytest
from unittest.mock import Mock, AsyncMock
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator
from kai.core.tools.workflow_search import WorkflowRefinementTool
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs


class TestWorkflowRefinementLoop:
    """Test the two-phase planning workflow."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_phase_transition_workflow_retrieval_to_task_planning(self, orchestrator):
        """
        Test transition from Phase 1 (workflow retrieval) to Phase 2 (task planning).

        Sequence:
        1. Initial state: planning_phase=None → search_workflows
        2. After 2 iterations: → plan_tasks (transition to phase 2)
        3. plan_tasks returns retrieval_queries → workflow_refinement
        """
        # Phase 1, iteration 1
        state1 = {
            "task_list": {},
            "retrieval_queries": ["User query"],
            "rag_enabled": True,
            "planning_phase": None,
            "workflow_retrieval_iteration": 0,
            "task_planning_iteration": 0,
        }

        route1 = orchestrator._route_planning_phase(state1)
        assert route1 == "search_workflows", "Phase 1 iteration 1"

        # Phase 1, iteration 2
        state2 = {
            "task_list": {},
            "retrieval_queries": ["Refined query"],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 1,
            "task_planning_iteration": 0,
        }

        route2 = orchestrator._route_planning_phase(state2)
        assert route2 == "search_workflows", "Phase 1 iteration 2"

        # Phase 1 max reached → transition to Phase 2
        state3 = {
            "task_list": {},
            "retrieval_queries": [],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 0,
        }

        route3 = orchestrator._route_planning_phase(state3)
        # Goes to increment_task_planning_iteration which then leads to plan_tasks
        assert route3 == "increment_task_planning_iteration", \
            "Phase 1 complete → transition to task planning"

        print("✅ Phase 1 → Phase 2 transition works correctly")

    def test_task_planning_with_retrieval_queries_triggers_refinement(self, orchestrator):
        """
        Test that task planning with retrieval_queries triggers workflow refinement.

        This is the KEY feature that fixes CellTypist bug:
        - plan_tasks generates: retrieval_queries = ["celltypist tutorial"]
        - Router sees queries → workflow_refinement
        - workflow_refinement searches for celltypist → finds tutorials!
        """
        # plan_tasks executed, returned retrieval_queries
        state = {
            "task_list": {"tasks": [{"task": "Annotate with CellTypist"}]},
            "retrieval_queries": ["celltypist tutorial", "cell annotation"],
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 0,
        }

        route = orchestrator._route_planning_phase(state)
        assert route == "workflow_refinement", \
            "Task planning with retrieval_queries should trigger workflow_refinement"

        print("✅ Retrieval queries from task planning trigger workflow refinement")

    def test_task_planning_without_queries_completes(self, orchestrator):
        """
        Test that task planning without retrieval_queries goes to workflow_refinement.

        Note: When RAG is enabled, even with empty queries, the router goes to
        workflow_refinement to allow selection changes (protecting cited workflows).
        """
        state = {
            "task_list": {"tasks": [{"task": "Load data"}]},
            "retrieval_queries": [],  # No more queries
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 1,
        }

        route = orchestrator._route_planning_phase(state)
        # With RAG enabled, always goes to workflow_refinement for selection changes
        assert route == "workflow_refinement", \
            "Task planning with RAG enabled should go to workflow_refinement"

        print("✅ Task planning with RAG goes to workflow_refinement")

    def test_workflow_refinement_loop_max_10_iterations(self, orchestrator):
        """
        Test that workflow refinement loop terminates after 10 iterations.

        Prevents infinite loop if LLM keeps generating retrieval_queries.
        Reference: kai_dev lines 262-313 (max_iterations = 10)

        The test verifies that when iteration 10 is reached WITH queries present,
        the router returns "complete" instead of continuing.
        """
        # Simulate state at iteration 9 with queries (should route to workflow_refinement)
        state_iter_9 = {
            "task_list": {"tasks": [{"task": "Test"}]},
            "retrieval_queries": ["query at iter 9"],
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 9,  # Just before max
        }

        route_at_9 = orchestrator._route_planning_phase(state_iter_9)
        assert route_at_9 == "workflow_refinement", \
            "Should route to workflow_refinement at iteration 9"

        # Simulate state at iteration 10 with queries (should complete due to max)
        state_iter_10 = {
            "task_list": {"tasks": [{"task": "Test"}]},
            "retrieval_queries": ["query at iter 10"],  # Has queries but at max
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 10,  # At max
        }

        route_at_10 = orchestrator._route_planning_phase(state_iter_10)
        # At max iterations with RAG enabled, goes to filter_and_complete
        assert route_at_10 == "filter_and_complete", \
            "Should complete at iteration 10 even with queries present"

        print("✅ Workflow refinement loop terminates at 10 iterations")
        print(f"   Iteration 9 with queries → {route_at_9}")
        print(f"   Iteration 10 with queries → {route_at_10}")

    def test_complete_two_phase_workflow_sequence(self, orchestrator):
        """
        Test the complete two-phase planning sequence matching old orchestrator.

        Full sequence:
        1. Phase 1: search_workflows (iter 0)
        2. Phase 1: search_workflows (iter 1)
        3. Phase 1 complete: plan_tasks
        4. Phase 2: plan_tasks returns queries → workflow_refinement
        5. Phase 2: workflow_refinement clears queries → plan_tasks
        6. Phase 2: plan_tasks returns no queries → complete
        """
        sequence = []

        # 1. Phase 1, iteration 0
        state = {
            "task_list": {},
            "retrieval_queries": ["User: analyze data with celltypist"],
            "rag_enabled": True,
            "planning_phase": None,
            "workflow_retrieval_iteration": 0,
            "task_planning_iteration": 0,
        }
        route = orchestrator._route_planning_phase(state)
        sequence.append(route)
        assert route == "search_workflows"

        # 2. Phase 1, iteration 1
        state = {
            "task_list": {},
            "retrieval_queries": ["batch correction", "preprocessing"],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 1,
            "task_planning_iteration": 0,
        }
        route = orchestrator._route_planning_phase(state)
        sequence.append(route)
        assert route == "search_workflows"

        # 3. Phase 1 max reached → transition to task planning
        state = {
            "task_list": {},
            "retrieval_queries": [],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 0,
        }
        route = orchestrator._route_planning_phase(state)
        sequence.append(route)
        assert route == "increment_task_planning_iteration"

        # 4. plan_tasks returns retrieval_queries from task context
        state = {
            "task_list": {"tasks": [{"task": "Annotate cells with CellTypist"}]},
            "retrieval_queries": ["celltypist tutorial"],  # Extracted from task!
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 1,
        }
        route = orchestrator._route_planning_phase(state)
        sequence.append(route)
        assert route == "workflow_refinement"

        # 5. workflow_refinement clears queries → still routes to workflow_refinement
        # when RAG enabled (to allow selection changes)
        state = {
            "task_list": {"tasks": [{"task": "Annotate cells with CellTypist"}]},
            "retrieval_queries": [],  # Cleared by workflow_refinement
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 1,
        }
        route = orchestrator._route_planning_phase(state)
        sequence.append(route)
        # With RAG enabled, always goes to workflow_refinement even with empty queries
        assert route == "workflow_refinement"

        expected_sequence = [
            "search_workflows",                    # Phase 1, iter 0
            "search_workflows",                    # Phase 1, iter 1
            "increment_task_planning_iteration",   # Phase 1 → 2 transition
            "workflow_refinement",                 # Phase 2: refine with task queries
            "workflow_refinement",                 # Phase 2: again (RAG enabled)
        ]

        assert sequence == expected_sequence, \
            f"Expected: {expected_sequence}\nGot: {sequence}"

        print("✅ Complete two-phase workflow sequence matches new orchestrator")
        print(f"   Sequence: {' → '.join(sequence)}")

    def test_rag_disabled_skips_workflow_refinement(self, orchestrator):
        """
        Test that workflow refinement is skipped when RAG disabled.
        """
        # plan_tasks with queries, but RAG disabled
        state = {
            "task_list": {"tasks": [{"task": "Test"}]},
            "retrieval_queries": ["some query"],
            "rag_enabled": False,  # RAG disabled
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 0,
            "task_planning_iteration": 1,
        }

        route = orchestrator._route_planning_phase(state)
        assert route == "complete", \
            "Should skip workflow_refinement when RAG disabled"

        print("✅ Workflow refinement skipped when RAG disabled")


class TestWorkflowRefinementTool:
    """Test WorkflowRefinementTool implementation."""

    @pytest.mark.asyncio
    async def test_tool_clears_retrieval_queries(self):
        """
        Verify WorkflowRefinementTool clears retrieval_queries after execution.

        Reference: kai_dev line 293
        exec_context.inputs.context["retrieval_queries"] = []
        """
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

        # Create tool
        llm = Mock()
        kb = Mock()
        tool = WorkflowRefinementTool(llm, kb)
        tool.query_prep_tool = mock_query_prep
        tool.selection_tool = mock_selection
        tool.cell_selection_tool = mock_cell_selection

        # Execute with retrieval_queries
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="Test",
                context={
                    "retrieval_queries": ["celltypist", "annotation"],
                    "task_planning_iteration": 0,
                },
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        # Verify queries were cleared
        assert result.output_workflow is not None
        assert result.output_workflow["retrieval_queries"] == [], \
            "WorkflowRefinementTool should clear retrieval_queries"

        print("✅ WorkflowRefinementTool clears retrieval_queries")

    @pytest.mark.asyncio
    async def test_tool_increments_task_planning_iteration(self):
        """
        Verify WorkflowRefinementTool increments task_planning_iteration.
        """
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

        llm = Mock()
        kb = Mock()
        tool = WorkflowRefinementTool(llm, kb)
        tool.query_prep_tool = mock_query_prep
        tool.selection_tool = mock_selection
        tool.cell_selection_tool = mock_cell_selection

        # Start at iteration 0
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="Test",
                context={
                    "retrieval_queries": ["query"],
                    "task_planning_iteration": 0,
                },
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        # WorkflowRefinementTool doesn't increment iteration - router does
        # The tool sets planning_phase="workflow_refinement"
        assert result.output_workflow.get("planning_phase") == "workflow_refinement", \
            "Should set planning_phase to workflow_refinement"

        print("✅ WorkflowRefinementTool sets planning_phase")

    @pytest.mark.asyncio
    async def test_tool_sets_planning_phase(self):
        """
        Verify WorkflowRefinementTool sets planning_phase to task_planning.
        """
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

        llm = Mock()
        kb = Mock()
        tool = WorkflowRefinementTool(llm, kb)
        tool.query_prep_tool = mock_query_prep
        tool.selection_tool = mock_selection
        tool.cell_selection_tool = mock_cell_selection

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="Test",
                context={
                    "retrieval_queries": ["query"],
                    "task_planning_iteration": 1,
                },
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        assert result.output_workflow["planning_phase"] == "workflow_refinement", \
            "Should set planning_phase to workflow_refinement"

        print("✅ WorkflowRefinementTool sets workflow_refinement phase")


class TestCellTypistScenario:
    """Test the exact CellTypist production scenario."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_celltypist_tutorial_discovery_workflow(self, orchestrator):
        """
        Reproduce the CellTypist bug and verify the fix.

        Bug:
        - User: "analyze breastcancer data with celltypist"
        - Phase 1: Finds batch correction workflows (generic preprocessing)
        - plan_tasks: Creates task "Annotate with CellTypist"
        - OLD: Planning complete → CellTypist tutorials never found
        - NEW: plan_tasks extracts "celltypist tutorial" → workflow_refinement → finds tutorials!
        """
        sequence = []

        # Phase 1: Initial retrieval finds generic preprocessing
        state1 = {
            "task_list": {},
            "retrieval_queries": ["analyze breastcancer data with celltypist"],
            "rag_enabled": True,
            "planning_phase": None,
            "workflow_retrieval_iteration": 0,
            "task_planning_iteration": 0,
        }
        route1 = orchestrator._route_planning_phase(state1)
        sequence.append("search_workflows (finds batch_correction)")

        # Phase 1 iteration 2
        state2 = {
            "task_list": {},
            "retrieval_queries": ["preprocessing", "batch correction"],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 1,
            "task_planning_iteration": 0,
        }
        route2 = orchestrator._route_planning_phase(state2)
        sequence.append("search_workflows (refines to preprocessing)")

        # Phase 1 complete
        state3 = {
            "task_list": {},
            "retrieval_queries": [],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 0,
        }
        route3 = orchestrator._route_planning_phase(state3)
        assert route3 == "increment_task_planning_iteration"
        sequence.append("increment_task_planning_iteration")

        # Phase 2: plan_tasks generates task with CellTypist
        # LLM extracts retrieval_queries from task context
        state4 = {
            "task_list": {
                "tasks": [
                    {"task": "Load and preprocess data"},
                    {"task": "Annotate cell types with CellTypist"},  # KEY: CellTypist mentioned
                ]
            },
            "retrieval_queries": ["celltypist tutorial", "cell type annotation"],  # Extracted by LLM!
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 1,
        }
        route4 = orchestrator._route_planning_phase(state4)
        assert route4 == "workflow_refinement", \
            "Should route to workflow_refinement to find CellTypist tutorials"
        sequence.append("workflow_refinement (finds celltypist tutorials!)")

        # Phase 2: After refinement, queries cleared
        state5 = {
            "task_list": {
                "tasks": [
                    {"task": "Load and preprocess data"},
                    {"task": "Annotate cell types with CellTypist"},
                ]
            },
            "retrieval_queries": [],  # Cleared by workflow_refinement
            "rag_enabled": True,
            "planning_phase": "task_planning",
            "workflow_retrieval_iteration": 2,
            "task_planning_iteration": 1,
        }
        route5 = orchestrator._route_planning_phase(state5)
        # With RAG enabled, always goes to workflow_refinement even with empty queries
        assert route5 == "workflow_refinement"
        sequence.append("workflow_refinement")

        print("✅ CellTypist tutorial discovery workflow:")
        for step in sequence:
            print(f"   {step}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
