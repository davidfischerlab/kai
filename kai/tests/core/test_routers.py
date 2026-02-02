"""Comprehensive tests for all routing logic in the orchestrator.

This file tests all possible paths through the routing functions.
Each router function has its own test class for clarity.

Router hierarchy:
    route_deterministic (entry point)
    ├── route_first_execution
    │   └── (reasoning or code generation paths)
    └── route_standard_execution
        ├── route_standard_continue_branch (Branch 2)
        ├── route_standard_retry_branch (Branch 3)
        └── route_backtracking_branch (Branch 4)

    route_planning_phase (planning subgraph)
"""

import pytest

from kai.core.orchestration.routers import (
    route_deterministic,
    route_planning_phase,
)
from kai.core.orchestration.routers.first_execution import route_first_execution
from kai.core.orchestration.routers.standard_execution import route_standard_execution
from kai.core.orchestration.routers.standard_continue import route_standard_continue_branch
from kai.core.orchestration.routers.standard_retry import route_standard_retry_branch
from kai.core.orchestration.routers.backtracking import route_backtracking_branch


def mock_send_message(msg: str) -> None:
    """Mock send_message callback for testing."""
    pass


# =============================================================================
# route_deterministic tests
# =============================================================================

class TestRouteDeterministic:
    """Test the main deterministic router entry point."""

    def test_no_tasks_returns_complete(self):
        """When no tasks exist, should return complete."""
        state = {"task_list": {"tasks": []}}
        assert route_deterministic(state) == "complete"

    def test_empty_task_list_returns_complete(self):
        """When task_list is empty dict, should return complete."""
        state = {"task_list": {}}
        assert route_deterministic(state) == "complete"

    def test_first_iteration_all_pending_activates_task(self):
        """First iteration with all pending tasks should activate first task."""
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "pending"},
                {"id": 2, "status": "pending"},
            ]},
            "autonomous_mode_continue": False,
            "auto_mode_first_execution_done": False,
        }
        assert route_deterministic(state) == "mark_next_task_active"

    def test_first_iteration_has_active_confirm_plan_true_exits(self):
        """First iteration with active task and confirm_plan=True should exit to UI."""
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "active"},
                {"id": 2, "status": "pending"},
            ]},
            "autonomous_mode_continue": False,
            "auto_mode_first_execution_done": False,
            "confirm_plan": True,
        }
        assert route_deterministic(state) == "complete"

    def test_first_iteration_learning_mode_exits_to_ui(self):
        """First iteration with learning_mode=True should still exit to UI.

        ARCHITECTURE NOTE: After refactor, learning explanation runs in a
        SEPARATE graph AFTER code execution, not during routing. The main
        execution graph is identical regardless of learning_mode.
        """
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "active"},
                {"id": 2, "status": "pending"},
            ]},
            "autonomous_mode_continue": False,
            "auto_mode_first_execution_done": False,
            "confirm_plan": True,
            "learning_mode": True,
        }
        # Learning mode no longer affects routing - exits to UI same as non-learning
        assert route_deterministic(state) == "complete"

    def test_first_iteration_has_active_confirm_plan_false_continues(self):
        """First iteration with active task and confirm_plan=False continues to first_execution."""
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "active"},
                {"id": 2, "status": "pending"},
            ]},
            "autonomous_mode_continue": False,
            "auto_mode_first_execution_done": False,
            "confirm_plan": False,
            "active_task_objective": "Test task",
        }
        # Should continue to first_execution logic, which needs positioning
        result = route_deterministic(state)
        assert result == "cell_positioning"

    def test_first_iteration_no_active_or_pending_returns_complete(self):
        """First iteration with no active/pending tasks returns complete."""
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "completed"},
            ]},
            "autonomous_mode_continue": False,
            "auto_mode_first_execution_done": False,
        }
        assert route_deterministic(state) == "complete"

    def test_all_tasks_complete_returns_complete(self):
        """When all tasks are completed, should return complete."""
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "completed"},
                {"id": 2, "status": "completed"},
            ]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
        }
        assert route_deterministic(state) == "complete"

    def test_first_execution_not_done_routes_to_first_execution(self):
        """When first execution not done, should route to first_execution logic."""
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "active"},
            ]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": False,
        }
        # first_execution needs positioning
        assert route_deterministic(state) == "cell_positioning"

    def test_standard_execution_after_first_done(self):
        """After first execution done, should route to standard_execution."""
        state = {
            "task_list": {"tasks": [
                {"id": 1, "status": "active"},
            ]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            # standard_execution first analyzes completion
        }
        assert route_deterministic(state) == "autonomous_mark_completion"


# =============================================================================
# route_first_execution tests
# =============================================================================

class TestRouteFirstExecution:
    """Test first execution phase routing."""

    def test_already_done_returns_complete(self):
        """Defensive: if first_exec already done, return complete."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": True,
        }
        assert route_first_execution(state) == "complete"

    def test_no_active_task_activates_next(self):
        """When no active task, should activate next."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "pending"}]},
            "auto_mode_first_execution_done": False,
        }
        assert route_first_execution(state) == "mark_next_task_active"

    def test_no_positioning_gets_positioning(self):
        """When no positioning, should get cell_positioning."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": None,
        }
        assert route_first_execution(state) == "cell_positioning"

    # --- Code task paths ---
    def test_code_task_no_code_generates(self):
        """Code task without code should generate."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": False,
            "generated_code": None,
        }
        assert route_first_execution(state) == "code_generation_with_guidance"

    def test_code_task_with_code_marks_done(self):
        """Code task with code should mark first execution done."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": False,
            "generated_code": "print('hello')",
        }
        assert route_first_execution(state) == "mark_first_execution_done"

    # --- Reasoning task paths ---
    def test_reasoning_task_no_response_generates(self):
        """Reasoning task without response should generate."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": None,
            "reasoning_grade": None,
            "reasoning_evaluation_iteration": 0,
        }
        assert route_first_execution(state) == "reasoning_response_with_guidance"

    def test_reasoning_task_response_no_grade_evaluates(self):
        """Reasoning task with response but no grade should evaluate."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Some reasoning",
            "reasoning_grade": None,
            "reasoning_evaluation_iteration": 0,
        }
        assert route_first_execution(state) == "reasoning_evaluator"

    def test_reasoning_task_rejected_regenerates(self):
        """Reasoning task with REJECTED grade should regenerate."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Some reasoning",
            "reasoning_grade": "REJECTED",
            "reasoning_evaluation_iteration": 1,
        }
        assert route_first_execution(state) == "reasoning_response_with_guidance"

    def test_reasoning_task_approved_marks_complete(self):
        """Reasoning task with APPROVED should mark reasoning completed."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Some reasoning",
            "reasoning_grade": "APPROVED",
            "reasoning_evaluation_iteration": 1,
        }
        assert route_first_execution(state, mock_send_message) == "mark_reasoning_completed"

    def test_reasoning_task_max_iterations_marks_complete(self):
        """Reasoning task at max iterations should mark reasoning completed."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Some reasoning",
            "reasoning_grade": "REJECTED",  # Still not approved
            "reasoning_evaluation_iteration": 2,  # Max reached
        }
        assert route_first_execution(state, mock_send_message) == "mark_reasoning_completed"

    # --- Learning mode tests (ARCHITECTURE NOTE) ---
    # After refactor, learning explanation runs in SEPARATE learning graph
    # AFTER code execution, not during first_execution routing.
    # The execution graph is identical regardless of learning_mode.
    # These tests verify learning_mode does NOT affect first_execution routing.

    def test_learning_mode_does_not_affect_first_execution_routing(self):
        """Learning mode should NOT affect first_execution routing.

        After refactor, learning runs in separate graph after execution.
        """
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": None,
            "learning_mode": True,
        }
        # Should go straight to positioning regardless of learning_mode
        assert route_first_execution(state) == "cell_positioning"

    def test_learning_mode_false_goes_to_positioning(self):
        """When learning mode is off, should go straight to positioning."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "auto_mode_first_execution_done": False,
            "positioning_info": None,
            "learning_mode": False,
        }
        assert route_first_execution(state) == "cell_positioning"


# =============================================================================
# route_standard_execution tests
# =============================================================================

class TestRouteStandardExecution:
    """Test standard execution phase routing."""

    def test_not_analyzed_analyzes_completion(self):
        """When completion not analyzed, should analyze."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": False,
        }
        assert route_standard_execution(state) == "autonomous_mark_completion"

    def test_analyzed_no_backup_backs_up(self):
        """After analysis, should backup task list before update."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": False,
            "task_list_backup": None,
            "retry_objective": None,
            "recovery_objective": None,
            "last_execution_failed": False,
        }
        assert route_standard_execution(state) == "backup_task_list"

    def test_analyzed_with_backup_updates_tasks(self):
        """After backup, should update tasks."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": False,
            "task_list_backup": {"tasks": []},
            "retry_objective": None,
            "recovery_objective": None,
            "last_execution_failed": False,
        }
        assert route_standard_execution(state) == "autonomous_update_tasks"

    def test_standard_retry_skips_task_update(self):
        """Standard retry should skip task updates and go to mark_next_task_active."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": False,
            "last_execution_failed": True,  # Standard retry
            "retry_objective": None,
            "recovery_objective": None,
            "next_task_activated": False,
        }
        assert route_standard_execution(state) == "mark_next_task_active"

    def test_all_complete_returns_complete(self):
        """When all tasks completed, should return complete."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "completed"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "retry_objective": None,
            "recovery_objective": None,
            "last_execution_failed": False,
        }
        assert route_standard_execution(state) == "complete"

    def test_rag_retry_assembles_query(self):
        """Standard retry with RAG enabled should assemble query first."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "last_execution_failed": True,
            "retry_objective": None,
            "recovery_objective": None,
            "rag_enabled": True,
            "rag_query_assembled": False,
        }
        assert route_standard_execution(state) == "assemble_rag_query"

    def test_rag_retry_retrieves_after_assembly(self):
        """After RAG query assembled, should retrieve."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "last_execution_failed": True,
            "retry_objective": None,
            "recovery_objective": None,
            "rag_enabled": True,
            "rag_query_assembled": True,
            "rag_retrieval": None,
        }
        assert route_standard_execution(state) == "rag_retrieval"

    def test_backtracking_branch_detection(self):
        """Recovery objective triggers backtracking branch."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "recovery_objective": "Need to backtrack",  # Triggers backtracking
            "last_execution_failed": False,
            "retry_objective": None,
        }
        # Backtracking branch first does backtrack_recovery
        assert route_standard_execution(state) == "backtrack_recovery"

    def test_standard_retry_branch_detection(self):
        """Error without recovery_objective triggers standard retry."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "last_execution_failed": True,
            "retry_objective": None,
            "recovery_objective": None,
            "rag_enabled": False,
        }
        # Standard retry first does error_recovery
        assert route_standard_execution(state) == "error_recovery"

    def test_standard_continue_branch_detection(self):
        """No error or retry triggers standard continue."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "last_execution_failed": False,
            "retry_objective": None,
            "recovery_objective": None,
        }
        # Standard continue needs positioning
        assert route_standard_execution(state) == "set_positioning_from_last_cell"

    # =========================================================================
    # Learning mode tests (ARCHITECTURE NOTE)
    # =========================================================================
    # After refactor, learning explanation runs in SEPARATE learning graph
    # AFTER code execution, not during standard_execution routing.
    # The execution graph is identical regardless of learning_mode.
    # These tests verify learning_mode does NOT affect standard_execution routing.

    def test_learning_mode_does_not_affect_standard_execution_routing(self):
        """Learning mode should NOT affect standard_execution routing.

        After refactor, learning runs in separate graph after execution.
        """
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "last_execution_failed": False,
            "retry_objective": None,
            "recovery_objective": None,
            "learning_mode": True,
        }
        # Should go to positioning regardless of learning_mode
        assert route_standard_execution(state) == "set_positioning_from_last_cell"

    def test_learning_mode_disabled_goes_to_positioning(self):
        """When learning mode is off, should go to positioning."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": True,
            "next_task_activated": True,
            "last_execution_failed": False,
            "retry_objective": None,
            "recovery_objective": None,
            "learning_mode": False,
        }
        # Should go directly to standard_continue branch (positioning)
        assert route_standard_execution(state) == "set_positioning_from_last_cell"

    def test_learning_mode_does_not_affect_error_handling(self):
        """Learning mode should NOT affect error handling routing."""
        state = {
            "task_list": {"tasks": [{"id": 1, "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": False,  # Skipped in retry
            "next_task_activated": False,
            "last_execution_failed": True,  # Error occurred
            "retry_objective": None,
            "recovery_objective": None,
            "learning_mode": True,
        }
        # Error path should be the same regardless of learning_mode
        assert route_standard_execution(state) == "mark_next_task_active"


# =============================================================================
# route_standard_continue_branch tests
# =============================================================================

class TestRouteStandardContinueBranch:
    """Test standard continue branch (Branch 2)."""

    def test_no_positioning_sets_positioning(self):
        """When no positioning, should set from last cell."""
        state = {"positioning_info": None}
        assert route_standard_continue_branch(state) == "set_positioning_from_last_cell"

    def test_code_task_no_code_generates(self):
        """Code task without code should generate."""
        state = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": False,
            "generated_code": None,
        }
        assert route_standard_continue_branch(state) == "code_generation_with_guidance"

    def test_code_task_with_code_completes(self):
        """Code task with code should complete."""
        state = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": False,
            "generated_code": "print('hello')",
        }
        assert route_standard_continue_branch(state) == "complete"


# =============================================================================
# route_standard_retry_branch tests
# =============================================================================

class TestRouteStandardRetryBranch:
    """Test standard retry branch (Branch 3)."""

    def test_no_strategy_determines_strategy(self):
        """When no recovery strategy, should determine it."""
        state = {"error_recovery_strategy": None}
        assert route_standard_retry_branch(state) == "error_recovery"

    def test_no_positioning_sets_positioning(self):
        """After strategy, should set positioning."""
        state = {
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": None,
        }
        assert route_standard_retry_branch(state) == "set_positioning_from_last_cell"

    def test_code_task_no_code_updates(self):
        """Code task without update should call code_update."""
        state = {
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "Some code task",
            "generated_code": None,
            "reasoning_response": None,
        }
        assert route_standard_retry_branch(state) == "code_update"

    def test_code_task_with_code_completes(self):
        """Code task with updated code should complete."""
        state = {
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "Some code task",
            "generated_code": "print('fixed')",
        }
        assert route_standard_retry_branch(state) == "complete"

    def test_reasoning_task_detected_from_objective(self):
        """Reasoning task should be detected from active_task_objective."""
        state = {
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "[reasoning] Explain something",
            "reasoning_response": None,
            "reasoning_grade": None,
            "reasoning_evaluation_iteration": 0,
        }
        assert route_standard_retry_branch(state) == "reasoning_response_with_guidance"

    def test_reasoning_max_iterations_marks_complete(self):
        """Reasoning at max iterations should mark complete."""
        state = {
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "[reasoning] Explain something",
            "reasoning_response": "Some reasoning",
            "reasoning_grade": "REJECTED",
            "reasoning_evaluation_iteration": 2,  # Max reached
        }
        assert route_standard_retry_branch(state, mock_send_message) == "mark_reasoning_completed"

    def test_code_update_workflow_output_allows_completion(self):
        """Regression test: router should exit to 'complete' after code_update.

        This tests the scenario where code_update has completed and set
        generated_code. The router should detect this and return "complete"
        regardless of what other fields are cleared.

        Previously, code_update was clearing error_recovery_strategy, which
        caused the router to call error_recovery again before exiting.
        """
        # Simulate state AFTER code_update's workflow_output is applied
        state = {
            "error_recovery_strategy": "REPLACE_AND_RETRY",  # Still set from before
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "Install packages",
            "generated_code": "# Fixed code",  # Set by code_update
            # These are cleared by code_update workflow_output:
            "cells_to_delete": None,
            "cells_deleted": None,
            "backtrack_recovery_done": None,
            "recovery_objective": None,
            "retry_objective": None,
        }
        # Router should exit to "complete" because generated_code is set
        assert route_standard_retry_branch(state) == "complete"


# =============================================================================
# route_backtracking_branch tests
# =============================================================================

class TestRouteBacktrackingBranch:
    """Test backtracking branch (Branch 4)."""

    def test_no_recovery_done_recovers(self):
        """When backtrack not started, should call backtrack_recovery."""
        state = {"backtrack_recovery_done": False}
        assert route_backtracking_branch(state) == "backtrack_recovery"

    def test_no_cells_to_delete_selects_cells(self):
        """After recovery, should select cells to delete."""
        state = {
            "backtrack_recovery_done": True,
            "cells_to_delete": None,
        }
        assert route_backtracking_branch(state) == "cell_selection_deletion"

    def test_cells_selected_deletes(self):
        """After selection, should delete cells."""
        state = {
            "backtrack_recovery_done": True,
            "cells_to_delete": [1, 2],
            "cells_deleted": False,
        }
        assert route_backtracking_branch(state) == "cell_deletion"

    def test_cells_deleted_positions(self):
        """After deletion, should get positioning."""
        state = {
            "backtrack_recovery_done": True,
            "cells_to_delete": [1, 2],
            "cells_deleted": True,
            "positioning_info": None,
        }
        assert route_backtracking_branch(state) == "cell_positioning"

    def test_positioned_generates_code(self):
        """After positioning, should generate code."""
        state = {
            "backtrack_recovery_done": True,
            "cells_to_delete": [1, 2],
            "cells_deleted": True,
            "positioning_info": {"target_cell": 3},
            "generated_code": None,
        }
        assert route_backtracking_branch(state) == "code_generation_with_guidance"

    def test_code_generated_completes(self):
        """After code generation, should complete."""
        state = {
            "backtrack_recovery_done": True,
            "cells_to_delete": [1, 2],
            "cells_deleted": True,
            "positioning_info": {"target_cell": 3},
            "generated_code": "print('new code')",
        }
        assert route_backtracking_branch(state) == "complete"


# =============================================================================
# route_planning_phase tests
# =============================================================================

class TestRoutePlanningPhase:
    """Test planning phase routing."""

    def test_initial_with_rag_searches(self):
        """Initial phase with RAG enabled should search workflows."""
        state = {
            "planning_phase": None,
            "rag_enabled": True,
            "retrieval_queries": ["test query"],
        }
        assert route_planning_phase(state) == "search_workflows"

    def test_initial_without_rag_increments(self):
        """Initial phase without RAG should increment iteration."""
        state = {
            "planning_phase": None,
            "rag_enabled": False,
        }
        assert route_planning_phase(state) == "increment_task_planning_iteration"

    def test_initial_with_rag_no_queries_increments(self):
        """Initial phase with RAG but no queries should increment."""
        state = {
            "planning_phase": None,
            "rag_enabled": True,
            "retrieval_queries": [],
        }
        assert route_planning_phase(state) == "increment_task_planning_iteration"

    def test_workflow_retrieval_max_iterations_increments(self):
        """Workflow retrieval at max should increment."""
        state = {
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 2,  # Max
        }
        assert route_planning_phase(state) == "increment_task_planning_iteration"

    def test_workflow_retrieval_continues(self):
        """Workflow retrieval with queries should continue."""
        state = {
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 1,
            "retrieval_queries": ["more queries"],
        }
        assert route_planning_phase(state) == "search_workflows"

    def test_task_planning_with_rag_refines(self):
        """Task planning with RAG should run workflow_refinement."""
        state = {
            "planning_phase": "task_planning",
            "task_planning_iteration": 1,
            "rag_enabled": True,
        }
        assert route_planning_phase(state, max_task_planning_iterations=10) == "workflow_refinement"

    def test_task_planning_without_rag_evaluation_enabled_evaluates(self):
        """Task planning without RAG but evaluation enabled should evaluate."""
        state = {
            "planning_phase": "task_planning",
            "task_planning_iteration": 1,
            "rag_enabled": False,
            "use_critique": True,
            "task_list_grade": None,
        }
        assert route_planning_phase(state, max_task_planning_iterations=10) == "task_list_evaluator"

    def test_task_planning_approved_completes(self):
        """Task planning with approval should complete."""
        state = {
            "planning_phase": "task_planning",
            "task_planning_iteration": 1,
            "rag_enabled": False,
            "use_critique": True,
            "task_list_grade": "APPROVED",
        }
        assert route_planning_phase(state, max_task_planning_iterations=10) == "complete"

    def test_task_planning_max_iterations_completes(self):
        """Task planning at max iterations should complete."""
        state = {
            "planning_phase": "task_planning",
            "task_planning_iteration": 10,  # Max
            "rag_enabled": False,
        }
        assert route_planning_phase(state, max_task_planning_iterations=10) == "complete"

    def test_task_planning_max_iterations_with_rag_filters(self):
        """Task planning at max with RAG should filter_and_complete."""
        state = {
            "planning_phase": "task_planning",
            "task_planning_iteration": 10,
            "rag_enabled": True,
        }
        assert route_planning_phase(state, max_task_planning_iterations=10) == "filter_and_complete"

    def test_evaluation_approved_completes(self):
        """After evaluation with approval should complete."""
        state = {
            "planning_phase": "task_list_evaluation",
            "task_list_grade": "APPROVED",
            "rag_enabled": False,
        }
        assert route_planning_phase(state) == "complete"

    def test_evaluation_rejected_increments(self):
        """After evaluation with rejection should increment."""
        state = {
            "planning_phase": "task_list_evaluation",
            "task_list_grade": "REJECTED",
        }
        assert route_planning_phase(state) == "increment_task_planning_iteration"

    def test_evaluation_no_result_completes_anyway(self):
        """After evaluation with no result should complete anyway."""
        state = {
            "planning_phase": "task_list_evaluation",
            "task_list_grade": None,
            "rag_enabled": False,
        }
        assert route_planning_phase(state) == "complete"

    def test_workflow_refinement_with_queries_increments(self):
        """After workflow refinement with queries should increment."""
        state = {
            "planning_phase": "workflow_refinement",
            "had_retrieval_queries_before_refinement": True,
        }
        assert route_planning_phase(state) == "increment_task_planning_iteration"

    def test_workflow_refinement_no_queries_evaluates(self):
        """After workflow refinement without queries should evaluate."""
        state = {
            "planning_phase": "workflow_refinement",
            "had_retrieval_queries_before_refinement": False,
            "use_critique": True,
        }
        assert route_planning_phase(state) == "task_list_evaluator"

    def test_workflow_refinement_no_queries_no_critique_completes(self):
        """After refinement without queries and no critique should complete."""
        state = {
            "planning_phase": "workflow_refinement",
            "had_retrieval_queries_before_refinement": False,
            "use_critique": False,
            "rag_enabled": False,
        }
        assert route_planning_phase(state) == "complete"

    def test_ready_to_generate_generates(self):
        """Ready to generate phase should generate task list."""
        state = {
            "planning_phase": "ready_to_generate",
            "task_planning_iteration": 1,
        }
        assert route_planning_phase(state, max_task_planning_iterations=10) == "task_list_generation"

    def test_ready_to_generate_max_iterations_completes(self):
        """Ready to generate at max iterations should complete."""
        state = {
            "planning_phase": "ready_to_generate",
            "task_planning_iteration": 10,
            "rag_enabled": False,
        }
        assert route_planning_phase(state, max_task_planning_iterations=10) == "complete"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
