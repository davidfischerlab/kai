"""Comprehensive tests for deterministic router covering all branches."""

import pytest
from unittest.mock import Mock
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator


class TestPlanningPhaseRouting:
    """Tests for _route_planning_phase."""

    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_initial_planning_with_rag_routes_to_search_workflows(self, orchestrator):
        """First planning iteration with RAG enabled → search_workflows."""
        state = {
            "task_list": {},
            "retrieval_queries": ["Analyze data"],
            "rag_enabled": True,
            "planning_phase": None,  # First time
            "workflow_retrieval_iteration": 0,
        }
        result = orchestrator._route_planning_phase(state)
        assert result == "search_workflows"

    def test_initial_planning_without_rag_routes_to_plan_tasks(self, orchestrator):
        """First planning iteration with RAG disabled → increment_task_planning_iteration."""
        state = {
            "task_list": {},
            "retrieval_queries": [],
            "rag_enabled": False,
            "planning_phase": None,
            "workflow_retrieval_iteration": 0,
        }
        result = orchestrator._route_planning_phase(state)
        # Matches kai_dev: initial planning → increment counter → task_list_generation
        assert result == "increment_task_planning_iteration"

    def test_workflow_retrieval_max_iterations_exits(self, orchestrator):
        """After 2 iterations → increment_task_planning_iteration."""
        state = {
            "task_list": {},
            "retrieval_queries": ["More queries"],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 2,  # Max reached
        }
        result = orchestrator._route_planning_phase(state)
        # Matches kai_dev: workflow retrieval complete → increment counter → task_list_generation
        assert result == "increment_task_planning_iteration"

    def test_workflow_retrieval_empty_queries_exits(self, orchestrator):
        """Empty queries → increment_task_planning_iteration."""
        state = {
            "task_list": {},
            "retrieval_queries": [],  # Empty
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 1,
        }
        result = orchestrator._route_planning_phase(state)
        # Matches kai_dev: workflow retrieval complete (no more queries) → increment counter → task_list_generation
        assert result == "increment_task_planning_iteration"

    def test_workflow_retrieval_continues_iteration_1(self, orchestrator):
        """Iteration 1 with queries → continues to search_workflows."""
        state = {
            "task_list": {},
            "retrieval_queries": ["Query"],
            "rag_enabled": True,
            "planning_phase": "workflow_retrieval",
            "workflow_retrieval_iteration": 1,
        }
        result = orchestrator._route_planning_phase(state)
        assert result == "search_workflows"


class TestFirstExecutionRouting:
    """Tests for _route_first_execution."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_need_positioning(self, orchestrator):
        """No positioning info → cell_positioning."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": False,
            "positioning_info": None,  # Missing
            "generated_code": None,
        }
        result = orchestrator._route_first_execution(state)
        assert result == "cell_positioning"

    def test_need_code_generation(self, orchestrator):
        """Has positioning, no code → code_generation_with_guidance."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 0},
            "generated_code": None,  # Missing
        }
        result = orchestrator._route_first_execution(state)
        assert result == "code_generation_with_guidance"

    def test_mark_first_exec_done(self, orchestrator):
        """Has positioning and code, but flag not set → mark_first_execution_done."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": False,  # Not set yet
            "positioning_info": {"target_cell": 0},
            "generated_code": "print('hello')",
        }
        result = orchestrator._route_first_execution(state)
        assert result == "mark_first_execution_done"

    def test_first_execution_complete(self, orchestrator):
        """Everything done → complete."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,  # Set
            "positioning_info": {"target_cell": 0},
            "generated_code": "print('hello')",
        }
        result = orchestrator._route_first_execution(state)
        assert result == "complete"


class TestStandardExecutionBranches:
    """Tests for _route_standard_execution and its branches."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_analyze_completion_first(self, orchestrator):
        """First action is always analyze completion."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": False,  # Not analyzed yet
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "autonomous_mark_completion"

    def test_all_complete_exits(self, orchestrator):
        """All tasks completed → complete."""
        state = {
            "task_list": {"tasks": [
                {"status": "completed"},
                {"status": "completed"}
            ]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": True,  # Need this set
            "tasks_updated": True,  # And this
            "update_approved": True,  # And this
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "complete"

    def test_standard_continue_needs_positioning(self, orchestrator):
        """Standard continue without positioning → set_positioning_from_last_cell."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": True,
            "tasks_updated": True,
            "update_approved": True,
            "positioning_info": None,  # Missing
            "last_cell_modified_in_auto_mode": 5,
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "set_positioning_from_last_cell"

    def test_standard_continue_code_task_needs_code(self, orchestrator):
        """Code task without generated code → code_generation_with_guidance."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": True,
            "tasks_updated": True,
            "update_approved": True,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": False,
            "generated_code": None,  # Missing
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "code_generation_with_guidance"

    def test_standard_continue_reasoning_task_needs_response(self, orchestrator):
        """Reasoning task without response → reasoning_response_with_guidance."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": True,
            "tasks_updated": True,
            "update_approved": True,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": None,  # Missing
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "reasoning_response_with_guidance"

    def test_standard_retry_needs_error_recovery(self, orchestrator):
        """Error without recovery strategy → error_recovery."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "last_execution_failed": True,  # Error
            "next_task_activated": True,
            "tasks_updated": True,
            "update_approved": True,
            "error_recovery_strategy": None,  # Not set yet
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "error_recovery"

    def test_standard_retry_with_rag_needs_retrieval(self, orchestrator):
        """Standard retry with RAG enabled → rag_retrieval."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "last_execution_failed": True,
            "retry_objective": "Fix error",
            "next_task_activated": True,
            "tasks_updated": True,
            "update_approved": True,
            "rag_enabled": True,
            "rag_retrieval": None,  # Not retrieved yet
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "rag_retrieval"

    def test_backtracking_needs_recovery(self, orchestrator):
        """Backtracking without recovery done → backtrack_recovery."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "recovery_objective": "Backtrack to task 2",  # Backtracking
            "next_task_activated": True,
            "tasks_updated": True,
            "update_approved": True,
            "backtrack_recovery_done": False,  # Not done yet
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "backtrack_recovery"

    def test_task_update_needs_backup(self, orchestrator):
        """Before task update → backup_task_list."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "tasks_updated": False,  # Not updated yet
            "task_list_backup": None,  # No backup yet
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "backup_task_list"

    def test_task_update_with_backup(self, orchestrator):
        """Has backup → autonomous_update_tasks."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "tasks_updated": False,
            "task_list_backup": {"tasks": []},  # Has backup
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "autonomous_update_tasks"

    def test_task_update_critique_needed(self, orchestrator):
        """Task updated but not approved → autonomous_update_critique."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "task_list_update_rule": "UPDATE",
            "update_approved": False,  # Not approved
            "critique_iteration": 0,
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "autonomous_update_critique"

    def test_task_update_max_iterations_revert(self, orchestrator):
        """Max critique iterations without approval → revert_task_list."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "task_list_update_rule": "UPDATE",
            "update_approved": False,
            "critique_iteration": 3,  # Max reached
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "revert_task_list"

    def test_activate_next_task(self, orchestrator):
        """After completion analysis → mark_next_task_active."""
        state = {
            "task_list": {"tasks": [{"status": "active"}]},
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_completion_analyzed": True,
            "next_task_activated": False,  # Not activated yet
            "tasks_updated": True,
            "update_approved": True,
        }
        result = orchestrator._route_standard_execution(state)
        assert result == "mark_next_task_active"


class TestBacktrackingBranch:
    """Tests for _route_backtracking_branch."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_backtrack_needs_recovery(self, orchestrator):
        """First step → backtrack_recovery."""
        state = {
            "recovery_objective": "Backtrack",
            "backtrack_recovery_done": False,
        }
        result = orchestrator._route_backtracking_branch(state)
        assert result == "backtrack_recovery"

    def test_backtrack_needs_cell_selection(self, orchestrator):
        """After recovery → cell_selection_deletion."""
        state = {
            "recovery_objective": "Backtrack",
            "backtrack_recovery_done": True,
            "cells_to_delete": None,  # Not selected
        }
        result = orchestrator._route_backtracking_branch(state)
        assert result == "cell_selection_deletion"

    def test_backtrack_needs_deletion(self, orchestrator):
        """After selection → cell_deletion."""
        state = {
            "recovery_objective": "Backtrack",
            "backtrack_recovery_done": True,
            "cells_to_delete": [5, 6],
            "cells_deleted": False,  # Not deleted
        }
        result = orchestrator._route_backtracking_branch(state)
        assert result == "cell_deletion"

    def test_backtrack_needs_positioning(self, orchestrator):
        """After deletion → cell_positioning."""
        state = {
            "recovery_objective": "Backtrack",
            "backtrack_recovery_done": True,
            "cells_to_delete": [5, 6],
            "cells_deleted": True,
            "positioning_info": None,  # Not set
        }
        result = orchestrator._route_backtracking_branch(state)
        assert result == "cell_positioning"

    def test_backtrack_needs_restart(self, orchestrator):
        """Restart required → restart_and_rerun_prompt."""
        state = {
            "recovery_objective": "Backtrack",
            "backtrack_recovery_done": True,
            "cells_to_delete": [5, 6],
            "cells_deleted": True,
            "positioning_info": {"target_cell": 4},
            "restart_required": True,  # Restart needed
        }
        result = orchestrator._route_backtracking_branch(state)
        assert result == "restart_and_rerun_prompt"

    def test_backtrack_needs_code_generation(self, orchestrator):
        """After restart → code_generation_with_guidance."""
        state = {
            "recovery_objective": "Backtrack",
            "backtrack_recovery_done": True,
            "cells_to_delete": [5, 6],
            "cells_deleted": True,
            "positioning_info": {"target_cell": 4},
            "restart_required": False,  # Already done
            "generated_code": None,  # Need code
        }
        result = orchestrator._route_backtracking_branch(state)
        assert result == "code_generation_with_guidance"

    def test_backtrack_complete(self, orchestrator):
        """All done → complete."""
        state = {
            "recovery_objective": "Backtrack",
            "backtrack_recovery_done": True,
            "cells_to_delete": [5, 6],
            "cells_deleted": True,
            "positioning_info": {"target_cell": 4},
            "restart_required": False,
            "generated_code": "print('new code')",
        }
        result = orchestrator._route_backtracking_branch(state)
        assert result == "complete"


class TestStandardRetryBranch:
    """Tests for _route_standard_retry_branch."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_retry_needs_strategy(self, orchestrator):
        """No strategy → error_recovery."""
        state = {
            "last_execution_failed": True,
            "error_recovery_strategy": None,
        }
        result = orchestrator._route_standard_retry_branch(state)
        assert result == "error_recovery"

    def test_retry_needs_positioning(self, orchestrator):
        """Has strategy, no positioning → set_positioning_from_last_cell."""
        state = {
            "last_execution_failed": True,
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": None,
        }
        result = orchestrator._route_standard_retry_branch(state)
        assert result == "set_positioning_from_last_cell"

    def test_retry_replace_and_restart_needs_restart(self, orchestrator):
        """REPLACE_AND_RESTART without restart → restart_and_rerun_prompt."""
        state = {
            "last_execution_failed": True,
            "error_recovery_strategy": "REPLACE_AND_RESTART",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "Load data",
            "restart_required": True,  # Not done yet (tool sets to False after)
        }
        result = orchestrator._route_standard_retry_branch(state)
        assert result == "restart_and_rerun_prompt"

    def test_retry_replace_and_restart_needs_code(self, orchestrator):
        """After restart → code_update."""
        state = {
            "last_execution_failed": True,
            "error_recovery_strategy": "REPLACE_AND_RESTART",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "Load data",
            "restart_required": False,  # Done
            "generated_code": None,
        }
        result = orchestrator._route_standard_retry_branch(state)
        assert result == "code_update"

    def test_retry_replace_and_retry_needs_code(self, orchestrator):
        """REPLACE_AND_RETRY → code_update."""
        state = {
            "last_execution_failed": True,
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "Load data",
            "generated_code": None,
        }
        result = orchestrator._route_standard_retry_branch(state)
        assert result == "code_update"

    def test_retry_reasoning_task_needs_response(self, orchestrator):
        """Reasoning task → reasoning_response_with_guidance."""
        state = {
            "last_execution_failed": True,
            "error_recovery_strategy": "REPLACE_AND_RETRY",
            "positioning_info": {"target_cell": 5},
            "active_task_objective": "[Reasoning] Analyze results",
            "reasoning_response": None,
        }
        result = orchestrator._route_standard_retry_branch(state)
        assert result == "reasoning_response_with_guidance"


class TestStandardContinueBranch:
    """Tests for _route_standard_continue_branch."""

    @pytest.fixture
    def orchestrator(self):
        llm = Mock()
        kb = Mock()
        return LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    def test_continue_needs_positioning(self, orchestrator):
        """No positioning → set_positioning_from_last_cell."""
        state = {
            "positioning_info": None,
            "last_cell_modified_in_auto_mode": 5,
        }
        result = orchestrator._route_standard_continue_branch(state)
        assert result == "set_positioning_from_last_cell"

    def test_continue_code_task_needs_code(self, orchestrator):
        """Code task → code_generation_with_guidance."""
        state = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": False,
            "generated_code": None,
        }
        result = orchestrator._route_standard_continue_branch(state)
        assert result == "code_generation_with_guidance"

    def test_continue_code_task_complete(self, orchestrator):
        """Code generated → complete."""
        state = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": False,
            "generated_code": "print('hello')",
        }
        result = orchestrator._route_standard_continue_branch(state)
        assert result == "complete"

    def test_continue_reasoning_needs_response(self, orchestrator):
        """Reasoning task → reasoning_response_with_guidance."""
        state = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": None,
        }
        result = orchestrator._route_standard_continue_branch(state)
        assert result == "reasoning_response_with_guidance"

    def test_continue_reasoning_needs_critique(self, orchestrator):
        """Reasoning not approved → reasoning_critique."""
        state = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Here is my reasoning...",
            "reasoning_approval": "NOT_APPROVED",
            "critique_iteration": 0,
        }
        result = orchestrator._route_standard_continue_branch(state)
        assert result == "reasoning_critique"

    def test_continue_reasoning_complete(self, orchestrator):
        """Reasoning approved → mark_reasoning_completed (marks task complete and exits)."""
        state = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Here is my reasoning...",
            "reasoning_approval": "APPROVED",
            "critique_iteration": 1,
        }
        result = orchestrator._route_standard_continue_branch(state)
        assert result == "mark_reasoning_completed"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
