"""Tests for critique loops, guards, and iteration limits.

This module tests loop control mechanisms in the LangGraph orchestrator:
- Task update critique loop: UPDATE -> CRITIQUE -> (if rejected) UPDATE -> CRITIQUE
- Reasoning critique loop: similar pattern for reasoning tasks
- Max iteration limits (guards against infinite loops)
- Approval flows and regeneration after rejection
- Counter isolation (reasoning vs autonomous update)
- Completed status preservation
"""

import pytest
from unittest.mock import Mock

from kai.core.orchestration.routers import (
    route_standard_continue_branch,
)
from kai.core.orchestration.routers.standard_execution import route_standard_execution


def mock_send_message(msg: str) -> None:
    """Mock send_message callback for testing."""
    pass


class TestTaskUpdateCritiqueLoop:
    """Test task update critique loop routing."""

    def test_first_critique_called_when_approval_is_none(self):
        """First critique should be called when autonomous_update_approval is None."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "autonomous_update_critique_iteration": 0,
            "autonomous_update_approval": None,  # Not yet critiqued
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "autonomous_update_critique", \
            "Should call critique when autonomous_update_approval is None"

    def test_regenerate_called_when_approval_is_modify(self):
        """Regeneration should be called when critique returns MODIFY."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "autonomous_update_critique_iteration": 1,  # Already had one critique
            "autonomous_update_approval": "MODIFY",  # Critique rejected
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "autonomous_update_tasks", \
            "Should call autonomous_update_tasks when approval is MODIFY"

    def test_critique_called_after_regeneration(self):
        """Critique should be called after regeneration clears approval to None."""
        # State after regeneration (autonomous_update_tasks cleared approval)
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "autonomous_update_critique_iteration": 1,  # Still on iteration 1 (critique increments it)
            "autonomous_update_approval": None,  # Cleared by regeneration
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "autonomous_update_critique", \
            "Should call critique after regeneration clears approval to None"

    def test_approved_skips_further_critique(self):
        """When approved, should not call critique or regenerate."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "autonomous_update_critique_iteration": 1,
            "autonomous_update_approval": "APPROVED",
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
            "next_task_activated": False,
        }

        result = route_standard_execution(state, mock_send_message)
        # Should proceed past critique loop (not call critique or regenerate)
        assert result not in ["autonomous_update_critique", "autonomous_update_tasks", "revert_task_list"], \
            f"Should not call critique/regenerate when approved, got {result}"

    def test_max_iterations_triggers_revert(self):
        """Max iterations (3) should trigger revert to backup."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "autonomous_update_critique_iteration": 3,  # Max reached
            "autonomous_update_approval": "MODIFY",  # Still not approved
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "revert_task_list", \
            "Should revert to backup when max iterations reached"


class TestCritiqueLoopProgression:
    """Test full critique loop progression without infinite loops."""

    def test_critique_loop_progression_simulation(self):
        """Simulate full critique loop: UPDATE → CRITIQUE → UPDATE → CRITIQUE → APPROVE."""
        # Simulate the progression of states through the critique loop
        progression = []

        # Step 1: Initial state after first task update
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "autonomous_update_critique_iteration": 0,
            "autonomous_update_approval": None,
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=0, approval=None", result))
        assert result == "autonomous_update_critique"

        # Step 2: After first critique (rejected)
        state["autonomous_update_critique_iteration"] = 1
        state["autonomous_update_approval"] = "MODIFY"

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=1, approval=MODIFY", result))
        assert result == "autonomous_update_tasks"

        # Step 3: After regeneration (tool clears approval)
        state["autonomous_update_approval"] = None

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=1, approval=None", result))
        assert result == "autonomous_update_critique"

        # Step 4: After second critique (approved)
        state["autonomous_update_critique_iteration"] = 2
        state["autonomous_update_approval"] = "APPROVED"
        state["next_task_activated"] = False

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=2, approval=APPROVED", result))
        # Should proceed past critique loop (not call critique/regenerate)
        assert result not in [
            "autonomous_update_critique",
            "autonomous_update_tasks",
            "revert_task_list"
        ], f"Should proceed past critique loop when approved, got {result}"

    def test_no_infinite_loop_with_repeated_rejection(self):
        """Verify that repeated rejections eventually hit max iterations."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        actions = []
        critique_iteration = 0

        # Simulate up to 10 iterations (should hit max at 3)
        for i in range(10):
            state["autonomous_update_critique_iteration"] = critique_iteration
            state["autonomous_update_approval"] = None if i % 2 == 0 else "MODIFY"

            result = route_standard_execution(state, mock_send_message)
            actions.append(result)

            if result == "revert_task_list":
                break
            elif result == "autonomous_update_critique":
                # Critique increments counter
                critique_iteration += 1

        # Should have hit revert before 10 iterations
        assert "revert_task_list" in actions, \
            "Should eventually revert when max iterations reached"
        assert len(actions) <= 7, \
            f"Should not take more than 7 actions to reach revert (got {len(actions)})"


class TestReasoningCritiqueLoop:
    """Test reasoning critique loop routing.

    NOTE: These tests use route_standard_continue_branch directly because
    that's where the reasoning critique loop logic lives. The full router
    (route_standard_execution) has many pre-conditions (positioning, task updates, etc.)
    that would need to be satisfied first.
    """

    def test_reasoning_first_critique_called_when_approval_is_none(self):
        """First critique should be called when reasoning_approval is None."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test reasoning task", "status": "active"}]},
            "reasoning_critique_iteration": 0,
            "reasoning_response": "Some generated reasoning",
            "reasoning_approval": None,  # Not yet critiqued
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},  # Required for branch to process
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "reasoning_critique", \
            "Should call reasoning_critique when reasoning_approval is None and reasoning_response exists"

    def test_reasoning_regenerate_called_when_approval_is_modify(self):
        """Regeneration should be called when reasoning critique returns MODIFY."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test reasoning task", "status": "active"}]},
            "reasoning_critique_iteration": 1,
            "reasoning_response": "Some generated reasoning",
            "reasoning_approval": "MODIFY",  # Critique rejected
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "reasoning_response_with_guidance", \
            "Should call reasoning_response_with_guidance when approval is MODIFY"

    def test_reasoning_critique_called_after_regeneration(self):
        """Critique should be called after regeneration clears approval to None.

        This tests the fix for the infinite loop bug where reasoning_response_with_guidance
        wasn't clearing reasoning_approval, causing the router to keep regenerating.
        """
        # State after regeneration (reasoning_response_with_guidance cleared approval)
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test reasoning task", "status": "active"}]},
            "reasoning_critique_iteration": 1,  # Still on iteration 1 (critique increments it)
            "reasoning_response": "Regenerated reasoning",
            "reasoning_approval": None,  # CRITICAL: Must be cleared by regeneration tool
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "reasoning_critique", \
            "Should call reasoning_critique after regeneration clears approval to None"

    def test_reasoning_approved_marks_complete(self):
        """When approved, should proceed to mark_reasoning_completed."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test reasoning task", "status": "active"}]},
            "reasoning_critique_iteration": 1,
            "reasoning_response": "Approved reasoning",
            "reasoning_approval": "APPROVED",
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "mark_reasoning_completed", \
            "Should proceed to mark_reasoning_completed when approved"

    def test_reasoning_max_iterations_proceeds_anyway(self):
        """Max iterations (2) should proceed to mark_reasoning_completed anyway."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test reasoning task", "status": "active"}]},
            "reasoning_critique_iteration": 2,  # Max reached
            "reasoning_response": "Still not approved reasoning",
            "reasoning_approval": "MODIFY",  # Still not approved
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "mark_reasoning_completed", \
            "Should proceed to mark_reasoning_completed when max iterations reached"

    def test_reasoning_tool_clears_approval_for_next_critique(self):
        """Verify reasoning_response_with_guidance tool sets reasoning_approval to None."""
        import asyncio
        from kai.core.tools import ReasoningResponseWithGuidanceTool

        llm_mock = Mock()
        tool = ReasoningResponseWithGuidanceTool(llm_mock)

        state = {
            "user_query": "Continue",
            "positioning_info": {"target_cell": 5},
            "reasoning_critique": "Some critique",  # Indicates regeneration
            "task_list": {"tasks": []},
            "backtracking_context": None
        }

        # Call the tool's _process_response
        result = asyncio.get_event_loop().run_until_complete(
            tool._process_response("Regenerated reasoning", state)
        )

        # Verify it clears reasoning_approval for next critique
        assert "reasoning_approval" in result.output_workflow, \
            "Tool should output reasoning_approval"
        assert result.output_workflow["reasoning_approval"] is None, \
            "Tool should clear reasoning_approval to None for next critique"

        # Verify should_replace is True when reasoning_critique exists (critique rejected)
        assert result.output_ui["should_replace"] is True, \
            "should_replace should be True when regenerating after critique rejection"

    def test_reasoning_tool_should_replace_on_retry_objective(self):
        """Verify should_replace is True when retry_objective exists (incomplete addressed)."""
        import asyncio
        from kai.core.tools import ReasoningResponseWithGuidanceTool

        llm_mock = Mock()
        tool = ReasoningResponseWithGuidanceTool(llm_mock)

        state = {
            "user_query": "Continue",
            "positioning_info": {"target_cell": 5},
            "retry_objective": "Address the incomplete reasoning",  # Flagged as incomplete
            "task_list": {"tasks": []},
            "backtracking_context": None
        }

        result = asyncio.get_event_loop().run_until_complete(
            tool._process_response("Revised reasoning", state)
        )

        # Verify should_replace is True when retry_objective exists
        assert result.output_ui["should_replace"] is True, \
            "should_replace should be True when retry_objective indicates incomplete addressing"

    def test_reasoning_tool_should_not_replace_on_first_generation(self):
        """Verify should_replace is False on first generation (no critique or retry)."""
        import asyncio
        from kai.core.tools import ReasoningResponseWithGuidanceTool

        llm_mock = Mock()
        tool = ReasoningResponseWithGuidanceTool(llm_mock)

        state = {
            "user_query": "Continue",
            "positioning_info": {"target_cell": 5},
            # No reasoning_critique or retry_objective
            "task_list": {"tasks": []},
            "backtracking_context": None
        }

        result = asyncio.get_event_loop().run_until_complete(
            tool._process_response("Initial reasoning", state)
        )

        # Verify should_replace is False on first generation
        assert result.output_ui["should_replace"] is False, \
            "should_replace should be False on first generation"


class TestCompletedStatusPreservation:
    """Test that completed task status is preserved through autonomous updates."""

    def test_autonomous_update_preserves_completed_status(self):
        """Verify autonomous_update_tasks preserves completed status even if LLM returns different status."""
        from kai.core.tools import AutonomousUpdateTasksTool
        from kai.core.tools.common_schemas import TaskItem
        from kai.core.tools.autonomous_update_tasks import AutonomousTaskUpdate

        # Create mock LLM interface
        llm_mock = Mock()
        tool = AutonomousUpdateTasksTool(llm_mock)

        # Original task list with task 1 marked as COMPLETED
        original_task_list = {
            "tasks": [
                {"id": 1, "task": "First task", "status": "completed"},  # Already completed
                {"id": 2, "task": "Second task", "status": "active"},
                {"id": 3, "task": "Third task", "status": "pending"},
            ]
        }

        state = {
            "user_query": "Continue",
            "rag_enabled": False,
            "last_execution_failed": False,
            "task_list": original_task_list,
            "backtracking_context": None
        }

        # LLM returns task 1 with status="active" (trying to revert it)
        llm_result = AutonomousTaskUpdate(
            tasks=[
                TaskItem(id=1, task="First task (modified)", status="active"),  # LLM tries to set active
                TaskItem(id=2, task="Second task", status="active"),
            ],
            retrieval_queries=[],
            update_rationale="Test update",
            update_rule="UPDATE"
        )

        # Process the result
        result = tool._process_structured_result(llm_result, state)

        # Verify task 1 kept its "completed" status
        updated_tasks = result.output_workflow["task_list"]["tasks"]
        task_1 = next(t for t in updated_tasks if t["id"] == 1)
        assert task_1["status"] == "completed", \
            f"Task 1 should remain completed, but got status={task_1['status']}"

        # Verify task 2 can still be updated
        task_2 = next(t for t in updated_tasks if t["id"] == 2)
        assert task_2["status"] == "active"

        # Verify task 3 is preserved (not in LLM output)
        task_3 = next(t for t in updated_tasks if t["id"] == 3)
        assert task_3["status"] == "pending"


class TestCritiqueIterationCounterIsolation:
    """Test that reasoning and autonomous update critique counters are independent.

    This is a regression test for a bug where both critique loops shared a single
    'critique_iteration' counter, causing the reasoning critique loop to be
    prematurely terminated when autonomous_update_critique had already incremented
    the counter.
    """

    def test_autonomous_update_counter_does_not_affect_reasoning(self):
        """Autonomous update critique counter should not affect reasoning loop."""
        # State after autonomous_update_critique has run (counter = 1)
        # but reasoning task is just starting (reasoning counter should be 0)
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test", "status": "active"}]},
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Initial reasoning",
            "reasoning_approval": None,  # Not yet critiqued
            # Autonomous update has been critiqued (counter = 1)
            "autonomous_update_critique_iteration": 1,
            # Reasoning critique should start at 0
            "reasoning_critique_iteration": 0,
        }

        result = route_standard_continue_branch(state, mock_send_message)
        # Should still call critique even though autonomous_update_critique_iteration is 1
        assert result == "reasoning_critique", \
            "Reasoning critique should run regardless of autonomous_update_critique_iteration"

    def test_reasoning_counter_does_not_affect_autonomous_update(self):
        """Reasoning critique counter should not affect autonomous update loop."""
        # State where reasoning has had 2 iterations but autonomous update is just starting
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            # Reasoning has maxed out (counter = 2)
            "reasoning_critique_iteration": 2,
            # Autonomous update should start at 0
            "autonomous_update_critique_iteration": 0,
            "autonomous_update_approval": None,
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        # Should still call critique even though reasoning_critique_iteration is 2
        assert result == "autonomous_update_critique", \
            "Autonomous update critique should run regardless of reasoning_critique_iteration"

    def test_counters_increment_independently(self):
        """Each counter should only be incremented by its respective tool."""
        import asyncio
        from kai.core.tools import ReasoningCritiqueTool, AutonomousUpdateCritiqueTool
        from unittest.mock import Mock

        # Create mock LLMs that return approval
        reasoning_llm = Mock()
        reasoning_llm.get_llm_for_tool.return_value.provider_name = "test"
        reasoning_llm.get_llm_for_tool.return_value.model = "test"
        reasoning_llm.get_llm_for_tool.return_value.use_structured_output = True

        update_llm = Mock()
        update_llm.get_llm_for_tool.return_value.provider_name = "test"
        update_llm.get_llm_for_tool.return_value.model = "test"
        update_llm.get_llm_for_tool.return_value.use_structured_output = True

        reasoning_tool = ReasoningCritiqueTool(reasoning_llm)
        update_tool = AutonomousUpdateCritiqueTool(update_llm)

        # State with both counters at 0
        state = {
            "user_query": "test",
            "reasoning_critique_iteration": 0,
            "autonomous_update_critique_iteration": 0,
            "task_list": {"tasks": []},
        }

        # Simulate reasoning critique output
        from kai.core.tools.reasoning_critique import ReasoningCritique
        reasoning_result = Mock(spec=ReasoningCritique)
        reasoning_result.approval = "MODIFY"
        reasoning_result.critique = "Need improvement"
        result1 = reasoning_tool._process_structured_result(reasoning_result, state)

        # Verify only reasoning counter incremented
        assert result1.output_workflow["reasoning_critique_iteration"] == 1
        assert "autonomous_update_critique_iteration" not in result1.output_workflow, \
            "Reasoning critique should not touch autonomous_update counter"

        # Simulate autonomous update critique output
        from kai.core.tools.autonomous_update_critique import AutonomousUpdateCritique
        update_result = Mock(spec=AutonomousUpdateCritique)
        update_result.approval = "MODIFY"
        update_result.critique = "Need changes"
        result2 = update_tool._process_structured_result(update_result, state)

        # Verify only update counter incremented
        assert result2.output_workflow["autonomous_update_critique_iteration"] == 1
        assert "reasoning_critique_iteration" not in result2.output_workflow, \
            "Autonomous update critique should not touch reasoning counter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
