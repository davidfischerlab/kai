"""Tests for evaluation loops, guards, and iteration limits.

This module tests loop control mechanisms in the LangGraph orchestrator:
- Task update evaluation loop: UPDATE -> EVALUATE -> (if rejected) UPDATE -> EVALUATE
- Reasoning evaluation loop: similar pattern for reasoning tasks
- Max iteration limits (guards against infinite loops)
- Approval flows and regeneration after rejection
- Counter isolation (reasoning vs task update)
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


class TestTaskUpdateEvaluationLoop:
    """Test task update evaluation loop routing."""

    def test_first_evaluation_called_when_grade_is_none(self):
        """First evaluation should be called when task_update_grade is None."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "task_update_evaluation_iteration": 0,
            "task_update_grade": None,  # Not yet evaluated
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "task_update_evaluator", \
            "Should call evaluator when task_update_grade is None"

    def test_regenerate_called_when_grade_is_rejected(self):
        """Regeneration should be called when evaluation returns REJECTED."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "task_update_evaluation_iteration": 1,  # Already had one evaluation
            "task_update_grade": "REJECTED",  # Evaluation rejected
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "autonomous_update_tasks", \
            "Should call autonomous_update_tasks when grade is REJECTED"

    def test_evaluation_called_after_regeneration(self):
        """Evaluation should be called after regeneration clears grade to None."""
        # State after regeneration (autonomous_update_tasks cleared grade)
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "task_update_evaluation_iteration": 1,  # Still on iteration 1
            "task_update_grade": None,  # Cleared by regeneration
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "task_update_evaluator", \
            "Should call evaluator after regeneration clears grade to None"

    def test_approved_skips_further_evaluation(self):
        """When approved, should not call evaluator or regenerate."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "task_update_evaluation_iteration": 1,
            "task_update_grade": "APPROVED",
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
            "next_task_activated": False,
        }

        result = route_standard_execution(state, mock_send_message)
        # Should proceed past evaluation loop
        assert result not in ["task_update_evaluator", "autonomous_update_tasks", "revert_task_list"], \
            f"Should not call evaluator/regenerate when approved, got {result}"

    def test_max_iterations_triggers_revert(self):
        """Max iterations (3) should trigger revert to backup."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "task_update_evaluation_iteration": 3,  # Max reached
            "task_update_grade": "REJECTED",  # Still not approved
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        assert result == "revert_task_list", \
            "Should revert to backup when max iterations reached"


class TestEvaluationLoopProgression:
    """Test full evaluation loop progression without infinite loops."""

    def test_evaluation_loop_progression_simulation(self):
        """Simulate full loop: UPDATE → EVALUATE → UPDATE → EVALUATE → APPROVE."""
        # Simulate the progression of states through the evaluation loop
        progression = []

        # Step 1: Initial state after first task update
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            "task_update_evaluation_iteration": 0,
            "task_update_grade": None,
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=0, grade=None", result))
        assert result == "task_update_evaluator"

        # Step 2: After first evaluation (rejected)
        state["task_update_evaluation_iteration"] = 1
        state["task_update_grade"] = "REJECTED"

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=1, grade=REJECTED", result))
        assert result == "autonomous_update_tasks"

        # Step 3: After regeneration (tool clears grade)
        state["task_update_grade"] = None

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=1, grade=None", result))
        assert result == "task_update_evaluator"

        # Step 4: After second evaluation (approved)
        state["task_update_evaluation_iteration"] = 2
        state["task_update_grade"] = "APPROVED"
        state["next_task_activated"] = False

        result = route_standard_execution(state, mock_send_message)
        progression.append(("iteration=2, grade=APPROVED", result))
        # Should proceed past evaluation loop
        assert result not in [
            "task_update_evaluator",
            "autonomous_update_tasks",
            "revert_task_list"
        ], f"Should proceed past loop when approved, got {result}"

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
        evaluation_iteration = 0

        # Simulate up to 10 iterations (should hit max at 3)
        for i in range(10):
            state["task_update_evaluation_iteration"] = evaluation_iteration
            state["task_update_grade"] = None if i % 2 == 0 else "REJECTED"

            result = route_standard_execution(state, mock_send_message)
            actions.append(result)

            if result == "revert_task_list":
                break
            elif result == "task_update_evaluator":
                # Evaluation increments counter
                evaluation_iteration += 1

        # Should have hit revert before 10 iterations
        assert "revert_task_list" in actions, \
            "Should eventually revert when max iterations reached"
        assert len(actions) <= 7, \
            f"Should not take more than 7 actions to reach revert (got {len(actions)})"


class TestReasoningEvaluationLoop:
    """Test reasoning evaluation loop routing.

    NOTE: These tests use route_standard_continue_branch directly because
    that's where the reasoning evaluation loop logic lives.
    """

    def test_reasoning_first_evaluation_called_when_grade_is_none(self):
        """First evaluation should be called when reasoning_grade is None."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test", "status": "active"}]},
            "reasoning_evaluation_iteration": 0,
            "reasoning_response": "Some generated reasoning",
            "reasoning_grade": None,  # Not yet evaluated
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "reasoning_evaluator", \
            "Should call reasoning_evaluator when grade is None and response exists"

    def test_reasoning_regenerate_called_when_grade_is_rejected(self):
        """Regeneration should be called when reasoning evaluation returns REJECTED."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test", "status": "active"}]},
            "reasoning_evaluation_iteration": 1,
            "reasoning_response": "Some generated reasoning",
            "reasoning_grade": "REJECTED",  # Evaluation rejected
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "reasoning_response_with_guidance", \
            "Should call reasoning_response_with_guidance when grade is REJECTED"

    def test_reasoning_evaluation_called_after_regeneration(self):
        """Evaluation should be called after regeneration clears grade to None."""
        # State after regeneration
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test", "status": "active"}]},
            "reasoning_evaluation_iteration": 1,
            "reasoning_response": "Regenerated reasoning",
            "reasoning_grade": None,  # Cleared by regeneration
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "reasoning_evaluator", \
            "Should call reasoning_evaluator after regeneration clears grade to None"

    def test_reasoning_approved_marks_complete(self):
        """When approved, should proceed to mark_reasoning_completed."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test", "status": "active"}]},
            "reasoning_evaluation_iteration": 1,
            "reasoning_response": "Approved reasoning",
            "reasoning_grade": "APPROVED",
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "mark_reasoning_completed", \
            "Should proceed to mark_reasoning_completed when approved"

    def test_reasoning_max_iterations_proceeds_anyway(self):
        """Max iterations (2) should proceed to mark_reasoning_completed anyway."""
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test", "status": "active"}]},
            "reasoning_evaluation_iteration": 2,  # Max reached
            "reasoning_response": "Still not approved reasoning",
            "reasoning_grade": "REJECTED",  # Still not approved
            "is_reasoning_task": True,
            "positioning_info": {"target_cell": 5},
        }

        result = route_standard_continue_branch(state, mock_send_message)
        assert result == "mark_reasoning_completed", \
            "Should proceed to mark_reasoning_completed when max iterations reached"

    def test_reasoning_tool_clears_grade_for_next_evaluation(self):
        """Verify reasoning_response_with_guidance tool sets reasoning_grade to None."""
        import asyncio
        from kai.core.tools import ReasoningResponseWithGuidanceTool

        llm_mock = Mock()
        tool = ReasoningResponseWithGuidanceTool(llm_mock)

        state = {
            "user_query": "Continue",
            "positioning_info": {"target_cell": 5},
            "reasoning_feedback": "Some feedback",  # Indicates regeneration
            "task_list": {"tasks": []},
            "backtracking_context": None
        }

        # Call the tool's _process_response
        result = asyncio.get_event_loop().run_until_complete(
            tool._process_response("Regenerated reasoning", state)
        )

        # Verify it clears reasoning_grade for next evaluation
        assert "reasoning_grade" in result.output_workflow, \
            "Tool should output reasoning_grade"
        assert result.output_workflow["reasoning_grade"] is None, \
            "Tool should clear reasoning_grade to None for next evaluation"

        # Verify should_replace is True when reasoning_feedback exists
        assert result.output_ui["should_replace"] is True, \
            "should_replace should be True when regenerating after feedback"

    def test_reasoning_tool_should_replace_on_retry_objective(self):
        """Verify should_replace is True when retry_objective exists."""
        import asyncio
        from kai.core.tools import ReasoningResponseWithGuidanceTool

        llm_mock = Mock()
        tool = ReasoningResponseWithGuidanceTool(llm_mock)

        state = {
            "user_query": "Continue",
            "positioning_info": {"target_cell": 5},
            "retry_objective": "Address the incomplete reasoning",
            "task_list": {"tasks": []},
            "backtracking_context": None
        }

        result = asyncio.get_event_loop().run_until_complete(
            tool._process_response("Revised reasoning", state)
        )

        # Verify should_replace is True when retry_objective exists
        assert result.output_ui["should_replace"] is True, \
            "should_replace should be True when retry_objective exists"

    def test_reasoning_tool_should_not_replace_on_first_generation(self):
        """Verify should_replace is False on first generation (no feedback or retry)."""
        import asyncio
        from kai.core.tools import ReasoningResponseWithGuidanceTool

        llm_mock = Mock()
        tool = ReasoningResponseWithGuidanceTool(llm_mock)

        state = {
            "user_query": "Continue",
            "positioning_info": {"target_cell": 5},
            # No reasoning_feedback or retry_objective
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
        """Verify autonomous_update_tasks preserves completed status."""
        from kai.core.tools import AutonomousUpdateTasksTool
        from kai.core.tools.common_schemas import TaskItem
        from kai.core.tools.autonomous_update_tasks import AutonomousTaskUpdate

        # Create mock LLM interface
        llm_mock = Mock()
        tool = AutonomousUpdateTasksTool(llm_mock)

        # Original task list with task 1 marked as COMPLETED
        original_task_list = {
            "tasks": [
                {"id": 1, "task": "First task", "status": "completed"},
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
                TaskItem(id=1, task="First task (modified)", status="active"),
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


class TestEvaluationIterationCounterIsolation:
    """Test that reasoning and task update evaluation counters are independent.

    This is a regression test for a bug where both evaluation loops shared a single
    counter, causing the reasoning evaluation loop to be prematurely terminated.
    """

    def test_task_update_counter_does_not_affect_reasoning(self):
        """Task update evaluation counter should not affect reasoning loop."""
        # State after task_update_evaluator has run (counter = 1)
        # but reasoning task is just starting (reasoning counter should be 0)
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] Test", "status": "active"}]},
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_response": "Initial reasoning",
            "reasoning_grade": None,  # Not yet evaluated
            # Task update has been evaluated (counter = 1)
            "task_update_evaluation_iteration": 1,
            # Reasoning evaluation should start at 0
            "reasoning_evaluation_iteration": 0,
        }

        result = route_standard_continue_branch(state, mock_send_message)
        # Should still call evaluator even though task_update counter is 1
        assert result == "reasoning_evaluator", \
            "Reasoning evaluator should run regardless of task_update counter"

    def test_reasoning_counter_does_not_affect_task_update(self):
        """Reasoning evaluation counter should not affect task update loop."""
        # State where reasoning has had 2 iterations but task update is just starting
        state = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "task_completion_analyzed": True,
            "tasks_updated": True,
            "update_approved": False,
            "task_list_update_rule": "UPDATE",
            # Reasoning has maxed out (counter = 2)
            "reasoning_evaluation_iteration": 2,
            # Task update should start at 0
            "task_update_evaluation_iteration": 0,
            "task_update_grade": None,
            "retry_objective": None,
            "recovery_objective": None,
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "task_list_backup": {"tasks": []},
        }

        result = route_standard_execution(state, mock_send_message)
        # Should still call evaluator even though reasoning counter is 2
        assert result == "task_update_evaluator", \
            "Task update evaluator should run regardless of reasoning counter"

    def test_counters_increment_independently(self):
        """Each counter should only be incremented by its respective tool."""
        from kai.core.tools import ReasoningEvaluatorTool, TaskUpdateEvaluatorTool
        from unittest.mock import Mock

        # Create mock LLMs
        reasoning_llm = Mock()
        reasoning_llm.get_llm_for_tool.return_value.provider_name = "test"
        reasoning_llm.get_llm_for_tool.return_value.model = "test"
        reasoning_llm.get_llm_for_tool.return_value.use_structured_output = True

        update_llm = Mock()
        update_llm.get_llm_for_tool.return_value.provider_name = "test"
        update_llm.get_llm_for_tool.return_value.model = "test"
        update_llm.get_llm_for_tool.return_value.use_structured_output = True

        reasoning_tool = ReasoningEvaluatorTool(reasoning_llm)
        update_tool = TaskUpdateEvaluatorTool(update_llm)

        # State with both counters at 0
        state = {
            "user_query": "test",
            "reasoning_evaluation_iteration": 0,
            "task_update_evaluation_iteration": 0,
            "task_list": {"tasks": []},
        }

        # Simulate reasoning evaluation output
        from kai.core.tools.reasoning_evaluator import ReasoningEvaluation
        reasoning_result = Mock(spec=ReasoningEvaluation)
        reasoning_result.grade = "REJECTED"
        reasoning_result.feedback = "Need improvement"
        result1 = reasoning_tool._process_structured_result(reasoning_result, state)

        # Verify only reasoning counter incremented
        assert result1.output_workflow["reasoning_evaluation_iteration"] == 1
        assert "task_update_evaluation_iteration" not in result1.output_workflow, \
            "Reasoning evaluation should not touch task_update counter"

        # Simulate task update evaluation output
        from kai.core.tools.task_update_evaluator import TaskUpdateEvaluation
        update_result = Mock(spec=TaskUpdateEvaluation)
        update_result.grade = "REJECTED"
        update_result.feedback = "Need changes"
        result2 = update_tool._process_structured_result(update_result, state)

        # Verify only update counter incremented
        assert result2.output_workflow["task_update_evaluation_iteration"] == 1
        assert "reasoning_evaluation_iteration" not in result2.output_workflow, \
            "Task update evaluation should not touch reasoning counter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
