"""Tests for prompt assembly - verifying all context parts arrive in prompts.

Tests scenarios:
- Reference workflow content flows to prompts
- RAG retrieval flows to prompts
- Task list flows to prompts
- Execution context fields are accessible
"""

import pytest
from kai.core.prompt_manager import PromptManager, PromptScenario


def create_state(**kwargs):
    """Create state dict with specified fields."""
    default_state = {
        "user_query": kwargs.get("user_query", "Test query"),
        "current_cell": "",
        "error_message": "",
        "execution_history": [],
        "conversation_history": [],
        "notebook_structure": {
            "totalCells": 3,
            "allCells": ["# Cell 1", "# Cell 2", "# Cell 3"]
        },
        "last_execution_failed": False,
        "autonomous_mode": True,
        "notebook_cells": [],
        "task_list": kwargs.get("task_list", {"tasks": []}),
        "backtracking_context": None,
        "excluded_workflows": kwargs.get("excluded_workflows", []),
        "session_id": "test"
    }
    # Merge kwargs into default state (allows overriding defaults)
    default_state.update(kwargs)

    return default_state


class TestReferenceWorkflowInPrompts:
    """Test reference_workflow_content flows to prompts."""

    def test_reference_workflow_content_in_code_generation_prompt(self):
        """Verify reference_workflow_content appears in code generation prompts."""
        pm = PromptManager()

        # Create context with reference workflow content
        state = create_state(
            reference_workflow_content={
                "workflow_1": "> Notebook ID: example_org/test_workflow\nCell 0: import scanpy as sc\nCell 1: adata = sc.read_h5ad('data.h5ad')",
                "workflow_2": "> Notebook ID: another_org/tutorial\nCell 0: # Tutorial notebook"
            },
            active_task_objective="Load and preprocess single-cell data",
            task_list={"tasks": [{"id": 1, "task": "Load data", "status": "active"}]}
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.CODE_GENERATION_WITH_GUIDANCE
        )

        # Verify reference workflow content appears
        assert "example_org/test_workflow" in user_prompt, \
            "Reference workflow ID should appear in prompt"
        assert "import scanpy as sc" in user_prompt, \
            "Reference workflow cell content should appear in prompt"
        assert "another_org/tutorial" in user_prompt, \
            "Second workflow ID should appear in prompt"

    def test_empty_reference_workflow_handled_gracefully(self):
        """Verify empty reference_workflow_content doesn't break prompts."""
        pm = PromptManager()

        state = create_state(
            reference_workflow_content={},
            active_task_objective="Load data"
        )

        # Should not raise
        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.CODE_GENERATION_WITH_GUIDANCE
        )

        assert user_prompt is not None
        assert "Reference workflows" not in user_prompt or "Session context: Reference workflows" not in user_prompt

    def test_missing_reference_workflow_handled_gracefully(self):
        """Verify missing reference_workflow_content doesn't break prompts."""
        pm = PromptManager()

        # Don't include reference_workflow_content at all
        state = create_state(
            active_task_objective="Load data"
        )

        # Should not raise
        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.CODE_GENERATION_WITH_GUIDANCE
        )

        assert user_prompt is not None


class TestRagRetrievalInPrompts:
    """Test rag_retrieval flows to prompts."""

    def test_rag_retrieval_in_code_update_prompt(self):
        """Verify rag_retrieval appears in code update prompts."""
        pm = PromptManager()

        state = create_state(
            rag_retrieval="API Documentation: sc.pp.filter_cells(adata, min_genes=200)",
            retry_objective="Fix the cell filtering error",
            active_task_objective="Filter low quality cells",
            error_message="KeyError: 'n_genes'",
            last_execution_failed=True
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.CODE_UPDATE_WITH_GUIDANCE
        )

        # Verify RAG retrieval appears
        assert "sc.pp.filter_cells" in user_prompt, \
            "RAG retrieval content should appear in prompt"

    def test_empty_rag_retrieval_handled_gracefully(self):
        """Verify empty rag_retrieval doesn't break prompts."""
        pm = PromptManager()

        state = create_state(
            rag_retrieval="",
            retry_objective="Fix the error"
        )

        # Should not raise
        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.CODE_UPDATE_WITH_GUIDANCE
        )

        assert user_prompt is not None

    def test_none_rag_retrieval_handled_gracefully(self):
        """Verify None rag_retrieval doesn't break prompts."""
        pm = PromptManager()

        state = create_state(
            rag_retrieval=None,
            retry_objective="Fix the error"
        )

        # Should not raise
        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.CODE_UPDATE_WITH_GUIDANCE
        )

        assert user_prompt is not None


class TestTaskListInPrompts:
    """Test task_list flows to prompts."""

    def test_task_list_in_autonomous_update_prompt(self):
        """Verify task_list appears in autonomous update prompts."""
        pm = PromptManager()

        task_list = {
            "tasks": [
                {"id": 1, "task": "Load single-cell data", "status": "completed"},
                {"id": 2, "task": "Quality control", "status": "active"},
                {"id": 3, "task": "Normalize data", "status": "pending"}
            ]
        }

        state = create_state(
            task_list=task_list,
            active_task_objective="Quality control"
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.AUTONOMOUS_UPDATE_TASKS
        )

        # Verify task content appears
        assert "Load single-cell data" in user_prompt, \
            "Task descriptions should appear in prompt"
        assert "Quality control" in user_prompt
        assert "Normalize data" in user_prompt


class TestPutativeWorkflowsInPrompts:
    """Test putative reference workflow summaries flow to prompts."""

    def test_putative_workflows_in_selection_prompt(self):
        """Verify putative workflow summaries appear in reference workflow selection prompts."""
        pm = PromptManager()

        state = create_state(
            putative_reference_workflow_summaries="Notebook 1: example_org/pbmc_tutorial - PBMC analysis\nNotebook 2: another_org/metacells - Metacell analysis",
            user_query="Analyze PBMC single-cell data"
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.REFERENCE_WORKFLOW_SELECTION
        )

        # Verify putative workflows appear
        assert "pbmc_tutorial" in user_prompt or "PBMC analysis" in user_prompt, \
            "Putative workflow summaries should appear in selection prompt"


class TestExcludedWorkflowsInPrompts:
    """Test excluded workflows flow to prompts."""

    def test_excluded_workflows_in_selection_prompt(self):
        """Verify excluded workflows appear in reference workflow selection prompts."""
        pm = PromptManager()

        state = create_state(
            putative_reference_workflow_summaries="Some workflows",
            excluded_workflows=["example_org/old_workflow", "another_org/deprecated_tutorial"]
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.REFERENCE_WORKFLOW_SELECTION
        )

        # Verify excluded workflows appear
        assert "example_org/old_workflow" in user_prompt, \
            "Excluded workflows should appear in selection prompt"
        assert "another_org/deprecated_tutorial" in user_prompt


class TestContextFieldPropagation:
    """Test that state dict properly propagates fields to prompts."""

    def test_state_dict_contains_all_fields(self):
        """Verify state dict contains all provided fields."""
        state = create_state(
            reference_workflow_content={"wf1": "content1"},
            rag_retrieval="some rag content",
            active_task_objective="Do something",
            custom_field="custom_value"
        )

        # State dict should contain all provided fields
        assert state.get("reference_workflow_content") == {"wf1": "content1"}
        assert state.get("rag_retrieval") == "some rag content"
        assert state.get("active_task_objective") == "Do something"
        assert state.get("custom_field") == "custom_value"


class TestPromptScenarioCoverage:
    """Test that key scenarios have required sections."""

    @pytest.mark.parametrize("scenario,expected_section", [
        (PromptScenario.CODE_GENERATION_WITH_GUIDANCE, "reference_workflow_section"),
        (PromptScenario.TASK_LIST_GENERATION, "reference_workflow_section"),
        (PromptScenario.AUTONOMOUS_UPDATE_TASKS, "reference_workflow_section"),
        (PromptScenario.CODE_UPDATE_WITH_GUIDANCE, "rag_section"),
    ])
    def test_scenario_includes_expected_section(self, scenario, expected_section):
        """Verify scenarios include expected sections in template."""
        pm = PromptManager()
        template = pm.PROMPT_TEMPLATES.get(scenario)

        assert template is not None, f"Missing template for {scenario}"
        user_template = template.get("user_template", "")

        # The template should reference the section (either directly or via placeholder)
        # Section placeholders are like {reference_workflow_section}
        assert expected_section in user_template or f"{{{expected_section}}}" in user_template, \
            f"Scenario {scenario} should include {expected_section}"


class TestCheckpointRestoration:
    """Test that critical fields are restored from checkpoint."""

    def test_reference_workflow_content_restored_from_checkpoint(self):
        """Verify reference_workflow_content is restored from checkpoint on subsequent iterations."""
        # Simulate checkpoint with reference_workflow_content
        checkpoint_values = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "active"}]},
            "reference_workflow_content": {"wf1": "workflow content here"},
            "excluded_workflows": ["old_wf"],
            "auto_mode_first_execution_done": True,
            "planning_phase": "complete",
            "active_task_objective": "Test task",
        }

        # Initial state WITHOUT reference_workflow_content (simulating UI call)
        initial_state = {
            "user_query": "Continue analysis",
            "autonomous_mode": True,
            "task_list": {},  # Empty - should be restored from checkpoint
        }

        # Apply the checkpoint restoration logic (same as in langgraph_orchestrator.py)
        # For task_list
        checkpoint_task_list = checkpoint_values.get("task_list", {})
        initial_task_list = initial_state.get("task_list", {})
        if checkpoint_task_list.get("tasks") and not initial_task_list.get("tasks"):
            initial_state["task_list"] = checkpoint_task_list

        # For reference_workflow_content and excluded_workflows
        for field in ["reference_workflow_content", "excluded_workflows"]:
            checkpoint_val = checkpoint_values.get(field)
            initial_val = initial_state.get(field)
            if checkpoint_val and (field not in initial_state or not initial_val):
                initial_state[field] = checkpoint_val

        # Verify restoration
        assert initial_state.get("reference_workflow_content") == {"wf1": "workflow content here"}, \
            "reference_workflow_content should be restored from checkpoint"
        assert initial_state.get("excluded_workflows") == ["old_wf"], \
            "excluded_workflows should be restored from checkpoint"
        assert initial_state.get("task_list", {}).get("tasks"), \
            "task_list should be restored from checkpoint"

    def test_reference_workflow_content_not_overwritten_if_provided(self):
        """Verify reference_workflow_content is NOT overwritten if caller provides it."""
        checkpoint_values = {
            "reference_workflow_content": {"old_wf": "old content"},
            "excluded_workflows": ["old_excluded"],
        }

        # Initial state WITH reference_workflow_content (caller provided fresh data)
        initial_state = {
            "reference_workflow_content": {"new_wf": "new content"},
            "excluded_workflows": ["new_excluded"],
        }

        # Apply restoration logic
        for field in ["reference_workflow_content", "excluded_workflows"]:
            checkpoint_val = checkpoint_values.get(field)
            initial_val = initial_state.get(field)
            if checkpoint_val and (field not in initial_state or not initial_val):
                initial_state[field] = checkpoint_val

        # Verify NOT overwritten
        assert initial_state.get("reference_workflow_content") == {"new_wf": "new content"}, \
            "reference_workflow_content should NOT be overwritten when caller provides it"
        assert initial_state.get("excluded_workflows") == ["new_excluded"], \
            "excluded_workflows should NOT be overwritten when caller provides it"


class TestNoneValueHandling:
    """Test that None values in LangGraph state don't break prompt assembly.

    In LangGraph, state fields are defined in TypedDict and can have None values
    even when the key exists. The prompt manager must check for BOTH key existence
    AND non-None values to avoid joining None into strings.
    """

    def test_reasoning_instructions_with_none_values_doesnt_crash(self):
        """Verify _build_reasoning_instructions_section handles None values.

        In LangGraph state, reasoning_feedback and reasoning_response keys may exist
        but have None values. The function should return empty string, not crash.
        """
        pm = PromptManager()

        # Simulate LangGraph state where keys exist but values are None
        state = create_state(
            reasoning_feedback=None,  # Key exists but value is None
            reasoning_response=None,  # Key exists but value is None
        )

        # Should not crash - should return empty string
        result = pm._build_reasoning_instructions_section(state)
        assert result == "", "Should return empty string when values are None"

    def test_reasoning_evaluation_instructions_with_none_response_doesnt_crash(self):
        """Verify _build_reasoning_evaluation_instructions_section handles None reasoning_response."""
        pm = PromptManager()

        state = create_state(
            reasoning_response=None,  # Key exists but value is None
        )

        # Should not crash
        result = pm._build_reasoning_evaluation_instructions_section(state)
        assert result == "", "Should return empty string when reasoning_response is None"

    def test_task_list_update_evaluation_instructions_with_none_values(self):
        """Verify _build_task_list_update_evaluation_instructions_section handles None values."""
        pm = PromptManager()

        state = create_state(
            task_list_update_rationale=None,  # Key exists but value is None
            task_list_backup=None,
        )

        # Should not crash
        result = pm._build_task_list_update_evaluation_instructions_section(state)
        assert result == "", "Should return empty string when rationale is None"

    def test_task_list_update_instructions_with_none_feedback(self):
        """Verify _build_task_list_update_instructions_section handles None feedback.

        When feedback is None, this function falls through to the else branch
        which shows the current task list (standard first pass behavior).
        """
        pm = PromptManager()

        state = create_state(
            task_update_feedback=None,  # Key exists but value is None
            task_list_backup=None,
            task_list={"tasks": [{"id": 1, "task": "Test task", "status": "pending"}]},
        )

        # Should not crash - should return current task list section (first pass)
        result = pm._build_task_list_update_instructions_section(state)
        # With None feedback, it falls through to the else branch which shows current task list
        assert "current task list" in result.lower(), "Should show current task list when feedback is None"
        assert "Test task" in result, "Should include task content"

    def test_task_list_generation_evaluation_instructions_with_none_values(self):
        """Verify _build_task_list_generation_evaluation_instructions_section handles None task_text_old."""
        pm = PromptManager()

        state = create_state(
            task_text_old=None,  # Key exists but value is None
            task_list={"tasks": [{"id": 1, "task": "Test", "status": "pending"}]},
        )

        # Should not crash - task_text_old should be skipped when None
        result = pm._build_task_list_generation_evaluation_instructions_section(state)
        # Should still have current version, just not the old version
        assert "Test" in result or "current version" in result.lower()

    def test_task_list_generation_instructions_with_none_feedback(self):
        """Verify _build_task_list_generation_instructions_section handles None feedback."""
        pm = PromptManager()

        state = create_state(
            task_list_feedback=None,  # Key exists but value is None
            task_list={"tasks": [{"id": 1, "task": "Test", "status": "pending"}]},
        )

        # Should not crash - should return empty string when feedback is None
        result = pm._build_task_list_generation_instructions_section(state)
        assert result.strip() == "", "Should return empty when feedback is None"


class TestEvaluatorLoopPromptAssembly:
    """Test that evaluator loops properly include old/new versions in prompts.

    Evaluator loops require showing both the original and updated versions
    so the LLM can compare them. These tests verify that:
    1. task_update_evaluation receives both task lists
    2. autonomous_update_tasks (after evaluation) receives both task lists + feedback
    3. task_list_evaluation receives both task lists
    4. task_list_generation (after evaluation) receives both + feedback
    """

    def test_task_update_evaluation_includes_both_task_lists(self):
        """Verify task_update_evaluation prompt includes original and updated task lists.

        After autonomous_update_tasks runs, the evaluation prompt should show:
        - Original task list (from task_list_backup)
        - Updated task list (current task_list)
        - Update rationale
        """
        pm = PromptManager()

        # Original task list (backed up before update)
        original_tasks = {
            "tasks": [
                {"id": 1, "task": "Load data from file", "status": "completed"},
                {"id": 2, "task": "Filter cells", "status": "pending"},
            ]
        }

        # Updated task list (after autonomous_update_tasks)
        updated_tasks = {
            "tasks": [
                {"id": 1, "task": "Load data from file", "status": "completed"},
                {"id": 2, "task": "Filter cells by quality metrics", "status": "pending"},
                {"id": 3, "task": "Normalize expression", "status": "pending"},
            ]
        }

        state = create_state(
            task_list=updated_tasks,
            task_list_backup=original_tasks,  # Set by _backup_task_list_node
            task_list_update_rationale="Added normalization step and clarified filtering criteria",
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.TASK_UPDATE_EVALUATION
        )

        # Verify BOTH task lists appear
        assert "Load data from file" in user_prompt, \
            "Original task should appear in evaluation prompt"
        assert "Filter cells by quality metrics" in user_prompt, \
            "Updated task should appear in evaluation prompt"
        assert "Normalize expression" in user_prompt, \
            "New task from update should appear in evaluation prompt"

        # Verify rationale appears
        assert "Added normalization step" in user_prompt, \
            "Update rationale should appear in evaluation prompt"

        # Verify structure - should show "original" and "updated/draft"
        assert "original task list" in user_prompt.lower(), \
            "Prompt should label the original task list"
        assert "draft" in user_prompt.lower() or "updated" in user_prompt.lower(), \
            "Prompt should label the updated task list"

    def test_task_update_evaluation_without_rationale_returns_empty(self):
        """Verify evaluation prompt is empty when no update rationale is present.

        This happens on first execution before any updates have been made.
        """
        pm = PromptManager()

        state = create_state(
            task_list={"tasks": [{"id": 1, "task": "Test", "status": "pending"}]},
            # No task_list_update_rationale - means no update has been made yet
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.TASK_UPDATE_EVALUATION
        )

        # The evaluation instructions section should be empty/minimal when no update has occurred
        # Check that it doesn't contain task list comparison markers
        evaluation_section = pm._build_task_list_update_evaluation_instructions_section(state)
        assert evaluation_section == "", \
            "Evaluation section should be empty when no update rationale is present"

    def test_autonomous_update_tasks_after_evaluation_includes_all_context(self):
        """Verify autonomous_update_tasks prompt (after evaluation rejection) includes full context.

        After evaluation returns REJECTED, the regeneration prompt should show:
        - Original task list (from task_list_backup)
        - Current draft (task_list)
        - Evaluation feedback
        """
        pm = PromptManager()

        original_tasks = {
            "tasks": [
                {"id": 1, "task": "Load data", "status": "completed"},
                {"id": 2, "task": "Analyze", "status": "pending"},
            ]
        }

        current_draft = {
            "tasks": [
                {"id": 1, "task": "Load data", "status": "completed"},
                {"id": 2, "task": "Bad task description", "status": "pending"},
            ]
        }

        state = create_state(
            task_list=current_draft,
            task_list_backup=original_tasks,
            task_update_feedback="Task 2 description is too vague. Be more specific about the analysis type.",
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.AUTONOMOUS_UPDATE_TASKS
        )

        # Verify original task appears
        assert "Load data" in user_prompt

        # Verify evaluation feedback appears
        assert "too vague" in user_prompt, \
            "Evaluation feedback should appear in regeneration prompt"
        assert "Be more specific" in user_prompt

    def test_autonomous_update_tasks_first_pass_shows_current_only(self):
        """Verify autonomous_update_tasks first pass only shows current task list.

        On first pass (no evaluation yet), should only show current task list
        without comparison context.
        """
        pm = PromptManager()

        current_tasks = {
            "tasks": [
                {"id": 1, "task": "Load data", "status": "completed"},
                {"id": 2, "task": "Process data", "status": "active"},
            ]
        }

        state = create_state(
            task_list=current_tasks,
            # No task_update_feedback - first pass
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.AUTONOMOUS_UPDATE_TASKS
        )

        # Should have current task list
        assert "Load data" in user_prompt
        assert "Process data" in user_prompt

        # Should NOT have "original" comparison context on first pass
        update_section = pm._build_task_list_update_instructions_section(state)
        assert "original task list" not in update_section.lower(), \
            "First pass should not show 'original' comparison - only current"

    def test_task_list_evaluation_includes_both_versions(self):
        """Verify task_list_evaluation prompt includes previous and current versions.

        During initial planning, the evaluation should compare the previous draft
        with the current draft.
        """
        pm = PromptManager()

        current_tasks = {
            "tasks": [
                {"id": 1, "task": "Load PBMC data", "status": "pending"},
                {"id": 2, "task": "QC filtering", "status": "pending"},
            ]
        }

        state = create_state(
            task_list=current_tasks,
            task_text_old="Task ID=1: Load data\nTask ID=2: Process",  # Previous version
        )

        system_prompt, user_prompt = pm.generate_prompt(
            state,
            PromptScenario.TASK_LIST_EVALUATION
        )

        # Should show both versions
        assert "Load PBMC data" in user_prompt, \
            "Current task list should appear"
        # Previous version via task_text_old
        evaluation_section = pm._build_task_list_generation_evaluation_instructions_section(state)
        assert "Load data" in evaluation_section or "previous version" in evaluation_section.lower(), \
            "Previous version should appear in evaluation"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
