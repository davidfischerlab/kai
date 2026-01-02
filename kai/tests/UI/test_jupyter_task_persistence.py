"""Test that JupyterInterface correctly persists task_list between iterations.

This was a critical production bug: task_list created during planning was lost
on subsequent iterations, causing "No actions taken" loops.
"""

import pytest
import tempfile
import json
import nbformat
from UI.jupyter.context_builder import ContextBuilder


class TestJupyterTaskPersistence:
    """Test task_list persistence in JupyterInterface."""

    @pytest.fixture
    def context_builder(self):
        """Create a ContextBuilder with minimal notebook."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            f.write('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
            temp_nb = f.name

        notebook = nbformat.read(temp_nb, as_version=4)
        return ContextBuilder(notebook, temp_nb)

    def test_task_list_persists_across_iterations(self, context_builder):
        """Test that task_list set on context_builder is included in subsequent contexts.

        Production bug: After planning created task_list, it was lost on next iteration.
        """
        # First iteration - no task_list yet
        context1 = context_builder.get_context(
            autonomous_mode=True,
            autonomous_mode_continue=False,
            rag_enabled=False,
            turbo_enabled=False
        )

        # Should NOT have taskList initially
        assert 'taskList' not in context1, "Initial context should not have taskList"

        # Simulate planning completing and returning task_list
        # (This is what JupyterInterface does when it receives task_list_display message)
        task_list = {
            "tasks": [
                {"id": 1, "task": "Load CSV", "status": "pending"},
                {"id": 2, "task": "Analyze data", "status": "pending"}
            ]
        }
        context_builder.task_list = task_list

        # Second iteration - should include task_list
        context2 = context_builder.get_context(
            autonomous_mode=True,
            autonomous_mode_continue=True,  # Continue mode
            rag_enabled=False,
            turbo_enabled=False
        )

        # MUST have taskList now
        assert 'taskList' in context2, "Context must include taskList after it's been set"
        assert context2['taskList'] == task_list
        assert len(context2['taskList']['tasks']) == 2

        print("✅ task_list correctly persists across iterations")

    def test_task_list_none_does_not_include_field(self, context_builder):
        """Test that taskList field is omitted when task_list is None."""
        # Don't set task_list
        context = context_builder.get_context(
            autonomous_mode=True,
            autonomous_mode_continue=False,
            rag_enabled=False,
            turbo_enabled=False
        )

        # Should NOT have taskList key
        assert 'taskList' not in context, "taskList should not be in context when None"

        print("✅ taskList omitted when None")

    def test_task_list_empty_dict_is_included(self, context_builder):
        """Test that empty task_list dict is still included in context."""
        context_builder.task_list = {}

        context = context_builder.get_context(
            autonomous_mode=True,
            autonomous_mode_continue=False,
            rag_enabled=False,
            turbo_enabled=False
        )

        # Should have taskList even if empty
        assert 'taskList' in context
        assert context['taskList'] == {}

        print("✅ Empty taskList dict is included")

    def test_production_scenario_task_list_flow(self, context_builder):
        """Simulate exact production flow: plan → save task_list → next iteration uses it.

        This is what happens in full_agent_test/run_single_test.py.
        """
        # Iteration 1: Planning (autonomous_mode_continue=False)
        context_iter1 = context_builder.get_context(
            autonomous_mode=True,
            autonomous_mode_continue=False,
            rag_enabled=False,
            turbo_enabled=False
        )

        assert 'taskList' not in context_iter1, "Iteration 1: No taskList yet"

        # Simulate orchestrator returning task_list via task_list_display message
        # This mimics what happens in JupyterInterface._process_pending_messages()
        returned_task_list = {
            "tasks": [
                {"id": 1, "task": "Quality control", "status": "pending"},
                {"id": 2, "task": "Normalization", "status": "pending"},
                {"id": 3, "task": "Clustering", "status": "pending"}
            ]
        }

        # JupyterInterface saves it
        context_builder.task_list = returned_task_list

        # Iteration 2: Execution (autonomous_mode_continue=True)
        context_iter2 = context_builder.get_context(
            autonomous_mode=True,
            autonomous_mode_continue=True,
            rag_enabled=False,
            turbo_enabled=False
        )

        # MUST include task_list from iteration 1
        assert 'taskList' in context_iter2, "Iteration 2: Must have taskList from planning"
        assert len(context_iter2['taskList']['tasks']) == 3
        assert context_iter2['taskList']['tasks'][0]['task'] == "Quality control"

        # Iteration 3: Continue execution
        context_iter3 = context_builder.get_context(
            autonomous_mode=True,
            autonomous_mode_continue=True,
            rag_enabled=False,
            turbo_enabled=False
        )

        # MUST STILL include task_list
        assert 'taskList' in context_iter3, "Iteration 3: Must retain taskList"
        assert len(context_iter3['taskList']['tasks']) == 3

        print("✅ Production scenario: task_list flows correctly across all iterations")
