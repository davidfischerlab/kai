"""Regression tests for ExecutionContext.from_dict()

These tests verify that session_metadata fields are correctly preserved
when converting from LangGraph state dict to ExecutionContext.

This was a critical bug - missing session_timestamp caused KeyError in production.
"""

import pytest
from kai.core.orchestration.execution_context import ExecutionContext


class TestExecutionContextRegression:
    """Test ExecutionContext.from_dict() session_metadata handling."""

    def test_session_metadata_from_nested_dict(self):
        """Test that nested session_metadata dict is preserved (VSCode/Jupyter format)."""

        # This is how VSCode/Jupyter passes session_metadata
        state = {
            "user_query": "test query",
            "task_list": {},
            "autonomous_mode": True,
            "session_metadata": {
                "session_id": "test_session",
                "session_timestamp": "2025-01-01_12-00-00",
                "iteration_timestamp": "12-01-00",
                "iteration_counter": 1,
                "notebook_uri": "file:///test.ipynb",
                "active": True,
                "request_id": "req123"
            }
        }

        exec_context = ExecutionContext.from_dict(state)

        # Verify all session_metadata fields are preserved
        assert exec_context.session_metadata["session_id"] == "test_session"
        assert exec_context.session_metadata["session_timestamp"] == "2025-01-01_12-00-00"
        assert exec_context.session_metadata["iteration_timestamp"] == "12-01-00"
        assert exec_context.session_metadata["iteration_counter"] == 1
        assert exec_context.session_metadata["notebook_uri"] == "file:///test.ipynb"
        assert exec_context.session_metadata["active"] is True
        assert exec_context.session_metadata["request_id"] == "req123"
        assert exec_context.session_metadata["autonomous_mode"] is True

        print("✅ Nested session_metadata preserved correctly")

    def test_session_metadata_from_flat_dict(self):
        """Test that flat state dict builds session_metadata (test format)."""

        # This is how some tests pass state (flat structure)
        state = {
            "user_query": "test query",
            "task_list": {},
            "autonomous_mode": True,
            "session_id": "test_session",
            "session_timestamp": "2025-01-01_12-00-00",
            "iteration_timestamp": "12-01-00",
            "iteration_counter": 1,
            "notebook_uri": "file:///test.ipynb",
            "request_id": "req123"
        }

        exec_context = ExecutionContext.from_dict(state)

        # Verify session_metadata is built from flat fields
        assert exec_context.session_metadata["session_id"] == "test_session"
        assert exec_context.session_metadata["session_timestamp"] == "2025-01-01_12-00-00"
        assert exec_context.session_metadata["iteration_timestamp"] == "12-01-00"
        assert exec_context.session_metadata["iteration_counter"] == 1
        assert exec_context.session_metadata["notebook_uri"] == "file:///test.ipynb"
        assert exec_context.session_metadata["request_id"] == "req123"
        assert exec_context.session_metadata["autonomous_mode"] is True

        print("✅ Flat state converted to session_metadata correctly")

    def test_session_timestamp_regression(self):
        """Regression test: session_timestamp must be available (production bug).

        Production error:
        KeyError: 'session_timestamp' in prompt_tools.py when logging prompts
        """

        # Minimal state that caused the production error
        state = {
            "user_query": "Analyze blood dataset",
            "task_list": {},
            "autonomous_mode": True,
            "session_metadata": {
                "session_id": "prod_session",
                "session_timestamp": "2025-12-06_21-17-30",
                "iteration_timestamp": "21-17-56",
                "iteration_counter": 1,
                "notebook_uri": "file:///blood.ipynb",
                "active": True
            }
        }

        exec_context = ExecutionContext.from_dict(state)

        # This should NOT raise KeyError
        session_timestamp = exec_context.session_metadata["session_timestamp"]
        assert session_timestamp == "2025-12-06_21-17-30"

        # Verify other required fields are also present
        assert "iteration_timestamp" in exec_context.session_metadata
        assert "notebook_uri" in exec_context.session_metadata

        print(f"✅ session_timestamp accessible: {session_timestamp}")

    def test_minimal_state_has_defaults(self):
        """Test that minimal state dict doesn't crash (graceful defaults)."""

        state = {
            "user_query": "test",
            "task_list": {}
        }

        exec_context = ExecutionContext.from_dict(state)

        # Should have defaults, not crash
        assert "session_id" in exec_context.session_metadata
        assert "session_timestamp" in exec_context.session_metadata
        assert exec_context.session_metadata["session_timestamp"] == ""  # Empty default

        print("✅ Minimal state handled with defaults")
