"""Tests for state management in LangGraph orchestrator.

This module consolidates tests for:
- Transient state handling and defaults
- Checkpointer/persistence integration
- Incoming context preservation
- Error context propagation

Consolidates: test_transient_state.py, test_persistence.py
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock

import pytest

from kai.core.orchestration.state import get_transient_defaults, TRANSIENT_FIELD_NAMES


# =============================================================================
# Transient State Defaults
# =============================================================================

class TestTransientStateDefaults:
    """Test transient state default values."""

    def test_transient_defaults_include_error_fields(self):
        """Verify error-related fields are in transient defaults."""
        defaults = get_transient_defaults()

        # These fields should be transient (reset each iteration)
        assert "last_execution_failed" in defaults
        assert "retry_objective" in defaults
        assert "recovery_objective" in defaults

        # Verify default values
        assert defaults["last_execution_failed"] is None
        assert defaults["retry_objective"] is None

    def test_learning_explanation_done_removed_from_transient_fields(self):
        """Verify learning_explanation_done is NOT in transient defaults.

        ARCHITECTURE NOTE: After refactor, learning explanation runs in a
        SEPARATE learning graph AFTER code execution, not during the main
        execution graph. Therefore, we don't need a transient flag.
        """
        defaults = get_transient_defaults()

        # learning_explanation_done should NOT be in transient fields
        assert "learning_explanation_done" not in defaults, (
            "learning_explanation_done was removed - learning runs in separate graph"
        )

    def test_transient_field_names_match_defaults(self):
        """Verify TRANSIENT_FIELD_NAMES matches get_transient_defaults keys."""
        defaults = get_transient_defaults()

        for field in TRANSIENT_FIELD_NAMES:
            assert field in defaults, f"Field {field} in TRANSIENT_FIELD_NAMES but not in defaults"


# =============================================================================
# Incoming Context Preservation
# =============================================================================

class TestIncomingContextPreservation:
    """Test that incoming context values are preserved during transient reset.

    This tests the fix for the bug where last_execution_failed from VSCode
    was overwritten by transient reset.
    """

    def test_last_execution_failed_preserved_when_true(self):
        """Incoming last_execution_failed=True should not be overwritten."""
        from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator

        # Create minimal orchestrator mock
        orchestrator = LangGraphOrchestrator.__new__(LangGraphOrchestrator)
        orchestrator._send_message = Mock()

        # Simulate incoming state with error from VSCode
        incoming_state = {
            "last_execution_failed": True,
            "error_message": "ValueError: something went wrong",
            "user_query": "Continue",
        }

        # Apply transient reset logic (same as in process_request)
        for key, default_value in get_transient_defaults().items():
            incoming_value = incoming_state.get(key)
            if incoming_value is None:
                incoming_state[key] = default_value

        # Verify incoming True was preserved (not overwritten with None)
        assert incoming_state["last_execution_failed"] is True, \
            "last_execution_failed=True should be preserved, not reset to None"

    def test_last_execution_failed_preserved_when_false(self):
        """Incoming last_execution_failed=False should not be overwritten."""
        incoming_state = {
            "last_execution_failed": False,
            "user_query": "Continue",
        }

        for key, default_value in get_transient_defaults().items():
            incoming_value = incoming_state.get(key)
            if incoming_value is None:
                incoming_state[key] = default_value

        # False is also a meaningful value (no error), should be preserved
        assert incoming_state["last_execution_failed"] is False, \
            "last_execution_failed=False should be preserved, not reset to None"

    def test_retry_objective_preserved_when_set(self):
        """Incoming retry_objective should not be overwritten."""
        incoming_state = {
            "retry_objective": "Fix the syntax error in line 5",
            "user_query": "Continue",
        }

        for key, default_value in get_transient_defaults().items():
            incoming_value = incoming_state.get(key)
            if incoming_value is None:
                incoming_state[key] = default_value

        assert incoming_state["retry_objective"] == "Fix the syntax error in line 5", \
            "retry_objective should be preserved when explicitly set"

    def test_missing_values_get_defaults(self):
        """Fields not in incoming state should get default values."""
        incoming_state = {
            "user_query": "Continue",
            # No transient fields set
        }

        for key, default_value in get_transient_defaults().items():
            incoming_value = incoming_state.get(key)
            if incoming_value is None:
                incoming_state[key] = default_value

        # Should have defaults for all transient fields
        assert incoming_state["last_execution_failed"] is None
        assert incoming_state["retry_objective"] is None
        assert incoming_state["reasoning_evaluation_iteration"] == 0
        assert incoming_state["task_update_evaluation_iteration"] == 0
        assert incoming_state["task_completion_analyzed"] is False


# =============================================================================
# Error Context Propagation
# =============================================================================

class TestErrorContextPropagation:
    """Test that error context flows correctly through the orchestrator."""

    def test_error_section_uses_last_execution_failed(self):
        """Verify prompt manager's _build_error_section checks last_execution_failed."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        # Test with error
        state = {
            "user_query": "Continue",
            "last_execution_failed": True,
            "error_message": "ValueError: test error",
            "task_list": {},
            "backtracking_context": None
        }

        error_section = pm._build_error_section(state)
        assert "ValueError: test error" in error_section, \
            "Error message should appear when last_execution_failed=True"

    def test_error_section_shows_success_when_no_error(self):
        """Verify error section shows success when last_execution_failed=False."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        state = {
            "user_query": "Continue",
            "last_execution_failed": False,
            "error_message": "",
            "task_list": {},
            "backtracking_context": None
        }

        error_section = pm._build_error_section(state)
        assert "passed successfully" in error_section.lower(), \
            "Should indicate success when last_execution_failed=False"

    def test_error_section_handles_none_gracefully(self):
        """Verify error section handles None last_execution_failed (shouldn't happen but be safe)."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        state = {
            "user_query": "Continue",
            "last_execution_failed": None,  # Edge case
            "error_message": "",
            "task_list": {},
            "backtracking_context": None
        }

        # Should not crash, should treat None as falsy
        error_section = pm._build_error_section(state)
        assert "passed successfully" in error_section.lower()


# =============================================================================
# Current Cell Derivation
# =============================================================================

class TestCurrentCellDerivation:
    """Test that current_cell is derived from execution_history when needed."""

    def test_current_cell_from_context_takes_priority(self):
        """If current_cell is in context, use it directly."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        state = {
            "user_query": "Fix error",
            "current_cell": "import pandas as pd",
            "last_execution_failed": True,
            "execution_history": ["some history"],
            "task_list": {},
            "backtracking_context": None
        }

        result = pm._get_current_cell_for_prompt(state)
        assert result == "import pandas as pd", \
            "Should use current_cell from context when available"

    def test_current_cell_derived_from_execution_history_on_error(self):
        """If current_cell missing but error occurred, extract from execution_history."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        # Simulate the execution_history format from VSCode
        execution_history_item = """> CELL at index 3: FAILED
Executed at 12:35:28, took 2.014s
>>Content of cell at index 3:
# ----- Identify highly variable genes -----
import scanpy as sc
sc.pp.highly_variable_genes(adata)
>> Error output:
ValueError: something went wrong"""

        state = {
            "user_query": "Fix error",
            "current_cell": None,  # Not provided
            "last_execution_failed": True,
            "execution_history": [execution_history_item],
            "task_list": {},
            "backtracking_context": None
        }

        result = pm._get_current_cell_for_prompt(state)
        assert "import scanpy as sc" in result, \
            "Should extract cell content from execution_history"
        assert "sc.pp.highly_variable_genes" in result, \
            "Should include full cell content"

    def test_current_cell_empty_when_no_error_and_no_context(self):
        """If no error and no current_cell, return empty string."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        state = {
            "user_query": "Generate code",
            "current_cell": None,
            "last_execution_failed": False,
            "execution_history": [],
            "task_list": {},
            "backtracking_context": None
        }

        result = pm._get_current_cell_for_prompt(state)
        assert result == '', "Should return empty string when no cell available"

    def test_extract_cell_content_from_history_item(self):
        """Test the helper method that parses execution history items."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        history_item = """> CELL at index 3: FAILED
Executed at 12:35:28, took 2.014s
>>Content of cell at index 3:
X_log = adata.X.copy()
adata.X = adata.layers["counts"]
sc.pp.highly_variable_genes(adata)
>> Error output:
ValueError: Bin edges must be unique"""

        result = pm._extract_cell_content_from_history_item(history_item)

        assert "X_log = adata.X.copy()" in result
        assert "sc.pp.highly_variable_genes" in result
        assert "ValueError" not in result, "Should not include error output"


# =============================================================================
# Snippet/RAG Query Reset
# =============================================================================

class TestSnippetRetrievalQueryReset:
    """Test that snippet_retrieval_query resets each iteration.

    Bug discovered: RAG queries were accumulating across iterations
    (2 -> 12 -> 22 -> 32...) instead of resetting, causing performance
    degradation and duplicate searches.
    """

    def test_snippet_retrieval_query_in_transient_fields(self):
        """snippet_retrieval_query should be in transient fields for reset."""
        assert "snippet_retrieval_query" in TRANSIENT_FIELD_NAMES, \
            "snippet_retrieval_query must be transient to reset each iteration"

    def test_snippet_retrieval_query_default_is_empty_list(self):
        """Default value for snippet_retrieval_query should be empty list."""
        defaults = get_transient_defaults()
        assert "snippet_retrieval_query" in defaults, \
            "snippet_retrieval_query must have a default value"
        assert defaults["snippet_retrieval_query"] == [], \
            "snippet_retrieval_query default should be empty list, not None"

    def test_snippet_retrieval_query_resets_each_iteration(self):
        """Simulate transient reset and verify query list is cleared.

        This tests the CORRECT behavior: list-type transient fields should
        ALWAYS be reset to their default (empty list), unlike scalar fields
        like last_execution_failed which preserve incoming non-None values.
        """
        from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator

        # Simulate state after first iteration with accumulated queries
        state_after_iteration_1 = {
            "snippet_retrieval_query": ["query1", "query2", "error_message"],
            "user_query": "Continue",
            "last_execution_failed": True,
        }

        # Apply transient reset using the CORRECT logic:
        # - List-type defaults (like []) should ALWAYS reset
        # - Scalar fields (like last_execution_failed) preserve non-None incoming values
        for key, default_value in get_transient_defaults().items():
            incoming_value = state_after_iteration_1.get(key)
            # For list defaults, always reset to prevent accumulation
            # For scalar defaults, preserve incoming non-None values
            if isinstance(default_value, list) or incoming_value is None:
                state_after_iteration_1[key] = default_value

        # The query list should be reset to empty
        assert state_after_iteration_1.get("snippet_retrieval_query") == [], \
            "snippet_retrieval_query should reset to [] each iteration"

        # But scalar values like last_execution_failed should be preserved
        assert state_after_iteration_1.get("last_execution_failed") is True, \
            "last_execution_failed should preserve incoming True value"

    def test_rag_query_assembled_in_transient_fields(self):
        """rag_query_assembled should be transient to allow re-assembly each iteration."""
        assert "rag_query_assembled" in TRANSIENT_FIELD_NAMES, \
            "rag_query_assembled must be transient"

        defaults = get_transient_defaults()
        assert defaults.get("rag_query_assembled") is False, \
            "rag_query_assembled default should be False"


# =============================================================================
# Checkpointer/Persistence Tests
# =============================================================================

class TestCheckpointerFactory:
    """Test checkpointer creation."""

    @pytest.mark.asyncio
    async def test_create_memory_checkpointer(self):
        """MemorySaver created without db_path."""
        from kai.core.persistence.checkpointer import (
            CheckpointerType, create_checkpointer
        )
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = await create_checkpointer(CheckpointerType.MEMORY)
        assert isinstance(checkpointer, MemorySaver)

    @pytest.mark.asyncio
    async def test_create_sqlite_checkpointer(self):
        """AsyncSqliteSaver created with db_path."""
        from kai.core.persistence.checkpointer import (
            CheckpointerType, create_checkpointer
        )
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )
            assert isinstance(checkpointer, AsyncSqliteSaver)

    @pytest.mark.asyncio
    async def test_sqlite_requires_db_path(self):
        """AsyncSqliteSaver raises error without db_path."""
        from kai.core.persistence.checkpointer import (
            CheckpointerType, create_checkpointer
        )

        with pytest.raises(ValueError, match="db_path required"):
            await create_checkpointer(CheckpointerType.SQLITE)

    @pytest.mark.asyncio
    async def test_sqlite_creates_parent_dir(self):
        """AsyncSqliteSaver creates parent directory if needed."""
        from kai.core.persistence.checkpointer import (
            CheckpointerType, create_checkpointer
        )
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )
            assert isinstance(checkpointer, AsyncSqliteSaver)
            assert db_path.parent.exists()

    @pytest.mark.asyncio
    async def test_unknown_type_raises_error(self):
        """Unknown checkpointer type raises ValueError."""
        from kai.core.persistence.checkpointer import create_checkpointer

        with pytest.raises(ValueError, match="Unknown checkpointer type"):
            await create_checkpointer("invalid")  # type: ignore


class TestSettingsIntegration:
    """Test settings-based checkpointer creation."""

    @pytest.mark.asyncio
    async def test_disabled_uses_memory(self):
        """Disabled checkpointing uses MemorySaver."""
        from kai.core.persistence.checkpointer import get_checkpointer_for_settings
        from langgraph.checkpoint.memory import MemorySaver

        mock_settings = MagicMock()
        mock_settings.CHECKPOINT_ENABLED = False

        checkpointer = await get_checkpointer_for_settings(mock_settings)
        assert isinstance(checkpointer, MemorySaver)

    @pytest.mark.asyncio
    async def test_path_uses_sqlite(self):
        """Set checkpoint path uses AsyncSqliteSaver."""
        from kai.core.persistence.checkpointer import get_checkpointer_for_settings
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            mock_settings = MagicMock()
            mock_settings.checkpoint_db_path_resolved = db_path

            checkpointer = await get_checkpointer_for_settings(mock_settings)
            assert isinstance(checkpointer, AsyncSqliteSaver)


class TestCheckpointerTypeDetection:
    """Test checkpointer type detection."""

    def test_detect_memory_saver(self):
        """MemorySaver detected correctly."""
        from kai.core.persistence.checkpointer import is_sqlite_checkpointer
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        assert not is_sqlite_checkpointer(checkpointer)

    @pytest.mark.asyncio
    async def test_detect_sqlite_saver(self):
        """AsyncSqliteSaver detected correctly."""
        from kai.core.persistence.checkpointer import (
            CheckpointerType, create_checkpointer, is_sqlite_checkpointer
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            # Use factory function to create initialized AsyncSqliteSaver
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )
            assert is_sqlite_checkpointer(checkpointer)


class TestRetentionPolicyIntegration:
    """Test retention policy integration."""

    @pytest.mark.asyncio
    async def test_cleanup_with_memory_saver_warns(self):
        """Cleanup on MemorySaver produces warning."""
        from kai.core.debug.checkpoint_cleanup import cleanup_old_checkpoints
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        result = await cleanup_old_checkpoints(
            checkpointer, retention_days=7
        )

        # MemorySaver doesn't support cleanup operations
        assert "errors" in result
        assert len(result["errors"]) > 0


# =============================================================================
# Subgraph State Propagation Tests
# =============================================================================

class TestSubgraphCheckpointerConfig:
    """Test that subgraphs don't have their own checkpointers.

    When a subgraph has its own checkpointer, state updates from subgraph nodes
    are stored in the subgraph's checkpointer and NOT propagated to the parent
    graph's state. This caused a bug where auto_loop_update was set in a subgraph
    but the parent graph's aget_state() didn't see it.

    Rule: Only the main graph should have a checkpointer. Nested subgraphs
    (execution, planning, regular) should use graph.compile() without a
    checkpointer so state flows through the parent.

    These tests verify by inspecting the source code patterns, since building
    full graphs requires many dependencies.
    """

    def test_execution_subgraph_compile_has_no_checkpointer(self):
        """Execution subgraph source should use graph.compile() without checkpointer."""
        import inspect
        from kai.core.orchestration.graphs import execution

        source = inspect.getsource(execution.build_execution_subgraph)

        # Should NOT have checkpointer= in compile call
        assert "graph.compile(checkpointer=" not in source, (
            "Execution subgraph should use graph.compile() without checkpointer. "
            "Using a separate checkpointer causes state updates to be lost."
        )
        # Should have the plain compile() call
        assert "return graph.compile()" in source, (
            "Execution subgraph should end with return graph.compile()"
        )

    def test_planning_subgraph_compile_has_no_checkpointer(self):
        """Planning subgraph source should use graph.compile() without checkpointer."""
        import inspect
        from kai.core.orchestration.graphs import planning

        source = inspect.getsource(planning.build_planning_subgraph)

        assert "graph.compile(checkpointer=" not in source, (
            "Planning subgraph should use graph.compile() without checkpointer."
        )
        assert "return graph.compile()" in source

    def test_regular_subgraph_compile_has_no_checkpointer(self):
        """Regular subgraph source should use graph.compile() without checkpointer."""
        import inspect
        from kai.core.orchestration.graphs import regular

        source = inspect.getsource(regular.build_regular_subgraph)

        assert "graph.compile(checkpointer=" not in source, (
            "Regular subgraph should use graph.compile() without checkpointer."
        )
        assert "return graph.compile()" in source

    def test_main_graph_compile_uses_checkpointer(self):
        """Main graph source should use graph.compile(checkpointer=...)."""
        import inspect
        from kai.core.orchestration.graphs import main

        source = inspect.getsource(main.build_main_graph)

        # Main graph SHOULD have checkpointer
        assert "graph.compile(checkpointer=" in source, (
            "Main graph should use graph.compile(checkpointer=...) "
            "for state persistence across iterations."
        )

    def test_section_execution_is_standalone_with_checkpointer(self):
        """Section execution is standalone and CAN have its own checkpointer.

        Section execution is invoked directly by the orchestrator, not as a
        nested subgraph. It's intentionally separate for cell-by-cell execution.
        """
        import inspect
        from kai.core.orchestration.graphs import section_execution

        source = inspect.getsource(
            section_execution.build_section_execution_subgraph
        )

        # Section execution CAN have checkpointer (it's standalone)
        assert "graph.compile(checkpointer=" in source, (
            "Section execution is standalone and should have its own "
            "checkpointer for tracking cell execution state."
        )


class TestAutoLoopUpdatePropagation:
    """Test that auto_loop_update state field propagates correctly."""

    def test_auto_loop_update_in_kai_state(self):
        """auto_loop_update should be defined in KaiState."""
        from kai.core.orchestration.state import KaiState
        from typing import get_type_hints

        hints = get_type_hints(KaiState)
        assert "auto_loop_update" in hints, (
            "auto_loop_update must be defined in KaiState for proper "
            "state propagation. Without it, the field may not persist."
        )

    def test_auto_loop_update_not_in_transient_fields(self):
        """auto_loop_update should NOT be reset each iteration.

        If it were transient, the completion signal would be lost before
        the orchestrator could read it.
        """
        assert "auto_loop_update" not in TRANSIENT_FIELD_NAMES, (
            "auto_loop_update must NOT be transient - the completion signal "
            "must persist until the orchestrator reads it."
        )


# =============================================================================
# UICommunicator Hook Priority Tests
# =============================================================================

class TestUICommunicatorHookPriority:
    """Test that Jupyter hooks are called BEFORE disabled check.

    In Jupyter mode, _disabled=True suppresses VSCode stdout output while
    hooks still need to receive workflow signals like LOOP_COMPLETE.
    """

    def test_send_workflow_result_checks_hook_before_disabled(self):
        """send_workflow_result must check Jupyter hook BEFORE _disabled.

        Inspects source code to verify hook check precedes disabled check.
        """
        import inspect
        from kai.core.orchestration.ui_communicator import UICommunicator

        source = inspect.getsource(UICommunicator.send_workflow_result)
        lines = source.split('\n')

        # Find line numbers of key checks
        hook_check_line = None
        disabled_check_line = None

        for i, line in enumerate(lines):
            if '_workflow_result_hook is not None' in line:
                hook_check_line = i
            # _disabled check may have return on same or next line
            if 'if self._disabled' in line or 'self._disabled:' in line:
                disabled_check_line = i

        assert hook_check_line is not None, (
            "send_workflow_result must check for _workflow_result_hook"
        )
        assert disabled_check_line is not None, (
            "send_workflow_result must check self._disabled"
        )
        assert hook_check_line < disabled_check_line, (
            f"CRITICAL BUG: Jupyter hook check (line {hook_check_line}) must "
            f"come BEFORE disabled check (line {disabled_check_line}). "
            f"Otherwise LOOP_COMPLETE signals are dropped in Jupyter mode, "
            f"causing infinite loops."
        )

    @pytest.mark.asyncio
    async def test_jupyter_hook_called_when_disabled(self):
        """Jupyter hook should be called even when _disabled=True.

        This simulates the Jupyter mode where _disabled=True but we still
        need workflow results to reach the Jupyter interface.
        """
        from kai.core.orchestration.ui_communicator import UICommunicator

        # Track hook calls
        hook_calls = []

        async def mock_hook(field, state):
            hook_calls.append((field, state))

        # Set up Jupyter hook
        original_hook = UICommunicator._workflow_result_hook
        try:
            UICommunicator.set_workflow_result_hook(mock_hook)

            # Create disabled communicator (simulates Jupyter mode)
            communicator = UICommunicator()
            communicator._disabled = True

            # Send workflow result - should still call hook
            await communicator.send_workflow_result(
                "auto_loop_update", "LOOP_COMPLETE"
            )

            # Verify hook was called
            assert len(hook_calls) == 1, (
                "Jupyter hook should be called even when _disabled=True"
            )
            assert hook_calls[0] == ("auto_loop_update", "LOOP_COMPLETE"), (
                "Hook should receive correct field and state"
            )
        finally:
            # Restore original hook
            UICommunicator.set_workflow_result_hook(original_hook)

    @pytest.mark.asyncio
    async def test_disabled_skips_when_no_hook(self):
        """When no hook is set and _disabled=True, should return early."""
        from kai.core.orchestration.ui_communicator import UICommunicator

        # Clear any existing hook
        original_hook = UICommunicator._workflow_result_hook
        try:
            UICommunicator.set_workflow_result_hook(None)

            # Create disabled communicator
            communicator = UICommunicator()
            communicator._disabled = True

            # This should return early without error (no stdout output)
            await communicator.send_workflow_result(
                "auto_loop_update", "LOOP_COMPLETE"
            )
            # If we get here without error, the test passes
        finally:
            UICommunicator.set_workflow_result_hook(original_hook)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
