"""Tests for time-travel debugging utilities.

Tests:
- Checkpoint listing
- State retrieval at step
- Error checkpoint finding
- State comparison
"""

import pytest
from unittest.mock import MagicMock, AsyncMock

from kai.core.debug.time_travel import (
    TimeTravelDebugger,
    resume_from_checkpoint,
    _truncate_value,
)


class TestTimeTravelDebugger:
    """Test TimeTravelDebugger class."""

    def test_init_with_checkpointer(self):
        """Debugger initializes with checkpointer."""
        mock_checkpointer = MagicMock()
        debugger = TimeTravelDebugger(mock_checkpointer)
        assert debugger.checkpointer is mock_checkpointer

    @pytest.mark.asyncio
    async def test_list_checkpoints(self):
        """List checkpoints returns checkpoint summaries."""
        mock_checkpointer = MagicMock()

        # Create mock state history
        mock_state1 = MagicMock()
        mock_state1.config = {"configurable": {"checkpoint_id": "cp1"}}
        mock_state1.metadata = {"ts": "2025-01-01T00:00:00"}
        mock_state1.next = ["node1"]
        mock_state1.values = {"key": "value"}

        mock_state2 = MagicMock()
        mock_state2.config = {"configurable": {"checkpoint_id": "cp2"}}
        mock_state2.metadata = {"ts": "2025-01-01T00:01:00"}
        mock_state2.next = []
        mock_state2.values = {"key": "value2"}

        async def mock_history(*args, **kwargs):
            yield mock_state1
            yield mock_state2

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        checkpoints = await debugger.list_checkpoints("test-thread")

        assert len(checkpoints) == 2
        assert checkpoints[0]["checkpoint_id"] == "cp1"
        assert checkpoints[1]["checkpoint_id"] == "cp2"

    @pytest.mark.asyncio
    async def test_list_checkpoints_with_limit(self):
        """List checkpoints respects limit."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            for i in range(10):
                state = MagicMock()
                state.config = {"configurable": {"checkpoint_id": f"cp{i}"}}
                state.metadata = {"ts": f"2025-01-01T00:0{i}:00"}
                state.next = []
                state.values = {}
                yield state

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        checkpoints = await debugger.list_checkpoints("test-thread", limit=3)

        assert len(checkpoints) == 3

    @pytest.mark.asyncio
    async def test_get_state_at_step(self):
        """Get state at specific step."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            for i in range(3):
                state = MagicMock()
                state.values = {"step": i, "data": f"value{i}"}
                yield state

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        state = await debugger.get_state_at_step("test-thread", step=1)

        assert state is not None
        assert state["step"] == 1
        assert state["data"] == "value1"

    @pytest.mark.asyncio
    async def test_get_state_at_step_not_found(self):
        """Get state returns None for invalid step."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            state = MagicMock()
            state.values = {"step": 0}
            yield state

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        state = await debugger.get_state_at_step("test-thread", step=99)

        assert state is None

    @pytest.mark.asyncio
    async def test_find_error_checkpoint(self):
        """Find error checkpoint returns error info."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            # First state - no error
            state1 = MagicMock()
            state1.values = {"key": "value"}
            state1.config = {"configurable": {"checkpoint_id": "cp1"}}
            state1.metadata = {"ts": "2025-01-01T00:00:00"}
            yield state1

            # Second state - has error
            state2 = MagicMock()
            state2.values = {"error_context": {"msg": "Test error"}}
            state2.config = {"configurable": {"checkpoint_id": "cp2"}}
            state2.metadata = {"ts": "2025-01-01T00:01:00"}
            yield state2

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        error_info = await debugger.find_error_checkpoint("test-thread")

        assert error_info is not None
        assert error_info["step"] == 1
        assert error_info["checkpoint_id"] == "cp2"
        assert "Test error" in str(error_info["error_context"])

    @pytest.mark.asyncio
    async def test_find_error_checkpoint_no_errors(self):
        """Find error checkpoint returns None when no errors."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            state = MagicMock()
            state.values = {"key": "value"}  # No error_context
            yield state

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        error_info = await debugger.find_error_checkpoint("test-thread")

        assert error_info is None

    @pytest.mark.asyncio
    async def test_get_resume_config(self):
        """Get resume config creates correct structure."""
        mock_checkpointer = MagicMock()
        debugger = TimeTravelDebugger(mock_checkpointer)

        config = await debugger.get_resume_config(
            "test-thread", "checkpoint-123"
        )

        assert config["configurable"]["thread_id"] == "test-thread"
        assert config["configurable"]["checkpoint_id"] == "checkpoint-123"

    @pytest.mark.asyncio
    async def test_compare_states(self):
        """Compare states finds differences."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            state1 = MagicMock()
            state1.values = {"same": "value", "different": "old"}
            yield state1

            state2 = MagicMock()
            state2.values = {"same": "value", "different": "new"}
            yield state2

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        comparison = await debugger.compare_states("test-thread", 0, 1)

        assert "different" in comparison["differences"]
        assert "same" in comparison["unchanged_keys"]


class TestTruncateValue:
    """Test value truncation helper."""

    def test_truncate_long_string(self):
        """Long strings are truncated."""
        long_str = "a" * 500
        result = _truncate_value(long_str, max_length=100)
        assert len(result) == 103  # 100 + "..."
        assert result.endswith("...")

    def test_truncate_short_string(self):
        """Short strings are not truncated."""
        short_str = "hello"
        result = _truncate_value(short_str)
        assert result == "hello"

    def test_truncate_long_list(self):
        """Long lists are truncated."""
        long_list = list(range(20))
        result = _truncate_value(long_list)
        assert len(result) == 6  # 5 items + "..."
        assert result[-1] == "..."

    def test_truncate_dict(self):
        """Large dicts show summary."""
        large_dict = {f"key{i}": f"value{i}" for i in range(100)}
        result = _truncate_value(large_dict, max_length=50)
        assert "100 keys" in result


@pytest.mark.asyncio
class TestResumeFromCheckpoint:
    """Test resume_from_checkpoint function."""

    async def test_resume_without_modifications(self):
        """Resume from checkpoint without state changes."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"result": "done"})

        result = await resume_from_checkpoint(
            mock_graph,
            thread_id="test-thread",
            checkpoint_id="cp-123",
        )

        assert result == {"result": "done"}
        mock_graph.ainvoke.assert_called_once()

        # Check config was correct
        call_args = mock_graph.ainvoke.call_args
        config = call_args[0][1]
        assert config["configurable"]["thread_id"] == "test-thread"
        assert config["configurable"]["checkpoint_id"] == "cp-123"

    async def test_resume_with_modifications(self):
        """Resume from checkpoint with state modifications."""
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"result": "modified"})

        modifications = {"new_field": "new_value"}
        result = await resume_from_checkpoint(
            mock_graph,
            thread_id="test-thread",
            checkpoint_id="cp-123",
            modified_state=modifications,
        )

        assert result == {"result": "modified"}

        # Check modifications were passed
        call_args = mock_graph.ainvoke.call_args
        passed_state = call_args[0][0]
        assert passed_state == modifications


class TestTimeTravelDebuggerEdgeCases:
    """Edge case tests for TimeTravelDebugger."""

    @pytest.mark.asyncio
    async def test_list_checkpoints_empty(self):
        """List checkpoints handles empty history."""
        mock_checkpointer = MagicMock()

        async def empty_history(*args, **kwargs):
            return
            yield

        mock_checkpointer.aget_state_history = empty_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        checkpoints = await debugger.list_checkpoints("test-thread")

        assert checkpoints == []

    @pytest.mark.asyncio
    async def test_compare_states_missing_step(self):
        """compare_states handles missing step gracefully."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            state = MagicMock()
            state.values = {"key": "value"}
            yield state

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        comparison = await debugger.compare_states("test-thread", 0, 99)

        # Should handle gracefully - one state exists, other doesn't
        assert comparison is not None

    @pytest.mark.asyncio
    async def test_find_error_via_error_context(self):
        """find_error_checkpoint detects error_context."""
        mock_checkpointer = MagicMock()

        async def mock_history(*args, **kwargs):
            # First state - no error
            state1 = MagicMock()
            state1.values = {"key": "value"}
            state1.config = {"configurable": {"checkpoint_id": "cp1"}}
            state1.metadata = {"ts": "2025-01-01T00:00:00"}
            yield state1

            # Second state - has error_context (what find_error_checkpoint looks for)
            state2 = MagicMock()
            state2.values = {"error_context": {"message": "Runtime error", "type": "ValueError"}}
            state2.config = {"configurable": {"checkpoint_id": "cp2"}}
            state2.metadata = {"ts": "2025-01-01T00:01:00"}
            yield state2

        mock_checkpointer.aget_state_history = mock_history

        debugger = TimeTravelDebugger(mock_checkpointer)
        error_info = await debugger.find_error_checkpoint("test-thread")

        assert error_info is not None
        assert error_info["checkpoint_id"] == "cp2"
        assert error_info["error_context"]["type"] == "ValueError"


class TestTruncateValueEdgeCases:
    """Edge case tests for _truncate_value."""

    def test_truncate_none(self):
        """Truncate handles None."""
        result = _truncate_value(None)
        assert result is None

    def test_truncate_number(self):
        """Truncate handles numbers."""
        result = _truncate_value(42)
        assert result == 42

    def test_truncate_boolean(self):
        """Truncate handles booleans."""
        result = _truncate_value(True)
        assert result is True

    def test_truncate_nested_dict(self):
        """Truncate handles nested dicts."""
        nested = {"level1": {"level2": {"level3": "value"}}}
        result = _truncate_value(nested)
        # Should handle without crashing
        assert result is not None

    def test_truncate_empty_list(self):
        """Truncate handles empty list."""
        result = _truncate_value([])
        assert result == []

    def test_truncate_empty_dict(self):
        """Truncate handles empty dict."""
        result = _truncate_value({})
        assert result == {}
