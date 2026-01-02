"""Tests for debugging and persistence utilities.

Tests:
- Checkpoint export functionality
- Checkpoint cleanup policies
"""

import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.core.debug.checkpoint_exporter import (
    CheckpointDebugExporter,
)


# =============================================================================
# Checkpoint Exporter Tests
# =============================================================================

class TestCheckpointExporter:
    """Test checkpoint export functionality."""

    def test_init_creates_output_dir(self):
        """Exporter creates output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "exports"
            exporter = CheckpointDebugExporter(output_dir)
            assert exporter.output_dir == output_dir
            assert output_dir.exists()

    def test_serialize_state_handles_simple_values(self):
        """State serialization handles simple values."""
        exporter = CheckpointDebugExporter()
        state = {
            "string": "test",
            "number": 42,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
        }
        serialized = exporter._serialize_state(state)
        assert serialized == state

    def test_serialize_state_handles_non_serializable(self):
        """State serialization handles non-JSON-serializable values."""
        exporter = CheckpointDebugExporter()

        class CustomObject:
            def __str__(self):
                return "CustomObject()"

        state = {
            "normal": "value",
            "custom": CustomObject(),
        }
        serialized = exporter._serialize_state(state)
        assert serialized["normal"] == "value"
        assert "CustomObject" in str(serialized["custom"])

    def test_format_as_markdown(self):
        """Markdown formatting works."""
        exporter = CheckpointDebugExporter()
        data = {
            "session_id": "test-session",
            "exported_at": "2025-01-01T00:00:00",
            "total_steps": 2,
            "history": [
                {
                    "checkpoint_id": "cp1",
                    "step": 0,
                    "timestamp": "2025-01-01T00:00:00",
                    "next_nodes": ["planning"],
                    "state": {"key": "value1"},
                },
                {
                    "checkpoint_id": "cp2",
                    "step": 1,
                    "timestamp": "2025-01-01T00:01:00",
                    "next_nodes": [],
                    "state": {"key": "value2"},
                },
            ],
        }
        md = exporter._format_as_markdown(data)

        assert "# Session Trace: test-session" in md
        assert "## Step 0" in md
        assert "## Step 1" in md
        assert "cp1" in md
        assert "cp2" in md


# =============================================================================
# Checkpoint Cleanup Tests
# =============================================================================

class TestCheckpointCleanup:
    """Test checkpoint retention and cleanup."""

    @pytest.mark.asyncio
    async def test_cleanup_with_memory_saver_returns_error(self):
        """Cleanup on MemorySaver returns error (not supported)."""
        from kai.core.debug.checkpoint_cleanup import cleanup_old_checkpoints
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        result = await cleanup_old_checkpoints(checkpointer, retention_days=7)

        assert "errors" in result
        assert len(result["errors"]) > 0
        assert "SqliteSaver" in result["errors"][0]

    @pytest.mark.asyncio
    async def test_get_checkpoint_stats_memory_saver(self):
        """get_checkpoint_stats returns error for MemorySaver."""
        from kai.core.debug.checkpoint_cleanup import get_checkpoint_stats
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        stats = await get_checkpoint_stats(checkpointer)

        # MemorySaver doesn't support listing without config
        # Class name may be MemorySaver or InMemorySaver depending on version
        assert "MemorySaver" in stats["checkpoint_type"]
        # Should have error since list() without args not supported
        assert "error" in stats or stats["total_threads"] == 0

    @pytest.mark.asyncio
    async def test_cleanup_stats_include_cutoff_date(self):
        """Cleanup stats include cutoff date and retention days."""
        from kai.core.debug.checkpoint_cleanup import cleanup_old_checkpoints
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
        result = await cleanup_old_checkpoints(
            checkpointer,
            retention_days=14,
            dry_run=True
        )

        assert result["retention_days"] == 14
        assert "cutoff_date" in result
        assert result["dry_run"] is True


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.asyncio
class TestDebugIntegration:
    """Integration tests for debug utilities."""

    async def test_exporter_handles_empty_history(self):
        """Exporter handles graph with no checkpoint history."""
        exporter = CheckpointDebugExporter()

        # Mock graph that returns empty history
        mock_graph = MagicMock()

        async def empty_history(*args, **kwargs):
            return
            yield  # Empty async generator

        mock_graph.aget_state_history = empty_history

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = Path(tmpdir)
            path = await exporter.export_session_trace(
                mock_graph,
                "test-session",
                format="json"
            )

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["session_id"] == "test-session"
            assert data["history"] == []

    async def test_get_state_at_step_not_found(self):
        """get_state_at_step returns None for missing step."""
        exporter = CheckpointDebugExporter()

        mock_graph = MagicMock()

        async def empty_history(*args, **kwargs):
            return
            yield

        mock_graph.aget_state_history = empty_history

        result = await exporter.get_state_at_step(mock_graph, "session", 5)
        assert result is None

    async def test_export_markdown_format(self):
        """Exporter can export in markdown format."""
        exporter = CheckpointDebugExporter()

        mock_graph = MagicMock()

        async def mock_history(*args, **kwargs):
            state = MagicMock()
            state.config = {"configurable": {"checkpoint_id": "cp1"}}
            state.metadata = {"step": 0, "ts": "2025-01-01T00:00:00"}
            state.next = ["next_node"]
            state.values = {"task_list": {"tasks": []}, "user_query": "test"}
            yield state

        mock_graph.aget_state_history = mock_history

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = Path(tmpdir)
            path = await exporter.export_session_trace(
                mock_graph,
                "test-session",
                format="markdown"
            )

            assert path.exists()
            assert path.suffix == ".md"

            content = path.read_text()
            assert "# Session Trace: test-session" in content
            assert "## Step 0" in content
            assert "cp1" in content

    async def test_get_state_at_specific_step(self):
        """get_state_at_step returns correct state."""
        exporter = CheckpointDebugExporter()

        mock_graph = MagicMock()

        async def mock_history(*args, **kwargs):
            for i in range(3):
                state = MagicMock()
                state.metadata = {"step": i}
                state.values = {"iteration": i, "data": f"value_{i}"}
                yield state

        mock_graph.aget_state_history = mock_history

        result = await exporter.get_state_at_step(mock_graph, "session", 1)
        assert result is not None
        assert result["iteration"] == 1
        assert result["data"] == "value_1"

    async def test_replay_from_checkpoint(self):
        """replay_from_checkpoint invokes graph correctly."""
        exporter = CheckpointDebugExporter()

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value={"result": "replayed"})

        result = await exporter.replay_from_checkpoint(
            mock_graph,
            session_id="test-session",
            checkpoint_id="cp-123"
        )

        assert result == {"result": "replayed"}
        mock_graph.ainvoke.assert_called_once()

        # Verify config was correct
        call_args = mock_graph.ainvoke.call_args
        config = call_args[0][1]
        assert config["configurable"]["thread_id"] == "test-session"
        assert config["configurable"]["checkpoint_id"] == "cp-123"

    async def test_export_invalid_format_raises(self):
        """Exporter raises for unknown format."""
        exporter = CheckpointDebugExporter()

        mock_graph = MagicMock()

        async def empty_history(*args, **kwargs):
            return
            yield

        mock_graph.aget_state_history = empty_history

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter.output_dir = Path(tmpdir)
            with pytest.raises(ValueError, match="Unknown format"):
                await exporter.export_session_trace(
                    mock_graph,
                    "test-session",
                    format="invalid"
                )

    async def test_top_level_export_session_trace(self):
        """Top-level export_session_trace function works."""
        from kai.core.debug.checkpoint_exporter import export_session_trace

        mock_graph = MagicMock()

        async def empty_history(*args, **kwargs):
            return
            yield

        mock_graph.aget_state_history = empty_history

        with tempfile.TemporaryDirectory() as tmpdir:
            path = await export_session_trace(
                mock_graph,
                "test-session",
                output_dir=Path(tmpdir),
                format="json"
            )

            assert path.exists()
            with open(path) as f:
                data = json.load(f)
            assert data["session_id"] == "test-session"
