"""Tests for checkpoint persistence and session restart.

Tests:
- AsyncSqliteSaver storage location and schema
- CheckpointMode TRANSIENT vs PERSISTENT behavior
- Session cleanup on completion
- Graph state restoration from checkpoint
- Integration with restart_session workflow
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from kai.core.persistence.checkpointer import (
    CheckpointerType,
    CheckpointMode,
    create_checkpointer,
    get_checkpointer_for_settings,
    get_checkpoint_mode,
    clear_session_checkpoints,
    get_default_checkpoint_path,
    is_sqlite_checkpointer,
)


class TestCheckpointerCreation:
    """Test checkpointer factory functions."""

    @pytest.mark.asyncio
    async def test_create_memory_checkpointer(self):
        """MemorySaver is created correctly."""
        checkpointer = await create_checkpointer(CheckpointerType.MEMORY)
        assert isinstance(checkpointer, MemorySaver)

    @pytest.mark.asyncio
    async def test_create_sqlite_checkpointer(self):
        """AsyncSqliteSaver is created with database path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_checkpoints.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )
            assert isinstance(checkpointer, AsyncSqliteSaver)
            assert db_path.exists()

    @pytest.mark.asyncio
    async def test_sqlite_checkpointer_requires_path(self):
        """AsyncSqliteSaver requires db_path parameter."""
        with pytest.raises(ValueError, match="db_path required"):
            await create_checkpointer(CheckpointerType.SQLITE)

    @pytest.mark.asyncio
    async def test_sqlite_checkpointer_creates_parent_dirs(self):
        """AsyncSqliteSaver creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "nested" / "dir" / "checkpoints.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )
            assert isinstance(checkpointer, AsyncSqliteSaver)
            assert db_path.parent.exists()


class TestCheckpointMode:
    """Test checkpoint mode configuration."""

    def test_transient_mode_value(self):
        """TRANSIENT mode has correct value."""
        assert CheckpointMode.TRANSIENT.value == "transient"

    def test_persistent_mode_value(self):
        """PERSISTENT mode has correct value."""
        assert CheckpointMode.PERSISTENT.value == "persistent"

    def test_get_checkpoint_mode_transient(self):
        """get_checkpoint_mode returns TRANSIENT for transient setting."""
        mock_settings = MagicMock()
        mock_settings.CHECKPOINT_MODE = "transient"
        mode = get_checkpoint_mode(mock_settings)
        assert mode == CheckpointMode.TRANSIENT

    def test_get_checkpoint_mode_persistent(self):
        """get_checkpoint_mode returns PERSISTENT for persistent setting."""
        mock_settings = MagicMock()
        mock_settings.CHECKPOINT_MODE = "persistent"
        mode = get_checkpoint_mode(mock_settings)
        assert mode == CheckpointMode.PERSISTENT

    def test_get_checkpoint_mode_invalid_defaults_transient(self):
        """Invalid mode string defaults to TRANSIENT."""
        mock_settings = MagicMock()
        mock_settings.CHECKPOINT_MODE = "invalid_mode"
        mode = get_checkpoint_mode(mock_settings)
        assert mode == CheckpointMode.TRANSIENT

    def test_get_checkpoint_mode_case_insensitive(self):
        """Mode is case insensitive."""
        mock_settings = MagicMock()
        mock_settings.CHECKPOINT_MODE = "PERSISTENT"
        mode = get_checkpoint_mode(mock_settings)
        assert mode == CheckpointMode.PERSISTENT


class TestCheckpointerForSettings:
    """Test checkpointer creation from settings."""

    @pytest.mark.asyncio
    async def test_default_path_used_when_not_specified(self):
        """Default path is used when CHECKPOINT_DB_PATH is None."""
        mock_settings = MagicMock()
        mock_settings.CHECKPOINT_DB_PATH = None

        with tempfile.TemporaryDirectory() as tmpdir:
            default_path = Path(tmpdir) / "checkpoints.db"
            mock_settings.checkpoint_db_path_resolved = default_path

            checkpointer = await get_checkpointer_for_settings(mock_settings)
            assert isinstance(checkpointer, AsyncSqliteSaver)
            assert default_path.exists()

    @pytest.mark.asyncio
    async def test_explicit_path_used(self):
        """Explicit CHECKPOINT_DB_PATH is used."""
        mock_settings = MagicMock()

        with tempfile.TemporaryDirectory() as tmpdir:
            explicit_path = Path(tmpdir) / "custom_checkpoints.db"
            mock_settings.checkpoint_db_path_resolved = explicit_path

            checkpointer = await get_checkpointer_for_settings(mock_settings)
            assert isinstance(checkpointer, AsyncSqliteSaver)
            assert explicit_path.exists()

    @pytest.mark.asyncio
    async def test_memory_saver_when_disabled(self):
        """MemorySaver used when checkpointing is disabled."""
        mock_settings = MagicMock()
        mock_settings.CHECKPOINT_ENABLED = False

        checkpointer = await get_checkpointer_for_settings(mock_settings)
        assert isinstance(checkpointer, MemorySaver)


class TestIsCheckpointer:
    """Test checkpointer type detection."""

    @pytest.mark.asyncio
    async def test_is_sqlite_checkpointer_true(self):
        """is_sqlite_checkpointer returns True for AsyncSqliteSaver."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )
            assert is_sqlite_checkpointer(checkpointer) is True

    @pytest.mark.asyncio
    async def test_is_sqlite_checkpointer_false_for_memory(self):
        """is_sqlite_checkpointer returns False for MemorySaver."""
        checkpointer = await create_checkpointer(CheckpointerType.MEMORY)
        assert is_sqlite_checkpointer(checkpointer) is False


class TestDefaultCheckpointPath:
    """Test default checkpoint path."""

    def test_get_default_checkpoint_path(self):
        """Default path is in ~/.kai_agent/."""
        path = get_default_checkpoint_path()
        assert path.name == "checkpoints.db"
        assert ".kai_agent" in str(path) or "kai_agent" in str(path)


class TestClearSessionCheckpoints:
    """Test session checkpoint cleanup."""

    @pytest.mark.asyncio
    async def test_clear_sqlite_checkpoints(self):
        """AsyncSqliteSaver checkpoints are cleared correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Insert test data directly (async methods for aiosqlite)
            conn = checkpointer.conn
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("test-session", "", "cp1", None, "checkpoint", b"{}", b"{}")
            )
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("test-session", "", "cp2", "cp1", "checkpoint", b"{}", b"{}")
            )
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("other-session", "", "cp3", None, "checkpoint", b"{}", b"{}")
            )
            await conn.commit()

            # Verify data exists
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test-session",)
            )
            row = await cursor.fetchone()
            assert row[0] == 2

            # Clear session
            result = await clear_session_checkpoints(checkpointer, "test-session")
            assert result is True

            # Verify test-session checkpoints are gone
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("test-session",)
            )
            row = await cursor.fetchone()
            assert row[0] == 0

            # Verify other-session checkpoints remain
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("other-session",)
            )
            row = await cursor.fetchone()
            assert row[0] == 1

    @pytest.mark.asyncio
    async def test_clear_memory_saver_noop(self):
        """MemorySaver clear is a no-op (returns True)."""
        checkpointer = await create_checkpointer(CheckpointerType.MEMORY)
        result = await clear_session_checkpoints(checkpointer, "test-session")
        assert result is True

    @pytest.mark.asyncio
    async def test_clear_nonexistent_session(self):
        """Clearing nonexistent session succeeds."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            result = await clear_session_checkpoints(
                checkpointer, "nonexistent-session"
            )
            assert result is True


class TestSqliteSchema:
    """Test SQLite database schema."""

    @pytest.mark.asyncio
    async def test_schema_has_checkpoints_table(self):
        """SQLite database has checkpoints table."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            conn = checkpointer.conn
            cursor = await conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='checkpoints'"
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "checkpoints"

    @pytest.mark.asyncio
    async def test_schema_supports_thread_id_queries(self):
        """Checkpoints can be queried by thread_id."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            conn = checkpointer.conn
            # Insert and query by thread_id
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("thread-abc", "", "cp1", None, "checkpoint", b"{}", b"{}")
            )
            await conn.commit()

            cursor = await conn.execute(
                "SELECT checkpoint_id FROM checkpoints WHERE thread_id = ?",
                ("thread-abc",)
            )
            result = await cursor.fetchone()
            assert result is not None
            assert result[0] == "cp1"


class TestCheckpointRestoration:
    """Test checkpoint state restoration scenarios."""

    @pytest.mark.asyncio
    async def test_state_persists_across_checkpointer_recreation(self):
        """State persists when checkpointer is recreated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            thread_id = "test-thread"

            # Create first checkpointer and save state
            checkpointer1 = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Insert checkpoint data
            conn1 = checkpointer1.conn
            await conn1.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", "cp-initial", None, "checkpoint", b'{"state": "saved"}', b'{"step": 1}')
            )
            await conn1.commit()
            await conn1.close()

            # Recreate checkpointer (simulating process restart)
            checkpointer2 = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Verify state is still there
            conn2 = checkpointer2.conn
            cursor = await conn2.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                (thread_id, "cp-initial")
            )
            result = await cursor.fetchone()
            assert result is not None
            assert b"saved" in result[0]


class TestTransientModeWorkflow:
    """Test TRANSIENT mode workflow (clear on completion)."""

    @pytest.mark.asyncio
    async def test_transient_mode_clears_on_completion(self):
        """TRANSIENT mode clears checkpoints when session completes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            thread_id = "transient-session"

            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Simulate checkpoint being saved during execution
            conn = checkpointer.conn
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", "cp1", None, "checkpoint", b"{}", b"{}")
            )
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", "cp2", "cp1", "checkpoint", b"{}", b"{}")
            )
            await conn.commit()

            # Verify checkpoints exist
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            row = await cursor.fetchone()
            assert row[0] == 2

            # Simulate TRANSIENT mode cleanup on completion
            mode = CheckpointMode.TRANSIENT
            if mode == CheckpointMode.TRANSIENT:
                await clear_session_checkpoints(checkpointer, thread_id)

            # Verify checkpoints are gone
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            row = await cursor.fetchone()
            assert row[0] == 0

    @pytest.mark.asyncio
    async def test_persistent_mode_keeps_checkpoints(self):
        """PERSISTENT mode keeps checkpoints after completion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            thread_id = "persistent-session"

            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Simulate checkpoint being saved during execution
            conn = checkpointer.conn
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", "cp1", None, "checkpoint", b"{}", b"{}")
            )
            await conn.commit()

            # Simulate PERSISTENT mode - no cleanup on completion
            mode = CheckpointMode.PERSISTENT
            if mode == CheckpointMode.TRANSIENT:
                await clear_session_checkpoints(checkpointer, thread_id)

            # Verify checkpoints remain
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            row = await cursor.fetchone()
            assert row[0] == 1


class TestOrchestratorIntegration:
    """Test integration with LangGraphOrchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_cleanup_with_transient_mode(self):
        """Orchestrator clears checkpoints in TRANSIENT mode."""
        from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create mock settings
            mock_settings = MagicMock()
            mock_settings.checkpoint_db_path_resolved = db_path
            mock_settings.CHECKPOINT_MODE = "transient"

            # Create checkpointer
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Create orchestrator with mocked components
            orchestrator = LangGraphOrchestrator.__new__(LangGraphOrchestrator)
            orchestrator.checkpointer = checkpointer
            orchestrator.checkpoint_mode = CheckpointMode.TRANSIENT

            # Add some checkpoint data
            conn = checkpointer.conn
            thread_id = "test-session-123"
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", "cp1", None, "checkpoint", b"{}", b"{}")
            )
            await conn.commit()

            # Verify data exists
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            row = await cursor.fetchone()
            assert row[0] == 1

            # Call cleanup method
            result = await orchestrator.clear_session_on_completion(thread_id)
            assert result is True

            # Verify checkpoints are cleared
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            row = await cursor.fetchone()
            assert row[0] == 0

    @pytest.mark.asyncio
    async def test_orchestrator_no_cleanup_with_persistent_mode(self):
        """Orchestrator keeps checkpoints in PERSISTENT mode."""
        from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            # Create checkpointer
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Create orchestrator with PERSISTENT mode
            orchestrator = LangGraphOrchestrator.__new__(LangGraphOrchestrator)
            orchestrator.checkpointer = checkpointer
            orchestrator.checkpoint_mode = CheckpointMode.PERSISTENT

            # Add some checkpoint data
            conn = checkpointer.conn
            thread_id = "test-session-456"
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", "cp1", None, "checkpoint", b"{}", b"{}")
            )
            await conn.commit()

            # Call cleanup method - should not clear in PERSISTENT mode
            result = await orchestrator.clear_session_on_completion(thread_id)
            assert result is True  # Returns True but doesn't delete

            # Verify checkpoints remain
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                (thread_id,)
            )
            row = await cursor.fetchone()
            assert row[0] == 1


class TestRestartSessionSimulation:
    """Test restart session scenarios."""

    @pytest.mark.asyncio
    async def test_checkpoint_survives_simulated_crash(self):
        """Checkpoint data survives simulated process crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            thread_id = "crash-test-session"

            # Phase 1: Create checkpoint before "crash"
            checkpointer1 = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Save state representing mid-execution
            conn1 = checkpointer1.conn
            state_data = b'{"task_list": [{"name": "Task 1", "status": "completed"}, {"name": "Task 2", "status": "in_progress"}], "last_cell_modified": 5}'
            await conn1.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (thread_id, "", "cp-before-crash", None, "checkpoint", state_data, b'{"step": 10}')
            )
            await conn1.commit()
            await conn1.close()

            # Phase 2: "Crash" - close connection without cleanup
            # (simulated by just closing the connection)

            # Phase 3: Restart - recreate checkpointer
            checkpointer2 = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Verify state is recoverable
            conn2 = checkpointer2.conn
            cursor = await conn2.execute(
                "SELECT checkpoint FROM checkpoints WHERE thread_id = ? ORDER BY checkpoint_id DESC LIMIT 1",
                (thread_id,)
            )
            result = await cursor.fetchone()
            assert result is not None
            assert b"Task 2" in result[0]
            assert b"in_progress" in result[0]

    @pytest.mark.asyncio
    async def test_multiple_sessions_isolated(self):
        """Multiple sessions have isolated checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            conn = checkpointer.conn

            # Create checkpoints for session A
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("session-A", "", "cp-A1", None, "checkpoint", b'{"session": "A"}', b"{}")
            )
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("session-A", "", "cp-A2", "cp-A1", "checkpoint", b'{"session": "A", "step": 2}', b"{}")
            )

            # Create checkpoints for session B
            await conn.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                ("session-B", "", "cp-B1", None, "checkpoint", b'{"session": "B"}', b"{}")
            )
            await conn.commit()

            # Clear session A only
            await clear_session_checkpoints(checkpointer, "session-A")

            # Verify session A is gone
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("session-A",)
            )
            row = await cursor.fetchone()
            assert row[0] == 0

            # Verify session B is untouched
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM checkpoints WHERE thread_id = ?",
                ("session-B",)
            )
            row = await cursor.fetchone()
            assert row[0] == 1


class TestListResumableSessions:
    """Test listing resumable sessions functionality."""

    @pytest.mark.asyncio
    async def test_list_sessions_from_sqlite(self):
        """Can list distinct sessions from SQLite database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            conn = checkpointer.conn

            # Create checkpoints for multiple sessions
            for session_id in ["session-1", "session-2", "session-3"]:
                await conn.execute(
                    "INSERT INTO checkpoints "
                    "(thread_id, checkpoint_ns, checkpoint_id, parent_checkpoint_id, "
                    "type, checkpoint, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, "", f"cp-{session_id}", None, "checkpoint", b"{}", b"{}")
                )
            await conn.commit()

            # Query distinct session IDs
            cursor = await conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints"
            )
            rows = await cursor.fetchall()
            sessions = [row[0] for row in rows]

            assert len(sessions) == 3
            assert "session-1" in sessions
            assert "session-2" in sessions
            assert "session-3" in sessions


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_clear_with_closed_connection(self):
        """Handles gracefully when connection has issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"

            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            # Close connection to simulate error condition
            await checkpointer.conn.close()

            # Should handle error gracefully
            result = await clear_session_checkpoints(checkpointer, "test")
            assert result is False  # Returns False on error

    @pytest.mark.asyncio
    async def test_invalid_checkpointer_type(self):
        """Invalid checkpointer type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown checkpointer type"):
            await create_checkpointer("invalid_type")

    @pytest.mark.asyncio
    async def test_clear_empty_thread_id(self):
        """Clearing empty thread_id works without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            checkpointer = await create_checkpointer(
                CheckpointerType.SQLITE, db_path=db_path
            )

            result = await clear_session_checkpoints(checkpointer, "")
            assert result is True  # No error, just no-op
