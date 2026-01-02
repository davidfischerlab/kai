"""Checkpointer factory and management.

Provides factory functions for creating LangGraph checkpointers:
- MemorySaver: In-memory (development, testing)
- AsyncSqliteSaver: Persistent SQLite with async support (production)

Checkpoint Modes:
- TRANSIENT: Checkpoints cleared when session completes (default)
  Good for: Normal runs where restart is only needed for interruptions
- PERSISTENT: Checkpoints kept indefinitely
  Good for: Debugging, analysis, long-term session history

Storage Location:
- Default: ~/.kai_agent/checkpoints.db (AsyncSqliteSaver)
- MemorySaver: In-memory only, lost on process exit

Usage:
    from kai.core.persistence import create_checkpointer, CheckpointerType, CheckpointMode

    # Explicit type with mode (async)
    checkpointer = await create_checkpointer(
        CheckpointerType.SQLITE,
        db_path="~/.kai_agent/checkpoints.db",
        mode=CheckpointMode.TRANSIENT
    )

    # From settings (async)
    from kai.config.settings import settings
    checkpointer = await get_checkpointer_for_settings(settings)
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from kai.utils import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from kai.config.settings import Settings


class CheckpointerType(Enum):
    """Supported checkpointer types."""

    MEMORY = "memory"  # In-memory, lost on restart (dev/testing)
    SQLITE = "sqlite"  # Persistent SQLite database (production)


class CheckpointMode(Enum):
    """Checkpoint retention modes.

    TRANSIENT: Checkpoints cleared when session completes successfully.
        - Good for normal operation where restart is only for interruptions
        - Reduces database size by cleaning up completed sessions
        - Default mode

    PERSISTENT: Checkpoints kept indefinitely.
        - Good for debugging, analysis, session history
        - Allows resuming old sessions or reviewing past state
        - Use checkpoint_cleanup utility for manual cleanup when needed
    """

    TRANSIENT = "transient"  # Clear on completion (default)
    PERSISTENT = "persistent"  # Keep indefinitely


async def create_checkpointer(
    checkpointer_type: CheckpointerType,
    db_path: Optional[Union[str, Path]] = None,
) -> Union[MemorySaver, AsyncSqliteSaver]:
    """Create a checkpointer instance.

    Args:
        checkpointer_type: Type of checkpointer to create
        db_path: Path to SQLite database (required for SQLITE type)

    Returns:
        Configured checkpointer instance

    Raises:
        ValueError: If SQLITE type requested without db_path
    """
    if checkpointer_type == CheckpointerType.MEMORY:
        logger.info("[PERSISTENCE] Using MemorySaver (in-memory checkpoints)")
        return MemorySaver()

    elif checkpointer_type == CheckpointerType.SQLITE:
        if db_path is None:
            raise ValueError("db_path required for AsyncSqliteSaver")

        db_path = Path(db_path)

        # Ensure parent directory exists
        db_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"[PERSISTENCE] Using AsyncSqliteSaver at {db_path}")

        # Create async SQLite connection and saver
        import aiosqlite
        conn = await aiosqlite.connect(str(db_path))

        # Add is_alive method for LangGraph compatibility
        # (aiosqlite 0.22+ removed is_alive, but LangGraph 3.0 expects it)
        if not hasattr(conn, 'is_alive'):
            conn.is_alive = lambda: True  # Connection is alive after connect

        saver = AsyncSqliteSaver(conn)
        await saver.setup()  # Initialize the database schema
        return saver

    else:
        raise ValueError(f"Unknown checkpointer type: {checkpointer_type}")


async def get_checkpointer_for_settings(
    settings: "Settings",
) -> Union[MemorySaver, AsyncSqliteSaver]:
    """Create checkpointer based on settings configuration.

    Behavior:
    - Uses AsyncSqliteSaver by default with db at ~/.kai_agent/checkpoints.db
    - CHECKPOINT_MODE controls retention (TRANSIENT clears on completion)
    - Set CHECKPOINT_ENABLED=False explicitly for MemorySaver (testing only)

    Args:
        settings: Application settings

    Returns:
        Configured checkpointer instance
    """
    # Check if explicitly disabled (for testing)
    if hasattr(settings, 'CHECKPOINT_ENABLED') and not settings.CHECKPOINT_ENABLED:
        return await create_checkpointer(CheckpointerType.MEMORY)

    # Use default path if not specified
    checkpoint_path = settings.checkpoint_db_path_resolved

    return await create_checkpointer(CheckpointerType.SQLITE, db_path=checkpoint_path)


def get_checkpoint_mode(settings: "Settings") -> CheckpointMode:
    """Get checkpoint mode from settings.

    Args:
        settings: Application settings

    Returns:
        CheckpointMode (TRANSIENT or PERSISTENT)
    """
    mode_str = getattr(settings, 'CHECKPOINT_MODE', 'transient').lower()
    try:
        return CheckpointMode(mode_str)
    except ValueError:
        logger.warning(f"Unknown checkpoint mode '{mode_str}', using TRANSIENT")
        return CheckpointMode.TRANSIENT


async def run_retention_cleanup(
    checkpointer: AsyncSqliteSaver,
    retention_days: int = 7,
    dry_run: bool = False,
) -> dict:
    """Run checkpoint retention cleanup (utility for manual cleanup).

    This is a utility function for manually cleaning up old checkpoints.
    Not called automatically - use when you want to reclaim database space.

    Args:
        checkpointer: SqliteSaver instance to clean
        retention_days: Days to keep checkpoints (delete older ones)
        dry_run: If True, report what would be deleted without deleting

    Returns:
        Summary of cleanup operation
    """
    from kai.core.debug.checkpoint_cleanup import cleanup_old_checkpoints

    return await cleanup_old_checkpoints(
        checkpointer=checkpointer,
        retention_days=retention_days,
        dry_run=dry_run,
    )


def is_sqlite_checkpointer(
    checkpointer: Union[MemorySaver, AsyncSqliteSaver],
) -> bool:
    """Check if checkpointer is AsyncSqliteSaver (supports cleanup)."""
    return isinstance(checkpointer, AsyncSqliteSaver)


async def clear_session_checkpoints(
    checkpointer: Union[MemorySaver, AsyncSqliteSaver],
    thread_id: str,
) -> bool:
    """Clear all checkpoints for a specific session.

    Used by TRANSIENT mode to clean up after successful completion.

    Args:
        checkpointer: LangGraph checkpointer
        thread_id: Session/thread ID to clear

    Returns:
        True if cleared successfully, False otherwise
    """
    if not is_sqlite_checkpointer(checkpointer):
        # MemorySaver - nothing to clear (will be garbage collected)
        return True

    try:
        # AsyncSqliteSaver stores checkpoints in a SQLite database
        # We need to delete all checkpoints for this thread
        conn = checkpointer.conn

        # Delete from checkpoints table (async)
        await conn.execute(
            "DELETE FROM checkpoints WHERE thread_id = ?",
            (thread_id,)
        )
        # Delete from writes table (if exists)
        try:
            await conn.execute(
                "DELETE FROM writes WHERE thread_id = ?",
                (thread_id,)
            )
        except Exception:
            pass  # writes table may not exist in older schemas

        await conn.commit()
        logger.info(f"[PERSISTENCE] Cleared checkpoints for session {thread_id}")
        return True

    except Exception as e:
        logger.error(f"Failed to clear checkpoints for {thread_id}: {e}")
        return False


def get_default_checkpoint_path() -> Path:
    """Get default checkpoint database path.

    Returns:
        Path to ~/.kai_agent/checkpoints.db
    """
    from kai.config.paths import AGENT_BASE_DIR
    return AGENT_BASE_DIR / "checkpoints.db"
