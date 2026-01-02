"""Checkpoint retention and cleanup utilities.

Implements retention policies for LangGraph checkpoints to prevent
unbounded database growth.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, TYPE_CHECKING

from kai.utils import setup_logger

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver

logger = setup_logger(__name__)


async def cleanup_old_checkpoints(
    checkpointer: 'BaseCheckpointSaver',
    retention_days: int = 7,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Remove checkpoints older than retention period.

    Args:
        checkpointer: LangGraph checkpointer (AsyncSqliteSaver recommended)
        retention_days: Number of days to retain checkpoints
        dry_run: If True, only report what would be deleted

    Returns:
        Dict with cleanup statistics
    """
    cutoff = datetime.now() - timedelta(days=retention_days)
    stats = {
        "cutoff_date": cutoff.isoformat(),
        "retention_days": retention_days,
        "dry_run": dry_run,
        "threads_scanned": 0,
        "threads_deleted": 0,
        "threads_kept": 0,
        "deleted_thread_ids": [],
        "errors": [],
    }

    # Check if this is an AsyncSqliteSaver (supports cleanup)
    # MemorySaver doesn't persist data, so cleanup is meaningless
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    if not isinstance(checkpointer, AsyncSqliteSaver):
        stats["errors"].append(
            "Checkpointer does not support listing threads. "
            "Use AsyncSqliteSaver for cleanup functionality."
        )
        logger.warning(stats["errors"][-1])
        return stats

    try:
        # Get all threads from SqliteSaver
        threads = list(checkpointer.list(None))
        stats["threads_scanned"] = len(threads)

        for thread_info in threads:
            thread_id = thread_info.get("thread_id", "")
            if not thread_id:
                continue

            try:
                config = {"configurable": {"thread_id": thread_id}}

                # Get most recent checkpoint timestamp
                latest_ts = None
                async for state in checkpointer.aget_state_history(config):
                    ts_str = state.metadata.get("ts", "")
                    if ts_str:
                        try:
                            latest_ts = datetime.fromisoformat(ts_str)
                            break  # Only need the most recent
                        except (ValueError, TypeError):
                            pass

                # Check if thread should be deleted
                if latest_ts and latest_ts < cutoff:
                    if not dry_run:
                        # Delete the thread
                        await _delete_thread(checkpointer, thread_id)

                    stats["threads_deleted"] += 1
                    stats["deleted_thread_ids"].append(thread_id)
                    logger.info(
                        f"{'Would delete' if dry_run else 'Deleted'} "
                        f"thread {thread_id} (last activity: {latest_ts})"
                    )
                else:
                    stats["threads_kept"] += 1

            except Exception as e:
                error_msg = f"Error processing thread {thread_id}: {e}"
                stats["errors"].append(error_msg)
                logger.error(error_msg)

    except (AttributeError, TypeError):
        # Unexpected error accessing checkpointer
        stats["errors"].append(
            "Unexpected error accessing checkpointer. "
            "Verify SqliteSaver is properly configured."
        )
        logger.warning(stats["errors"][-1])

    return stats


async def _delete_thread(
    checkpointer: 'BaseCheckpointSaver',
    thread_id: str
) -> bool:
    """Delete a thread from the checkpointer.

    Args:
        checkpointer: LangGraph checkpointer
        thread_id: Thread to delete

    Returns:
        True if deleted successfully
    """
    try:
        # SqliteSaver has put method that can clear by passing None
        # But standard way is to use the checkpointer's deletion mechanism
        if hasattr(checkpointer, 'delete'):
            config = {"configurable": {"thread_id": thread_id}}
            await checkpointer.delete(config)
        else:
            # Fallback: clear the state
            config = {"configurable": {"thread_id": thread_id}}
            await checkpointer.aput(config, {}, {})
        return True
    except Exception as e:
        logger.error(f"Failed to delete thread {thread_id}: {e}")
        return False


async def get_checkpoint_stats(
    checkpointer: 'BaseCheckpointSaver',
) -> Dict[str, Any]:
    """Get statistics about stored checkpoints.

    Args:
        checkpointer: LangGraph checkpointer

    Returns:
        Dict with checkpoint statistics
    """
    stats = {
        "total_threads": 0,
        "oldest_checkpoint": None,
        "newest_checkpoint": None,
        "threads_by_age": {
            "last_24h": 0,
            "last_7d": 0,
            "last_30d": 0,
            "older": 0,
        },
        "checkpoint_type": type(checkpointer).__name__,
    }

    try:
        # Try to list all threads (SqliteSaver pattern)
        try:
            threads = list(checkpointer.list(None))
        except TypeError:
            raise AttributeError("list() requires config - not SqliteSaver")

        stats["total_threads"] = len(threads)

        now = datetime.now()
        oldest = None
        newest = None

        for thread_info in threads:
            thread_id = thread_info.get("thread_id", "")
            if not thread_id:
                continue

            try:
                config = {"configurable": {"thread_id": thread_id}}

                # Get most recent checkpoint timestamp
                async for state in checkpointer.aget_state_history(config):
                    ts_str = state.metadata.get("ts", "")
                    if ts_str:
                        try:
                            ts = datetime.fromisoformat(ts_str)

                            # Track oldest/newest
                            if oldest is None or ts < oldest:
                                oldest = ts
                            if newest is None or ts > newest:
                                newest = ts

                            # Categorize by age
                            age = now - ts
                            if age < timedelta(days=1):
                                stats["threads_by_age"]["last_24h"] += 1
                            elif age < timedelta(days=7):
                                stats["threads_by_age"]["last_7d"] += 1
                            elif age < timedelta(days=30):
                                stats["threads_by_age"]["last_30d"] += 1
                            else:
                                stats["threads_by_age"]["older"] += 1

                            break  # Only need most recent
                        except (ValueError, TypeError):
                            pass

            except Exception as e:
                logger.debug(f"Error getting stats for {thread_id}: {e}")

        stats["oldest_checkpoint"] = oldest.isoformat() if oldest else None
        stats["newest_checkpoint"] = newest.isoformat() if newest else None

    except (AttributeError, TypeError):
        stats["error"] = (
            "Checkpointer does not support listing. "
            "Use SqliteSaver for statistics."
        )

    return stats
