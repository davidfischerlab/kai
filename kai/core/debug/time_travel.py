"""Time-travel debugging utilities for LangGraph checkpoints.

Provides:
- List checkpoints for a session
- Resume from any checkpoint
- Find error checkpoints
- Compare execution branches

These utilities leverage LangGraph's checkpoint system for debugging.
"""

from typing import Dict, Any, List, Optional, AsyncIterator, TYPE_CHECKING
from datetime import datetime

from kai.utils import setup_logger

logger = setup_logger(__name__)

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


class TimeTravelDebugger:
    """Time-travel debugging for LangGraph sessions.

    Enables:
    - Viewing checkpoint history
    - Resuming from specific steps
    - Finding error states
    - Comparing branches
    """

    def __init__(self, checkpointer: "BaseCheckpointSaver"):
        """Initialize with a checkpointer.

        Args:
            checkpointer: LangGraph checkpointer (SqliteSaver recommended)
        """
        self.checkpointer = checkpointer

    async def list_checkpoints(
        self,
        thread_id: str,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """List checkpoints for a session.

        Args:
            thread_id: Session to inspect
            limit: Maximum checkpoints to return

        Returns:
            List of checkpoint summaries (newest first)
        """
        config = {"configurable": {"thread_id": thread_id}}
        checkpoints = []

        try:
            async for state in self.checkpointer.aget_state_history(config):
                checkpoint_info = {
                    "checkpoint_id": state.config.get(
                        "configurable", {}
                    ).get("checkpoint_id"),
                    "step": len(checkpoints),
                    "timestamp": state.metadata.get("ts"),
                    "next_nodes": list(state.next) if state.next else [],
                    "state_keys": list(state.values.keys()) if state.values else [],
                }
                checkpoints.append(checkpoint_info)

                if len(checkpoints) >= limit:
                    break

        except Exception as e:
            logger.error(f"Failed to list checkpoints: {e}")

        return checkpoints

    async def get_state_at_step(
        self,
        thread_id: str,
        step: int,
    ) -> Optional[Dict[str, Any]]:
        """Get full state at a specific step.

        Args:
            thread_id: Session to inspect
            step: Step number (0 = first)

        Returns:
            State dict at that step, or None if not found
        """
        config = {"configurable": {"thread_id": thread_id}}
        current_step = 0

        try:
            async for state in self.checkpointer.aget_state_history(config):
                if current_step == step:
                    return dict(state.values) if state.values else {}
                current_step += 1

        except Exception as e:
            logger.error(f"Failed to get state at step {step}: {e}")

        return None

    async def find_error_checkpoint(
        self,
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Find checkpoint where error occurred.

        Args:
            thread_id: Session to inspect

        Returns:
            Error checkpoint info or None if no errors found
        """
        config = {"configurable": {"thread_id": thread_id}}
        step = 0

        try:
            async for state in self.checkpointer.aget_state_history(config):
                state_dict = dict(state.values) if state.values else {}

                # Check for error indicators
                if state_dict.get("error_context"):
                    return {
                        "step": step,
                        "checkpoint_id": state.config.get(
                            "configurable", {}
                        ).get("checkpoint_id"),
                        "error_context": state_dict.get("error_context"),
                        "timestamp": state.metadata.get("ts"),
                    }

                step += 1

        except Exception as e:
            logger.error(f"Failed to find error checkpoint: {e}")

        return None

    async def get_resume_config(
        self,
        thread_id: str,
        checkpoint_id: str,
    ) -> Dict[str, Any]:
        """Get config for resuming from a checkpoint.

        Args:
            thread_id: Session to resume
            checkpoint_id: Specific checkpoint to resume from

        Returns:
            Config dict for graph.ainvoke()
        """
        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }

    async def compare_states(
        self,
        thread_id: str,
        step1: int,
        step2: int,
    ) -> Dict[str, Any]:
        """Compare states at two different steps.

        Args:
            thread_id: Session to inspect
            step1: First step to compare
            step2: Second step to compare

        Returns:
            Dict with differences between states
        """
        state1 = await self.get_state_at_step(thread_id, step1)
        state2 = await self.get_state_at_step(thread_id, step2)

        if state1 is None or state2 is None:
            return {"error": "One or both steps not found"}

        # Find differences
        all_keys = set(state1.keys()) | set(state2.keys())
        differences = {}

        for key in all_keys:
            val1 = state1.get(key)
            val2 = state2.get(key)

            if val1 != val2:
                differences[key] = {
                    f"step_{step1}": _truncate_value(val1),
                    f"step_{step2}": _truncate_value(val2),
                }

        return {
            "step1": step1,
            "step2": step2,
            "differences": differences,
            "unchanged_keys": [k for k in all_keys if k not in differences],
        }


def _truncate_value(value: Any, max_length: int = 200) -> Any:
    """Truncate value for display."""
    if isinstance(value, str) and len(value) > max_length:
        return value[:max_length] + "..."
    if isinstance(value, list) and len(value) > 5:
        return value[:5] + ["..."]
    if isinstance(value, dict) and len(str(value)) > max_length:
        return f"<dict with {len(value)} keys>"
    return value


async def resume_from_checkpoint(
    graph: Any,
    thread_id: str,
    checkpoint_id: str,
    modified_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resume execution from a specific checkpoint.

    Args:
        graph: Compiled LangGraph
        thread_id: Session to resume
        checkpoint_id: Checkpoint to resume from
        modified_state: Optional state modifications before resuming

    Returns:
        Final state after execution
    """
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id,
        }
    }

    logger.info(
        f"[TIME_TRAVEL] Resuming from checkpoint {checkpoint_id}"
    )

    # If state modifications provided, update before resuming
    if modified_state:
        result = await graph.ainvoke(modified_state, config)
    else:
        # Resume with None to continue from checkpoint
        result = await graph.ainvoke(None, config)

    return result
