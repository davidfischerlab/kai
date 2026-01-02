"""Planning phase node functions."""

import copy
from typing import Any, Callable, Dict, Optional

from kai.utils import setup_logger

logger = setup_logger(__name__)


async def increment_task_planning_iteration(
    state: dict,
    max_task_planning_iterations: int = 10,
    send_message: Optional[Callable[[str], None]] = None,
) -> dict:
    """Set task_planning_iteration counter for current iteration.

    The counter is initialized to -1, so first increment gives 0.
    Sets planning_phase to "ready_to_generate" to signal router.

    Args:
        state: Current graph state
        max_task_planning_iterations: Maximum planning iterations allowed
        send_message: Optional callback to send UI messages
    """
    current = state.get("task_planning_iteration", -1)
    new_value = current + 1

    if new_value < max_task_planning_iterations:
        logger.info(
            f"[PLANNING ITERATION {new_value + 1}/{max_task_planning_iterations}]"
        )
        if send_message:
            send_message(
                f"[KAI] Planning iteration {new_value + 1}/{max_task_planning_iterations}"
            )

    return {
        "task_planning_iteration": new_value,
        "planning_phase": "ready_to_generate",
        "task_list_approval": None,
    }


async def backup_task_list(state: dict) -> dict:
    """Backup task list before autonomous_update_tasks (for reversion if critique fails).

    The backup is also used by prompt_manager to format the "original" task list
    for critique comparison (formatted on-demand from task_list_backup).
    """
    task_list = state.get("task_list", {})
    logger.debug(
        f"[BACKUP] Saving task list backup: {len(task_list.get('tasks', []))} tasks"
    )
    return {"task_list_backup": copy.deepcopy(task_list)}


async def revert_task_list(state: dict) -> dict:
    """Revert task list to backup (critique failed after max iterations)."""
    backup = state.get("task_list_backup")
    if backup:
        logger.warning(
            f"[REVERT] Reverting task list to backup: {len(backup.get('tasks', []))} tasks"
        )
        return {
            "task_list": backup,
            "task_list_backup": None,
            "tasks_updated": True,
            "update_approved": True,
            "autonomous_update_critique_iteration": 0,  # Reset task update counter
            "task_list_update_rule": None,
            "autonomous_update_critique": None,
        }
    else:
        logger.error("[REVERT] No backup found! Cannot revert task list")
        return {
            "tasks_updated": True,
            "update_approved": True,
            "autonomous_update_critique_iteration": 0,  # Reset task update counter
            "task_list_update_rule": None,
            "autonomous_update_critique": None,
        }
