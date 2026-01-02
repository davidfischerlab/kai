"""Main deterministic routing for execution subgraph.

This is the entry point router that delegates to phase-specific routers.
"""

from typing import Callable, Optional

from kai.utils import setup_logger, safe_get
from .first_execution import route_first_execution
from .standard_execution import route_standard_execution

logger = setup_logger(__name__)


def route_deterministic(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None
) -> str:
    """
    Main deterministic router for execution subgraph.

    Determines execution phase and routes accordingly:
    1. First iteration: Activate first task, show user (or continue)
    2. First execution: After user approval, generate and execute first code
    3. Standard execution: Execute remaining tasks with error handling

    Args:
        state: Current graph state
        send_message: Optional callback to send UI messages

    Returns:
        Next node name
    """
    task_list = safe_get(state, "task_list", {})
    tasks = safe_get(task_list, "tasks", [])
    autonomous_mode_continue = safe_get(state, "autonomous_mode_continue")

    auto_mode_first_execution_done = safe_get(
        state, "auto_mode_first_execution_done", False
    )
    logger.debug(
        f"[DET ROUTER] tasks={len(tasks)}, continue={autonomous_mode_continue}, "
        f"first_exec_done={auto_mode_first_execution_done}"
    )

    # ===== Sanity check: Should have tasks from planning graph =====
    if not tasks:
        logger.error(
            "[DET ROUTER] No tasks found! Planning graph should have created tasks."
        )
        return "complete"

    # ===== PHASE 2: FIRST ITERATION (after planning) =====
    if not autonomous_mode_continue and not auto_mode_first_execution_done:
        all_pending = all(safe_get(t, "status") == "pending" for t in tasks)
        has_active = any(safe_get(t, "status") == "active" for t in tasks)

        logger.debug(
            f"[DET ROUTER] FIRST_ITER check: all_pending={all_pending}, "
            f"has_active={has_active}, auto_continue={autonomous_mode_continue}, "
            f"first_exec_done={auto_mode_first_execution_done}"
        )

        if all_pending:
            logger.debug(
                "[DET ROUTER] FIRST_ITER: activating first task → mark_next_task_active"
            )
            return "mark_next_task_active"
        elif has_active:
            confirm_plan = safe_get(state, "confirm_plan", True)
            logger.debug(
                f"[DET ROUTER] FIRST_ITER: task active, confirm_plan={confirm_plan}"
            )
            if confirm_plan:
                logger.debug(
                    "[DET ROUTER] FIRST_ITER: exiting to show user (VSCode) → complete"
                )
                return "complete"
            else:
                logger.debug(
                    "[DET ROUTER] FIRST_ITER: continuing to first execution (Jupyter)"
                )
        else:
            logger.debug("[DET ROUTER] FIRST_ITER: no tasks active or pending → complete")
            return "complete"

    # ===== Check completion first =====
    all_complete = all(safe_get(t, "status") == "completed" for t in tasks)
    if all_complete:
        logger.debug("[DET ROUTER] All tasks complete!")
        return "complete"

    # ===== PHASE 3: FIRST EXECUTION =====
    if not auto_mode_first_execution_done:
        return route_first_execution(state, send_message)

    # ===== PHASE 4: STANDARD EXECUTION =====
    return route_standard_execution(state, send_message)
