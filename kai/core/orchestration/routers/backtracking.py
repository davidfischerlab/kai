"""Backtracking branch routing (Branch 4)."""

from typing import Callable, Optional

from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def route_backtracking_branch(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None
) -> str:
    """Branch 4: Backtracking - delete cells and regenerate.

    Args:
        state: Current graph state
        send_message: Optional callback to send UI messages (unused but kept for consistency)

    Returns:
        Next node name
    """
    logger.debug("[DET ROUTER] Branch 4: BACKTRACKING")

    # Backtracking sequence:
    # 1. backtrack_recovery (determine restart need, sets restart_required in state)
    # 2. cell_selection_deletion
    # 3. cell_deletion
    # 4. cell_positioning
    # 5. code_generation_with_guidance (passes restart_required to VSCode if set)
    # When restart_required=True, VSCode restarts kernel and runs all cells up to
    # the new cell before executing it.

    backtrack_recovery_done = safe_get(state, "backtrack_recovery_done")
    cells_to_delete = safe_get(state, "cells_to_delete")
    cells_deleted = safe_get(state, "cells_deleted")
    has_positioning = safe_get(state, "positioning_info") is not None
    generated_code = safe_get(state, "generated_code")

    if not backtrack_recovery_done:
        logger.debug("[DET ROUTER] BACKTRACK: determining recovery strategy")
        return "backtrack_recovery"

    if not cells_to_delete:
        logger.debug("[DET ROUTER] BACKTRACK: selecting cells to delete")
        return "cell_selection_deletion"

    if not cells_deleted:
        logger.debug("[DET ROUTER] BACKTRACK: deleting cells")
        return "cell_deletion"

    if not has_positioning:
        logger.debug("[DET ROUTER] BACKTRACK: positioning after deletion")
        return "cell_positioning"

    if not generated_code:
        logger.debug("[DET ROUTER] BACKTRACK: generating new code")
        return "code_generation_with_guidance"

    logger.debug("[DET ROUTER] BACKTRACK: complete, exiting")
    return "complete"
