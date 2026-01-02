"""Standard retry branch routing (Branch 3)."""

from typing import Callable, Optional

from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def route_standard_retry_branch(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None
) -> str:
    """Branch 3: Error or LLM detected issue - fix and retry.

    Args:
        state: Current graph state
        send_message: Optional callback to send UI messages

    Returns:
        Next node name
    """
    logger.debug("[DET ROUTER] Branch 3: STANDARD RETRY")

    # Step 1: Determine error recovery strategy
    error_recovery_strategy = safe_get(state, "error_recovery_strategy")

    if not error_recovery_strategy:
        logger.debug("[DET ROUTER] STANDARD RETRY: determining recovery strategy")
        return "error_recovery"

    # Log the strategy being used
    # Both REPLACE_AND_RETRY and REPLACE_AND_RESTART follow the same router path.
    # The strategy is passed to VSCode via code_update tool, and VSCode handles
    # the kernel restart when REPLACE_AND_RESTART is specified.
    logger.info(f"[DET ROUTER] STANDARD RETRY: using {error_recovery_strategy} strategy")

    # Step 2: Set positioning to failed cell
    # Use set_positioning_from_last_cell which uses last_cell_modified_in_auto_mode
    has_positioning = safe_get(state, "positioning_info") is not None
    if not has_positioning:
        logger.debug("[DET ROUTER] STANDARD RETRY: setting positioning from last cell")
        return "set_positioning_from_last_cell"

    # Step 3: Re-check if reasoning task (flag may be stale!)
    active_task_objective = safe_get(state, "active_task_objective", "")
    is_reasoning_task = "[reasoning]" in active_task_objective.lower()

    # Step 4: Execute recovery based on strategy and task type
    code_updated = (
        safe_get(state, "generated_code") or safe_get(state, "reasoning_response")
    )

    if is_reasoning_task:
        # Reasoning critique loop
        # Flow: generate → critique → (if MODIFY) regenerate → critique → ...
        reasoning_approval = safe_get(state, "reasoning_approval")
        critique_iteration = safe_get(state, "reasoning_critique_iteration", 0)

        # Check max iterations FIRST (before regenerating)
        if critique_iteration >= 2:
            if reasoning_approval == "APPROVED":
                logger.info(
                    f"[REASONING] ✅ Approved after {critique_iteration} critique iterations"
                )
                if send_message:
                    send_message(
                        f"✅ Reasoning approved after {critique_iteration} critique iterations"
                    )
            else:
                logger.info(
                    f"[REASONING] ⚠️ Auto-accepting after max iterations ({critique_iteration}) - "
                    f"critique did not approve"
                )
                if send_message:
                    send_message(
                        f"⚠️ Auto-accepting reasoning after max critique iterations ({critique_iteration}) reached"
                    )
            logger.debug(
                "[DET ROUTER] STANDARD RETRY: reasoning complete "
                "(max iterations) → mark_reasoning_completed"
            )
            return "mark_reasoning_completed"

        # Generate reasoning if not exists OR if critique rejected (needs regeneration)
        reasoning_response = safe_get(state, "reasoning_response")
        if not reasoning_response or reasoning_approval == "MODIFY":
            logger.debug("[DET ROUTER] STANDARD RETRY: regenerating reasoning")
            return "reasoning_response_with_guidance"

        # Run critique if reasoning exists but not yet critiqued (approval is None)
        if reasoning_approval is None:
            logger.debug(
                f"[DET ROUTER] STANDARD RETRY: reasoning critique "
                f"(iter {critique_iteration + 1})"
            )
            return "reasoning_critique"

        # Approved - mark complete
        if send_message:
            send_message(
                f"[KAI] Reasoning approved after {critique_iteration} critique iterations"
            )
        logger.debug(
            "[DET ROUTER] STANDARD RETRY: reasoning complete → mark_reasoning_completed"
        )
        return "mark_reasoning_completed"
    else:
        # Code task - code_update handles both REPLACE_AND_RETRY and REPLACE_AND_RESTART
        if not code_updated:
            logger.debug("[DET ROUTER] STANDARD RETRY: updating code")
            return "code_update"
        else:
            logger.debug("[DET ROUTER] STANDARD RETRY: code updated, exiting")
            return "complete"
