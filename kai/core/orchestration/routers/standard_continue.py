"""Standard continue branch routing (Branch 2)."""

from typing import Callable, Optional

from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def route_standard_continue_branch(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None
) -> str:
    """Branch 2: Standard continue - no errors, normal progression.

    Args:
        state: Current graph state
        send_message: Optional callback to send UI messages

    Returns:
        Next node name
    """
    logger.debug("[DET ROUTER] Branch 2: STANDARD_CONTINUE")

    # Set positioning from last_cell_modified_in_auto_mode
    # This is deterministic - we always add after the last modified cell
    has_positioning = safe_get(state, "positioning_info") is not None
    if not has_positioning:
        logger.debug("[DET ROUTER] STANDARD_CONTINUE: setting positioning from last cell")
        return "set_positioning_from_last_cell"

    # Check if reasoning task
    is_reasoning = safe_get(state, "is_reasoning_task")
    generated_code = safe_get(state, "generated_code")
    reasoning_response = safe_get(state, "reasoning_response")

    if is_reasoning:
        # Reasoning critique loop (max 2 iterations)
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
                "[DET ROUTER] STANDARD_CONTINUE: reasoning complete "
                "(max iterations) → mark_reasoning_completed"
            )
            return "mark_reasoning_completed"

        # Generate reasoning if not exists OR if critique rejected (needs regeneration)
        if not reasoning_response or reasoning_approval == "MODIFY":
            logger.debug("[DET ROUTER] STANDARD_CONTINUE: generating reasoning response")
            return "reasoning_response_with_guidance"

        # Run critique if reasoning exists but not yet critiqued (approval is None)
        if reasoning_approval is None:
            logger.debug(
                f"[DET ROUTER] STANDARD_CONTINUE: reasoning critique "
                f"(iter {critique_iteration + 1})"
            )
            return "reasoning_critique"

        # Approved - mark complete
        if send_message:
            send_message(
                f"Reasoning approved after {critique_iteration} critique iterations"
            )
        logger.debug(
            "[DET ROUTER] STANDARD_CONTINUE: reasoning complete → mark_reasoning_completed"
        )
        return "mark_reasoning_completed"
    else:
        if not generated_code:
            logger.debug("[DET ROUTER] STANDARD_CONTINUE: generating code")
            return "code_generation_with_guidance"

        # Code generated, exit to UI for execution
        logger.debug("[DET ROUTER] STANDARD_CONTINUE: code generated, exiting to UI")
        return "complete"
