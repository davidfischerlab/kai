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

    # Note: Learning explanation is handled AFTER code execution via learning graph
    # in langgraph_orchestrator.py, not during the execution subgraph.

    # Set positioning from last_cell_modified_in_auto_mode
    # This is deterministic - we always add after the last modified cell
    has_positioning = safe_get(state, "positioning_info") is not None
    if not has_positioning:
        logger.debug("[DET ROUTER] STANDARD_CONTINUE: setting positioning from last cell")
        return "set_positioning_from_last_cell"

    # Check if reasoning task
    # Note: The flag may have been set by mark_next_task_active but could be lost
    # when execution starts as a new request. Recalculate from active task objective.
    is_reasoning = safe_get(state, "is_reasoning_task")
    active_task_objective = safe_get(state, "active_task_objective", "")
    if not is_reasoning and active_task_objective:
        is_reasoning = "[reasoning]" in active_task_objective.lower()
        if is_reasoning:
            logger.debug("[DET ROUTER] STANDARD_CONTINUE: detected reasoning task from objective")

    generated_code = safe_get(state, "generated_code")
    reasoning_response = safe_get(state, "reasoning_response")

    if is_reasoning:
        # Reasoning evaluation loop (max 2 iterations)
        # Flow: generate → evaluator → (if REJECTED) regenerate → evaluator → ...
        reasoning_grade = safe_get(state, "reasoning_grade")
        evaluation_iteration = safe_get(state, "reasoning_evaluation_iteration", 0)

        # Check max iterations FIRST (before regenerating)
        if evaluation_iteration >= 2:
            if reasoning_grade == "APPROVED":
                logger.info(
                    f"[REASONING] Approved after {evaluation_iteration} evaluation "
                    f"iterations"
                )
                if send_message:
                    send_message(
                        f"Reasoning approved after {evaluation_iteration} "
                        f"evaluation iterations"
                    )
            else:
                logger.info(
                    f"[REASONING] Auto-accepting after max iterations "
                    f"({evaluation_iteration}) - evaluator did not approve"
                )
                if send_message:
                    send_message(
                        f"Auto-accepting reasoning after max evaluation iterations "
                        f"({evaluation_iteration}) reached"
                    )
            logger.debug(
                "[DET ROUTER] STANDARD_CONTINUE: reasoning complete "
                "(max iterations) → mark_reasoning_completed"
            )
            return "mark_reasoning_completed"

        # Generate reasoning if not exists OR if evaluator rejected (needs regen)
        if not reasoning_response or reasoning_grade == "REJECTED":
            logger.debug(
                "[DET ROUTER] STANDARD_CONTINUE: generating reasoning response"
            )
            return "reasoning_response_with_guidance"

        # Run evaluator if reasoning exists but not yet evaluated (grade is None)
        if reasoning_grade is None:
            logger.debug(
                f"[DET ROUTER] STANDARD_CONTINUE: reasoning evaluator "
                f"(iter {evaluation_iteration + 1})"
            )
            return "reasoning_evaluator"

        # Approved - mark complete
        if send_message:
            send_message(
                f"Reasoning approved after {evaluation_iteration} evaluation "
                f"iterations"
            )
        logger.debug(
            "[DET ROUTER] STANDARD_CONTINUE: reasoning complete → "
            "mark_reasoning_completed"
        )
        return "mark_reasoning_completed"
    else:
        if not generated_code:
            logger.debug("[DET ROUTER] STANDARD_CONTINUE: generating code")
            return "code_generation_with_guidance"

        # Code generated, exit to UI for execution
        logger.debug("[DET ROUTER] STANDARD_CONTINUE: code generated, exiting to UI")
        return "complete"
