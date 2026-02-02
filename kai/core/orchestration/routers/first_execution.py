"""First execution phase routing."""

from typing import Callable, Optional

from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def route_first_execution(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None
) -> str:
    """
    First execution phase routing.
    Sequence: mark_next_task_active (if needed) → cell_positioning → code_generation_with_guidance → exit

    Args:
        state: Current graph state
        send_message: Optional callback to send UI messages

    Returns:
        Next node name
    """
    logger.debug("[DET ROUTER] Phase: FIRST_EXECUTION")

    # Check if we have an active task
    task_list = safe_get(state, "task_list", {})
    tasks = safe_get(task_list, "tasks", [])
    has_active_task = any(safe_get(t, "status") == "active" for t in tasks)
    active_task_objective = safe_get(state, "active_task_objective")

    has_positioning = safe_get(state, "positioning_info") is not None
    first_exec_done = safe_get(state, "auto_mode_first_execution_done", False)

    logger.debug(
        f"[DET ROUTER] FIRST_EXEC state: has_active={has_active_task}, "
        f"active_objective={bool(active_task_objective)}, "
        f"has_positioning={has_positioning}, first_exec_done={first_exec_done}"
    )

    # Defensive: if first execution already done, shouldn't be here - exit
    if first_exec_done:
        logger.debug(
            "[DET ROUTER] FIRST_EXEC: first exec already done (unexpected state) → complete"
        )
        return "complete"

    # First, ensure we have an active task
    if not has_active_task:
        logger.debug("[DET ROUTER] FIRST_EXEC: no active task → mark_next_task_active")
        return "mark_next_task_active"

    if not has_positioning:
        logger.debug("[DET ROUTER] FIRST_EXEC: need positioning → cell_positioning")
        return "cell_positioning"

    # Check if this is a reasoning task
    # Note: The flag may have been set by mark_next_task_active during planning,
    # but could be lost when execution starts as a new request. Recalculate from
    # the active task objective to be safe.
    is_reasoning = safe_get(state, "is_reasoning_task", False)
    if not is_reasoning and active_task_objective:
        # Recalculate from task description (like standard_retry does)
        is_reasoning = "[reasoning]" in active_task_objective.lower()
        if is_reasoning:
            logger.debug("[DET ROUTER] FIRST_EXEC: detected reasoning task from objective")

    generated_code = safe_get(state, "generated_code")
    reasoning_response = safe_get(state, "reasoning_response")

    if is_reasoning:
        # Reasoning evaluation loop
        # Flow: generate → evaluator → (if REJECTED) regenerate → evaluator → ...
        reasoning_grade = safe_get(state, "reasoning_grade")
        evaluation_iteration = safe_get(state, "reasoning_evaluation_iteration", 0)

        # Check max iterations FIRST (before regenerating)
        if evaluation_iteration >= 2:
            # Max iterations reached - stop without regenerating
            if reasoning_grade == "APPROVED":
                logger.info(
                    f"[REASONING] Approved after {evaluation_iteration} "
                    f"evaluation iterations"
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
                "[DET ROUTER] FIRST_EXEC: reasoning complete "
                "(max iterations) → mark_reasoning_completed"
            )
            return "mark_reasoning_completed"

        # Generate reasoning if not exists OR if evaluator rejected
        if not reasoning_response or reasoning_grade == "REJECTED":
            logger.debug(
                "[DET ROUTER] FIRST_EXEC: reasoning task, generating reasoning"
            )
            return "reasoning_response_with_guidance"

        # Run evaluator if reasoning exists but not yet evaluated (grade is None)
        if reasoning_grade is None:
            logger.debug(
                f"[DET ROUTER] FIRST_EXEC: reasoning evaluator "
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
            "[DET ROUTER] FIRST_EXEC: reasoning complete → mark_reasoning_completed"
        )
        return "mark_reasoning_completed"
    else:
        # Code task flow
        if not generated_code:
            logger.debug(
                "[DET ROUTER] FIRST_EXEC: need code generation → code_generation_with_guidance"
            )
            return "code_generation_with_guidance"
        else:
            # Code has been generated - exit to UI for execution
            logger.debug(
                "[DET ROUTER] FIRST_EXEC: code generated, marking done and exiting to UI"
            )
            return "mark_first_execution_done"
