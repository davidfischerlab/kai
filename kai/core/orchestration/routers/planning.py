"""Planning phase routing function."""

from typing import Callable, Optional

from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def route_planning_phase(
    state: dict,
    max_task_planning_iterations: int = 10,
    send_message: Optional[Callable[[str], None]] = None,
    log_task_list_summary: Optional[Callable[[dict], None]] = None,
) -> str:
    """
    Route planning phase based on current state.

    PHASE 1: Initial workflow retrieval (max 2 iterations)
    - Entry → search_workflows (if RAG enabled)
    - search_workflows → router checks queries → search_workflows OR task_list_generation

    PHASE 2: Task generation + refinement loop (max iterations)
    For each iteration:
    1. task_list_generation (generates tasks ONCE)
    2. Router checks retrieval_queries:
       - If has queries: workflow_refinement → back to task_list_generation
       - If no queries: task_list_critique (if enabled)
    3. Router checks critique approval:
       - If APPROVED: complete
       - If REJECTED: back to task_list_generation
    4. Max iterations total

    Args:
        state: Current graph state
        max_task_planning_iterations: Maximum planning iterations
        send_message: Optional callback to send UI messages
        log_task_list_summary: Optional callback to log task list

    Returns:
        Next node name
    """
    planning_phase = safe_get(state, "planning_phase")
    retrieval_queries = safe_get(state, "retrieval_queries", [])
    workflow_iteration = safe_get(state, "workflow_retrieval_iteration", 0)
    task_planning_iteration = safe_get(state, "task_planning_iteration", 0)
    rag_enabled = safe_get(state, "rag_enabled", False)
    use_critique = safe_get(state, "use_critique", False)
    task_list_approval = safe_get(state, "task_list_approval")

    if send_message:
        # Show pending queries (not yet searched) instead of total accumulated
        searched = set(safe_get(state, "searched_retrieval_queries", []))
        pending_queries = [q for q in retrieval_queries if q not in searched] if retrieval_queries else []
        pending_count = len(pending_queries)
        send_message(f"[KAI] PLANNING ROUTER: phase={planning_phase}, rag={rag_enabled}, pending_queries={pending_count}")

    logger.debug(
        f"[PLANNING ROUTER] phase={planning_phase}, rag={rag_enabled}, "
        f"use_critique={use_critique}, workflow_iter={workflow_iteration}, "
        f"task_iter={task_planning_iteration}, "
        f"queries={len(retrieval_queries) if retrieval_queries else 0}, "
        f"approval={task_list_approval}"
    )

    # ===== PHASE 1: Initial workflow retrieval (max 2 iterations) =====
    if planning_phase is None:
        if rag_enabled and retrieval_queries:
            if send_message:
                send_message("[KAI] → search_workflows (initial)")
            return "search_workflows"
        else:
            if send_message:
                send_message("[KAI] → increment_task_planning_iteration (no RAG or no queries)")
            return "increment_task_planning_iteration"

    if planning_phase == "workflow_retrieval":
        # Check termination conditions
        if workflow_iteration >= 2:
            return "increment_task_planning_iteration"

        if not retrieval_queries or len(retrieval_queries) == 0:
            return "increment_task_planning_iteration"

        # Continue retrieval
        return "search_workflows"

    # ===== PHASE 2: Task generation + workflow refinement loop =====
    if planning_phase == "task_planning":
        # Check max iterations FIRST
        if task_planning_iteration >= max_task_planning_iterations:
            logger.info(
                f"[PLANNING ROUTER] Max task planning iterations "
                f"({max_task_planning_iterations}) reached → "
                f"{'filter_and_complete' if rag_enabled else 'complete'}"
            )
            if log_task_list_summary:
                log_task_list_summary(state)
            return "filter_and_complete" if rag_enabled else "complete"

        # ALWAYS run workflow refinement when rag_enabled
        # This runs even without new queries to allow selection changes
        if rag_enabled:
            return "workflow_refinement"

        # No RAG - run critique if enabled
        if use_critique:
            # Check if we just came from critique
            if task_list_approval is not None:
                # We have a critique result
                if task_list_approval == "APPROVED":
                    logger.info(
                        f"[PLANNING ROUTER] Task list approved after "
                        f"{task_planning_iteration} iterations → "
                        f"{'filter_and_complete' if rag_enabled else 'complete'}"
                    )
                    if log_task_list_summary:
                        log_task_list_summary(state)
                    return "filter_and_complete" if rag_enabled else "complete"
                else:
                    # Rejected - generate again (increment iteration counter first)
                    logger.info(
                        f"[PLANNING ROUTER] Task list rejected, iteration "
                        f"{task_planning_iteration + 1}/{max_task_planning_iterations} "
                        f"→ increment_task_planning_iteration"
                    )
                    return "increment_task_planning_iteration"
            else:
                # No critique result yet - run critique
                logger.info(
                    f"[PLANNING ROUTER] Task planning iteration "
                    f"{task_planning_iteration + 1}/{max_task_planning_iterations}: "
                    f"running critique → task_list_critique"
                )
                return "task_list_critique"

        # No critique enabled and no retrieval queries - done
        logger.info(
            f"[PLANNING ROUTER] Task planning complete (no critique, no queries) → "
            f"{'filter_and_complete' if rag_enabled else 'complete'}"
        )
        if log_task_list_summary:
            log_task_list_summary(state)
        return "filter_and_complete" if rag_enabled else "complete"

    # ===== PHASE 3: After task list critique =====
    if planning_phase == "task_list_critique":
        if task_list_approval is not None:
            # We have a valid critique result
            if task_list_approval == "APPROVED":
                if send_message:
                    send_message(
                        f"[KAI] Task list approved after "
                        f"{task_planning_iteration + 1} iterations"
                    )
                logger.info(
                    f"[PLANNING ROUTER] Task list approved after "
                    f"{task_planning_iteration + 1} iterations → "
                    f"{'filter_and_complete' if rag_enabled else 'complete'}"
                )
                if log_task_list_summary:
                    log_task_list_summary(state)
                return "filter_and_complete" if rag_enabled else "complete"
            else:
                # Rejected - generate again (increment iteration counter first)
                return "increment_task_planning_iteration"
        else:
            # Critique tool returned invalid/no result - proceed anyway
            logger.warning(
                f"[PLANNING ROUTER] Task list critique returned no approval status "
                f"(LLM error) - proceeding anyway → "
                f"{'filter_and_complete' if rag_enabled else 'complete'}"
            )
            if log_task_list_summary:
                log_task_list_summary(state)
            return "filter_and_complete" if rag_enabled else "complete"

    # ===== PHASE 4: After workflow refinement =====
    if planning_phase == "workflow_refinement":
        had_queries = safe_get(state, "had_retrieval_queries_before_refinement", False)
        if had_queries:
            # Had queries → continue to next iteration (skip critique)
            return "increment_task_planning_iteration"
        else:
            # No queries → proceed to critique (if enabled)
            if use_critique:
                return "task_list_critique"
            else:
                # No critique enabled → complete
                logger.info(
                    f"[PLANNING ROUTER] Workflow refinement complete (no critique) → "
                    f"{'filter_and_complete' if rag_enabled else 'complete'}"
                )
                if log_task_list_summary:
                    log_task_list_summary(state)
                return "filter_and_complete" if rag_enabled else "complete"

    # ===== PHASE 5: After increment =====
    if planning_phase == "ready_to_generate":
        if task_planning_iteration >= max_task_planning_iterations:
            if send_message:
                send_message(
                    f"[KAI] Task list generation reached max iterations "
                    f"({max_task_planning_iterations}) without approval - proceeding anyway"
                )
            logger.info(
                f"[PLANNING ROUTER] Max task planning iterations "
                f"({max_task_planning_iterations}) reached → "
                f"{'filter_and_complete' if rag_enabled else 'complete'}"
            )
            if log_task_list_summary:
                log_task_list_summary(state)
            return "filter_and_complete" if rag_enabled else "complete"
        return "task_list_generation"

    # Phase complete (shouldn't reach here in normal flow)
    logger.info("[PLANNING ROUTER] Planning phase complete (unexpected state)")
    if log_task_list_summary:
        log_task_list_summary(state)
    return "filter_and_complete" if rag_enabled else "complete"
