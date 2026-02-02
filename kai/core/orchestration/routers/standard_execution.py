"""Standard execution phase routing."""

from typing import Callable, Optional

from kai.utils import setup_logger, safe_get
from .standard_continue import route_standard_continue_branch
from .standard_retry import route_standard_retry_branch
from .backtracking import route_backtracking_branch

logger = setup_logger(__name__)


def route_standard_execution(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None
) -> str:
    """
    Standard execution phase routing with 4 branches.

    Phase 1: Analyze completion & update tasks
    Phase 2: Branch based on execution state:
      - Branch 1: All Complete
      - Branch 2: Standard Continue (no errors, normal progression)
      - Branch 3: Standard Retry (error or LLM detected issue)
      - Branch 4: Backtracking

    Args:
        state: Current graph state
        send_message: Optional callback to send UI messages

    Returns:
        Next node name
    """
    logger.debug("[DET ROUTER] Phase: STANDARD_EXECUTION")

    # ==== SUB-PHASE 1: Analyze Completion & Update Tasks ====

    task_completion_analyzed = safe_get(state, "task_completion_analyzed")

    # Step 1: Analyze completion
    if not task_completion_analyzed:
        logger.debug("[DET ROUTER] STANDARD_EXEC: analyzing completion")
        return "autonomous_mark_completion"

    # Branch detection - autonomous_mark_completion sets these flags
    has_error = safe_get(state, "last_execution_failed")
    retry_objective = safe_get(state, "retry_objective")
    recovery_objective = safe_get(state, "recovery_objective")

    is_backtracking = recovery_objective is not None
    is_standard_retry = (has_error or retry_objective) and not is_backtracking

    logger.debug(
        f"[DET ROUTER] Branch detection: error={has_error}, "
        f"retry={bool(retry_objective)}, backtrack={is_backtracking}"
    )

    # Step 2: Update task list (if NOT standard retry and NOT complete)
    tasks_updated = safe_get(state, "tasks_updated")
    update_approved = safe_get(state, "update_approved")

    task_list = safe_get(state, "task_list", {})
    tasks = safe_get(task_list, "tasks", [])
    all_complete = all(safe_get(t, "status") == "completed" for t in tasks)

    # Skip task updates only in standard retry
    if (task_completion_analyzed and not is_standard_retry and
            not all_complete and not tasks_updated):
        # Check if we need to backup task list first
        has_backup = safe_get(state, "task_list_backup") is not None
        if not has_backup:
            logger.debug("[DET ROUTER] STANDARD_EXEC: backing up task list before update")
            return "backup_task_list"
        else:
            logger.debug("[DET ROUTER] STANDARD_EXEC: updating tasks")
            return "autonomous_update_tasks"

    # Step 3: Task update evaluation loop
    # Pattern: UPDATE → EVALUATOR → (if rejected) UPDATE → EVALUATOR → ... (max 3)
    task_list_update_rule = safe_get(state, "task_list_update_rule")
    evaluation_iteration = safe_get(state, "task_update_evaluation_iteration", 0)

    if tasks_updated and task_list_update_rule == "UPDATE" and not update_approved:
        if evaluation_iteration < 3:  # Max 3 iterations
            task_update_grade = safe_get(state, "task_update_grade")
            if task_update_grade == "APPROVED":
                if send_message:
                    send_message(
                        f"[KAI] Task list update approved after "
                        f"{evaluation_iteration} evaluation iterations"
                    )
                logger.debug("[DET ROUTER] Task update approved")
                # Log the approved task list
                tasks = safe_get(task_list, "tasks", [])
                logger.info(f"Updated task list ({len(tasks)} tasks):")
                for i, task in enumerate(tasks, 1):
                    task_text = task.get("task", task.get("objective", "No objective"))
                    logger.info(f"  {i}. {task_text}")
                # Continue to next step
            elif task_update_grade is None:
                # No grade yet - run evaluator
                logger.debug(
                    f"[DET ROUTER] STANDARD_EXEC: task update evaluator "
                    f"(iter {evaluation_iteration + 1})"
                )
                return "task_update_evaluator"
            else:
                # Evaluator returned REJECTED - regenerate task list
                logger.debug(
                    f"[DET ROUTER] Task update not approved "
                    f"(iter {evaluation_iteration}), regenerating task list"
                )
                return "autonomous_update_tasks"
        else:
            # Max iterations reached without approval - revert to backup
            if send_message:
                send_message(
                    f"[KAI] Task list update evaluation reached max iterations (3) "
                    f"without approval - reverting to previous task list"
                )
            logger.warning(
                f"[DET ROUTER] Task update evaluation reached max iterations "
                f"({evaluation_iteration}) without approval - reverting to backup"
            )
            return "revert_task_list"

    # Step 4: Activate next task
    next_task_activated = safe_get(state, "next_task_activated")

    if task_completion_analyzed and not next_task_activated:
        # Skip if we're still in task update flow
        if not tasks_updated or update_approved or task_list_update_rule != "UPDATE":
            logger.debug("[DET ROUTER] STANDARD_EXEC: activating next task")
            return "mark_next_task_active"

    # Step 5: RAG retrieval (if standard retry and RAG enabled)
    rag_enabled = safe_get(state, "rag_enabled", False)
    rag_query_assembled = safe_get(state, "rag_query_assembled", False)
    rag_retrieved = safe_get(state, "rag_retrieval") is not None

    logger.debug(
        f"[DET ROUTER] RAG check: enabled={rag_enabled}, "
        f"assembled={rag_query_assembled}, retrieved={rag_retrieved}, "
        f"retry={is_standard_retry}, next_activated={next_task_activated}"
    )

    if next_task_activated and is_standard_retry and rag_enabled:
        # Step 5a: First assemble the query
        if not rag_query_assembled:
            logger.debug("[DET ROUTER] STANDARD_EXEC: assembling RAG query")
            return "assemble_rag_query"
        # Step 5b: Then retrieve using the assembled query
        if not rag_retrieved:
            logger.debug("[DET ROUTER] STANDARD_EXEC: RAG retrieval for error recovery")
            return "rag_retrieval"

    # Debug: Log if RAG was skipped
    if is_standard_retry and rag_enabled:
        if rag_retrieved:
            logger.debug("[DET ROUTER] RAG already retrieved, proceeding to error recovery")
        elif not next_task_activated:
            logger.debug("[DET ROUTER] Skipping RAG - next task not activated yet")

    # ==== SUB-PHASE 2: Branch Based on Execution State ====

    # Re-check completion after task updates
    tasks = safe_get(task_list, "tasks", [])
    all_complete = all(safe_get(t, "status") == "completed" for t in tasks)

    # BRANCH 1: All Complete
    if all_complete:
        logger.debug("[DET ROUTER] Branch 1: ALL COMPLETE")
        return "complete"

    # BRANCH 4: Backtracking
    if is_backtracking:
        return route_backtracking_branch(state, send_message)

    # BRANCH 3: Standard Retry
    if is_standard_retry:
        return route_standard_retry_branch(state, send_message)

    # BRANCH 2: Standard Continue (normal progression, no errors)
    return route_standard_continue_branch(state, send_message)
