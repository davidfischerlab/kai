"""Execution phase node functions."""

import copy
import json
from typing import Any, Callable, Optional

from kai.utils import setup_logger

logger = setup_logger(__name__)


async def mark_first_execution_done(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None,
) -> dict:
    """Mark first execution as complete.

    Args:
        state: Current graph state (unused but required for node signature)
        send_message: Optional callback to send UI messages
    """
    if send_message:
        send_message("[KAI] Marking first execution as done")
    return {"auto_mode_first_execution_done": True}


async def mark_reasoning_completed(
    state: dict,
    send_message: Optional[Callable[[str], None]] = None,
    send_task_list: Optional[Callable[[dict], Any]] = None,
) -> dict:
    """Mark the active reasoning task as completed.

    This is called after reasoning is approved to ensure the task is marked
    completed before the next iteration starts. Without this, the next iteration's
    autonomous_mark_completion would see the task still as 'active' and try to
    re-evaluate it, potentially setting a retry_objective.

    Also marks first execution as done if we're still in first execution phase.

    Args:
        state: Current graph state
        send_message: Optional callback to send UI messages
        send_task_list: Optional callback to send task list update to UI
    """
    task_list = state.get("task_list", {})
    if not task_list or "tasks" not in task_list:
        logger.warning("[MARK_REASONING_COMPLETED] No task list found")
        return {}

    # Deep copy to avoid mutating original
    updated_task_list = copy.deepcopy(task_list)

    # Find and mark the active task as completed
    active_task_id = None
    for task in updated_task_list["tasks"]:
        if task.get("status") == "active":
            task["status"] = "completed"
            active_task_id = task.get("id")
            logger.info(
                f"[MARK_REASONING_COMPLETED] Marked task {active_task_id} "
                f"as completed: {task.get('task', '')[:50]}..."
            )
            break

    if active_task_id is None:
        logger.warning("[MARK_REASONING_COMPLETED] No active task found")
        return {}

    # Send simple message to UI
    if send_message:
        send_message(f"[KAI] Reasoning task {active_task_id} completed")

    # Send task list update to UI so completion is visible
    if send_task_list:
        # Include formatted reference workflows for proper display in VSCode
        task_list_for_ui = updated_task_list.copy()
        reference_workflow_percentages = state.get("reference_workflow_percentages", {})
        if reference_workflow_percentages:
            # Format with 📚 emoji and percentages (same as ReferenceWorkflowCellSelectionTool)
            formatted_lines = [
                f"📚 {full_id} (considering {percentage:.0f}% of file)"
                for full_id, percentage in sorted(reference_workflow_percentages.items())
            ]
            task_list_for_ui["reference_workflow_ids"] = "\n".join(formatted_lines)
        await send_task_list(task_list_for_ui)

    # Build result - clear transient reasoning state for next task
    result = {
        "task_list": updated_task_list,
        "reasoning_response": None,
        "reasoning_approval": None,
        "reasoning_critique": None,
        "reasoning_critique_iteration": 0,  # Reset reasoning-specific counter
        "active_task": None,
        "active_task_objective": None,
        "is_reasoning_task": False,
    }

    # Also mark first execution as done if we're still in first execution
    if not state.get("auto_mode_first_execution_done"):
        result["auto_mode_first_execution_done"] = True
        logger.debug("[MARK_REASONING_COMPLETED] Also marking first execution done")

    # Check if ALL tasks are now complete - signal LOOP_COMPLETE to stop the UI loop
    all_complete = all(
        t.get("status") == "completed" for t in updated_task_list.get("tasks", [])
    )
    if all_complete:
        result["auto_loop_update"] = "LOOP_COMPLETE"
        logger.info(
            "[MARK_REASONING_COMPLETED] All tasks complete - signaling LOOP_COMPLETE"
        )
        if send_message:
            send_message("[KAI] All tasks completed! 🎉")

    return result


async def assemble_rag_query(state: dict) -> dict:
    """Assemble RAG retrieval query for error recovery.

    This ensures the RAG retrieval gets relevant context for error recovery.
    Builds a FRESH list each time to prevent accumulation if called multiple times.
    """
    # Build fresh list - do NOT read from existing state to prevent accumulation
    snippet_retrieval_query = []

    # Add feedback on last attempt if given
    retry_objective = state.get("retry_objective")
    if retry_objective:
        snippet_retrieval_query.append(retry_objective)

    # Add error message if error occurred
    has_error = state.get("last_execution_failed", False)
    if has_error:
        error_message = state.get("error_message", "")
        if error_message:
            snippet_retrieval_query.append(error_message)

    if snippet_retrieval_query:
        logger.debug(
            f"[RAG QUERY] Assembled {len(snippet_retrieval_query)} queries: "
            f"{snippet_retrieval_query[:2]}..."
        )
        return {
            "snippet_retrieval_query": snippet_retrieval_query,
            "rag_query_assembled": True,
        }
    else:
        logger.debug("[RAG QUERY] No queries to assemble")
        return {"rag_query_assembled": True}
