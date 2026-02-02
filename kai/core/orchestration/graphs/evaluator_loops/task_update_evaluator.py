"""Task update evaluator-optimizer loop subgraph.

This subgraph handles the task update and evaluation loop during execution:
- Optimizer: task_update_optimizer (updates task list based on progress)
- Evaluator: task_update_evaluator (assesses update quality, provides feedback)

Special features:
- Backup/revert mechanism: backs up task list before updates, reverts on max iterations
- Conditional entry: skips backup if already has one (for loop iterations)

Pattern:
    [backup] → optimizer → evaluator → conditional_edge(APPROVED: END, REJECTED: optimizer, MAX_ITER: revert)
"""

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState
from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def _route_after_evaluation(state: dict, max_iterations: int = 3) -> str:
    """Route after task update evaluation.

    Args:
        state: Current graph state
        max_iterations: Maximum evaluation iterations before revert

    Returns:
        "complete" if approved, "revert" if max iterations, "optimizer" to loop back
    """
    grade = safe_get(state, "task_update_grade")
    iteration = safe_get(state, "task_update_evaluation_iteration", 0)

    if grade == "APPROVED":
        logger.info(f"[TASK_UPDATE_EVAL] Approved after {iteration} iteration(s)")
        return "complete"

    if iteration >= max_iterations:
        logger.warning(
            f"[TASK_UPDATE_EVAL] Max iterations ({max_iterations}) reached "
            "without approval - reverting to backup"
        )
        return "revert"

    logger.debug(
        f"[TASK_UPDATE_EVAL] Rejected (iteration {iteration}), "
        "looping back to optimizer"
    )
    return "optimizer"


def _route_entry(state: dict) -> str:
    """Route at entry: check if backup exists.

    If backup exists, we're in a loop iteration - go directly to optimizer.
    If no backup, this is first entry - backup first.

    Args:
        state: Current graph state

    Returns:
        "backup" if no backup exists, "optimizer" if already backed up
    """
    has_backup = safe_get(state, "task_list_backup") is not None

    if has_backup:
        logger.debug("[TASK_UPDATE_EVAL] Entry: backup exists, going to optimizer")
        return "optimizer"
    else:
        logger.debug("[TASK_UPDATE_EVAL] Entry: no backup, backing up first")
        return "backup"


def build_task_update_evaluator_loop(
    optimizer_tool: Any,
    evaluator_tool: Any,
    backup_node: Callable[[dict], dict],
    revert_node: Callable[[dict], dict],
    max_iterations: int = 3,
    send_message: Optional[Callable[[str], None]] = None,
) -> Any:
    """Build the task update evaluator-optimizer loop subgraph.

    This subgraph follows LangGraph's evaluator-optimizer pattern with
    additional backup/revert mechanism:
    - Conditional entry: backup if needed
    - Direct edge from optimizer to evaluator
    - Conditional routing after evaluator with revert option

    Args:
        optimizer_tool: TaskUpdateOptimizerTool (autonomous_update_tasks) instance
        evaluator_tool: TaskUpdateEvaluatorTool instance
        backup_node: Node function to backup task list
        revert_node: Node function to revert to backup
        max_iterations: Maximum iterations before revert (default 3)
        send_message: Optional callback to send UI messages

    Returns:
        Compiled StateGraph for the evaluator loop
    """
    graph = StateGraph(KaiState)

    # Add nodes
    graph.add_node("backup", backup_node)
    graph.add_node("optimizer", optimizer_tool.as_graph_node())
    graph.add_node("evaluator", evaluator_tool.as_graph_node())
    graph.add_node("revert", revert_node)

    # Conditional entry point: backup if needed, otherwise straight to optimizer
    graph.set_conditional_entry_point(
        _route_entry,
        {"backup": "backup", "optimizer": "optimizer"}
    )

    # backup → optimizer
    graph.add_edge("backup", "optimizer")

    # Direct edge: optimizer → evaluator (no router between them!)
    graph.add_edge("optimizer", "evaluator")

    # Conditional edge after evaluator
    def route_decision(state: dict) -> str:
        result = _route_after_evaluation(state, max_iterations)
        if send_message:
            grade = safe_get(state, "task_update_grade", "UNKNOWN")
            iteration = safe_get(state, "task_update_evaluation_iteration", 0)
            if result == "revert":
                send_message(
                    f"[KAI] Task update evaluation reached max iterations "
                    f"({max_iterations}) - reverting to backup"
                )
            else:
                send_message(
                    f"[KAI] Task update evaluation: {grade} (iteration {iteration})"
                )
        return result

    graph.add_conditional_edges(
        "evaluator",
        route_decision,
        {"complete": END, "revert": "revert", "optimizer": "optimizer"}
    )

    # revert → END
    graph.add_edge("revert", END)

    return graph.compile()
