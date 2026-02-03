"""Task list evaluator-optimizer loop subgraph.

This subgraph handles the task list generation and evaluation loop during planning:
- Optimizer: task_list_generation (generates/improves task list)
- Evaluator: task_list_evaluator (assesses quality, provides feedback)

Pattern:
    optimizer → evaluator → conditional_edge(APPROVED: END, REJECTED: optimizer)
"""

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState
from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def _route_after_evaluation(state: dict, max_iterations: int = 10) -> str:
    """Route after task list evaluation.

    Args:
        state: Current graph state
        max_iterations: Maximum evaluation iterations

    Returns:
        "complete" if approved or max iterations reached, "optimizer" to loop back
    """
    grade = safe_get(state, "task_list_grade")
    iteration = safe_get(state, "task_list_evaluation_iteration", 0)

    if grade == "APPROVED":
        logger.info(f"[TASK_LIST_EVAL] Approved after {iteration} iteration(s)")
        return "complete"

    if iteration >= max_iterations:
        logger.warning(
            f"[TASK_LIST_EVAL] Max iterations ({max_iterations}) reached, "
            "auto-approving"
        )
        return "complete"

    logger.debug(
        f"[TASK_LIST_EVAL] Rejected (iteration {iteration}), looping back to optimizer"
    )
    return "optimizer"


def build_task_list_evaluator_loop(
    optimizer_tool: Any,
    evaluator_tool: Any,
    max_iterations: int = 10,
    send_message: Optional[Callable[[str], None]] = None,
) -> Any:
    """Build the task list evaluator-optimizer loop subgraph.

    This subgraph follows LangGraph's evaluator-optimizer pattern:
    - Direct edge from optimizer to evaluator
    - Conditional routing only after evaluator

    Args:
        optimizer_tool: TaskListGenerationTool (or similar) instance
        evaluator_tool: TaskListEvaluatorTool instance
        max_iterations: Maximum iterations before auto-approval (default 10)
        send_message: Optional callback to send UI messages

    Returns:
        Compiled StateGraph for the evaluator loop
    """
    graph = StateGraph(KaiState)

    # Add nodes
    graph.add_node("optimizer", optimizer_tool.as_graph_node())
    graph.add_node("evaluator", evaluator_tool.as_graph_node())

    # Entry point: always start with optimizer
    graph.set_entry_point("optimizer")

    # Direct edge: optimizer → evaluator (no router between them!)
    graph.add_edge("optimizer", "evaluator")

    # Conditional edge after evaluator
    def route_decision(state: dict) -> str:
        result = _route_after_evaluation(state, max_iterations)
        if send_message:
            grade = safe_get(state, "task_list_grade", "UNKNOWN")
            iteration = safe_get(state, "task_list_evaluation_iteration", 0)
            send_message(
                f"[KAI] Task list evaluation: {grade} (iteration {iteration})"
            )
        return result

    graph.add_conditional_edges(
        "evaluator",
        route_decision,
        {"complete": END, "optimizer": "optimizer"}
    )

    return graph.compile()
