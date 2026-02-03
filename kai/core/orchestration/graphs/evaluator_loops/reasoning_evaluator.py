"""Reasoning evaluator-optimizer loop subgraph.

This subgraph handles the reasoning response generation and evaluation loop:
- Optimizer: reasoning_optimizer (generates/improves reasoning response)
- Evaluator: reasoning_evaluator (assesses quality, provides feedback)

Pattern:
    optimizer → evaluator → conditional_edge(APPROVED: END, REJECTED: optimizer)
"""

from typing import Any, Callable, Optional

from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState
from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


def _route_after_evaluation(state: dict, max_iterations: int = 2) -> str:
    """Route after reasoning evaluation.

    Args:
        state: Current graph state
        max_iterations: Maximum evaluation iterations

    Returns:
        "complete" if approved or max iterations reached, "optimizer" to loop back
    """
    grade = safe_get(state, "reasoning_grade")
    iteration = safe_get(state, "reasoning_evaluation_iteration", 0)

    if grade == "APPROVED":
        logger.info(f"[REASONING_EVAL] Approved after {iteration} iteration(s)")
        return "complete"

    if iteration >= max_iterations:
        logger.warning(
            f"[REASONING_EVAL] Max iterations ({max_iterations}) reached, "
            "auto-approving"
        )
        return "complete"

    logger.debug(
        f"[REASONING_EVAL] Rejected (iteration {iteration}), looping back to optimizer"
    )
    return "optimizer"


def build_reasoning_evaluator_loop(
    optimizer_tool: Any,
    evaluator_tool: Any,
    max_iterations: int = 2,
    send_message: Optional[Callable[[str], None]] = None,
) -> Any:
    """Build the reasoning evaluator-optimizer loop subgraph.

    This subgraph follows LangGraph's evaluator-optimizer pattern:
    - Direct edge from optimizer to evaluator
    - Conditional routing only after evaluator

    Args:
        optimizer_tool: ReasoningResponseWithGuidanceTool instance
        evaluator_tool: ReasoningEvaluatorTool instance
        max_iterations: Maximum iterations before auto-approval (default 2)
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
            grade = safe_get(state, "reasoning_grade", "UNKNOWN")
            iteration = safe_get(state, "reasoning_evaluation_iteration", 0)
            send_message(
                f"[KAI] Reasoning evaluation: {grade} (iteration {iteration})"
            )
        return result

    graph.add_conditional_edges(
        "evaluator",
        route_decision,
        {"complete": END, "optimizer": "optimizer"}
    )

    return graph.compile()
