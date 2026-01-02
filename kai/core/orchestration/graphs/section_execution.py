"""Section execution subgraph builder for running cell ranges with error recovery."""

from typing import Any, Callable, Dict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import SectionState
from kai.core.tools.section_code_review import SectionCodeReviewTool


def build_section_execution_subgraph(
    llm: Any,
    section_check_position_node: Callable,
    section_route_from_position_check: Callable,
    section_execute_cell_node: Callable,
    section_check_execution_node: Callable,
    section_route_from_execution_check: Callable,
    section_advance_cell_node: Callable,
    section_route_fix_operation: Callable,
    section_apply_delete_node: Callable,
    section_apply_replace_node: Callable,
    section_apply_insert_node: Callable,
    section_check_fix_result_node: Callable,
    section_route_from_fix_check: Callable,
    section_complete_success_node: Callable,
    section_complete_failure_node: Callable,
) -> Any:
    """Build subgraph for section execution with error recovery.

    This subgraph handles running a specific range of cells, detecting errors,
    and applying fixes through intelligent code review.

    Args:
        llm: LLM interface for section code review
        section_check_position_node: Node to check current position
        section_route_from_position_check: Router after position check
        section_execute_cell_node: Node to execute current cell
        section_check_execution_node: Node to check execution result
        section_route_from_execution_check: Router after execution check
        section_advance_cell_node: Node to advance to next cell
        section_route_fix_operation: Router to determine fix operation
        section_apply_delete_node: Node to apply delete fix
        section_apply_replace_node: Node to apply replace fix
        section_apply_insert_node: Node to apply insert fix
        section_check_fix_result_node: Node to check fix result
        section_route_from_fix_check: Router after fix check
        section_complete_success_node: Node to mark success
        section_complete_failure_node: Node to mark failure

    Returns:
        Compiled StateGraph with checkpointer
    """
    # Initialize section-specific tools
    section_review_tool = SectionCodeReviewTool(llm)

    graph = StateGraph(SectionState)

    # Add nodes
    graph.add_node("check_position", section_check_position_node)
    graph.add_node("execute_cell", section_execute_cell_node)
    graph.add_node("check_execution", section_check_execution_node)
    graph.add_node("advance_cell", section_advance_cell_node)
    graph.add_node("section_code_review", section_review_tool.as_graph_node())
    graph.add_node("apply_delete", section_apply_delete_node)
    graph.add_node("apply_replace", section_apply_replace_node)
    graph.add_node("apply_insert", section_apply_insert_node)
    graph.add_node("check_fix_result", section_check_fix_result_node)
    graph.add_node("complete_success", section_complete_success_node)
    graph.add_node("complete_failure", section_complete_failure_node)

    # Entry point
    graph.set_entry_point("check_position")

    # Conditional edges from check_position
    graph.add_conditional_edges(
        "check_position",
        section_route_from_position_check,
        {
            "execute": "execute_cell",
            "complete": "complete_success",
        },
    )

    # execute_cell → check_execution
    graph.add_edge("execute_cell", "check_execution")

    # Conditional edges from check_execution
    graph.add_conditional_edges(
        "check_execution",
        section_route_from_execution_check,
        {
            "success": "advance_cell",
            "error": "section_code_review",
        },
    )

    # advance_cell → check_position (loop)
    graph.add_edge("advance_cell", "check_position")

    # section_code_review → route to fix operation
    graph.add_conditional_edges(
        "section_code_review",
        section_route_fix_operation,
        {
            "delete": "apply_delete",
            "replace": "apply_replace",
            "insert": "apply_insert",
            "fail": "complete_failure",
        },
    )

    # All fix operations → check_fix_result
    graph.add_edge("apply_delete", "check_fix_result")
    graph.add_edge("apply_replace", "check_fix_result")
    graph.add_edge("apply_insert", "check_fix_result")

    # Conditional edges from check_fix_result
    graph.add_conditional_edges(
        "check_fix_result",
        section_route_from_fix_check,
        {
            "retry": "execute_cell",
            "fail": "complete_failure",
        },
    )

    # Terminal nodes
    graph.add_edge("complete_success", END)
    graph.add_edge("complete_failure", END)

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
