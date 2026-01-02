"""Execution subgraph builder for autonomous mode task execution."""

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState
from kai.core.orchestration.nodes import (
    mark_first_execution_done,
    mark_reasoning_completed,
    backup_task_list,
    revert_task_list,
    assemble_rag_query,
)
from kai.core.orchestration.routers import route_deterministic

# Tools used in execution subgraph
AUTONOMOUS_TOOLS = [
    # Core execution tools
    "mark_next_task_active",
    "autonomous_mark_completion",
    "cell_positioning",
    "code_generation_with_guidance",
    "reasoning_response_with_guidance",
    "reasoning_critique",
    # Positioning tools
    "set_positioning_from_last_cell",
    # Task update tools
    "autonomous_update_tasks",
    "autonomous_update_critique",
    # Error recovery tools
    "error_recovery",
    "code_update",
    # Backtracking tools
    "backtrack_recovery",
    "cell_selection_deletion",
    "cell_deletion",
    # RAG tools
    "rag_retrieval",
    # Execution tools
    "execute_cell",
]


def build_execution_subgraph(
    tools: Dict[str, Any],
    send_message: Optional[Callable[[str], None]] = None,
    send_task_list: Optional[Callable[[dict], Any]] = None,
) -> Any:
    """Build subgraph for autonomous mode execution (after planning).

    This subgraph handles task execution with checkpointing for persistence.

    Args:
        tools: Dict of tool name to tool instance
        send_message: Optional callback to send UI messages
        send_task_list: Optional callback to send task list updates to UI

    Returns:
        Compiled StateGraph with checkpointer
    """
    graph = StateGraph(KaiState)

    # Create node functions with send_message bound
    async def _mark_first_execution_done_node(state: dict) -> dict:
        return await mark_first_execution_done(state, send_message=send_message)

    async def _mark_reasoning_completed_node(state: dict) -> dict:
        return await mark_reasoning_completed(
            state, send_message=send_message, send_task_list=send_task_list
        )

    def _route_deterministic(state: dict) -> str:
        return route_deterministic(state, send_message)

    # Helper nodes
    graph.add_node("mark_first_execution_done", _mark_first_execution_done_node)
    graph.add_node("mark_reasoning_completed", _mark_reasoning_completed_node)
    graph.add_node("backup_task_list", backup_task_list)
    graph.add_node("revert_task_list", revert_task_list)
    graph.add_node("assemble_rag_query", assemble_rag_query)

    # Add tool nodes
    for name in AUTONOMOUS_TOOLS:
        if name in tools:
            graph.add_node(name, tools[name].as_graph_node())

    # Build routing map
    routing_map = {tool: tool for tool in AUTONOMOUS_TOOLS}
    routing_map["mark_first_execution_done"] = "mark_first_execution_done"
    routing_map["mark_reasoning_completed"] = "mark_reasoning_completed"
    routing_map["backup_task_list"] = "backup_task_list"
    routing_map["revert_task_list"] = "revert_task_list"
    routing_map["assemble_rag_query"] = "assemble_rag_query"
    routing_map["complete"] = END

    # Entry point and tool edges
    graph.set_conditional_entry_point(_route_deterministic, routing_map)
    for tool_name in AUTONOMOUS_TOOLS:
        graph.add_conditional_edges(tool_name, _route_deterministic, routing_map)

    # Helper node edges
    graph.add_edge("mark_first_execution_done", END)
    graph.add_edge("mark_reasoning_completed", END)
    graph.add_conditional_edges("backup_task_list", _route_deterministic, routing_map)
    graph.add_conditional_edges("revert_task_list", _route_deterministic, routing_map)
    graph.add_conditional_edges("assemble_rag_query", _route_deterministic, routing_map)

    # NOTE: Don't use a subgraph-specific checkpointer - state should flow
    # through the parent graph's checkpointer. Using a separate checkpointer
    # here causes state updates (like auto_loop_update) to be lost when the
    # parent graph reads state via aget_state().
    return graph.compile()


def build_execution_subgraph_for_studio(
    tools: Dict[str, Any],
    send_message: Optional[Callable[[str], None]] = None,
    send_task_list: Optional[Callable[[dict], Any]] = None,
) -> Any:
    """Build execution subgraph WITHOUT checkpointer for LangGraph Studio.

    Studio provides its own persistence layer.

    Args:
        tools: Dict of tool name to tool instance
        send_message: Optional callback to send UI messages
        send_task_list: Optional callback to send task list updates to UI

    Returns:
        Compiled StateGraph (no checkpointer)
    """
    graph = StateGraph(KaiState)

    # Create node functions with send_message bound
    async def _mark_first_execution_done_node(state: dict) -> dict:
        return await mark_first_execution_done(state, send_message=send_message)

    async def _mark_reasoning_completed_node(state: dict) -> dict:
        return await mark_reasoning_completed(
            state, send_message=send_message, send_task_list=send_task_list
        )

    def _route_deterministic(state: dict) -> str:
        return route_deterministic(state, send_message)

    # Helper nodes
    graph.add_node("mark_first_execution_done", _mark_first_execution_done_node)
    graph.add_node("mark_reasoning_completed", _mark_reasoning_completed_node)
    graph.add_node("backup_task_list", backup_task_list)
    graph.add_node("revert_task_list", revert_task_list)
    graph.add_node("assemble_rag_query", assemble_rag_query)

    # Add tool nodes
    for name in AUTONOMOUS_TOOLS:
        if name in tools:
            graph.add_node(name, tools[name].as_graph_node())

    # Build routing map
    routing_map = {tool: tool for tool in AUTONOMOUS_TOOLS}
    routing_map["mark_first_execution_done"] = "mark_first_execution_done"
    routing_map["mark_reasoning_completed"] = "mark_reasoning_completed"
    routing_map["backup_task_list"] = "backup_task_list"
    routing_map["revert_task_list"] = "revert_task_list"
    routing_map["assemble_rag_query"] = "assemble_rag_query"
    routing_map["complete"] = END

    # Entry point and tool edges
    graph.set_conditional_entry_point(_route_deterministic, routing_map)
    for tool_name in AUTONOMOUS_TOOLS:
        graph.add_conditional_edges(tool_name, _route_deterministic, routing_map)

    # Helper node edges
    graph.add_edge("mark_first_execution_done", END)
    graph.add_edge("mark_reasoning_completed", END)
    graph.add_conditional_edges("backup_task_list", _route_deterministic, routing_map)
    graph.add_conditional_edges("revert_task_list", _route_deterministic, routing_map)
    graph.add_conditional_edges("assemble_rag_query", _route_deterministic, routing_map)

    return graph.compile()
