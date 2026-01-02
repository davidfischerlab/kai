"""Planning subgraph builder for workflow retrieval + task generation."""

from typing import Any, Callable, Dict, Union
from collections.abc import Coroutine

from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState


def build_planning_subgraph(
    tools: Dict[str, Any],
    increment_task_planning_iteration_node: Callable[[dict], Union[dict, Coroutine[Any, Any, dict]]],
    route_planning_phase: Callable[[dict], str],
) -> Any:
    """Build subgraph for planning phase (workflow retrieval + task generation).

    This subgraph handles:
    1. Initial workflow retrieval (max 2 iterations)
    2. Task list generation + critique loop (max 10 iterations)
    3. Workflow refinement based on retrieval queries from task generation

    Args:
        tools: Dict of tool name to tool instance
        increment_task_planning_iteration_node: Node function for incrementing iteration
        route_planning_phase: Router function for planning phase

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(KaiState)

    # Add iteration counter node
    graph.add_node(
        "increment_task_planning_iteration",
        increment_task_planning_iteration_node,
    )

    # Add planning tools
    planning_tools = [
        "search_workflows",
        "workflow_refinement",
        "task_list_generation",
        "task_list_critique",
    ]

    for name in planning_tools:
        if name in tools:
            graph.add_node(name, tools[name].as_graph_node())

    # Add filter tool for cleanup after planning
    if "filter_unused_reference_workflows" in tools:
        graph.add_node(
            "filter_unused_reference_workflows",
            tools["filter_unused_reference_workflows"].as_graph_node(),
        )

    # Build routing map for all destinations
    routing_map = {tool: tool for tool in planning_tools}
    routing_map["increment_task_planning_iteration"] = "increment_task_planning_iteration"
    routing_map["filter_and_complete"] = "filter_unused_reference_workflows"
    routing_map["complete"] = END

    # Entry point routes directly to first action (no router node)
    graph.set_conditional_entry_point(route_planning_phase, routing_map)

    # Each tool routes to next action via conditional edges
    for tool_name in planning_tools:
        graph.add_conditional_edges(tool_name, route_planning_phase, routing_map)

    graph.add_conditional_edges(
        "increment_task_planning_iteration",
        route_planning_phase,
        routing_map,
    )

    # Filter tool goes to END (last step)
    graph.add_edge("filter_unused_reference_workflows", END)

    return graph.compile()
