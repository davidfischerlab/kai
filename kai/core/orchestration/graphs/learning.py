"""Learning graph for post-execution explanations.

This graph runs AFTER code execution completes successfully in learning mode.
It generates an explanation of the just-completed task with execution context.

The separation from the main execution graph allows:
1. Learning explanation to have access to execution results
2. The main execution graph to be identical regardless of learning mode
3. Clean separation of concerns between code execution and explanation generation
"""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState


def build_learning_graph(tools: Dict[str, Any]) -> Any:
    """Build minimal graph that runs learning_explanation tool.

    This graph is invoked by the orchestrator AFTER the main execution graph
    completes and cell execution succeeds.

    Args:
        tools: Dict of tool name to tool instance (must include 'learning_explanation')

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(KaiState)

    # Add learning explanation tool node
    learning_tool = tools.get("learning_explanation")
    if learning_tool:
        graph.add_node("learning_explanation", learning_tool.as_graph_node())
        graph.set_entry_point("learning_explanation")
        graph.add_edge("learning_explanation", END)
    else:
        # Fallback: empty graph that just ends
        # This shouldn't happen in practice but prevents errors
        async def noop(state: dict) -> dict:
            return {}
        graph.add_node("noop", noop)
        graph.set_entry_point("noop")
        graph.add_edge("noop", END)

    return graph.compile()
