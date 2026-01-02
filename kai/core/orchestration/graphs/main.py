"""Main graph builder for orchestrating planning/execution/regular subgraphs.

Iteration Model (for restart/resume understanding):
==================================================

Each LangGraph iteration is one call to graph.astream(). Within an iteration,
multiple nodes execute and checkpoints are saved after each node.

    Iteration N:
    ├── [nodes: routing, tool selection, code_generation, etc.]
    ├── code_generation returns ToolResult{code}
    ├── [checkpoint saved after each node]
    ├── ToolResult sent to Jupyter/VSCode (side effect)
    └── Iteration ends

    Jupyter/VSCode (external to LangGraph):
    ├── Receives execute_code message
    ├── Inserts cell into notebook
    ├── Executes cell
    └── Saves notebook

    Iteration N+1:
    ├── Context rebuilt from notebook (sees new cell + output)
    ├── Continues based on execution result
    └── ...

On restart after interruption:
1. Kernel restart + re-run all cells → notebook in consistent state
2. Resume LangGraph from SqliteSaver checkpoint
3. Continue with next iteration (context shows cell + result)

LangGraph doesn't track fine-grained Jupyter state (inserted vs executed).
The notebook save + kernel re-run pattern handles state reconciliation.
"""

from typing import Any, Callable, Optional

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState


def build_main_graph(
    planning_subgraph: Any,
    execution_subgraph: Any,
    regular_subgraph: Any,
    route_mode: Callable[[KaiState], str],
    route_after_planning: Callable[[dict], str],
    checkpointer: Optional[Any] = None,
) -> Any:
    """Build main graph that orchestrates planning/execution/regular subgraphs.

    This is the top-level graph that:
    1. Routes to appropriate subgraph based on mode
    2. Handles transitions between planning and execution
    3. Provides single entry point for all requests

    Args:
        planning_subgraph: Compiled planning subgraph
        execution_subgraph: Compiled execution subgraph
        regular_subgraph: Compiled regular subgraph
        route_mode: Function to determine which subgraph to use
        route_after_planning: Function to route after planning completes
        checkpointer: LangGraph checkpointer for state persistence.
            - None (default): Uses MemorySaver (in-memory, lost on restart)
            - SqliteSaver: Enables restart/resume across process restarts

    Returns:
        Compiled StateGraph with checkpointer attached
    """
    graph = StateGraph(KaiState)

    # Add subgraphs as nodes
    graph.add_node("planning", planning_subgraph)
    graph.add_node("execution", execution_subgraph)
    graph.add_node("regular", regular_subgraph)

    # Entry routing - determine which subgraph based on state
    graph.set_conditional_entry_point(
        route_mode,
        {
            "planning": "planning",
            "execution": "execution",
            "regular": "regular",
        },
    )

    # Planning → Execution transition (if tasks were generated)
    graph.add_conditional_edges(
        "planning",
        route_after_planning,
        {"execution": "execution", "complete": END},
    )

    # Execution exits to END (returns to caller for UI interaction)
    graph.add_edge("execution", END)
    graph.add_edge("regular", END)

    # Use provided checkpointer or default to MemorySaver
    # SqliteSaver enables restart/resume across process restarts
    # MemorySaver is in-memory only (development/testing)
    effective_checkpointer = checkpointer if checkpointer is not None else MemorySaver()
    return graph.compile(checkpointer=effective_checkpointer)


def build_main_graph_for_studio(
    planning_subgraph: Any,
    execution_subgraph: Any,
    regular_subgraph: Any,
    route_mode: Callable[[KaiState], str],
    route_after_planning: Callable[[dict], str],
) -> Any:
    """Build main graph WITHOUT checkpointer for LangGraph Studio.

    Studio provides its own persistence layer.

    Args:
        planning_subgraph: Compiled planning subgraph
        execution_subgraph: Compiled execution subgraph
        regular_subgraph: Compiled regular subgraph
        route_mode: Function to determine which subgraph to use
        route_after_planning: Function to route after planning completes

    Returns:
        Compiled StateGraph (no checkpointer)
    """
    graph = StateGraph(KaiState)

    graph.add_node("planning", planning_subgraph)
    graph.add_node("execution", execution_subgraph)
    graph.add_node("regular", regular_subgraph)

    graph.set_conditional_entry_point(
        route_mode,
        {
            "planning": "planning",
            "execution": "execution",
            "regular": "regular",
        },
    )

    graph.add_conditional_edges(
        "planning",
        route_after_planning,
        {"execution": "execution", "complete": END},
    )

    graph.add_edge("execution", END)
    graph.add_edge("regular", END)

    return graph.compile()
