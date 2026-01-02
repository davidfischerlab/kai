"""LangGraph Studio entry points for graph visualization and debugging.

This module exposes the compiled LangGraph graphs for use with LangGraph Studio.
Studio connects via `langgraph dev` which reads langgraph.json and loads these graphs.

Usage:
    1. Start the dev server: `langgraph dev`
    2. Open Studio: https://smith.langchain.com/studio
    3. Connect to: http://localhost:2024

The graphs exposed here mirror the orchestrator's internal subgraphs:
- main_graph: Top-level orchestrator routing between planning/execution/regular modes
- planning_graph: Workflow retrieval + task list generation subgraph
- execution_graph: Autonomous execution subgraph (after planning)

IMPORTANT: LangGraph Studio handles persistence automatically, so we expose
graphs compiled WITHOUT checkpointers. The orchestrator's internal graphs
use MemorySaver for standalone operation.
"""

from kai.core.agent import KaiAgent

# Create agent instance for Studio
# Uses default ollama provider - Studio is for development/debugging
_agent = KaiAgent(llm_provider='ollama')

# Get graphs compiled WITHOUT checkpointers for Studio
# Studio provides its own persistence layer
_studio_graphs = _agent.orchestrator.get_graphs_for_studio()

main_graph = _studio_graphs['main']
planning_graph = _studio_graphs['planning']
execution_graph = _studio_graphs['execution']
