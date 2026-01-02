"""Regular subgraph builder for non-autonomous mode."""

from typing import Any, Dict

from langgraph.graph import END, StateGraph

from kai.core.orchestration.state import KaiState


def build_regular_subgraph(tools: Dict[str, Any]) -> Any:
    """Build subgraph for regular (non-autonomous) mode.

    Args:
        tools: Dict of tool name to tool instance

    Returns:
        Compiled StateGraph
    """
    graph = StateGraph(KaiState)

    graph.add_node("classify_intent", tools["classify_intent"].as_graph_node())
    graph.add_node("rag_retrieval", tools["rag_retrieval"].as_graph_node())
    graph.add_node("code_generation", tools["code_generation"].as_graph_node())
    graph.add_node("answer_question", tools["answer_question"].as_graph_node())

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state.get("intent", "question_about_code"),
        {
            "generate_code": "rag_retrieval",
            "generate_code_in_place": "rag_retrieval",
            "question_about_code": "rag_retrieval",
            "remove_code": END,
        },
    )

    graph.add_conditional_edges(
        "rag_retrieval",
        lambda state: (
            "generate" if state.get("intent", "").startswith("generate_code") else "answer"
        ),
        {
            "generate": "code_generation",
            "answer": "answer_question",
        },
    )

    graph.add_edge("code_generation", END)
    graph.add_edge("answer_question", END)

    return graph.compile()
