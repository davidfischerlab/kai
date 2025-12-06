"""State definition for LangGraph orchestrator using TypedDict."""

from typing import TypedDict, List, Dict, Any, Optional
from typing_extensions import Annotated
from langgraph.graph import add_messages


class KaiState(TypedDict, total=False):
    """
    State for LangGraph-based workflow orchestration.

    Uses TypedDict (not Pydantic) as LangGraph's native state format.
    All fields are optional (total=False) for flexibility.
    """
    # Core request fields
    user_query: str
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # Session metadata
    session_id: str
    request_id: str
    autonomous_mode: bool
    rag_enabled: bool
    auto_mode_continue: bool
    error_message: str
    excluded_workflows: List[str]

    # Notebook context
    notebook_cells: List[Dict[str, Any]]
    notebook_structure: Dict[str, Any]
    current_cell: str
    current_cell_index: int
    execution_history: List[Any]

    # Task management
    task_list: Dict[str, Any]
    active_task: Optional[Dict[str, Any]]

    # RAG and retrieval
    retrieval_queries: List[str]
    snippet_retrieval_query: Optional[str]
    rag_text: Optional[str]
    reference_workflow_content: Dict[str, Any]

    # Code generation
    intent: str
    target_cell: Optional[int]
    generated_code: Optional[str]

    # Execution tracking
    just_executed: bool
    last_execution_failed: bool
    last_output: Optional[str]

    # Error handling
    error_context: Optional[Dict[str, Any]]
    backtracking_context: Optional[Dict[str, Any]]

    # Reasoning and conversation
    conversation_history: List[str]
    reasoning_response: Optional[str]

    # Control flow
    autonomous_mode_continue: bool
    next_action: Optional[str]
