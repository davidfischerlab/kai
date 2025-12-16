"""State definition for LangGraph orchestrator using TypedDict."""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from typing_extensions import Annotated
from langgraph.graph import add_messages

# Planning phase tracking for explicit control flow
PlanningPhase = Literal["workflow_retrieval", "task_planning", "workflow_refinement", "task_list_critique", "ready_to_generate", "complete"]


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
    confirm_plan: bool  # Whether to pause after planning for user approval (True in VSCode, False in Jupyter)
    error_message: str
    notebook_uri: Optional[str]  # Path to notebook for session tracking and debug folder naming
    session_timestamp: Optional[str]  # Session start time
    iteration_timestamp: Optional[str]  # Current iteration time
    iteration_counter: int  # Iteration count within session
    active: bool  # Whether session is active

    # Notebook context
    notebook_cells: List[Dict[str, Any]]
    notebook_structure: Dict[str, Any]
    current_cell: str
    current_cell_index: int
    execution_history: List[Any]

    # Task management
    task_list: Dict[str, Any]
    task_list_backup: Optional[Dict[str, Any]]  # Backup for reversion if critique fails
    active_task: Optional[Dict[str, Any]]

    # RAG and retrieval
    retrieval_queries: List[str]
    snippet_retrieval_query: List[str]  # Queries for RAG snippet retrieval (from task update tool)
    rag_text: Optional[str]
    rag_retrieval: Optional[str]  # RAG retrieval result (for router check)
    reference_workflow_content: Dict[str, Any]
    excluded_workflows: List[str]  # Workflows to exclude from retrieval (persists across iterations)

    # Code generation
    intent: str
    target_cell: Optional[int]
    generated_code: Optional[str]

    # Execution tracking
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

    # Deterministic routing phase tracking
    task_completion_analyzed: bool  # Set by autonomous_mark_completion
    next_task_activated: bool  # Set by mark_next_task_active

    # Error recovery and branching
    retry_objective: Optional[str]  # Set by autonomous_mark_completion when LLM detects issue
    recovery_objective: Optional[str]  # Set by autonomous_mark_completion for backtracking
    backtrack_to_task: Optional[Dict[str, Any]]  # Task to backtrack to
    error_recovery_strategy: Optional[str]  # "REPLACE_AND_RETRY" or "REPLACE_AND_RESTART"
    restart_required: bool  # Set by backtrack_recovery

    # Task updates
    task_list_update_rule: Optional[str]  # "UPDATE", "NO_UPDATE", or "BACKTRACK"
    task_text_old: Optional[str]  # Original task list text for critique
    tasks_updated: bool  # Set by autonomous_update_tasks
    update_approved: bool  # Set by autonomous_update_critique

    # Positioning
    positioning_info: Optional[Dict[str, Any]]  # {target_cell: int}
    last_cell_modified_in_auto_mode: Optional[int]

    # Active task details
    active_task_objective: Optional[str]  # String description of active task
    is_reasoning_task: bool  # Whether active task is reasoning
    next_pending_task_objective: Optional[str]  # Next task for context

    # Critique loops
    use_critique: bool  # Whether to use critique loops (default True in kai_dev)
    critique_iteration: int  # Current critique iteration count
    task_list_approval: Optional[str]  # "APPROVED" or rejection reason
    task_list_critique: Optional[str]  # Critique feedback for task list
    reasoning_approval: Optional[str]  # "APPROVED" or rejection reason
    reasoning_critique: Optional[str]  # Critique feedback for reasoning (CRITICAL for state propagation)
    autonomous_update_approval: Optional[str]  # "APPROVED" or rejection reason
    autonomous_update_critique: Optional[str]  # Critique feedback for autonomous updates

    # Backtracking
    reset_tasks: Optional[List[Dict[str, Any]]]  # Tasks to reset in backtracking
    cells_to_delete: Optional[List[int]]  # Cells selected for deletion
    cells_deleted: bool  # Whether cell deletion completed
    backtrack_recovery_done: bool  # Whether backtrack_recovery completed

    # Reference workflows (persistent across iterations)
    reference_workflow_ids: Optional[str]  # IDs of selected reference workflows

    # Orchestrator phase tracking (replaces instance variables)
    auto_mode_first_execution_done: bool  # Whether first autonomous execution completed

    # Planning phase tracking (for explicit control flow in planning)
    planning_phase: Optional[PlanningPhase]  # Current planning phase
    workflow_retrieval_iteration: int  # Iteration counter for workflow retrieval loop (max 2)
    task_planning_iteration: int  # Iteration counter for task generation + workflow refinement loop (max 10)
    had_retrieval_queries_before_refinement: bool  # Whether retrieval queries existed before workflow refinement cleared them

    # Internal tool communication (cleaned up after each node)
    _last_tool_result: Optional[Dict[str, Any]]  # Tool result for UI communication
