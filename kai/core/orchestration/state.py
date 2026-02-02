"""State definition for LangGraph orchestrator using TypedDict.

This module defines state schemas for the LangGraph-based orchestration.

Memory Management (see docs/langgraph_improvements.md):
- Short-term memory (checkpointed via MemorySaver): task_list, reference_workflow_content,
  excluded_workflows - persists across iterations within same thread_id
- Transient state: tracked via iteration_id - nodes check if their output matches
  current iteration to determine staleness (see TRANSIENT_FIELD_NAMES)

State Classes:
- TaskOutput: Unified output from code/reasoning generation
- ExecutionEvent: Append-only event log entries
- KaiState: Full state schema for orchestrator graphs (70+ fields)

Reducer Strategy (see custom reducers below):
- add_messages: LangGraph's message list reducer for conversation history
- add_unique: Accumulate unique items (for retrieval_queries, excluded_workflows)
- replace_or_keep: Replace if not None, otherwise keep existing value
- No annotation (last-write-wins): For fields like execution_history where VSCode sends complete data
"""

from dataclasses import dataclass, field
from typing import TypedDict, List, Dict, Any, Optional, Literal, Mapping
from typing_extensions import Annotated
from langgraph.graph import add_messages

from kai.utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# BACKTRACKING CONTEXT (moved from execution_context.py)
# =============================================================================

@dataclass
class BacktrackingContext:
    """Context for backtracking operations.

    Provides typed access to backtracking state fields.
    Can be constructed from state dict via from_state().
    """
    recovery_objective: str
    backtrack_to_task: Dict[str, Any]
    deleted_cells: List[int] = field(default_factory=list)
    index_translation: Dict[int, int] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """Check if backtracking is active."""
        return bool(self.recovery_objective)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> Optional["BacktrackingContext"]:
        """Create BacktrackingContext from state dict if backtracking_context exists."""
        bt_dict = state.get("backtracking_context")
        if not bt_dict:
            return None
        return cls(
            recovery_objective=bt_dict.get("recovery_objective", ""),
            backtrack_to_task=bt_dict.get("backtrack_to_task", {}),
            deleted_cells=bt_dict.get("deleted_cells", []),
            index_translation=bt_dict.get("index_translation", {})
        )


# =============================================================================
# CUSTOM REDUCERS FOR STATE MANAGEMENT
# =============================================================================

def add_unique(left: list, right: list) -> list:
    """Accumulate unique items (for retrieval_queries, excluded_workflows).

    This reducer deduplicates items when merging lists, ensuring that
    repeated queries or workflow IDs are not accumulated.

    Args:
        left: Existing list in state
        right: New items to add

    Returns:
        Combined list with unique items only
    """
    if left is None:
        left = []
    if right is None:
        return left
    seen = set(left)
    return left + [x for x in right if x not in seen]


def replace_or_keep(left: Any, right: Any) -> Any:
    """Replace if right is not None, otherwise keep left.

    This reducer allows explicit updates while preserving state when
    nodes don't need to modify a field.

    Args:
        left: Existing value in state
        right: New value (None means "keep existing")

    Returns:
        right if right is not None, otherwise left
    """
    return right if right is not None else left


# =============================================================================
# TRANSIENT FIELD NAMES (reset each iteration)
# =============================================================================
# These fields track progress WITHIN a single iteration and are reset to
# default values at the start of each iteration in process_request().
# The iteration_id field is incremented each iteration for tracking purposes.

TRANSIENT_FIELD_NAMES = [
    # Progress tracking within iteration
    "task_completion_analyzed",
    "next_task_activated",
    "tasks_updated",
    "update_approved",
    # Evaluation iteration counters (evaluator-optimizer pattern)
    "reasoning_evaluation_iteration",
    "task_update_evaluation_iteration",
    "task_list_evaluation_iteration",
    # Generation outputs (fresh each iteration)
    "generated_code",
    "reasoning_response",
    "positioning_info",
    # Error/retry flags (set fresh based on execution results)
    "last_execution_failed",
    "retry_objective",
    "rag_retrieval",
    "rag_query_assembled",  # Whether RAG query has been assembled with retry context
    "snippet_retrieval_query",  # RAG queries - MUST reset each iteration to prevent accumulation
    # Recovery/backtracking (set by LLM analysis)
    "recovery_objective",
    "error_recovery_strategy",
    "restart_required",
    "backtrack_recovery_done",
    "cells_to_delete",
    "cells_deleted",
    # Evaluation grades and feedback (evaluator-optimizer pattern)
    "reasoning_grade",
    "reasoning_feedback",
    "task_update_grade",
    "task_update_feedback",
    "task_list_grade",
    "task_list_feedback",
    # Task update context
    "task_list_update_rule",
    "task_list_update_rationale",  # Reasoning for update, used in evaluation prompt
    "task_list_backup",  # Used for reversion AND for evaluation prompt comparison
    # Note: learning_explanation_done removed - learning now runs in separate graph
]


def get_transient_defaults() -> Dict[str, Any]:
    """Get default values for all transient state fields.

    Returns a dict that can be used to reset transient state.
    This function is kept for backwards compatibility during migration.
    """
    return {
        # Progress tracking within iteration
        "task_completion_analyzed": False,
        "next_task_activated": False,
        "tasks_updated": False,
        "update_approved": False,
        # Evaluation iteration counters (evaluator-optimizer pattern)
        "reasoning_evaluation_iteration": 0,
        "task_update_evaluation_iteration": 0,
        "task_list_evaluation_iteration": 0,
        # Generation outputs (fresh each iteration)
        "generated_code": None,
        "reasoning_response": None,
        "positioning_info": None,
        # Error/retry flags (set fresh based on execution results)
        "last_execution_failed": None,
        "retry_objective": None,
        "rag_retrieval": None,
        "rag_query_assembled": False,  # Whether RAG query has been assembled with retry context
        "snippet_retrieval_query": [],  # RAG queries - empty list default, MUST reset each iteration
        # Recovery/backtracking (set by LLM analysis)
        "recovery_objective": None,
        "error_recovery_strategy": None,
        "restart_required": None,
        "backtrack_recovery_done": None,
        "cells_to_delete": None,
        "cells_deleted": None,
        # Evaluation grades and feedback (evaluator-optimizer pattern)
        "reasoning_grade": None,
        "reasoning_feedback": None,
        "task_update_grade": None,
        "task_update_feedback": None,
        "task_list_grade": None,
        "task_list_feedback": None,
        # Task update context
        "task_list_update_rule": None,
        "task_list_update_rationale": None,  # Reasoning for update, used in evaluation prompt
        "task_list_backup": None,  # Used for reversion AND for evaluation prompt comparison
    }


# =============================================================================
# STATE INITIALIZATION
# =============================================================================

def initialize_state(
    state: "KaiState",
    context: Dict[str, Any],
    checkpoint: Optional[Any] = None,
    is_first_invocation: bool = True,
    max_task_planning_iterations: int = 10,
    max_workflow_retrieval_iterations: int = 2,
) -> "KaiState":
    """Initialize or restore state for a new iteration.

    Handles both first invocation (fresh state) and continuation
    (restore from checkpoint).

    Args:
        state: Current state dict to initialize
        context: Request context with session info
        checkpoint: Existing checkpoint (if any)
        is_first_invocation: Whether this is first invocation
        max_task_planning_iterations: Max planning iterations
        max_workflow_retrieval_iterations: Max workflow retrieval iterations

    Returns:
        Updated state dict
    """
    if is_first_invocation:
        logger.debug("[STATE] First invocation - initializing state")
        rag_enabled = context.get("rag_enabled", False)

        # Initialize task management fields
        state.setdefault("task_list", {})
        state.setdefault("reference_workflow_content", {})
        state.setdefault("excluded_workflows", [])

        # Initialize RAG queries
        if "retrieval_queries" not in state:
            message = state.get("user_query", "")
            if rag_enabled and message:
                state["retrieval_queries"] = [message]
            else:
                state["retrieval_queries"] = []

        # Initialize iteration counters
        state.setdefault("workflow_retrieval_iteration", 0)
        state.setdefault("task_planning_iteration", -1)
        state.setdefault("iteration_id", 0)

        # Initialize phase tracking
        state.setdefault("planning_phase", None)
        state.setdefault("auto_mode_first_execution_done", False)
    else:
        # Restore from checkpoint
        logger.debug("[STATE] Restoring from checkpoint")
        checkpoint_values = checkpoint.values

        # Restore task_list if checkpoint has tasks
        checkpoint_tasks = checkpoint_values.get("task_list", {})
        initial_tasks = state.get("task_list", {})
        if checkpoint_tasks.get("tasks") and not initial_tasks.get("tasks"):
            state["task_list"] = checkpoint_tasks

        # Restore other fields
        restore_fields = [
            "auto_mode_first_execution_done", "planning_phase",
            "active_task_objective", "is_reasoning_task",
            "reference_workflow_content", "excluded_workflows",
            "iteration_id", "user_query",
        ]
        for field in restore_fields:
            checkpoint_val = checkpoint_values.get(field)
            initial_val = state.get(field)
            if checkpoint_val is not None and not initial_val:
                state[field] = checkpoint_val

    # Initialize required fields for prompts
    state.setdefault("execution_history", [])
    state.setdefault("conversation_history", [])
    state.setdefault("notebook_cells", [])
    state.setdefault(
        "notebook_structure", {'totalCells': 0, 'allCells': []}
    )
    state.setdefault("current_cell", "")
    state.setdefault("current_cell_index", 0)
    state.setdefault("use_critique", True)

    # Pass config to state so tools can access iteration limits
    state["max_task_planning_iterations"] = max_task_planning_iterations
    state["max_workflow_retrieval_iterations"] = max_workflow_retrieval_iterations

    return state


# =============================================================================
# SUBGRAPH STATE SCHEMAS
# =============================================================================

# -----------------------------------------------------------------------------
# Supporting TypedDicts
# -----------------------------------------------------------------------------

class TaskOutput(TypedDict):
    """Unified output from both code and reasoning generation.

    Used by generate_code_node and generate_reasoning_node to produce
    a consistent structure that format_output_node can handle uniformly.
    """
    content: str
    cell_type: Literal["code", "markdown"]
    positioning_info: Dict[str, Any]
    should_replace: bool


class ExecutionEvent(TypedDict):
    """Append-only event log entry.

    Immutable events for execution tracking. State is derived by replaying.
    """
    event_type: Literal["cell_executed", "cell_backtracked", "task_completed", "task_failed", "error_occurred"]
    timestamp: str
    payload: Dict[str, Any]


class QueryWithMetadata(TypedDict):
    """Query with iteration tracking for list reducers.

    Allows distinguishing new queries from already-processed ones.
    """
    query: str
    iteration: int
    used: bool


# =============================================================================
# FULL ORCHESTRATOR STATE
# =============================================================================

# Planning phase tracking for explicit control flow
PlanningPhase = Literal["workflow_retrieval", "task_planning", "workflow_refinement", "task_list_evaluation", "ready_to_generate", "complete"]


class KaiState(TypedDict, total=False):
    """
    State for LangGraph-based workflow orchestration.

    Uses TypedDict (not Pydantic) as LangGraph's native state format.
    All fields are optional (total=False) for flexibility.

    Reducer Strategy:
    - Annotated[..., add_messages]: LangGraph's message list reducer
    - Annotated[..., add_unique]: Accumulate unique items (deduplication)
    - Annotated[..., replace_or_keep]: Replace if not None, otherwise keep
    - No annotation: Last-write-wins (default LangGraph behavior)
    """
    # Core request fields
    user_query: str
    messages: Annotated[List[Dict[str, Any]], add_messages]

    # Session metadata
    session_id: str
    request_id: str
    autonomous_mode: bool
    rag_enabled: bool
    learning_mode: bool  # Whether to pause and explain after each cell execution
    confirm_plan: bool  # Whether to pause after planning for user approval (True in VSCode, False in Jupyter)
    error_message: str
    notebook_uri: Optional[str]  # Path to notebook for session tracking and debug folder naming
    session_timestamp: Optional[str]  # Session start time
    iteration_timestamp: Optional[str]  # Current iteration time
    iteration_counter: int  # Iteration count within session
    iteration_id: Annotated[int, replace_or_keep]  # Iteration ID for transient state tracking
    active: bool  # Whether session is active

    # Notebook context
    notebook_cells: List[Dict[str, Any]]
    notebook_structure: Dict[str, Any]
    current_cell: str
    current_cell_index: int
    execution_history: List[Any]  # Complete history sent by VSCode each call

    # Task management (replace-on-write semantics)
    task_list: Annotated[Dict[str, Any], replace_or_keep]
    active_task: Optional[Dict[str, Any]]

    # RAG and retrieval (accumulating fields)
    retrieval_queries: Annotated[List[str], add_unique]  # Accumulate unique queries
    searched_retrieval_queries: Annotated[List[str], add_unique]  # Track which queries have been searched
    snippet_retrieval_query: List[str]  # Queries for RAG snippet retrieval (from task update tool)
    rag_text: Optional[str]
    rag_retrieval: Optional[str]  # RAG retrieval result (for router check)
    reference_workflow_content: Annotated[Dict[str, Any], replace_or_keep]
    excluded_workflows: Annotated[List[str], add_unique]  # Accumulate excluded workflow IDs

    # Code generation
    intent: str
    target_cell: Optional[int]
    generated_code: Optional[str]

    # Execution tracking
    last_execution_failed: bool
    last_output: Optional[str]
    execution_result: Optional[str]  # Output from last executed cell (for learning explanations)

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
    task_list_update_rationale: Optional[str]  # Reasoning for update, used in evaluation
    task_list_backup: Optional[Dict[str, Any]]  # Pre-update snapshot for reversion and comparison
    tasks_updated: bool  # Set by autonomous_update_tasks
    update_approved: bool  # Set by task_update_evaluator

    # Positioning
    positioning_info: Optional[Dict[str, Any]]  # {target_cell: int}
    last_cell_modified_in_auto_mode: Optional[int]

    # Active task details
    active_task_objective: Optional[str]  # String description of active task
    is_reasoning_task: bool  # Whether active task is reasoning
    next_pending_task_objective: Optional[str]  # Next task for context

    # Evaluator-optimizer loops
    use_critique: bool  # Whether to use evaluation loops (default True)
    # Evaluation iteration counters
    reasoning_evaluation_iteration: int  # Counter for reasoning evaluation loop
    task_update_evaluation_iteration: int  # Counter for task update evaluation loop
    task_list_evaluation_iteration: int  # Counter for task list evaluation loop
    # Evaluation grades and feedback
    reasoning_grade: Optional[str]  # "APPROVED" or "REJECTED"
    reasoning_feedback: Optional[str]  # Feedback for reasoning improvement
    task_update_grade: Optional[str]  # "APPROVED" or "REJECTED"
    task_update_feedback: Optional[str]  # Feedback for task update improvement
    task_list_grade: Optional[str]  # "APPROVED" or "REJECTED"
    task_list_feedback: Optional[str]  # Feedback for task list improvement

    # Backtracking
    reset_tasks: Optional[List[Dict[str, Any]]]  # Tasks to reset in backtracking
    cells_to_delete: Optional[List[int]]  # Cells selected for deletion
    cells_deleted: bool  # Whether cell deletion completed
    backtrack_recovery_done: bool  # Whether backtrack_recovery completed

    # Reference workflows (persistent across iterations)
    reference_workflow_ids: Optional[str]  # IDs of selected reference workflows
    reference_workflow_percentages: Annotated[Dict[str, float], replace_or_keep]  # {full_id: percentage}

    # Orchestrator phase tracking (replaces instance variables)
    auto_mode_first_execution_done: bool  # Whether first autonomous execution completed
    auto_loop_update: Optional[str]  # "LOOP_COMPLETE", "LEARNING_MODE_PENDING", "LOOP_INCOMPLETE", or None - signals state to UI

    # Note: learning_explanation_done removed - learning now runs in separate graph
    # after code execution, not during the main execution graph

    # Planning phase tracking (for explicit control flow in planning)
    planning_phase: Optional[PlanningPhase]  # Current planning phase
    workflow_retrieval_iteration: int  # Iteration counter for workflow retrieval loop (max 2)
    task_planning_iteration: int  # Iteration counter for task generation + workflow refinement loop (max 10)
    had_retrieval_queries_before_refinement: bool  # Whether retrieval queries existed before workflow refinement cleared them

    # Internal tool communication (cleaned up after each node)
    _last_tool_result: Optional[Dict[str, Any]]  # Tool result for UI communication

    # Orchestrator config (passed to state so tools/routers can access limits)
    max_task_planning_iterations: int
    max_workflow_retrieval_iterations: int


# =============================================================================
# SECTION EXECUTION STATE
# =============================================================================

class SectionState(TypedDict, total=False):
    """State for section execution subgraph.

    This is a separate state schema from KaiState, used by the section execution
    subgraph for running a specific range of cells with error recovery.

    Fields:
    - Section definition (immutable during execution)
    - Current execution position
    - Error tracking
    - Fix history
    - Context for prompts
    """
    # Section definition (set at start, not modified)
    start_cell: int
    end_cell: int  # inclusive
    section_code: List[str]  # Code content for each cell

    # Execution tracking
    current_cell_index: int
    execution_complete: bool
    execution_success: bool

    # Error state
    last_execution_failed: bool
    current_error: Optional[str]
    error_cell_index: Optional[int]

    # Fix tracking
    fix_attempts: List[Dict[str, Any]]
    max_fix_attempts: int  # Configurable limit

    # Context for prompts (passed through from main state)
    conversation_history: List[Dict[str, Any]]
    execution_history: List[Dict[str, Any]]
    notebook_structure: Dict[str, Any]

    # Tool output (for routing)
    fix_decision: Optional[Dict[str, Any]]  # Output from section_code_review
    fix_applied: Optional[bool]  # Whether fix was successfully applied

    # Session metadata (for UI communication)
    session_id: str
    request_id: str
