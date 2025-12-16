"""Test helpers for core orchestration tests."""

from typing import Dict, Any, Optional
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs


def create_full_execution_context(
    user_query: str = "Test query",
    task_list: Optional[Dict[str, Any]] = None,
    session_id: str = "test_session",
    autonomous_mode: bool = True,
    **context_overrides
) -> ExecutionContext:
    """
    Create a complete ExecutionContext with all required fields for testing.

    Args:
        user_query: The user's query
        task_list: Task list dict (default: empty)
        session_id: Session ID for metadata
        autonomous_mode: Whether in autonomous mode
        **context_overrides: Additional context fields to override defaults

    Returns:
        Fully initialized ExecutionContext ready for tool testing
    """
    # Base context with ALL required fields
    context = {
        # Core fields
        "autonomous_mode": autonomous_mode,
        "user_query": user_query,

        # Notebook fields
        "notebook_cells": [],
        "notebook_structure": {"totalCells": 0, "allCells": []},
        "current_cell": "",
        "current_cell_index": 0,

        # History fields
        "execution_history": [],
        "conversation_history": [],

        # Execution state
        "last_execution_failed": False,
        "just_executed": False,
        "last_output": None,

        # Task management
        "task_list": task_list or {},
        "active_task": None,
        "active_task_objective": None,

        # Code generation
        "generated_code": None,
        "target_cell": None,
        "positioning_info": None,

        # RAG
        "rag_enabled": False,
        "rag_text": None,
        "retrieval_queries": [],

        # Error handling
        "error_message": "",
        "error_context": None,
        "backtracking_context": None,

        # Reference workflows
        "reference_workflow_content": {},
        "reference_workflow_ids": None,

        # Flags
        "excluded_workflows": [],

        # Override with provided values
        **context_overrides
    }

    # Build ExecutionInputs
    inputs = ExecutionInputs(
        user_query=user_query,
        context=context,
        task_list=task_list or {},
        backtracking_context=None,
        excluded_workflows=[]
    )

    # Build session metadata
    session_metadata = {
        "session_id": session_id,
        "request_id": "test_request",
        "autonomous_mode": autonomous_mode,
        "session_timestamp": "2025-01-01_12-00-00",
        "iteration_timestamp": "12-01-00",
        "iteration_counter": 1,
        "notebook_uri": "file:///test.ipynb",
        "active": True,
    }

    return ExecutionContext(
        inputs=inputs,
        session_metadata=session_metadata
    )


def create_task_list(num_tasks: int = 3, active_index: int = 0) -> Dict[str, Any]:
    """
    Create a task list with specified number of tasks.

    Args:
        num_tasks: Number of tasks to create
        active_index: Which task should be active (0-indexed)

    Returns:
        Task list dict
    """
    tasks = []
    for i in range(num_tasks):
        status = "active" if i == active_index else "pending"
        tasks.append({
            "id": i + 1,
            "task": f"Task {i + 1}",
            "status": status
        })

    return {"tasks": tasks}


def create_positioning_info(target_cell: int = 5, positioning: str = "below") -> Dict[str, Any]:
    """
    Create positioning info for code generation.

    Args:
        target_cell: Target cell index
        positioning: Position relative to target ("above", "below", "replace")

    Returns:
        Positioning info dict
    """
    return {
        "target_cell_index": target_cell,
        "positioning": positioning
    }
