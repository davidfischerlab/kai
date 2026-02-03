"""Shared fixtures and test utilities for core tests.

This module provides:
- State builders for creating test states
- Mock fixtures for LLM interfaces, notebook selectors, etc.
- Pytest configuration and markers
"""

import os
import pytest
from typing import Dict, Any, Optional, List
from unittest.mock import Mock, MagicMock, AsyncMock, patch


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "llm: mark test as requiring LLM (ollama)")

    # Disable debug prompts during tests
    os.environ['KAI_DEBUG_PROMPTS'] = 'false'

    # Disable Turbo mode during tests
    os.environ['KAI_DISABLE_TURBO'] = 'true'


# =============================================================================
# State Builders
# =============================================================================

def create_base_state(
    user_query: str = "Test query",
    task_list: Optional[Dict[str, Any]] = None,
    session_id: str = "test_session",
    autonomous_mode: bool = True,
    **overrides
) -> Dict[str, Any]:
    """Create a complete state dict with all required fields for testing.

    This is the primary state builder - use this for most tests.

    Args:
        user_query: The user's query
        task_list: Task list dict (default: empty)
        session_id: Session ID for metadata
        autonomous_mode: Whether in autonomous mode
        **overrides: Additional state fields to override defaults

    Returns:
        Fully initialized state dict ready for tool/router testing
    """
    state = {
        # Core fields
        "autonomous_mode": autonomous_mode,
        "user_query": user_query,

        # Session metadata
        "session_id": session_id,
        "request_id": "test_request",
        "session_timestamp": "2025-01-01_12-00-00",
        "iteration_timestamp": "12-01-00",
        "iteration_counter": 1,
        "notebook_uri": "file:///test.ipynb",
        "active": True,

        # Notebook fields
        "notebook_cells": [],
        "notebook_structure": {"totalCells": 3, "allCells": ["# Cell 1", "# Cell 2", "# Cell 3"]},
        "current_cell": "",
        "current_cell_index": 0,

        # History fields
        "execution_history": [],
        "conversation_history": [],

        # Execution state
        "last_execution_failed": False,
        "last_output": None,
        "last_cell_modified_in_auto_mode": None,

        # Task management
        "task_list": task_list or {"tasks": []},
        "active_task": None,
        "active_task_objective": None,
        "task_completion_analyzed": False,
        "tasks_updated": False,
        "update_approved": False,
        "next_task_activated": False,

        # Code generation
        "generated_code": None,
        "target_cell": None,
        "positioning_info": None,
        "is_reasoning_task": False,

        # Reasoning
        "reasoning_response": None,
        "reasoning_grade": None,
        "reasoning_feedback": None,
        "reasoning_evaluation_iteration": 0,

        # RAG/Workflow
        "rag_enabled": False,
        "rag_text": None,
        "rag_retrieval": None,
        "rag_query_assembled": False,
        "retrieval_queries": [],
        "reference_workflow_content": {},
        "reference_workflow_ids": None,
        "reference_workflow_internal_ids": None,
        "reference_workflow_annotation": "",

        # Error handling
        "error_message": "",
        "error_context": None,
        "error_recovery_strategy": None,
        "retry_objective": None,
        "recovery_objective": None,

        # Backtracking
        "backtracking_context": None,
        "backtrack_recovery_done": False,
        "cells_to_delete": None,
        "cells_deleted": False,

        # Autonomous mode iteration flags
        "autonomous_mode_continue": False,
        "auto_mode_first_execution_done": False,
        "confirm_plan": False,

        # Evaluation loop state
        "task_list_backup": None,
        "task_list_update_rule": None,
        "task_update_evaluation_iteration": 0,
        "task_update_grade": None,
        "task_list_grade": None,

        # Planning state
        "planning_phase": None,
        "task_planning_iteration": 0,
        "workflow_retrieval_iteration": 0,
        "had_retrieval_queries_before_refinement": False,
        "use_critique": True,

        # Exclusions
        "excluded_workflows": [],
    }

    # Apply overrides
    state.update(overrides)
    return state


def create_task_list(
    num_tasks: int = 3,
    active_index: int = 0,
    task_prefix: str = "Task",
    include_reasoning: bool = False
) -> Dict[str, Any]:
    """Create a task list with specified number of tasks.

    Args:
        num_tasks: Number of tasks to create
        active_index: Which task should be active (0-indexed, -1 for none)
        task_prefix: Prefix for task descriptions
        include_reasoning: If True, make some tasks reasoning tasks

    Returns:
        Task list dict with tasks array
    """
    tasks = []
    for i in range(num_tasks):
        if active_index >= 0 and i < active_index:
            status = "completed"
        elif i == active_index:
            status = "active"
        else:
            status = "pending"

        task_desc = f"{task_prefix} {i + 1}"
        if include_reasoning and i % 3 == 2:  # Every 3rd task is reasoning
            task_desc = f"[reasoning] {task_desc}"

        tasks.append({
            "id": i + 1,
            "task": task_desc,
            "status": status
        })

    return {"tasks": tasks}


def create_positioning_info(
    target_cell: int = 5,
    positioning: str = "below"
) -> Dict[str, Any]:
    """Create positioning info for code generation.

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


def create_error_state(
    error_message: str = "Test error",
    error_type: str = "ValueError",
    **overrides
) -> Dict[str, Any]:
    """Create a state representing an execution error.

    Args:
        error_message: The error message
        error_type: The type of error
        **overrides: Additional state overrides

    Returns:
        State dict with error fields populated
    """
    return create_base_state(
        last_execution_failed=True,
        error_message=f"{error_type}: {error_message}",
        **overrides
    )


def create_rag_state(
    reference_workflows: Optional[Dict[str, str]] = None,
    retrieval_queries: Optional[List[str]] = None,
    **overrides
) -> Dict[str, Any]:
    """Create a state with RAG/workflow content.

    Args:
        reference_workflows: Dict of workflow_id -> content
        retrieval_queries: List of retrieval queries
        **overrides: Additional state overrides

    Returns:
        State dict with RAG fields populated
    """
    return create_base_state(
        rag_enabled=True,
        reference_workflow_content=reference_workflows or {
            "workflow_1": "> Notebook ID: test/workflow1.ipynb\nCell 0: import scanpy as sc",
            "workflow_2": "> Notebook ID: test/workflow2.ipynb\nCell 0: import anndata as ad"
        },
        retrieval_queries=retrieval_queries or ["query 1", "query 2"],
        **overrides
    )


# =============================================================================
# Mock Fixtures
# =============================================================================

@pytest.fixture
def mock_llm_interface():
    """Create a mock LLM interface for unit tests.

    This mock does NOT call any real LLM - use for fast unit tests.
    For LLM integration tests, use llm_interface fixture instead.
    """
    mock_llm = Mock()
    mock_provider = Mock()
    mock_provider.provider_name = "test_provider"
    mock_provider.model = "test_model"
    mock_provider.use_structured_output = True
    mock_provider.generate_structured = AsyncMock()
    mock_provider.generate = AsyncMock(return_value="Mock response")
    mock_llm.get_llm_for_tool.return_value = mock_provider
    mock_llm.get_reasoning_for_tool.return_value = "detailed"
    return mock_llm


@pytest.fixture
def mock_notebook_selector():
    """Create a mock notebook selector for workflow tests."""
    mock_selector = Mock()

    # Default notebook content
    notebooks = {
        "workflow_1": {
            "cells": [
                {"content": "import scanpy as sc", "order": 0, "section": "main"},
                {"content": "adata = sc.read_h5ad('data.h5ad')", "order": 1, "section": "main"},
            ],
            "metadata": {
                "source_repository": "test-org/test-repo",
                "workflow_filename": "workflow_1.ipynb",
                "title": "Test Workflow 1"
            }
        },
        "workflow_2": {
            "cells": [
                {"content": "import pandas as pd", "order": 0, "section": "main"},
            ],
            "metadata": {
                "source_repository": "test-org/test-repo",
                "workflow_filename": "workflow_2.ipynb",
                "title": "Test Workflow 2"
            }
        }
    }

    def get_selected(notebook_ids):
        return {nb_id: notebooks[nb_id] for nb_id in notebook_ids if nb_id in notebooks}

    mock_selector.get_selected_notebook_content = get_selected
    mock_selector.format_notebook_context.return_value = "Formatted workflow context"
    mock_selector.format_notebook_context_dict.return_value = {
        "workflow_1": "> Notebook ID: test-org/test-repo/workflow_1.ipynb\nContent here",
        "workflow_2": "> Notebook ID: test-org/test-repo/workflow_2.ipynb\nContent here"
    }

    return mock_selector


@pytest.fixture
def mock_summary_search():
    """Create a mock summary search for RAG tests."""
    mock_search = Mock()
    mock_search.search_summaries.return_value = [
        {
            "notebook_id": "workflow_1",
            "summary": "Single-cell analysis workflow",
            "metadata": {"source_repository": "test/repo1"},
            "similarity_score": 0.95
        },
        {
            "notebook_id": "workflow_2",
            "summary": "Data preprocessing workflow",
            "metadata": {"source_repository": "test/repo2"},
            "similarity_score": 0.87
        }
    ]
    return mock_search


@pytest.fixture
def llm_interface():
    """Create a REAL LLM interface for integration tests.

    Uses qwen3:0.6b for fast tests. Mark tests using this with @pytest.mark.llm.
    """
    from kai.core.agent import KaiAgent
    agent = KaiAgent(llm_provider='ollama', model="qwen3:0.6b")
    return agent.llm_interface


# =============================================================================
# Router Test Helpers
# =============================================================================

def mock_send_message(msg: str) -> None:
    """Mock send_message callback for router testing."""
    pass


# =============================================================================
# Assertion Helpers
# =============================================================================

def assert_tool_result_valid(result, expected_type=None):
    """Assert that a tool result has valid structure.

    Args:
        result: The ToolResult to validate
        expected_type: Optional expected ToolOutputType
    """
    assert result is not None, "Result should not be None"
    assert hasattr(result, 'output_ui'), "Result should have output_ui"
    assert hasattr(result, 'output_workflow'), "Result should have output_workflow"
    assert hasattr(result, 'output_type'), "Result should have output_type"

    if expected_type:
        from kai.core.tools.base import ToolOutputType
        assert result.output_type == expected_type, \
            f"Expected output_type {expected_type}, got {result.output_type}"


def assert_code_output_valid(result):
    """Assert that a code generation result has valid structure."""
    assert_tool_result_valid(result)

    if isinstance(result.output_ui, dict):
        assert 'code' in result.output_ui, "Code output should have 'code' field"
        assert 'positioning_info' in result.output_ui, "Should have positioning_info"
        assert 'should_replace' in result.output_ui, "Should have should_replace"
        assert 'cell_type' in result.output_ui, "Should have cell_type"
        assert len(result.output_ui['code']) > 0, "Code should not be empty"
