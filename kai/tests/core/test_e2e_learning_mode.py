"""End-to-end tests for learning mode flow.

These tests mock LLM responses and verify the full orchestrator flow
from VSCode input to VSCode output, capturing all messages sent to UI.

ARCHITECTURE NOTE (post-VSCode-trigger-refactor):
- Learning explanation is TRIGGERED BY VSCODE, not run automatically in Python
- Main graph sends LEARNING_MODE_PENDING signal (not LEARNING_LOOP)
- VSCode receives LEARNING_MODE_PENDING, executes the cell, then calls
  agent.run_learning_explanation() if execution succeeded
- This ensures learning explanation only runs AFTER actual cell execution succeeds

Test scenarios:
1. Planning phase: Same in learning and non-learning mode
2. Execution phase: Main graph sends LEARNING_MODE_PENDING in learning mode
3. Learning explanation: Tested via run_learning_explanation_for_vscode()
"""

import json
import pytest
from io import StringIO
from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch
from dataclasses import dataclass, field

from langgraph.checkpoint.memory import MemorySaver

from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator
from kai.core.orchestration.ui_communicator import UICommunicator
from kai.core.tools.base import ToolResult, ToolOutputType


# =============================================================================
# Test Infrastructure
# =============================================================================

@dataclass
class CapturedMessages:
    """Captures all messages sent to VSCode."""
    console_messages: List[str] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    workflow_results: List[Dict[str, Any]] = field(default_factory=list)

    def clear(self):
        self.console_messages.clear()
        self.tool_results.clear()
        self.workflow_results.clear()

    def get_tool_names(self) -> List[str]:
        """Get list of tool names that sent results."""
        return [r.get("tool_name") for r in self.tool_results]

    def has_learning_explanation(self) -> bool:
        """Check if learning_explanation was called by checking output_ui."""
        for r in self.tool_results:
            output_ui = r.get("output_ui", {})
            if isinstance(output_ui, dict):
                if output_ui.get("isLearningExplanation"):
                    return True
        return False

    def get_tool_types(self) -> List[str]:
        """Get tool types by examining output_ui content."""
        types = []
        for r in self.tool_results:
            output_ui = r.get("output_ui", {})
            if isinstance(output_ui, dict):
                if output_ui.get("isLearningExplanation"):
                    types.append("learning_explanation")
                elif "tasks" in output_ui:
                    types.append("task_list")
                elif "code" in output_ui:
                    types.append("code_generation")
                else:
                    types.append("unknown")
            else:
                types.append("text_response")
        return types

    def get_workflow_state(self) -> str:
        """Get final workflow state (LOOP_COMPLETE, LOOP_INCOMPLETE, LEARNING_LOOP, etc)."""
        if not self.workflow_results:
            return None
        last = self.workflow_results[-1]
        return last.get("auto_loop_update") or last.get("regular_chat_update")


class MockLLMProvider:
    """Mock LLM provider that returns predefined responses."""

    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.use_structured_output = True
        self.responses = {}  # tool_name -> response
        self.call_count = {}  # tool_name -> count

    def set_response(self, tool_name: str, response: Any):
        """Set the response for a specific tool."""
        self.responses[tool_name] = response
        self.call_count[tool_name] = 0

    async def generate_structured(self, prompt: str, schema: Any, **kwargs) -> Any:
        """Return mocked structured response based on tool context."""
        # Use tool_name which is passed by all tools
        tool_name = kwargs.get("tool_name", "unknown")

        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1

        if tool_name in self.responses:
            response = self.responses[tool_name]
            # If response is a callable, call it
            if callable(response):
                return response(prompt, schema, **kwargs)
            return response

        # Default: raise error to make missing mocks obvious
        raise ValueError(f"No mock response set for tool: {tool_name}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Return mocked unstructured response."""
        # UnstructuredPromptTool doesn't pass tool_name, use task_type
        tool_name = kwargs.get("tool_name")
        task_type = kwargs.get("task_type", "unknown")

        # Try tool_name first, then task_type
        key = tool_name if tool_name else task_type
        self.call_count[key] = self.call_count.get(key, 0) + 1

        if tool_name and tool_name in self.responses:
            return self.responses[tool_name]
        if task_type in self.responses:
            return self.responses[task_type]
        # Check if prompt contains learning explanation keywords
        if "learning" in prompt.lower() and "learning_explanation" in self.responses:
            return self.responses["learning_explanation"]
        raise ValueError(f"No mock response for tool={tool_name}, task_type={task_type}")


class MockLLMInterface:
    """Mock LLM interface that returns MockLLMProvider."""

    def __init__(self):
        self.provider = MockLLMProvider()

    def get_llm_for_tool(self, tool: Any) -> MockLLMProvider:
        return self.provider

    def get_reasoning_for_tool(self, tool: Any) -> str:
        return "detailed"

    def set_response(self, tool_name: str, response: Any):
        """Convenience method to set response on provider."""
        self.provider.set_response(tool_name, response)


@pytest.fixture
def captured_messages():
    """Fixture that captures all VSCode messages."""
    captured = CapturedMessages()

    # Patch UICommunicator to capture messages
    original_send_console = UICommunicator.send_console_message
    original_send_tool_result = UICommunicator.send_tool_result
    original_send_workflow_result = UICommunicator.send_workflow_result

    def mock_send_console(self, message: str):
        captured.console_messages.append(message)

    async def mock_send_tool_result(self, result: ToolResult, context):
        captured.tool_results.append({
            "tool_name": context.request_id if hasattr(context, 'request_id') else "unknown",
            "result": result,
            "output_ui": result.output_ui,
            "output_type": result.output_type.value,
        })

    async def mock_send_workflow_result(self, field: str, state: str):
        captured.workflow_results.append({field: state})

    UICommunicator.send_console_message = mock_send_console
    UICommunicator.send_tool_result = mock_send_tool_result
    UICommunicator.send_workflow_result = mock_send_workflow_result

    yield captured

    # Restore original methods
    UICommunicator.send_console_message = original_send_console
    UICommunicator.send_tool_result = original_send_tool_result
    UICommunicator.send_workflow_result = original_send_workflow_result


@pytest.fixture
def mock_llm():
    """Fixture that provides mock LLM interface."""
    return MockLLMInterface()


@pytest.fixture
def mock_knowledge_base():
    """Fixture that provides mock knowledge base."""
    mock_kb = Mock()
    mock_kb.search_summaries = AsyncMock(return_value=[])
    return mock_kb


def create_orchestrator(mock_llm, mock_kb) -> LangGraphOrchestrator:
    """Create orchestrator with mocked dependencies."""
    return LangGraphOrchestrator(
        llm_interface=mock_llm,
        knowledge_base=mock_kb,
        ui_communicator=UICommunicator(),
        graph_recursion_limit=50,
        max_task_planning_iterations=3,
        checkpointer=MemorySaver(),
    )


def create_vscode_context(
    session_id: str = "test_session",
    autonomous_mode: bool = True,
    learning_mode: bool = False,
    confirm_plan: bool = True,
    task_list: Dict = None,
    **overrides
) -> Dict[str, Any]:
    """Create context dict simulating VSCode input."""
    context = {
        "session_metadata": {
            "session_id": session_id,
            "request_id": "test_request",
        },
        "autonomous_mode": autonomous_mode,
        "learning_mode": learning_mode,
        "confirm_plan": confirm_plan,
        "task_list": task_list or {"tasks": []},
        "notebook_structure": {"totalCells": 3, "allCells": ["# Cell 1", "# Cell 2", "# Cell 3"]},
        "current_cell": "",
        "current_cell_index": 0,
        "execution_history": [],
        "conversation_history": [],
        "last_execution_failed": False,
        "error_message": "",
    }
    context.update(overrides)
    return context


# =============================================================================
# Mock Response Factories
# =============================================================================

def create_task_list_response(tasks: List[Dict] = None):
    """Create mock response for task_list_generation tool."""
    from kai.core.tools.task_list_generation import (
        TaskListGeneration, TaskItem
    )

    if tasks is None:
        tasks = [
            {"id": 1, "task": "Load the data", "status": "pending"},
            {"id": 2, "task": "Process the data", "status": "pending"},
        ]

    return TaskListGeneration(
        tasks=[TaskItem(**t) for t in tasks],
        retrieval_queries=[],
    )


def create_task_list_evaluation_response(approved: bool = True):
    """Create mock response for task_list_evaluator tool."""
    from kai.core.tools.task_list_evaluator import TaskListEvaluation
    return TaskListEvaluation(
        grade="APPROVED" if approved else "REJECTED",
        feedback="" if approved else "Needs improvement"
    )


def create_learning_explanation_response():
    """Create mock response for learning_explanation tool."""
    return "This step loaded your data using scanpy's read function."


def create_cell_positioning_response():
    """Create mock response for cell_positioning tool."""
    from kai.core.tools.cell_positioning import CellPositioning
    return CellPositioning(
        target_cell=1,
        reasoning="After the imports cell",
    )


def create_code_generation_response():
    """Create mock response for code_generation_with_guidance tool.

    Returns a string with code block - the tool extracts code from it.
    """
    return """```python
import scanpy as sc
adata = sc.read_h5ad('data.h5ad')
```"""


# =============================================================================
# E2E Tests - Architecture Verification
# =============================================================================

class TestLearningModeArchitecture:
    """Verify the new learning mode architecture."""

    def test_execution_graph_identical_regardless_of_learning_mode(self):
        """The execution graph should be identical regardless of learning_mode.

        After refactor, learning explanation runs in a SEPARATE graph,
        so the execution tools list should NOT include learning_explanation.
        """
        from kai.core.orchestration.graphs.execution import AUTONOMOUS_TOOLS

        assert "learning_explanation" not in AUTONOMOUS_TOOLS, (
            "learning_explanation should NOT be in execution graph tools - "
            "it runs in separate learning graph"
        )

    def test_learning_graph_exists_and_exports(self):
        """Verify learning graph builder exists and is exported."""
        from kai.core.orchestration.graphs import build_learning_graph

        # Should be importable and callable
        assert callable(build_learning_graph)

    def test_learning_mode_pending_signal_documented_in_state(self):
        """LEARNING_MODE_PENDING should be a valid auto_loop_update value."""
        from kai.core.orchestration.state import KaiState
        import inspect

        source = inspect.getsource(KaiState)
        assert "LEARNING_MODE_PENDING" in source, (
            "LEARNING_MODE_PENDING should be documented in KaiState as valid signal"
        )


class TestLearningModeFirstIteration:
    """Test learning mode on first iteration (after planning).

    ARCHITECTURE NOTE: Learning explanation does NOT run during first iteration.
    It runs AFTER code execution in a separate graph.
    """

    @pytest.mark.asyncio
    async def test_first_iteration_does_not_call_learning_explanation(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        E2E: Learning mode first iteration should:
        1. Run planning (task_list_generation)
        2. Activate first task (mark_next_task_active)
        3. Exit to show user (complete) - NO learning_explanation during planning

        Learning explanation runs AFTER code execution, not during planning.
        """
        # Setup mock responses
        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )
        # NOT setting learning_explanation response - it shouldn't be called

        # Create orchestrator
        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        # Create VSCode input context
        context = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,  # VSCode mode - pause after plan
            autonomous_mode=True,
            use_critique=False,  # Skip evaluator loop
        )

        # Execute
        result = await orchestrator.process_request("Generate analysis code", context)

        # Verify learning_explanation was NOT called during first iteration
        assert not captured_messages.has_learning_explanation(), (
            f"learning_explanation should NOT be called during first iteration. "
            f"It runs after execution. Tool types: {captured_messages.get_tool_types()}"
        )

        # Verify task was activated
        task_list = result.get("task_list", {})
        tasks = task_list.get("tasks", [])
        assert any(t.get("status") == "active" for t in tasks), (
            f"First task should be active. Tasks: {tasks}"
        )

    @pytest.mark.asyncio
    async def test_learning_mode_and_non_learning_mode_identical_first_iteration(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        E2E: First iteration should be IDENTICAL regardless of learning_mode.

        The execution graph is the same - only the post-execution handling differs.
        """
        # Setup mock responses
        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )

        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        # Test with learning_mode=True
        context_learning = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
            session_id="test_learning",
        )
        result_learning = await orchestrator.process_request(
            "Generate analysis code", context_learning
        )
        learning_task_count = len(
            result_learning.get("task_list", {}).get("tasks", [])
        )

        captured_messages.clear()

        # Test with learning_mode=False (need new orchestrator due to state)
        orchestrator2 = create_orchestrator(mock_llm, mock_knowledge_base)
        context_non_learning = create_vscode_context(
            learning_mode=False,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
            session_id="test_non_learning",
        )
        result_non_learning = await orchestrator2.process_request(
            "Generate analysis code", context_non_learning
        )
        non_learning_task_count = len(
            result_non_learning.get("task_list", {}).get("tasks", [])
        )

        # Both should produce same task count
        assert learning_task_count == non_learning_task_count, (
            "Task list should be identical regardless of learning_mode"
        )


class TestLearningModeContinue:
    """Test learning mode on continue (after user clicks Continue).

    ARCHITECTURE NOTE: Learning explanation runs AFTER code execution
    succeeds, in a separate learning graph.
    """

    @pytest.mark.skip(reason="Requires full execution mock including cell execution")
    @pytest.mark.asyncio
    async def test_learning_loop_returned_after_successful_execution(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        E2E: After successful code execution in learning mode:
        1. Code executes successfully
        2. Orchestrator runs learning graph
        3. Learning explanation is generated
        4. Returns LEARNING_LOOP (not LOOP_INCOMPLETE)

        This tests the post-execution learning flow.
        TODO: Implement when we have cell execution mocking.
        """
        pass


class TestLearningExplanationExecutionResult:
    """Test that execution_result is properly passed to learning explanation.

    BUG FIX: The learning explanation was receiving empty execution_result
    because the context was built BEFORE cell execution, not AFTER.

    The fix ensures context.executionResult is updated with the actual
    cell output right before calling requestLearningExplanation().
    """

    @pytest.mark.asyncio
    async def test_execution_result_passed_to_learning_graph(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Test that run_learning_explanation_for_vscode receives and uses
        the execution_result from context.

        This tests the Python side - verifying that when VSCode calls
        agent.run_learning_explanation(context) with an executionResult,
        that result ends up in the state used for the learning graph.
        """
        # Setup: First run planning to get a valid session state
        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )
        mock_llm.set_response(
            "learning_explanation", create_learning_explanation_response()
        )

        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        # Run planning first to establish session state
        planning_context = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
            session_id="test_exec_result",
        )
        await orchestrator.process_request("Generate code", planning_context)

        captured_messages.clear()

        # Now simulate what VSCode does: call run_learning_explanation_for_vscode
        # with the execution result from the cell
        EXPECTED_OUTPUT = "✅ Local CellTypist models:\n  - Immune_All_High.pkl\n  - Immune_All_Low.pkl"

        learning_context = {
            'execution_result': EXPECTED_OUTPUT,  # This is the key value being tested
            'execution_history': [],
            'conversation_history': [],
            'notebook_structure': {'totalCells': 3, 'allCells': []},
            'learning_mode': True,
        }

        session_metadata = {'session_id': 'test_exec_result'}

        # Capture what state is passed to the learning graph
        captured_state = {}
        original_run_learning_graph = orchestrator._run_learning_graph

        async def mock_run_learning_graph(state, session_metadata):
            captured_state.update(state)
            # Still run the original to get the full flow
            return await original_run_learning_graph(state, session_metadata)

        orchestrator._run_learning_graph = mock_run_learning_graph

        # Call the method that VSCode calls after successful execution
        await orchestrator.run_learning_explanation_for_vscode(
            context=learning_context,
            session_metadata=session_metadata
        )

        # CRITICAL ASSERTION: execution_result must be in the state
        # that gets passed to the learning graph
        assert 'execution_result' in captured_state, (
            "execution_result must be passed to learning graph state"
        )
        assert captured_state['execution_result'] == EXPECTED_OUTPUT, (
            f"execution_result should be the actual cell output. "
            f"Expected: {EXPECTED_OUTPUT!r}, "
            f"Got: {captured_state.get('execution_result')!r}"
        )

    @pytest.mark.asyncio
    async def test_execution_result_empty_string_still_passed(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Test that even empty execution_result is passed (not None).

        Some cells may have no output but should still be explained.
        """
        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )
        mock_llm.set_response(
            "learning_explanation", create_learning_explanation_response()
        )

        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        # Run planning first
        planning_context = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
            session_id="test_empty_output",
        )
        await orchestrator.process_request("Generate code", planning_context)

        # Simulate cell with no output (e.g., import statements)
        learning_context = {
            'execution_result': '',  # Empty but not None
            'execution_history': [],
            'conversation_history': [],
            'notebook_structure': {'totalCells': 3, 'allCells': []},
            'learning_mode': True,
        }

        session_metadata = {'session_id': 'test_empty_output'}

        captured_state = {}
        original_run_learning_graph = orchestrator._run_learning_graph

        async def mock_run_learning_graph(state, session_metadata):
            captured_state.update(state)
            return await original_run_learning_graph(state, session_metadata)

        orchestrator._run_learning_graph = mock_run_learning_graph

        await orchestrator.run_learning_explanation_for_vscode(
            context=learning_context,
            session_metadata=session_metadata
        )

        # Should be empty string, not None or missing
        assert captured_state.get('execution_result') == '', (
            f"Empty execution_result should be preserved. "
            f"Got: {captured_state.get('execution_result')!r}"
        )


class TestLearningExplanationDebugLogging:
    """Test that debug logging works for learning_explanation tool.

    The learning graph runs separately from the main graph, and the state
    must contain all required fields for prompt debug logging to work.
    """

    @pytest.mark.asyncio
    async def test_learning_graph_state_has_debug_logging_fields(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify that the state passed to learning graph has all fields
        required for prompt debug logging.

        Required fields: session_id, session_timestamp, iteration_timestamp,
        autonomous_mode, notebook_uri
        """
        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )
        mock_llm.set_response(
            "learning_explanation", create_learning_explanation_response()
        )

        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        # Run planning first to establish session state with proper metadata
        planning_context = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
            session_id="test_debug_fields",
        )
        await orchestrator.process_request("Generate code", planning_context)

        # Capture what state is passed to learning graph
        captured_state = {}
        original_run_learning_graph = orchestrator._run_learning_graph

        async def mock_run_learning_graph(state, session_metadata):
            captured_state.update(state)
            return await original_run_learning_graph(state, session_metadata)

        orchestrator._run_learning_graph = mock_run_learning_graph

        # Call learning explanation with execution result
        learning_context = {
            'execution_result': 'Test output',
            'execution_history': [],
            'conversation_history': [],
            'notebook_structure': {'totalCells': 3, 'allCells': []},
            'learning_mode': True,
        }

        # Session metadata with all fields needed for debug logging
        # In production, these are set by agent.py
        session_metadata = {
            'session_id': 'test_debug_fields',
            'session_timestamp': '2026-01-22_17-00-00-000',
            'iteration_timestamp': '17-00-00-000',
            'notebook_uri': 'file:///test/notebook.ipynb',
        }

        await orchestrator.run_learning_explanation_for_vscode(
            context=learning_context,
            session_metadata=session_metadata
        )

        # These fields are required for _log_prompt in prompt_base.py
        required_fields = [
            'session_id',
            'session_timestamp',
            'iteration_timestamp',
            'autonomous_mode',
            'notebook_uri',
        ]

        missing_fields = []
        for field in required_fields:
            if field not in captured_state:
                missing_fields.append(field)
            elif captured_state[field] is None and field not in ['notebook_uri']:
                # notebook_uri is allowed to be None
                missing_fields.append(f"{field} (is None)")

        assert not missing_fields, (
            f"Learning graph state missing required debug logging fields: {missing_fields}. "
            f"Available fields: {list(captured_state.keys())}"
        )

    @pytest.mark.asyncio
    async def test_session_metadata_merged_into_state_values(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify that session_metadata fields are merged into state_values.

        This tests the fix where session_metadata fields were not being
        merged into the state retrieved from the checkpointer.
        """
        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )
        mock_llm.set_response(
            "learning_explanation", create_learning_explanation_response()
        )

        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        # Run planning first
        planning_context = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
            session_id="test_merge_metadata",
        )
        await orchestrator.process_request("Generate code", planning_context)

        # Capture state
        captured_state = {}
        original_run_learning_graph = orchestrator._run_learning_graph

        async def mock_run_learning_graph(state, session_metadata):
            captured_state.update(state)
            return await original_run_learning_graph(state, session_metadata)

        orchestrator._run_learning_graph = mock_run_learning_graph

        # Session metadata with specific values to verify they're merged
        EXPECTED_SESSION_TIMESTAMP = "2026-01-22_18-30-45-123"
        EXPECTED_ITERATION_TIMESTAMP = "18-30-45-123"
        EXPECTED_NOTEBOOK_URI = "file:///Users/test/my_notebook.ipynb"

        session_metadata = {
            'session_id': 'test_merge_metadata',
            'session_timestamp': EXPECTED_SESSION_TIMESTAMP,
            'iteration_timestamp': EXPECTED_ITERATION_TIMESTAMP,
            'notebook_uri': EXPECTED_NOTEBOOK_URI,
        }

        await orchestrator.run_learning_explanation_for_vscode(
            context={
                'execution_result': 'Some output',
                'execution_history': [],
                'conversation_history': [],
                'notebook_structure': {'totalCells': 3, 'allCells': []},
                'learning_mode': True,
            },
            session_metadata=session_metadata
        )

        # Verify session_metadata values were merged into state
        assert captured_state.get('session_timestamp') == EXPECTED_SESSION_TIMESTAMP, (
            f"session_timestamp should be merged from session_metadata. "
            f"Expected: {EXPECTED_SESSION_TIMESTAMP}, Got: {captured_state.get('session_timestamp')}"
        )
        assert captured_state.get('iteration_timestamp') == EXPECTED_ITERATION_TIMESTAMP, (
            f"iteration_timestamp should be merged from session_metadata. "
            f"Expected: {EXPECTED_ITERATION_TIMESTAMP}, Got: {captured_state.get('iteration_timestamp')}"
        )
        assert captured_state.get('notebook_uri') == EXPECTED_NOTEBOOK_URI, (
            f"notebook_uri should be merged from session_metadata. "
            f"Expected: {EXPECTED_NOTEBOOK_URI}, Got: {captured_state.get('notebook_uri')}"
        )


class TestAgentLearningExplanationFlow:
    """Test the full learning explanation flow with execution_result.

    These tests verify that execution_result properly flows through
    to the learning_explanation prompt template.
    """

    @pytest.mark.asyncio
    async def test_execution_result_reaches_prompt_template(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Test that execution_result in learning graph state reaches the
        prompt template as last_execution_output.

        This captures the actual prompt sent to LLM to verify the
        execution output appears in the prompt text.
        """
        # Capture the actual prompt sent to LLM
        captured_prompts = []
        original_generate = mock_llm.provider.generate

        async def capture_prompt(prompt, **kwargs):
            captured_prompts.append(prompt)
            return create_learning_explanation_response()

        mock_llm.provider.generate = capture_prompt

        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )

        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        # Run planning first
        planning_context = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
            session_id="test_prompt_output",
        )
        await orchestrator.process_request("Generate code", planning_context)

        # Now call learning explanation with execution result
        EXPECTED_OUTPUT = "CellTypist models:\n  - Immune_All_High.pkl"

        learning_context = {
            'execution_result': EXPECTED_OUTPUT,
            'execution_history': [],
            'conversation_history': [],
            'notebook_structure': {'totalCells': 3, 'allCells': []},
            'learning_mode': True,
        }

        session_metadata = {
            'session_id': 'test_prompt_output',
            'session_timestamp': '2026-01-22_18-00-00-000',
            'iteration_timestamp': '18-00-00-000',
            'notebook_uri': 'file:///test/notebook.ipynb',
        }

        await orchestrator.run_learning_explanation_for_vscode(
            context=learning_context,
            session_metadata=session_metadata
        )

        # Check that the execution output appears in the prompt
        assert len(captured_prompts) > 0, "LLM should have been called"
        prompt = captured_prompts[-1]

        assert EXPECTED_OUTPUT in prompt, (
            f"execution_result should appear in prompt. "
            f"Expected to find: {EXPECTED_OUTPUT!r}\n"
            f"Prompt contains: {prompt[:500]}..."
        )


class TestToolCallSequence:
    """Test that tools are called in the correct sequence."""

    @pytest.mark.asyncio
    async def test_first_iteration_sequence_no_learning_explanation(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify tool sequence for first iteration (learning mode or not):
        1. task_list_generation (planning)
        2. mark_next_task_active
        3. complete (end) - NO learning_explanation

        Learning explanation runs after execution, not during planning.
        """
        mock_llm.set_response(
            "task_list_generation", create_task_list_response()
        )
        # NOT mocking learning_explanation - it shouldn't be called

        orchestrator = create_orchestrator(mock_llm, mock_knowledge_base)

        context = create_vscode_context(
            learning_mode=True,
            confirm_plan=True,
            autonomous_mode=True,
            use_critique=False,
        )

        await orchestrator.process_request("Generate code", context)

        # Check console messages for tool completion
        console_msgs = "\n".join(captured_messages.console_messages)

        # Verify planning was called
        assert "task_list_generation" in console_msgs.lower() or "planning" in console_msgs.lower(), (
            f"Planning should be logged. Console: {console_msgs}"
        )

        # Verify learning_explanation was NOT called
        assert not captured_messages.has_learning_explanation(), (
            "learning_explanation should NOT be called during first iteration"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
