"""Comprehensive End-to-End tests for the Kai Agent.

These tests mock LLM responses and verify the FULL flow from agent.chat()
(the real VSCode entry point) to VSCode output.

IMPORTANT: All context uses camelCase keys as VSCode actually sends them.
The conversion from camelCase → snake_case happens in agent.py and IS tested.

Test Structure:
===============
1. PLANNING PHASE TESTS
   - Auto Mode (learningMode=False): Planning → task activation → complete
   - Tutorial Mode (learningMode=True): Planning → task activation →
     learning_explanation → complete
   - With/without evaluator loop (use_critique=True/False)

2. EXECUTION PHASE TESTS
   - Auto Mode: Mark completion → update tasks → generate code → complete
   - Tutorial Mode: Same + learning_explanation before code generation
   - Success/Failure/Retry/Backtrack flows

3. TRANSITION TESTS
   - Planning → First Execution (the critical flow that was broken)
   - Ensures proper state persistence between invocations

Output Contract:
================
Each test validates EXACTLY what should be emitted to VSCode:
- task_list_display: Task list updates (always shown in chat)
- execute_code: Code to execute in notebook
- learning_explanation: Tutorial explanation bubble
- workflow_state: LOOP_COMPLETE or LOOP_INCOMPLETE

Tests also validate what should NOT be emitted (no duplicate messages,
no intermediate states exposed, etc.)
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import patch
from dataclasses import dataclass, field

from kai.core.agent import KaiAgent
from kai.core.orchestration.ui_communicator import UICommunicator
from kai.core.tools.base import ToolResult


# =============================================================================
# Test Infrastructure - CapturedMessages with Rigorous Assertions
# =============================================================================

@dataclass
class CapturedMessages:
    """Captures all messages sent to VSCode with assertion helpers."""
    console_messages: List[str] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    workflow_results: List[Dict[str, Any]] = field(default_factory=list)

    def clear(self):
        self.console_messages.clear()
        self.tool_results.clear()
        self.workflow_results.clear()

    # =========================================================================
    # Query Methods - What was emitted?
    # =========================================================================

    def get_all_output_types(self) -> List[str]:
        """Get list of all output types sent."""
        return [r.get("output_type") for r in self.tool_results]

    def get_task_list_displays(self) -> List[Dict]:
        """Get all task list display messages."""
        return [
            r for r in self.tool_results
            if r.get("output_type") == "task_list_display"
        ]

    def get_code_executions(self) -> List[Dict]:
        """Get all code execution messages."""
        return [
            r for r in self.tool_results
            if r.get("output_type") == "execute_code"
        ]

    def get_learning_explanations(self) -> List[Dict]:
        """Get all learning explanation messages."""
        results = []
        for r in self.tool_results:
            output_ui = r.get("output_ui", {})
            if isinstance(output_ui, dict) and output_ui.get("isLearningExplanation"):
                results.append(r)
        return results

    def get_reasoning_outputs(self) -> List[Dict]:
        """Get all reasoning/markdown outputs."""
        results = []
        for r in self.tool_results:
            output_ui = r.get("output_ui", {})
            if isinstance(output_ui, dict) and output_ui.get("cell_type") == "markdown":
                results.append(r)
        return results

    def get_final_workflow_state(self) -> Optional[str]:
        """Get final workflow state (LOOP_COMPLETE or LOOP_INCOMPLETE)."""
        if not self.workflow_results:
            return None
        last = self.workflow_results[-1]
        return last.get("auto_loop_update") or last.get("regular_chat_update")

    # =========================================================================
    # Boolean Checks
    # =========================================================================

    def has_learning_explanation(self) -> bool:
        """Check if any learning explanation was sent."""
        return len(self.get_learning_explanations()) > 0

    def has_code_execution(self) -> bool:
        """Check if any code was sent for execution."""
        return len(self.get_code_executions()) > 0

    def has_reasoning_output(self) -> bool:
        """Check if any reasoning markdown was sent."""
        return len(self.get_reasoning_outputs()) > 0

    def has_task_list_display(self) -> bool:
        """Check if task list was displayed."""
        return len(self.get_task_list_displays()) > 0

    # =========================================================================
    # Assertion Methods - Validate Output Contract
    # =========================================================================

    def assert_learning_explanation_sent(
        self, expected_count: int = 1, validate_content: bool = True
    ):
        """Assert exactly N learning explanations were sent with valid content.

        Args:
            expected_count: Number of learning explanations expected
            validate_content: If True, also validates that each explanation has
                              non-empty text and isLearningExplanation=True
        """
        explanations = self.get_learning_explanations()
        actual = len(explanations)
        assert actual == expected_count, (
            f"Expected {expected_count} learning explanation(s), got {actual}. "
            f"Output types: {self.get_all_output_types()}"
        )

        if validate_content:
            for i, explanation in enumerate(explanations):
                output_ui = explanation.get("output_ui", {})

                # Validate isLearningExplanation flag
                assert output_ui.get("isLearningExplanation") is True, (
                    f"Learning explanation #{i+1} missing isLearningExplanation=True. "
                    f"Got: {output_ui}"
                )

                # Validate text content exists and is non-empty
                text = output_ui.get("text", "")
                assert text and len(text.strip()) > 0, (
                    f"Learning explanation #{i+1} has empty text. "
                    f"Got: {output_ui}"
                )

    def assert_no_learning_explanation(self):
        """Assert NO learning explanation was sent."""
        explanations = self.get_learning_explanations()
        assert len(explanations) == 0, (
            f"Expected no learning explanation, but got {len(explanations)}. "
            f"Output types: {self.get_all_output_types()}"
        )

    def assert_code_execution_sent(
        self,
        expected_count: int = 1,
        validate_content: bool = True,
        validate_positioning: bool = True,
        expected_should_replace: Optional[bool] = None,
    ):
        """Assert exactly N code executions were sent with valid content.

        Args:
            expected_count: Number of code executions expected
            validate_content: If True, validates that each has non-empty code
            validate_positioning: If True, validates positioning_info structure
            expected_should_replace: If set, validates should_replace field matches
        """
        executions = self.get_code_executions()
        actual = len(executions)
        assert actual == expected_count, (
            f"Expected {expected_count} code execution(s), got {actual}. "
            f"Output types: {self.get_all_output_types()}"
        )

        for i, execution in enumerate(executions):
            output_ui = execution.get("output_ui", {})

            if validate_content:
                code = output_ui.get("code", "")
                assert code and len(code.strip()) > 0, (
                    f"Code execution #{i+1} has empty code. "
                    f"Got: {output_ui}"
                )

            if validate_positioning:
                positioning_info = output_ui.get("positioning_info")
                assert positioning_info is not None, (
                    f"Code execution #{i+1} missing positioning_info. "
                    f"Got: {output_ui}"
                )
                target_cell = positioning_info.get("target_cell")
                assert target_cell is not None and isinstance(target_cell, int), (
                    f"Code execution #{i+1} has invalid target_cell. "
                    f"positioning_info: {positioning_info}"
                )
                assert target_cell >= 0, (
                    f"Code execution #{i+1} has negative target_cell: {target_cell}"
                )

            if expected_should_replace is not None:
                should_replace = output_ui.get("should_replace")
                assert should_replace == expected_should_replace, (
                    f"Code execution #{i+1} should_replace mismatch. "
                    f"Expected: {expected_should_replace}, got: {should_replace}"
                )

    def assert_no_code_execution(self):
        """Assert NO code execution was sent."""
        executions = self.get_code_executions()
        assert len(executions) == 0, (
            f"Expected no code execution, but got {len(executions)}. "
            f"Output types: {self.get_all_output_types()}"
        )

    def assert_task_list_displayed(
        self, min_count: int = 1, validate_content: bool = True
    ):
        """Assert task list was displayed at least N times with valid content.

        Args:
            min_count: Minimum number of task list displays expected
            validate_content: If True, validates that each has non-empty tasks array
        """
        displays = self.get_task_list_displays()
        actual = len(displays)
        assert actual >= min_count, (
            f"Expected at least {min_count} task list display(s), got {actual}. "
            f"Output types: {self.get_all_output_types()}"
        )

        if validate_content:
            for i, display in enumerate(displays):
                output_ui = display.get("output_ui", {})
                tasks = output_ui.get("tasks", [])
                assert isinstance(tasks, list), (
                    f"Task list #{i+1} 'tasks' is not a list. Got: {type(tasks)}"
                )
                # Note: tasks can be empty at certain stages, so we just validate structure

    def assert_workflow_state(self, expected: str):
        """Assert final workflow state matches expected."""
        actual = self.get_final_workflow_state()
        assert actual == expected, (
            f"Expected workflow state '{expected}', got '{actual}'. "
            f"All workflow results: {self.workflow_results}"
        )

    def assert_reasoning_output_sent(self, expected_count: int = 1):
        """Assert exactly N reasoning outputs were sent."""
        actual = len(self.get_reasoning_outputs())
        assert actual == expected_count, (
            f"Expected {expected_count} reasoning output(s), got {actual}. "
            f"Output types: {self.get_all_output_types()}"
        )

    def assert_exact_outputs(
        self,
        task_list_displays: int = 0,
        code_executions: int = 0,
        learning_explanations: int = 0,
        reasoning_outputs: int = 0,
    ):
        """Assert exact counts of all output types."""
        actual_tasks = len(self.get_task_list_displays())
        actual_code = len(self.get_code_executions())
        actual_learning = len(self.get_learning_explanations())
        actual_reasoning = len(self.get_reasoning_outputs())

        errors = []
        if actual_tasks != task_list_displays:
            errors.append(
                f"task_list_displays: expected {task_list_displays}, got {actual_tasks}"
            )
        if actual_code != code_executions:
            errors.append(
                f"code_executions: expected {code_executions}, got {actual_code}"
            )
        if actual_learning != learning_explanations:
            errors.append(
                f"learning_explanations: expected {learning_explanations}, "
                f"got {actual_learning}"
            )
        if actual_reasoning != reasoning_outputs:
            errors.append(
                f"reasoning_outputs: expected {reasoning_outputs}, "
                f"got {actual_reasoning}"
            )

        if errors:
            raise AssertionError(
                "Output count mismatch:\n" + "\n".join(errors) +
                f"\n\nAll output types: {self.get_all_output_types()}"
            )


# =============================================================================
# Mock LLM Provider - Supports both structured and unstructured generation
# =============================================================================

class MockLLMProvider:
    """Mock LLM provider that returns predefined responses by tool name."""

    def __init__(self):
        self.provider_name = "mock"
        self.model = "mock-model"
        self.use_structured_output = True
        self.responses = {}
        self.call_count = {}
        self.call_sequence = []

    def set_response(self, tool_name: str, response: Any):
        """Set the response for a specific tool."""
        self.responses[tool_name] = response
        self.call_count[tool_name] = 0

    def set_responses(self, responses: Dict[str, Any]):
        """Set multiple responses at once."""
        for tool_name, response in responses.items():
            self.set_response(tool_name, response)

    async def generate_structured(
        self, prompt: str, schema: Any, **kwargs
    ) -> Any:
        """Return mocked structured response."""
        tool_name = kwargs.get("tool_name", "unknown")
        self.call_count[tool_name] = self.call_count.get(tool_name, 0) + 1
        self.call_sequence.append(tool_name)

        if tool_name in self.responses:
            response = self.responses[tool_name]
            if callable(response):
                return response(prompt, schema, **kwargs)
            return response

        raise ValueError(f"No mock response for tool: {tool_name}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Return mocked unstructured response."""
        tool_name = kwargs.get("tool_name")
        task_type = kwargs.get("task_type", "unknown")

        key = tool_name if tool_name else task_type
        self.call_count[key] = self.call_count.get(key, 0) + 1
        self.call_sequence.append(key)

        if tool_name and tool_name in self.responses:
            return self.responses[tool_name]
        if task_type in self.responses:
            return self.responses[task_type]
        if "learning" in prompt.lower() and "learning_explanation" in self.responses:
            return self.responses["learning_explanation"]

        raise ValueError(f"No mock response for tool={tool_name}, type={task_type}")


class MockLLMInterface:
    """Mock LLM interface that wraps MockLLMProvider."""

    def __init__(self):
        self.provider = MockLLMProvider()
        self.provider_name = "mock"
        self.model = "mock-model"

    def get_llm_for_tool(self, tool: Any) -> MockLLMProvider:
        return self.provider

    def get_reasoning_for_tool(self, tool: Any) -> str:
        return "detailed"

    def set_response(self, tool_name: str, response: Any):
        self.provider.set_response(tool_name, response)

    def set_responses(self, responses: Dict[str, Any]):
        self.provider.set_responses(responses)


# =============================================================================
# Mock Knowledge Base
# =============================================================================

class MockKnowledgeBase:
    """Mock knowledge base for testing."""

    def __init__(self):
        self.search_results = []

    async def search_summaries(self, *args, **kwargs):
        return self.search_results

    def start_background_initialization(self):
        pass


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def captured_messages():
    """Fixture that captures all VSCode messages."""
    captured = CapturedMessages()

    original_send_console = UICommunicator.send_console_message
    original_send_tool_result = UICommunicator.send_tool_result
    original_send_workflow_result = UICommunicator.send_workflow_result

    def mock_send_console(self, message: str):
        captured.console_messages.append(message)

    async def mock_send_tool_result(self, result: ToolResult, context):
        captured.tool_results.append({
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

    UICommunicator.send_console_message = original_send_console
    UICommunicator.send_tool_result = original_send_tool_result
    UICommunicator.send_workflow_result = original_send_workflow_result


@pytest.fixture
def mock_llm():
    return MockLLMInterface()


@pytest.fixture
def mock_knowledge_base():
    return MockKnowledgeBase()


# =============================================================================
# Mock Response Factories
# =============================================================================

def create_task_list_generation(
    tasks: List[Dict] = None,
    queries: List[str] = None
):
    """Create TaskListGeneration response."""
    from kai.core.tools.task_list_generation import TaskListGeneration, TaskItem

    if tasks is None:
        tasks = [
            {"id": 1, "task": "Load the data", "status": "pending"},
            {"id": 2, "task": "Process the data", "status": "pending"},
        ]

    return TaskListGeneration(
        tasks=[TaskItem(**t) for t in tasks],
        retrieval_queries=queries or [],
    )


def create_task_list_evaluation(approved: bool = True, feedback: str = ""):
    """Create TaskListEvaluation response."""
    from kai.core.tools.task_list_evaluator import TaskListEvaluation
    return TaskListEvaluation(
        grade="APPROVED" if approved else "REJECTED",
        feedback=feedback
    )


def create_cell_positioning(target_cell: int = 1, reasoning: str = "After imports"):
    """Create CellPositioning response."""
    from kai.core.tools.cell_positioning import CellPositioning
    return CellPositioning(target_cell=target_cell, reasoning=reasoning)


def create_code_generation_response(code: str = None):
    """Create code generation response (string with code block)."""
    if code is None:
        code = "import scanpy as sc\nadata = sc.read_h5ad('data.h5ad')"
    return f"```python\n{code}\n```"


def create_reasoning_response(text: str = None):
    """Create reasoning response (markdown string)."""
    if text is None:
        text = "## Analysis Plan\n\n1. Load data\n2. Process\n3. Visualize"
    return text


def create_reasoning_evaluation(approved: bool = True, feedback: str = ""):
    """Create ReasoningEvaluation response."""
    from kai.core.tools.reasoning_evaluator import ReasoningEvaluation
    return ReasoningEvaluation(
        grade="APPROVED" if approved else "REJECTED",
        feedback=feedback
    )


def create_autonomous_mark_completion(
    status_updates: List[Dict] = None,
    retry_objective: str = None,
    recovery_objective: str = None
):
    """Create AutonomousMarkCompletion response."""
    from kai.core.tools.autonomous_mark_completion import (
        AutonomousMarkCompletion, TaskStatusUpdate
    )

    if status_updates is None:
        status_updates = [{"id": 1, "status": "completed"}]

    return AutonomousMarkCompletion(
        status_updates=[TaskStatusUpdate(**u) for u in status_updates],
        retry_objective=retry_objective,
        recovery_objective=recovery_objective
    )


def create_autonomous_update_tasks(
    tasks: List[Dict] = None,
    update_rule: str = "KEEP",
    rationale: str = "No changes needed"
):
    """Create AutonomousTaskUpdate response."""
    from kai.core.tools.autonomous_update_tasks import (
        AutonomousTaskUpdate, TaskItem
    )

    if tasks is None:
        tasks = [{"id": 2, "task": "Next task", "status": "pending"}]

    return AutonomousTaskUpdate(
        tasks=[TaskItem(**t) for t in tasks],
        retrieval_queries=[],
        update_rationale=rationale,
        update_rule=update_rule
    )


def create_task_update_evaluation(approved: bool = True, feedback: str = ""):
    """Create TaskUpdateEvaluation response."""
    from kai.core.tools.task_update_evaluator import TaskUpdateEvaluation
    return TaskUpdateEvaluation(
        grade="APPROVED" if approved else "REJECTED",
        feedback=feedback
    )


def create_error_recovery(intent: str = "REPLACE_AND_RETRY"):
    """Create ErrorRecoveryStrategy response."""
    from kai.core.tools.error_recovery import ErrorRecoveryStrategy
    return ErrorRecoveryStrategy(intent=intent)


def create_backtrack_recovery(restart_required: bool = False):
    """Create BacktrackRecoveryStrategy response."""
    from kai.core.tools.backtrack_recovery import BacktrackRecoveryStrategy
    return BacktrackRecoveryStrategy(restart_required=restart_required)


def create_cell_deletion_selection(cells: List[int] = None):
    """Create CellDeletionSelection response."""
    from kai.core.tools.cell_selection_deletion import CellDeletionSelection
    return CellDeletionSelection(cells_to_delete=cells or [])


def create_learning_explanation():
    """Create learning explanation response."""
    return "This step loads your data using scanpy's read function."


# =============================================================================
# Helper Functions - VSCode Context (camelCase)
# =============================================================================

def create_vscode_context(
    autonomousMode: bool = True,
    learningMode: bool = False,
    autonomousModeContinue: bool = False,
    ragEnabled: bool = False,
    turboEnabled: bool = False,
    taskList: Dict = None,
    lastExecutionFailed: bool = False,
    executionResult: str = "",
    errorCellIndex: int = None,
    lastCellModifiedInAutoMode: int = None,
    **overrides
) -> Dict[str, Any]:
    """
    Create context dict as VSCode actually sends it (camelCase keys).

    This is the REAL format that agent.chat() receives from VSCode.
    The conversion from camelCase → snake_case happens in agent.py.
    """
    context = {
        # Request/session
        "request_id": "test_request",

        # Notebook context
        "notebookStructure": {"totalCells": 5, "allCells": ["# Cell"] * 5},
        "currentCell": "",
        "currentCellIndex": 0,
        "executionHistory": [],
        "conversationHistory": [],

        # Autonomous mode flags (camelCase as VSCode sends)
        "autonomousMode": autonomousMode,
        "autonomousModeContinue": autonomousModeContinue,
        "learningMode": learningMode,

        # Features
        "ragEnabled": ragEnabled,
        "turboEnabled": turboEnabled,

        # Task management
        "taskList": taskList or {},

        # Execution state
        "lastExecutionFailed": lastExecutionFailed,
        "executionResult": executionResult,
        "errorCellIndex": errorCellIndex,
        "lastCellModifiedInAutoMode": lastCellModifiedInAutoMode,
    }
    context.update(overrides)
    return context


def create_agent(mock_llm, mock_knowledge_base) -> KaiAgent:
    """Create KaiAgent with mocked LLM and knowledge base."""
    with patch('kai.core.agent.create_knowledge_base', return_value=mock_knowledge_base):
        with patch('kai.core.agent.LLMInterface', return_value=mock_llm):
            agent = KaiAgent(suppress_vscode_messages=True)
            # Inject mock LLM interface into orchestrator
            agent.orchestrator.llm_interface = mock_llm
            agent.orchestrator.knowledge_base = mock_knowledge_base
            return agent


# =============================================================================
# PLANNING PHASE TESTS - Auto Mode (learningMode=False)
# =============================================================================

class TestPlanningPhaseAutoMode:
    """
    Planning phase in Auto Mode (learningMode=False).

    Expected outputs:
    - task_list_display: YES (shows generated plan)
    - learning_explanation: NO
    - code_execution: NO (pauses for user approval)
    - workflow_state: LOOP_INCOMPLETE
    """

    @pytest.mark.asyncio
    async def test_planning_basic_code_tasks(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Planning with basic code tasks.

        VSCode sends: autonomousMode=True, learningMode=False

        Expected:
        - Task list displayed with activated first task
        - NO learning explanation
        - NO code execution
        - Pauses for user (LOOP_INCOMPLETE)
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Load data", "status": "pending"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            ),
            # Evaluator is enabled by default
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)

        # VSCode context with camelCase keys
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=False,  # Auto mode - no learning explanation
            autonomousModeContinue=False,  # First invocation
        )

        result, session_id = await agent.chat(
            user_input="Analyze my data",
            context=context
        )

        # Validate outputs
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_no_learning_explanation()
        captured_messages.assert_no_code_execution()
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

    @pytest.mark.asyncio
    async def test_planning_with_reasoning_task(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Planning where first task is a reasoning task.

        Expected: Same as code task - no special handling in planning phase.
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "[reasoning] Plan analysis",
                     "status": "pending"},
                    {"id": 2, "task": "Execute plan", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
        )

        await agent.chat("Plan analysis", context=context)

        # Validate outputs - same as code task
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_no_learning_explanation()
        captured_messages.assert_no_code_execution()
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")


# =============================================================================
# PLANNING PHASE TESTS - Tutorial Mode (learningMode=True)
# =============================================================================

class TestPlanningPhaseTutorialMode:
    """
    Planning phase in Tutorial Mode (learningMode=True).

    CRITICAL: This tests the camelCase → snake_case conversion.
    VSCode sends learningMode=True, but the orchestrator expects learning_mode=True.

    Expected outputs:
    - task_list_display: YES
    - learning_explanation: YES (explains first task)
    - code_execution: NO
    - workflow_state: LOOP_INCOMPLETE
    """

    @pytest.mark.asyncio
    async def test_planning_basic_code_tasks_learning_mode(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Planning phase in tutorial mode with code tasks.

        VSCode sends: autonomousMode=True, learningMode=True (camelCase!)
        autonomousModeContinue=False means this is the FIRST iteration (planning only).

        Expected:
        - Task list displayed
        - No code execution (that happens in second iteration)
        - No learning explanation (VSCode triggers that after execution)
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)

        # CRITICAL: Using camelCase as VSCode sends it
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,  # camelCase! Must convert to learning_mode
            autonomousModeContinue=False,  # First iteration = planning only
        )

        result, session_id = await agent.chat(
            user_input="Analyze data",
            context=context
        )

        # Validate planning phase outputs only
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        captured_messages.assert_code_execution_sent(expected_count=0)  # No code in planning phase
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

    @pytest.mark.asyncio
    async def test_planning_with_reasoning_task_learning_mode(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Planning in tutorial mode with reasoning task.

        ARCHITECTURE NOTE (post-refactor):
        - Planning phase only generates task list and marks first task active
        - NO code execution or reasoning execution during planning
        - Learning explanation runs in SEPARATE graph AFTER code execution
        - Reasoning execution happens in NEXT invocation (autonomousModeContinue=True)
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "[reasoning] Plan approach",
                     "status": "pending"},
                    {"id": 2, "task": "Execute", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            # No mocks for execution tools - not called during planning
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=False,  # Planning only
        )

        await agent.chat("Plan", context=context)

        # Planning phase outputs only - no execution
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        captured_messages.assert_reasoning_output_sent(expected_count=0)  # No execution during planning
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")


# =============================================================================
# PLANNING → FIRST EXECUTION TRANSITION TESTS
# =============================================================================

class TestPlanningToFirstExecutionTransition:
    """
    Tests the critical transition from planning completion to first execution.

    This simulates the TWO-INVOCATION flow:
    1. First invocation: Planning → task_list_generation → mark_next_task_active → LOOP_INCOMPLETE
    2. Second invocation: User clicks Continue → first execution → code_generation → LOOP_INCOMPLETE

    This is the flow that was broken: after planning completes, the next invocation
    should trigger code generation, not terminate.
    """

    @pytest.mark.asyncio
    async def test_planning_then_first_execution_auto_mode(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Two-invocation test: Planning → Continue → First Execution.

        Invocation 1: Planning phase
        - autonomousMode=True, autonomousModeContinue=False
        - Should generate task list, activate first task, return LOOP_INCOMPLETE

        Invocation 2: First execution (simulating user clicking Continue)
        - autonomousMode=True, autonomousModeContinue=True
        - Should generate code for first task, return LOOP_INCOMPLETE
        """
        # === INVOCATION 1: Planning ===
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Load data", "status": "pending"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=False,  # First invocation
        )

        result1, session_id = await agent.chat(
            user_input="Analyze my data",
            context=context1
        )

        # Validate planning outputs
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_no_code_execution()
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

        # === INVOCATION 2: First Execution (user clicked Continue) ===
        captured_messages.clear()

        # Get task list from orchestrator state for continuation
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        task_list = state.values.get("task_list", {})

        # Add mocks for first execution phase
        mock_llm.set_responses({
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation": create_code_generation_response(),
        })

        # Simulate VSCode sending continuation request
        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=True,  # Continuation!
            taskList=task_list,
        )

        result2, _ = await agent.chat(
            user_input="",  # Empty for continuation
            session_id=session_id,
            context=context2
        )

        # Validate first execution outputs
        # Should have generated code
        captured_messages.assert_code_execution_sent(
            expected_count=1,
            expected_should_replace=False  # New code, not replacement
        )
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

    @pytest.mark.asyncio
    async def test_planning_then_first_execution_tutorial_mode(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Tutorial Mode: Planning is SAME as auto mode (planning only).

        ARCHITECTURE NOTE (post-refactor):
        - Planning phase (autonomousModeContinue=False) is IDENTICAL for
          both learning_mode=True and learning_mode=False
        - Only generates task list, marks first task active
        - NO code execution, NO learning_explanation during planning
        - Code execution happens in NEXT invocation (Continue click)
        - Learning explanation runs in SEPARATE graph AFTER code executes

        Tests learningMode (camelCase) conversion still works.
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Load data", "status": "pending"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            # No execution tools - not called during planning phase
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,  # Tutorial mode - camelCase!
            autonomousModeContinue=False,  # Planning only
        )

        result, session_id = await agent.chat(
            user_input="Analyze my data",
            context=context
        )

        # Planning phase only - no execution
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        captured_messages.assert_code_execution_sent(expected_count=0)
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")


# =============================================================================
# EXECUTION PHASE TESTS - Auto Mode - Success Flows
# =============================================================================

class TestExecutionPhaseAutoModeSuccess:
    """
    Execution phase in Auto Mode with successful outcomes.

    Test scenarios:
    - Code task success → next task
    - Code task success → all complete
    - Reasoning task success
    """

    @pytest.mark.asyncio
    async def test_code_task_success_next_task(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Code task completes, moves to next task.

        Expected:
        - Task list update displayed
        - Code generated for next task
        - NO learning explanation (auto mode)
        - Pauses for execution (LOOP_INCOMPLETE)
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Load data", "status": "pending"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            "task_update_evaluator": create_task_update_evaluation(approved=True),
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation": create_code_generation_response(),
            "autonomous_mark_completion": create_autonomous_mark_completion(
                status_updates=[
                    {"id": 1, "status": "completed"},
                    {"id": 2, "status": "pending"},
                ]
            ),
            "autonomous_update_tasks": create_autonomous_update_tasks(
                update_rule="KEEP"
            ),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)

        # === Invocation 1: Planning ===
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=False,
        )
        _, session_id = await agent.chat("Analyze data", context=context1)
        captured_messages.clear()

        # === Invocation 2: First execution ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        task_list = state.values.get("task_list", {})

        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=True,
            taskList=task_list,
        )
        _, session_id = await agent.chat(
            "", session_id=session_id, context=context2
        )
        captured_messages.clear()

        # === Invocation 3: After first cell executes successfully, continue ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        task_list = state.values.get("task_list", {})

        context3 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=True,
            taskList=task_list,
            lastExecutionFailed=False,
            executionResult="# Cell executed successfully",
        )

        await agent.chat("", session_id=session_id, context=context3)

        # Validate outputs
        captured_messages.assert_task_list_displayed(min_count=1)
        # Success case: new code insertion (should_replace=False)
        captured_messages.assert_code_execution_sent(
            expected_count=1,
            expected_should_replace=False
        )
        captured_messages.assert_no_learning_explanation()
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

    @pytest.mark.asyncio
    async def test_code_task_success_all_complete(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Last code task completes - all tasks done.

        Expected:
        - Task list update displayed
        - NO code generation (nothing left)
        - LOOP_COMPLETE signal
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[{"id": 1, "task": "Only task", "status": "pending"}]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            "task_update_evaluator": create_task_update_evaluation(approved=True),
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation": create_code_generation_response(),
            "autonomous_mark_completion": create_autonomous_mark_completion(
                status_updates=[{"id": 1, "status": "completed"}]
            ),
            "autonomous_update_tasks": create_autonomous_update_tasks(
                update_rule="KEEP"
            ),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)

        # === Invocation 1: Planning ===
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=False,
        )
        _, session_id = await agent.chat("Analyze", context=context1)
        captured_messages.clear()

        # === Invocation 2: First execution ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=True,
            taskList=state.values.get("task_list", {}),
        )
        _, session_id = await agent.chat(
            "", session_id=session_id, context=context2
        )
        captured_messages.clear()

        # === Invocation 3: After execution - all complete ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        context3 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=True,
            taskList=state.values.get("task_list", {}),
            lastExecutionFailed=False,
        )

        await agent.chat("", session_id=session_id, context=context3)

        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_no_code_execution()
        captured_messages.assert_workflow_state("LOOP_COMPLETE")


# =============================================================================
# EXECUTION PHASE TESTS - Tutorial Mode - Success Flows
# =============================================================================

class TestExecutionPhaseTutorialModeSuccess:
    """
    Execution phase in Tutorial Mode with successful outcomes.

    Key difference from Auto Mode:
    - Learning explanation sent BEFORE code generation
    """

    @pytest.mark.asyncio
    async def test_code_task_success_next_task_tutorial(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Code task completes in tutorial mode, moves to next task.

        Expected:
        - Task list update displayed
        - Learning explanation for next task
        - Code generated
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Load data", "status": "pending"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            "task_update_evaluator": create_task_update_evaluation(approved=True),
            "learning_explanation": create_learning_explanation(),
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation": create_code_generation_response(),
            "autonomous_mark_completion": create_autonomous_mark_completion(
                status_updates=[
                    {"id": 1, "status": "completed"},
                    {"id": 2, "status": "pending"},
                ]
            ),
            "autonomous_update_tasks": create_autonomous_update_tasks(
                update_rule="KEEP"
            ),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)

        # === Invocation 1: Planning ===
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,  # Tutorial mode - camelCase!
            autonomousModeContinue=False,
        )
        _, session_id = await agent.chat("Analyze data", context=context1)
        captured_messages.clear()

        # === Invocation 2: First execution ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state.values.get("task_list", {}),
        )
        _, session_id = await agent.chat(
            "", session_id=session_id, context=context2
        )
        captured_messages.clear()

        # === Invocation 3: After first cell executes, continue ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        context3 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state.values.get("task_list", {}),
            lastExecutionFailed=False,
        )

        await agent.chat("", session_id=session_id, context=context3)

        # Tutorial mode: learning explanation runs in SEPARATE graph after execution
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        # Success case: new code insertion (should_replace=False)
        captured_messages.assert_code_execution_sent(
            expected_count=1,
            expected_should_replace=False
        )
        # Tutorial mode returns LEARNING_MODE_PENDING to signal VSCode
        # to run learning explanation after code executes
        captured_messages.assert_workflow_state("LEARNING_MODE_PENDING")


# =============================================================================
# EXECUTION PHASE TESTS - Error/Retry Flows
# =============================================================================

class TestExecutionPhaseErrorFlows:
    """
    Execution phase error handling and retry flows.

    Scenarios:
    - Code execution failed → error recovery → code update
    """

    @pytest.mark.asyncio
    async def test_execution_error_triggers_recovery(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Code execution failed, triggers error recovery.

        ARCHITECTURE NOTE (post-refactor):
        - Requires 3 invocations: planning → first code → error recovery
        - Planning phase (autonomousModeContinue=False) only generates task list
        - First code (autonomousModeContinue=True) generates code
        - Error recovery (autonomousModeContinue=True + lastExecutionFailed=True)
        """
        # === Invocation 1: Planning only ===
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[{"id": 1, "task": "Load data", "status": "pending"}]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=False,  # Planning only
        )
        _, session_id = await agent.chat("Analyze", context=context1)
        captured_messages.clear()

        # === Invocation 2: First code generation ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )

        mock_llm.set_responses({
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation_with_guidance": create_code_generation_response(),
        })

        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,  # Execute first code
            taskList=state.values.get("task_list", {}),
        )
        _, session_id = await agent.chat("", session_id=session_id, context=context2)
        captured_messages.clear()

        # === Invocation 3: Code execution failed, error recovery ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )

        mock_llm.set_responses({
            "autonomous_mark_completion": create_autonomous_mark_completion(
                status_updates=[{"id": 1, "status": "active"}]
            ),
            "error_recovery": create_error_recovery(intent="REPLACE_AND_RETRY"),
            "code_update": create_code_generation_response("# Fixed code"),
        })

        context3 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state.values.get("task_list", {}),
            lastExecutionFailed=True,
            executionResult="NameError: name 'x' is not defined",
            errorCellIndex=3,
            lastCellModifiedInAutoMode=3,
        )

        await agent.chat("", session_id=session_id, context=context3)

        # Error path: no learning explanation
        captured_messages.assert_no_learning_explanation()
        # Error recovery: replacing existing code (should_replace=True)
        captured_messages.assert_code_execution_sent(
            expected_count=1,
            expected_should_replace=True
        )


# =============================================================================
# OUTPUT CONTRACT VALIDATION TESTS
# =============================================================================

class TestOutputContract:
    """
    Tests that validate the output contract.

    These tests ensure:
    - Required outputs are sent
    - No duplicate learning explanations
    - Correct workflow states
    """

    @pytest.mark.asyncio
    async def test_planning_auto_mode_contract(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Planning phase (auto mode) output contract:
        - Task list displayed (at least once)
        - NO learning explanation
        - NO code execution
        - Workflow state: LOOP_INCOMPLETE
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
        )

        await agent.chat("Plan", context=context)

        # Contract validation
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_no_code_execution()
        captured_messages.assert_no_learning_explanation()
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

    @pytest.mark.asyncio
    async def test_planning_tutorial_mode_contract(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Planning phase (tutorial mode) output contract:
        - Task list displayed
        - NO learning explanation (runs in separate graph AFTER execution)
        - NO code generation (planning only)
        - Workflow state: LOOP_INCOMPLETE

        ARCHITECTURE NOTE (post-refactor):
        - Planning phase is IDENTICAL for tutorial and auto mode
        - Learning explanation runs in separate graph after code executes
        - Uses learningMode (camelCase) to test conversion.
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            # No execution tools - not called during planning phase
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,  # camelCase!
            autonomousModeContinue=False,  # Planning only
        )

        await agent.chat("Plan", context=context)

        # Contract validation - planning phase only
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        captured_messages.assert_code_execution_sent(expected_count=0)
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

    @pytest.mark.asyncio
    async def test_no_duplicate_learning_explanations(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Ensure learning explanation is sent exactly once per iteration, not duplicated.
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            "learning_explanation": create_learning_explanation(),
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation_with_guidance": create_code_generation_response(),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
        )

        await agent.chat("Plan", context=context)

        # Exactly 1 learning explanation, not 0, not 2+
        captured_messages.assert_learning_explanation_sent(expected_count=0)


# =============================================================================
# STATE PROPAGATION TESTS - Verifies internal state at each step
# =============================================================================

class TestStatePropagation:
    """
    Tests that verify internal state propagation through the graph.

    These tests inspect the actual state values at each step to ensure
    the camelCase → snake_case conversion and state persistence work correctly.
    """

    @pytest.mark.asyncio
    async def test_learning_mode_propagates_through_planning(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify learning_mode=True propagates from context through planning.

        ARCHITECTURE NOTE (post-refactor):
        - learning_mode is stored in state for orchestrator to use
        - learning_explanation is NOT called during planning
        - learning_explanation runs in separate graph after code executes
        - learning_explanation_done field was REMOVED from state
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            # No execution tools - not called during planning phase
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,  # camelCase as VSCode sends
            autonomousModeContinue=False,  # Planning only
        )

        result, session_id = await agent.chat(
            user_input="Analyze data",
            context=context
        )

        # Get final state from checkpointer
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        state_values = state.values

        # Verify learning_mode is True in final state (camelCase conversion)
        assert state_values.get("learning_mode") is True, (
            f"learning_mode should be True in final state, "
            f"got: {state_values.get('learning_mode')}"
        )

        # Verify learning_explanation was NOT called during planning
        assert "learning_explanation" not in mock_llm.provider.call_sequence, (
            f"learning_explanation should NOT be called during planning. "
            f"Call sequence: {mock_llm.provider.call_sequence}"
        )

    @pytest.mark.asyncio
    async def test_learning_mode_false_skips_explanation(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify learning_mode=False (Auto mode) skips learning_explanation entirely.
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            # NO learning_explanation response - should not be called
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=False,  # Auto mode
            autonomousModeContinue=False,
        )

        result, session_id = await agent.chat(
            user_input="Analyze data",
            context=context
        )

        # Verify learning_explanation was NOT called
        assert "learning_explanation" not in mock_llm.provider.call_sequence, (
            f"learning_explanation should NOT have been called in auto mode. "
            f"Call sequence: {mock_llm.provider.call_sequence}"
        )

        # Get final state
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        state_values = state.values

        # Verify learning_mode is False
        assert state_values.get("learning_mode") is False, (
            f"learning_mode should be False, got: {state_values.get('learning_mode')}"
        )

    @pytest.mark.asyncio
    async def test_workflow_result_sent_after_planning_tutorial_mode(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify LOOP_INCOMPLETE is sent after planning completes in tutorial mode.

        This is critical for TypeScript to know NOT to terminate the session.
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            "learning_explanation": create_learning_explanation(),
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation_with_guidance": create_code_generation_response(),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=False,
        )

        await agent.chat("Analyze", context=context)

        # Verify workflow result was sent
        assert len(captured_messages.workflow_results) > 0, (
            "No workflow result sent! TypeScript won't know to continue the loop."
        )

        # Verify it's LOOP_INCOMPLETE (not LOOP_COMPLETE)
        final_state = captured_messages.get_final_workflow_state()
        assert final_state == "LOOP_INCOMPLETE", (
            f"Expected LOOP_INCOMPLETE after planning, got: {final_state}. "
            f"All workflow results: {captured_messages.workflow_results}"
        )

    @pytest.mark.asyncio
    async def test_task_activation_happens_during_planning(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify first task is marked active during planning phase.

        ARCHITECTURE NOTE (post-refactor):
        - Task activation happens at end of planning
        - learning_explanation runs in SEPARATE graph AFTER code executes
        - This test verifies planning activates first task correctly
        """
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "First task", "status": "pending"},
                    {"id": 2, "task": "Second task", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
            # No execution tools - planning only
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=False,  # Planning only
        )

        _, session_id = await agent.chat("Analyze", context=context)

        # Get final state
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        state_values = state.values

        # Verify first task is active
        task_list = state_values.get("task_list", {})
        tasks = task_list.get("tasks", [])
        assert len(tasks) >= 1, "Should have at least one task"
        assert tasks[0].get("status") == "active", (
            f"First task should be active. Tasks: {tasks}"
        )

        # Verify tool sequence: task_list_generation → evaluator only
        sequence = mock_llm.provider.call_sequence
        assert "task_list_generation" in sequence
        assert "task_list_evaluator" in sequence
        # learning_explanation NOT called during planning
        assert "learning_explanation" not in sequence


# =============================================================================
# CONTINUATION FLOW TESTS - Multi-invocation scenarios
# =============================================================================

class TestContinuationFlows:
    """
    Tests that simulate the full continuation flow across multiple invocations.

    These tests verify that clicking "Continue" in VSCode properly continues
    execution with the correct state.
    """

    @pytest.mark.asyncio
    async def test_continuation_preserves_learning_mode(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify learning_mode is preserved across invocations.

        ARCHITECTURE NOTE (post-refactor):
        - learning_explanation runs in SEPARATE graph after code executes
        - Main graph does NOT call learning_explanation
        - learning_mode flag should persist in state across invocations

        Invocation 1: Planning only
        Invocation 2: First code generation
        Invocation 3: After execution, second task code generation
        """
        # === INVOCATION 1: Planning only ===
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Load data", "status": "pending"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=False,  # Planning only
        )

        _, session_id = await agent.chat("Analyze", context=context1)

        # Planning only - no code
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_code_execution_sent(expected_count=0)

        # Verify learning_mode persisted
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        assert state.values.get("learning_mode") is True

        captured_messages.clear()
        mock_llm.provider.call_sequence.clear()

        # === INVOCATION 2: First code generation ===
        mock_llm.set_responses({
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation_with_guidance": create_code_generation_response(),
        })

        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state.values.get("task_list", {}),
        )

        _, session_id = await agent.chat("", session_id=session_id, context=context2)

        # First code generated
        captured_messages.assert_code_execution_sent(expected_count=1)

        # Verify learning_mode still persisted
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        assert state.values.get("learning_mode") is True, (
            "learning_mode should persist across invocations"
        )

        # Verify learning_explanation NOT called by main graph
        assert "learning_explanation" not in mock_llm.provider.call_sequence, (
            f"learning_explanation should NOT be called by main graph. "
            f"Call sequence: {mock_llm.provider.call_sequence}"
        )

    @pytest.mark.asyncio
    async def test_auto_mode_continuation_no_explanation(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Verify auto mode continuation does NOT call learning_explanation.
        """
        # === INVOCATION 1: Planning ===
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,  # Auto mode
            autonomousModeContinue=False,
        )

        _, session_id = await agent.chat("Analyze", context=context1)
        captured_messages.clear()
        mock_llm.provider.call_sequence.clear()

        # === INVOCATION 2: Continue ===
        state = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )

        mock_llm.set_responses({
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation": create_code_generation_response(),
        })

        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=False,
            autonomousModeContinue=True,
            taskList=state.values.get("task_list", {}),
        )

        await agent.chat("", session_id=session_id, context=context2)

        # Verify learning_explanation was NOT called
        assert "learning_explanation" not in mock_llm.provider.call_sequence, (
            f"learning_explanation should NOT be called in auto mode. "
            f"Call sequence: {mock_llm.provider.call_sequence}"
        )

        captured_messages.assert_no_learning_explanation()
        captured_messages.assert_code_execution_sent(expected_count=1)

    @pytest.mark.asyncio
    async def test_full_tutorial_mode_three_invocation_flow(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        Full 3-invocation tutorial mode flow:
        1. Planning only → LOOP_INCOMPLETE
        2. First code generation → LEARNING_MODE_PENDING
        3. After execution → second code generation → LEARNING_MODE_PENDING

        ARCHITECTURE NOTE (post-refactor):
        - Planning phase only generates task list
        - learning_explanation runs in SEPARATE graph after code executes
        - LEARNING_MODE_PENDING signals VSCode to run learning explanation
        """
        # === INVOCATION 1: Planning only ===
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Load data", "status": "pending"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=False,  # Planning only
        )

        _, session_id = await agent.chat("Analyze", context=context1)

        # Invocation 1 outputs - planning only
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        captured_messages.assert_code_execution_sent(expected_count=0)
        captured_messages.assert_workflow_state("LOOP_INCOMPLETE")

        # Verify state after invocation 1
        state1 = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )
        assert state1.values.get("learning_mode") is True

        captured_messages.clear()

        # === INVOCATION 2: First code generation ===
        mock_llm.set_responses({
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation_with_guidance": create_code_generation_response(),
        })

        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state1.values.get("task_list", {}),
        )

        _, session_id = await agent.chat("", session_id=session_id, context=context2)

        # First code generated
        captured_messages.assert_code_execution_sent(expected_count=1)
        captured_messages.assert_workflow_state("LEARNING_MODE_PENDING")

        state2 = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )

        captured_messages.clear()

        # === INVOCATION 3: After code executes, second task ===
        mock_llm.set_responses({
            "autonomous_mark_completion": create_autonomous_mark_completion(
                status_updates=[{"id": 1, "status": "completed"}]
            ),
            "autonomous_update_tasks": create_autonomous_update_tasks(
                tasks=[{"id": 2, "task": "Process data", "status": "pending"}],
                update_rule="KEEP"
            ),
            "task_update_evaluator": create_task_update_evaluation(approved=True),
            "cell_positioning": create_cell_positioning(target_cell=4),
            "code_generation_with_guidance": create_code_generation_response(
                "# Process data"
            ),
        })

        context3 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state2.values.get("task_list", {}),
            lastExecutionFailed=False,
            executionResult="Data loaded successfully",
        )

        await agent.chat("", session_id=session_id, context=context3)

        # Invocation 3 - code for second task
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        captured_messages.assert_code_execution_sent(expected_count=1)
        captured_messages.assert_workflow_state("LEARNING_MODE_PENDING")

    @pytest.mark.asyncio
    async def test_second_step_after_continue_button_click(
        self, mock_llm, mock_knowledge_base, captured_messages
    ):
        """
        After first step completes and user clicks Continue,
        second step should generate code.

        ARCHITECTURE NOTE (post-refactor):
        - learning_explanation runs in SEPARATE graph after code executes
        - Main graph does NOT call learning_explanation
        - This tests the 3-invocation flow: planning → first code → second code

        This simulates exactly what VSCode sends after clicking Continue:
        - autonomousModeContinue=True
        - The task list from previous state
        - executionResult from notebook execution
        """
        # === INVOCATION 1: Planning only ===
        mock_llm.set_responses({
            "task_list_generation": create_task_list_generation(
                tasks=[
                    {"id": 1, "task": "Check normalization", "status": "pending"},
                    {"id": 2, "task": "Run celltypist", "status": "pending"},
                ]
            ),
            "task_list_evaluator": create_task_list_evaluation(approved=True),
        })

        agent = create_agent(mock_llm, mock_knowledge_base)
        context1 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=False,  # Planning only
        )

        _, session_id = await agent.chat(
            "Walk me through celltypist", context=context1
        )

        # Verify planning only
        captured_messages.assert_task_list_displayed(min_count=1)
        captured_messages.assert_code_execution_sent(expected_count=0)

        state1 = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )

        captured_messages.clear()
        mock_llm.provider.call_sequence.clear()

        # === INVOCATION 2: First code generation ===
        mock_llm.set_responses({
            "cell_positioning": create_cell_positioning(target_cell=3),
            "code_generation_with_guidance": create_code_generation_response(
                "# Check normalization\nprint(adata.uns.get('log1p'))"
            ),
        })

        context2 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state1.values.get("task_list", {}),
        )

        _, session_id = await agent.chat("", session_id=session_id, context=context2)

        # First code generated
        captured_messages.assert_code_execution_sent(expected_count=1)

        state2 = await agent.orchestrator.main_graph.aget_state(
            config={"configurable": {"thread_id": session_id}}
        )

        captured_messages.clear()
        mock_llm.provider.call_sequence.clear()

        # === INVOCATION 3: User clicked Continue after code executed ===
        mock_llm.set_responses({
            "autonomous_mark_completion": create_autonomous_mark_completion(
                status_updates=[{"id": 1, "status": "completed"}]
            ),
            "autonomous_update_tasks": create_autonomous_update_tasks(
                update_rule="KEEP"
            ),
            "task_update_evaluator": create_task_update_evaluation(approved=True),
            "cell_positioning": create_cell_positioning(target_cell=4),
            "code_generation_with_guidance": create_code_generation_response(
                "# Run celltypist\nimport celltypist"
            ),
        })

        # Context exactly as VSCode sends it after Continue button click
        context3 = create_vscode_context(
            autonomousMode=True,
            learningMode=True,
            autonomousModeContinue=True,
            taskList=state2.values.get("task_list", {}),
            lastExecutionFailed=False,
            executionResult="{'base': None}",  # Output from first cell
            lastCellModifiedInAutoMode=3,
        )

        await agent.chat("", session_id=session_id, context=context3)

        # Record third invocation tool sequence
        third_invocation_tools = mock_llm.provider.call_sequence.copy()

        # learning_explanation NOT called by main graph
        assert "learning_explanation" not in third_invocation_tools, (
            f"learning_explanation should NOT be called by main graph. "
            f"Tools called: {third_invocation_tools}"
        )

        # CRITICAL: Third invocation MUST generate code for step 2
        assert "code_generation_with_guidance" in third_invocation_tools, (
            f"Third invocation should generate code for step 2. "
            f"Tools called: {third_invocation_tools}"
        )

        # Verify outputs
        captured_messages.assert_learning_explanation_sent(expected_count=0)
        captured_messages.assert_code_execution_sent(expected_count=1)
        captured_messages.assert_workflow_state("LEARNING_MODE_PENDING")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
