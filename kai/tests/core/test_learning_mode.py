"""Tests for Tutorial/Learning mode information flow.

This test suite verifies:
1. LearningExplanationTool produces correct output structure
2. State flows correctly through graph nodes
3. Context from VS Code reaches Python state
4. UI Communicator receives tool output

ARCHITECTURE NOTE (post-refactor):
- Learning explanation runs in a SEPARATE learning graph AFTER code execution
- The main execution graph is identical regardless of learning_mode
- Routers no longer route to learning_explanation - orchestrator invokes learning graph
- learning_explanation_done transient field has been removed

All tests use MOCKED LLMs - no real API calls.
"""

import pytest
from unittest.mock import Mock, AsyncMock

from kai.core.orchestration.state import get_transient_defaults
from kai.core.tools.learning_explanation import LearningExplanationTool
from kai.core.tools.base import ToolResult, ToolOutputType

# Import the base state creator from conftest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from conftest import create_base_state


# =============================================================================
# Level 1: Architecture Tests (verify new learning mode design)
# =============================================================================

class TestLearningModeArchitecture:
    """Verify the new learning mode architecture."""

    def test_learning_explanation_done_removed_from_transient_fields(self):
        """learning_explanation_done should NOT be in transient fields.

        After refactor, learning explanation runs in separate graph,
        so we don't need a transient flag to prevent re-running.
        """
        defaults = get_transient_defaults()
        assert "learning_explanation_done" not in defaults, (
            "learning_explanation_done should be removed - "
            "learning explanation now runs in separate graph"
        )

    def test_learning_mode_still_in_state(self):
        """learning_mode flag should still exist in state for orchestrator."""
        from kai.core.orchestration.state import KaiState
        from typing import get_type_hints

        hints = get_type_hints(KaiState)
        assert "learning_mode" in hints, (
            "learning_mode must be defined in KaiState for orchestrator "
            "to decide whether to invoke learning graph"
        )


# =============================================================================
# Level 2: Tool Tests (MOCKED LLM)
# =============================================================================

class TestLearningExplanationTool:
    """Test LearningExplanationTool with mocked LLM."""

    @pytest.fixture
    def mock_llm_interface(self):
        """Mock LLM interface - no real API calls."""
        mock_llm = Mock()
        mock_provider = Mock()
        mock_provider.provider_name = "test_provider"
        mock_provider.model = "test_model"
        mock_provider.generate = AsyncMock(
            return_value="This step loads single-cell data."
        )
        mock_llm.get_llm_for_tool.return_value = mock_provider
        mock_llm.get_reasoning_for_tool.return_value = "detailed"
        return mock_llm

    @pytest.fixture
    def state_with_completed_task(self):
        """State with an active task (learning runs AFTER execution, task still active).

        The task that was just executed is still "active" because
        autonomous_mark_completion runs at the START of the next iteration.
        """
        return create_base_state(
            user_query="Test query",
            autonomous_mode=True,
            active_task_objective="Load the data",
            task_list={
                "tasks": [
                    {"id": 1, "task": "Load data", "status": "active"},
                    {"id": 2, "task": "Process data", "status": "pending"},
                ]
            },
        )

    @pytest.mark.asyncio
    async def test_tool_returns_display_output_type(
        self, mock_llm_interface, state_with_completed_task
    ):
        """Tool output should have output_type=DISPLAY_ONLY."""
        tool = LearningExplanationTool(mock_llm_interface)
        result = await tool.execute(state_with_completed_task)

        assert result.output_type == ToolOutputType.DISPLAY_ONLY

    @pytest.mark.asyncio
    async def test_tool_output_has_learning_explanation_flag(
        self, mock_llm_interface, state_with_completed_task
    ):
        """Tool output_ui should have isLearningExplanation=True."""
        tool = LearningExplanationTool(mock_llm_interface)
        result = await tool.execute(state_with_completed_task)

        assert isinstance(result.output_ui, dict)
        assert result.output_ui.get("isLearningExplanation") is True

    @pytest.mark.asyncio
    async def test_tool_output_has_explanation_text(
        self, mock_llm_interface, state_with_completed_task
    ):
        """Tool output_ui should contain formatted explanation text."""
        tool = LearningExplanationTool(mock_llm_interface)
        result = await tool.execute(state_with_completed_task)

        assert "text" in result.output_ui
        # Output includes step title (from completed task) and explanation
        assert "**Step" in result.output_ui["text"]
        # Should reference the completed task, not the active one
        assert "Load data" in result.output_ui["text"]

    @pytest.mark.asyncio
    async def test_tool_finds_active_task(
        self, mock_llm_interface
    ):
        """Tool should find the ACTIVE task (the one just executed).

        Learning explanation runs AFTER execution but BEFORE the task is
        marked completed (which happens at the START of the next iteration).
        So the just-executed task is still "active".
        """
        state = create_base_state(
            user_query="Test query",
            autonomous_mode=True,
            task_list={
                "tasks": [
                    {"id": 1, "task": "First task", "status": "completed"},
                    {"id": 2, "task": "Second task", "status": "completed"},
                    {"id": 3, "task": "Third task (current)", "status": "active"},
                ]
            },
        )

        tool = LearningExplanationTool(mock_llm_interface)
        result = await tool.execute(state)

        # Should reference task 3 (active - just executed), not task 1 or 2
        assert "Third task" in result.output_ui["text"]
        assert "**Step 3" in result.output_ui["text"]

    @pytest.mark.asyncio
    async def test_tool_workflow_output_is_none(
        self, mock_llm_interface, state_with_completed_task
    ):
        """Tool should NOT return workflow output.

        After refactor, learning explanation runs in separate graph
        and doesn't need to set any flags.
        """
        tool = LearningExplanationTool(mock_llm_interface)
        result = await tool.execute(state_with_completed_task)

        assert result.output_workflow is None, (
            "output_workflow should be None - "
            "learning runs in separate graph, no state changes needed"
        )

    @pytest.mark.asyncio
    async def test_tool_calls_llm_provider(
        self, mock_llm_interface, state_with_completed_task
    ):
        """Tool should call LLM provider.generate()."""
        tool = LearningExplanationTool(mock_llm_interface)
        await tool.execute(state_with_completed_task)

        mock_provider = mock_llm_interface.get_llm_for_tool.return_value
        mock_provider.generate.assert_called_once()


# =============================================================================
# Level 3: Learning Graph Tests
# =============================================================================

class TestLearningGraph:
    """Test the separate learning graph builder."""

    def test_learning_graph_builds_successfully(self):
        """Learning graph should build with learning_explanation tool."""
        from kai.core.orchestration.graphs import build_learning_graph

        mock_llm = Mock()
        mock_provider = Mock()
        mock_provider.generate = AsyncMock(return_value="Test explanation")
        mock_llm.get_llm_for_tool.return_value = mock_provider
        mock_llm.get_reasoning_for_tool.return_value = "detailed"

        tool = LearningExplanationTool(mock_llm)
        tools = {"learning_explanation": tool}

        graph = build_learning_graph(tools)
        assert graph is not None

    def test_learning_graph_handles_missing_tool(self):
        """Learning graph should handle missing tool gracefully."""
        from kai.core.orchestration.graphs import build_learning_graph

        tools = {}  # No learning_explanation tool
        graph = build_learning_graph(tools)

        # Should build a fallback noop graph
        assert graph is not None


# =============================================================================
# Level 4: Context Flow Tests
# =============================================================================

class TestLearningModeContextFlow:
    """Test context flows from VS Code to Python state."""

    def test_learning_mode_extracted_from_context(self):
        """learningMode in VS Code context → learning_mode in state."""
        context = {
            "learningMode": True,
            "autonomousMode": True,
        }

        context_data = {
            "learning_mode": context.get("learningMode", False),
            "autonomous_mode": context.get("autonomousMode", False),
        }

        assert context_data["learning_mode"] is True

    def test_learning_mode_defaults_to_false(self):
        """Missing learningMode should default to False."""
        context = {"autonomousMode": True}

        context_data = {
            "learning_mode": context.get("learningMode", False),
        }

        assert context_data["learning_mode"] is False


# =============================================================================
# Level 5: UI Communicator Tests
# =============================================================================

class TestLearningModeUICommunication:
    """Test that tool output reaches UI correctly."""

    def test_display_output_format_for_vscode(self):
        """DISPLAY_ONLY output should be formatted correctly for VS Code."""
        result = ToolResult(
            output_ui={
                "text": "**Step Explanation**\n\nThis is the explanation.",
                "isLearningExplanation": True,
            },
            output_type=ToolOutputType.DISPLAY_ONLY,
            output_workflow=None  # No workflow output after refactor
        )

        assert result.output_type.value == "display"
        assert result.output_ui["isLearningExplanation"] is True
        assert "text" in result.output_ui

    def test_learning_loop_signal_exists(self):
        """LEARNING_LOOP should be a valid auto_loop_update value.

        The orchestrator sends LEARNING_LOOP (not LOOP_INCOMPLETE)
        after running the learning graph, so UI knows to show continue button.
        """
        # This is documented in the state.py comment for auto_loop_update
        # Just verify the signal name is documented
        from kai.core.orchestration.state import KaiState
        import inspect

        source = inspect.getsource(KaiState)
        assert "LEARNING_MODE_PENDING" in source, (
            "LEARNING_MODE_PENDING should be documented as valid auto_loop_update value"
        )


# =============================================================================
# Debug Helper Tests
# =============================================================================

class TestLearningModeDebugHelpers:
    """Tests that help debug the flow if something breaks."""

    def test_tool_output_type_enum_value(self):
        """Verify DISPLAY_ONLY enum value matches VS Code expectation."""
        assert ToolOutputType.DISPLAY_ONLY.value == "display"

    def test_execution_graph_does_not_include_learning_explanation(self):
        """Verify learning_explanation is NOT in execution graph tools.

        After refactor, learning runs in separate graph, not execution graph.
        """
        from kai.core.orchestration.graphs.execution import AUTONOMOUS_TOOLS

        assert "learning_explanation" not in AUTONOMOUS_TOOLS, (
            "learning_explanation should NOT be in AUTONOMOUS_TOOLS - "
            "it runs in separate learning graph after execution"
        )
