"""Test state transitions between different task types.

These tests verify that state is properly cleared when transitioning between:
- code → code
- code → reasoning
- reasoning → code
- reasoning → reasoning

Key bugs these tests catch:
1. `reasoning_critique` not in TRANSIENT_STATE_FIELDS - persists across tasks
2. should_replace logic checking key presence instead of value - even when
   value is None, `"reasoning_critique" in context` returns True if the key
   exists, causing incorrect should_replace=true on first reasoning call.

The correct behavior:
- First reasoning call of a NEW task: should_replace=false (insert new cell)
- Regeneration after critique in SAME task: should_replace=true (replace cell)

IMPORTANT: These tests exercise the ACTUAL production code in
ReasoningResponseWithGuidanceTool. They should FAIL if the bug exists.
"""

import pytest
from unittest.mock import MagicMock
from kai.core.orchestration.langgraph_orchestrator import (
    LangGraphOrchestrator,
    TRANSIENT_STATE_FIELDS
)
from kai.core.orchestration.execution_context import (
    ExecutionContext,
    ExecutionInputs
)
from kai.core.orchestration.prompt_tools import (
    ReasoningResponseWithGuidanceTool,
    CodeGenerationWithGuidanceTool,
    CodeUpdateTool,
)
from kai.core.orchestration.deterministic_tools import SetPositioningFromLastCellTool


class TestTransientStateFields:
    """Verify all reasoning-related fields are in TRANSIENT_STATE_FIELDS."""

    def test_reasoning_critique_is_transient(self):
        """reasoning_critique must be transient to prevent cross-task persist."""
        assert "reasoning_critique" in TRANSIENT_STATE_FIELDS, \
            "reasoning_critique must be in TRANSIENT_STATE_FIELDS"

    def test_reasoning_response_is_transient(self):
        """reasoning_response must be transient."""
        assert "reasoning_response" in TRANSIENT_STATE_FIELDS

    def test_reasoning_approval_is_transient(self):
        """reasoning_approval must be transient."""
        assert "reasoning_approval" in TRANSIENT_STATE_FIELDS

    def test_critique_iteration_is_transient(self):
        """critique_iteration must be transient."""
        assert "critique_iteration" in TRANSIENT_STATE_FIELDS


class TestReasoningResponseWithGuidanceToolShouldReplace:
    """
    Test the ACTUAL should_replace logic in ReasoningResponseWithGuidanceTool.

    These tests call the real _process_response method and verify the
    should_replace_code field in the output.

    The critical bug: LangGraph state passes keys with None values, and the
    buggy code checks `"reasoning_critique" in context` which returns True
    even when the value is None.
    """

    @pytest.fixture
    def tool(self):
        """Create a ReasoningResponseWithGuidanceTool with mocked LLM."""
        mock_llm = MagicMock()
        return ReasoningResponseWithGuidanceTool(mock_llm)

    @pytest.mark.asyncio
    async def test_should_replace_false_when_reasoning_critique_is_none(
        self, tool
    ):
        """
        KEY BUG TEST: When reasoning_critique key exists but value is None,
        should_replace_code must be "false".

        This is the exact scenario that caused reasoning cells to not appear:
        - LangGraph state has reasoning_critique=None as a key
        - The full state dict is passed to the tool as context
        - Buggy: `"reasoning_critique" in context` returns True
        - Fixed: `context.get("reasoning_critique") is not None` returns False

        This test FAILS with the buggy code and PASSES with the fix.
        """
        # State exactly as LangGraph provides it - keys with None values
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_critique": None,  # Key EXISTS but value is None
            "retry_objective": None,  # Key EXISTS but value is None
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "## My reasoning response",
            exec_context
        )

        # The critical assertion - this is what failed in production
        assert result.output_ui["should_replace_code"] == "false", \
            "First reasoning call must have should_replace_code='false' " \
            "when reasoning_critique is None (new task, not regeneration)"

    @pytest.mark.asyncio
    async def test_should_replace_false_when_keys_absent(self, tool):
        """When reasoning_critique key is completely absent, should be false."""
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            # Keys not present at all
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "## My reasoning response",
            exec_context
        )

        assert result.output_ui["should_replace_code"] == "false"

    @pytest.mark.asyncio
    async def test_should_replace_true_when_reasoning_critique_has_value(
        self, tool
    ):
        """
        When reasoning_critique has an actual string value (critique said
        MODIFY), should_replace_code must be "true" to replace the cell.
        """
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_critique": "Please add more detail about batch effects",
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "## Updated reasoning response",
            exec_context
        )

        assert result.output_ui["should_replace_code"] == "true", \
            "After critique, should_replace_code must be 'true' to replace"

    @pytest.mark.asyncio
    async def test_should_replace_true_when_retry_objective_has_value(
        self, tool
    ):
        """When retry_objective has value, should_replace_code must be true."""
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "retry_objective": "Retry: fix the analysis",
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "## Retried reasoning response",
            exec_context
        )

        assert result.output_ui["should_replace_code"] == "true"

    @pytest.mark.asyncio
    async def test_cell_type_is_markdown(self, tool):
        """Verify reasoning responses have cell_type='markdown'."""
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "## My reasoning",
            exec_context
        )

        assert result.output_ui["cell_type"] == "markdown"


class TestReasoningToReasoningTransition:
    """Test that state is cleared when going reasoning task to reasoning task."""

    def test_router_routes_to_reasoning_generation_for_new_task(self):
        """
        After reasoning task 1 completes, task 2 should route to
        reasoning_response_with_guidance (not critique or mark_completed).
        """
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        # State after task 1 completed - keys exist with None values
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Task 1", "status": "completed"},
                    {"id": 2, "task": "[reasoning] Task 2", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            # Keys exist with None values (LangGraph state behavior)
            "reasoning_response": None,
            "reasoning_approval": None,
            "reasoning_critique": None,
            "retry_objective": None,
            "critique_iteration": 0,
            "active_task_objective": "[reasoning] Task 2",
        }

        result = orch._route_standard_continue_branch(state)

        assert result == "reasoning_response_with_guidance", \
            f"New reasoning task should route to generation, got {result}"


class TestCodeToReasoningTransition:
    """Test transition from code task to reasoning task."""

    @pytest.fixture
    def tool(self):
        """Create a ReasoningResponseWithGuidanceTool with mocked LLM."""
        mock_llm = MagicMock()
        return ReasoningResponseWithGuidanceTool(mock_llm)

    @pytest.mark.asyncio
    async def test_first_reasoning_after_code_has_should_replace_false(
        self, tool
    ):
        """
        After code task completes, first reasoning task should insert new cell.
        State will have reasoning_critique=None from transient clearing.
        """
        context = {
            "positioning_info": {"target_cell": 3},
            "active_task_objective": "[reasoning] Assess results",
            "is_reasoning_task": True,
            "generated_code": None,
            "reasoning_critique": None,  # Key exists with None value
            "retry_objective": None,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "## Assessment of results",
            exec_context
        )

        assert result.output_ui["should_replace_code"] == "false", \
            "First reasoning after code must insert new cell (false)"


class TestReasoningToCodeTransition:
    """Test transition from reasoning task to code task."""

    def test_reasoning_state_cleared_before_code_task(self):
        """
        After reasoning task completes, code task should route correctly.
        """
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Done", "status": "completed"},
                    {"id": 2, "task": "Run clustering", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 4},
            "is_reasoning_task": False,
            "generated_code": None,
            "reasoning_response": None,
            "reasoning_approval": None,
            "reasoning_critique": None,
            "critique_iteration": 0,
        }

        result = orch._route_standard_continue_branch(state)

        assert result == "code_generation_with_guidance", \
            f"Should route to code generation for code task, got {result}"


class TestRouterDoesNotLoopOnReasoning:
    """Test that router doesn't infinite loop on reasoning positioning."""

    def test_router_proceeds_to_critique_after_first_reasoning(self):
        """
        After first reasoning response is generated, router should proceed
        to critique, not get stuck in a loop.
        """
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Assess", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": "## Analysis\nHere is my reasoning...",
            "reasoning_approval": None,
            "critique_iteration": 0,
        }

        result = orch._route_standard_continue_branch(state)

        assert result == "reasoning_critique", \
            f"Should route to critique after first response, got {result}"

    def test_router_proceeds_to_regeneration_after_modify_critique(self):
        """After critique says MODIFY, router routes to reasoning regeneration."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Assess", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": None,
            "reasoning_approval": "MODIFY",
            "reasoning_critique": "Add more detail about batch effects",
            "critique_iteration": 1,
        }

        result = orch._route_standard_continue_branch(state)

        assert result == "reasoning_response_with_guidance", \
            f"Should regenerate reasoning after MODIFY, got {result}"


class TestCodeToCodeTransition:
    """Test transition from code task to code task."""

    def test_code_state_cleared_between_tasks(self):
        """After code task 1 completes, code task 2 should start fresh."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "Load data", "status": "completed"},
                    {"id": 2, "task": "Preprocess data", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": False,
            "generated_code": None,
            "active_task_objective": "Preprocess data",
        }

        result = orch._route_standard_continue_branch(state)

        assert result == "code_generation_with_guidance", \
            f"New code task should route to code generation, got {result}"


class TestSetPositioningFromLastCellTool:
    """
    Test the SetPositioningFromLastCellTool which sets positioning_info
    based on last_cell_modified_in_auto_mode.

    This is critical for correct cell insertion order.
    """

    @pytest.fixture
    def tool(self):
        """Create a SetPositioningFromLastCellTool."""
        return SetPositioningFromLastCellTool()

    @pytest.mark.asyncio
    async def test_positioning_uses_last_cell_modified(self, tool):
        """Primary case: use last_cell_modified_in_auto_mode."""
        context = {
            "last_cell_modified_in_auto_mode": 5,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        assert result.output_workflow["positioning_info"]["target_cell"] == 5

    @pytest.mark.asyncio
    async def test_positioning_fallback_to_error_cell(self, tool):
        """Fallback: use error_cell_index when last_cell is None."""
        context = {
            "last_cell_modified_in_auto_mode": None,
            "error_cell_index": 3,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        assert result.output_workflow["positioning_info"]["target_cell"] == 3

    @pytest.mark.asyncio
    async def test_positioning_fallback_to_notebook_last_cell(self, tool):
        """Ultimate fallback: use notebook's last cell index."""
        context = {
            "last_cell_modified_in_auto_mode": None,
            "error_cell_index": -1,  # No error cell
            "notebook_structure": {"totalCells": 10},
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        # Should use last cell index (totalCells - 1)
        assert result.output_workflow["positioning_info"]["target_cell"] == 9

    @pytest.mark.asyncio
    async def test_positioning_empty_notebook_fallback(self, tool):
        """Edge case: empty notebook should use cell 0."""
        context = {
            "last_cell_modified_in_auto_mode": None,
            "error_cell_index": -1,
            "notebook_structure": {"totalCells": 0},
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        assert result.output_workflow["positioning_info"]["target_cell"] == 0


class TestCodeGenerationWithGuidanceToolOutput:
    """
    Test CodeGenerationWithGuidanceTool output format.

    Code generation ALWAYS inserts new cells (should_replace_code="false").
    """

    @pytest.fixture
    def tool(self):
        """Create a CodeGenerationWithGuidanceTool with mocked LLM."""
        mock_llm = MagicMock()
        return CodeGenerationWithGuidanceTool(mock_llm)

    @pytest.mark.asyncio
    async def test_code_generation_always_inserts_new_cell(self, tool):
        """CodeGeneration must always have should_replace_code='false'."""
        context = {
            "positioning_info": {"target_cell": 3},
            "autonomous_mode": True,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        # Mock LLM response with code block
        result = await tool._process_response(
            "```python\nimport pandas as pd\n```",
            exec_context
        )

        assert result.output_ui["should_replace_code"] == "false", \
            "Code generation must insert new cells, not replace"

    @pytest.mark.asyncio
    async def test_code_generation_cell_type_is_code(self, tool):
        """CodeGeneration must have cell_type='code'."""
        context = {
            "positioning_info": {"target_cell": 3},
            "autonomous_mode": True,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "```python\nimport pandas as pd\n```",
            exec_context
        )

        assert result.output_ui["cell_type"] == "code"

    @pytest.mark.asyncio
    async def test_code_generation_preserves_positioning_info(self, tool):
        """Positioning info must be passed through to output."""
        positioning_info = {"target_cell": 7, "extra_field": "test"}
        context = {
            "positioning_info": positioning_info,
            "autonomous_mode": True,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "```python\nx = 1\n```",
            exec_context
        )

        assert result.output_ui["positioning_info"] == positioning_info


class TestCodeUpdateToolOutput:
    """
    Test CodeUpdateTool output format.

    Code update (error fixing) ALWAYS replaces cells (should_replace_code="true").
    """

    @pytest.fixture
    def tool(self):
        """Create a CodeUpdateTool with mocked LLM."""
        mock_llm = MagicMock()
        return CodeUpdateTool(mock_llm)

    @pytest.mark.asyncio
    async def test_code_update_always_replaces_cell(self, tool):
        """CodeUpdate must always have should_replace_code='true'."""
        context = {
            "positioning_info": {"target_cell": 3},
            "autonomous_mode": True,
            "last_execution_failed": True,
            "error_recovery_strategy": "fix_code",
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "```python\nimport pandas as pd  # fixed\n```",
            exec_context
        )

        assert result.output_ui["should_replace_code"] == "true", \
            "Code update must replace existing cell, not insert"

    @pytest.mark.asyncio
    async def test_code_update_cell_type_is_code(self, tool):
        """CodeUpdate must have cell_type='code'."""
        context = {
            "positioning_info": {"target_cell": 3},
            "autonomous_mode": True,
            "last_execution_failed": True,
            "error_recovery_strategy": "fix_code",
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "```python\nimport pandas as pd\n```",
            exec_context
        )

        assert result.output_ui["cell_type"] == "code"

    @pytest.mark.asyncio
    async def test_code_update_includes_recovery_strategy(self, tool):
        """CodeUpdate must include error_recovery_strategy in output."""
        context = {
            "positioning_info": {"target_cell": 3},
            "autonomous_mode": True,
            "last_execution_failed": True,
            "error_recovery_strategy": "simplify_approach",
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await tool._process_response(
            "```python\nx = 1\n```",
            exec_context
        )

        assert result.output_ui["error_recovery_strategy"] == "simplify_approach"


class TestCellTypeConsistency:
    """
    Test that cell_type is correctly set for different tool outputs.

    This prevents the bug where a code cell was converted to markdown
    because the wrong cell_type was returned.
    """

    @pytest.fixture
    def reasoning_tool(self):
        mock_llm = MagicMock()
        return ReasoningResponseWithGuidanceTool(mock_llm)

    @pytest.fixture
    def code_gen_tool(self):
        mock_llm = MagicMock()
        return CodeGenerationWithGuidanceTool(mock_llm)

    @pytest.fixture
    def code_update_tool(self):
        mock_llm = MagicMock()
        return CodeUpdateTool(mock_llm)

    @pytest.mark.asyncio
    async def test_reasoning_tool_returns_markdown_type(self, reasoning_tool):
        """Reasoning must return cell_type='markdown'."""
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await reasoning_tool._process_response(
            "## Analysis\nThis is reasoning.",
            exec_context
        )

        assert result.output_ui["cell_type"] == "markdown", \
            "Reasoning tool must return cell_type='markdown'"

    @pytest.mark.asyncio
    async def test_code_generation_returns_code_type(self, code_gen_tool):
        """Code generation must return cell_type='code'."""
        context = {
            "positioning_info": {"target_cell": 5},
            "autonomous_mode": True,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await code_gen_tool._process_response(
            "```python\nimport pandas\n```",
            exec_context
        )

        assert result.output_ui["cell_type"] == "code", \
            "Code generation must return cell_type='code'"

    @pytest.mark.asyncio
    async def test_code_update_returns_code_type(self, code_update_tool):
        """Code update must return cell_type='code'."""
        context = {
            "positioning_info": {"target_cell": 5},
            "autonomous_mode": True,
            "last_execution_failed": True,
            "error_recovery_strategy": "fix_code",
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result = await code_update_tool._process_response(
            "```python\nimport pandas\n```",
            exec_context
        )

        assert result.output_ui["cell_type"] == "code", \
            "Code update must return cell_type='code'"


class TestReasoningCritiqueOutcomes:
    """
    Test the different outcomes of reasoning critique:
    1. APPROVED - task completes, moves to next task
    2. MODIFY (not approved) - regenerate reasoning (up to 3 iterations)
    3. Max iterations reached - proceed anyway
    """

    @pytest.fixture
    def orchestrator(self):
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()
        return LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

    def test_approved_routes_to_mark_completed(self, orchestrator):
        """When reasoning_approval=APPROVED, should complete the task."""
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Analyze", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": "## Analysis\nMy reasoning here.",
            "reasoning_approval": "APPROVED",  # Critique approved
            "critique_iteration": 1,
        }

        result = orchestrator._route_standard_continue_branch(state)

        assert result == "mark_reasoning_completed", \
            f"APPROVED should route to mark_reasoning_completed, got {result}"

    def test_modify_iteration_1_routes_to_regenerate(self, orchestrator):
        """
        When critique says MODIFY (first iteration), should regenerate.
        Note: After critique sets reasoning_approval, reasoning_response is
        cleared so we regenerate before next critique.
        """
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Analyze", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": None,  # Cleared after MODIFY
            "reasoning_approval": "MODIFY",
            "reasoning_critique": "Add more detail about batch effects",
            "critique_iteration": 1,
        }

        result = orchestrator._route_standard_continue_branch(state)

        assert result == "reasoning_response_with_guidance", \
            f"MODIFY should regenerate reasoning, got {result}"

    def test_modify_iteration_2_routes_to_regenerate(self, orchestrator):
        """Second MODIFY iteration should still regenerate."""
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Analyze", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": None,
            "reasoning_approval": "MODIFY",
            "reasoning_critique": "Still needs improvement",
            "critique_iteration": 2,
        }

        result = orchestrator._route_standard_continue_branch(state)

        assert result == "reasoning_response_with_guidance", \
            f"MODIFY at iteration 2 should regenerate, got {result}"

    def test_max_iterations_proceeds_anyway(self, orchestrator):
        """At max iterations (3), should proceed even without approval."""
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Analyze", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": "## Analysis\nThird attempt.",
            "reasoning_approval": "MODIFY",  # Not approved
            "reasoning_critique": "Still not perfect",
            "critique_iteration": 3,  # Max reached
        }

        result = orchestrator._route_standard_continue_branch(state)

        assert result == "mark_reasoning_completed", \
            f"Max iterations should proceed to complete, got {result}"

    def test_no_approval_yet_routes_to_critique(self, orchestrator):
        """
        When reasoning_response exists but no approval yet,
        should route to critique (first critique iteration).
        """
        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Analyze", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": "## Analysis\nFirst attempt.",
            "reasoning_approval": None,  # Not critiqued yet
            "critique_iteration": 0,
        }

        result = orchestrator._route_standard_continue_branch(state)

        assert result == "reasoning_critique", \
            f"Uncritiqued response should route to critique, got {result}"

    def test_critique_iteration_increments_after_modify(self, orchestrator):
        """
        Verify that critique_iteration is correctly used to limit regenerations.
        At iteration 3, even MODIFY should not trigger more regenerations.
        """
        # This is a boundary test - iteration 2 should regenerate
        state_iter_2 = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] X", "status": "active"}]},
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": None,
            "reasoning_approval": "MODIFY",
            "critique_iteration": 2,
        }
        result_iter_2 = orchestrator._route_standard_continue_branch(state_iter_2)
        assert result_iter_2 == "reasoning_response_with_guidance", \
            "Iteration 2 should still regenerate"

        # Iteration 3 should NOT regenerate - proceed to complete
        state_iter_3 = {
            "task_list": {"tasks": [{"id": 1, "task": "[reasoning] X", "status": "active"}]},
            "auto_mode_first_execution_done": True,
            "positioning_info": {"target_cell": 3},
            "is_reasoning_task": True,
            "reasoning_response": "## Final",
            "reasoning_approval": "MODIFY",
            "critique_iteration": 3,
        }
        result_iter_3 = orchestrator._route_standard_continue_branch(state_iter_3)
        assert result_iter_3 == "mark_reasoning_completed", \
            "Iteration 3 should proceed to complete despite MODIFY"


class TestShouldReplaceVsInsertMatrix:
    """
    Comprehensive test matrix for should_replace_code across all scenarios.

    | Tool                      | Scenario                    | should_replace |
    |---------------------------|-----------------------------| ---------------|
    | CodeGenerationWithGuidance| New task, first exec        | false          |
    | CodeGenerationWithGuidance| Continue after success      | false          |
    | CodeUpdateTool            | Error recovery              | true           |
    | CodeUpdateTool            | Retry with feedback         | true           |
    | ReasoningResponse         | New reasoning task          | false          |
    | ReasoningResponse         | After critique (MODIFY)     | true           |
    | ReasoningResponse         | Retry reasoning task        | true           |
    """

    @pytest.fixture
    def code_gen_tool(self):
        mock_llm = MagicMock()
        return CodeGenerationWithGuidanceTool(mock_llm)

    @pytest.fixture
    def code_update_tool(self):
        mock_llm = MagicMock()
        return CodeUpdateTool(mock_llm)

    @pytest.fixture
    def reasoning_tool(self):
        mock_llm = MagicMock()
        return ReasoningResponseWithGuidanceTool(mock_llm)

    # CodeGeneration scenarios - ALWAYS insert (false)
    @pytest.mark.asyncio
    async def test_code_gen_first_execution_inserts(self, code_gen_tool):
        """First execution: insert new cell."""
        context = {
            "positioning_info": {"target_cell": 2},
            "autonomous_mode": True,
            "auto_mode_first_execution_done": False,
        }
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=context,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result = await code_gen_tool._process_response(
            "```python\nx=1\n```", exec_context
        )
        assert result.output_ui["should_replace_code"] == "false"

    @pytest.mark.asyncio
    async def test_code_gen_continue_after_success_inserts(self, code_gen_tool):
        """Continue after success: insert new cell."""
        context = {
            "positioning_info": {"target_cell": 4},
            "autonomous_mode": True,
            "auto_mode_first_execution_done": True,
            "last_execution_failed": False,
        }
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=context,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result = await code_gen_tool._process_response(
            "```python\ny=2\n```", exec_context
        )
        assert result.output_ui["should_replace_code"] == "false"

    # CodeUpdate scenarios - ALWAYS replace (true)
    @pytest.mark.asyncio
    async def test_code_update_error_recovery_replaces(self, code_update_tool):
        """Error recovery: replace failed cell."""
        context = {
            "positioning_info": {"target_cell": 3},
            "autonomous_mode": True,
            "last_execution_failed": True,
            "error_recovery_strategy": "fix_code",
        }
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=context,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result = await code_update_tool._process_response(
            "```python\nfixed=True\n```", exec_context
        )
        assert result.output_ui["should_replace_code"] == "true"

    # Reasoning scenarios
    @pytest.mark.asyncio
    async def test_reasoning_new_task_inserts(self, reasoning_tool):
        """New reasoning task: insert new markdown cell."""
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_critique": None,  # No critique yet
            "retry_objective": None,
        }
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=context,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result = await reasoning_tool._process_response(
            "## Analysis", exec_context
        )
        assert result.output_ui["should_replace_code"] == "false"

    @pytest.mark.asyncio
    async def test_reasoning_after_critique_replaces(self, reasoning_tool):
        """After critique (MODIFY): replace existing reasoning cell."""
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_critique": "Add more detail",  # Has critique
        }
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=context,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result = await reasoning_tool._process_response(
            "## Detailed Analysis", exec_context
        )
        assert result.output_ui["should_replace_code"] == "true"

    @pytest.mark.asyncio
    async def test_reasoning_retry_replaces(self, reasoning_tool):
        """Retry reasoning task: replace existing cell."""
        context = {
            "positioning_info": {"target_cell": 5},
            "is_reasoning_task": True,
            "reasoning_critique": None,
            "retry_objective": "Retry: improve the analysis",  # Retry
        }
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=context,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result = await reasoning_tool._process_response(
            "## Improved Analysis", exec_context
        )
        assert result.output_ui["should_replace_code"] == "true"


class TestReasoningCritiqueStateFlowIntegration:
    """
    Integration tests that simulate the FULL LangGraph state flow for reasoning critique.

    These tests verify that state propagates correctly through the tool chain:
    1. reasoning_response_with_guidance runs -> outputs reasoning_response
    2. reasoning_critique runs -> outputs reasoning_critique, reasoning_approval
    3. reasoning_response_with_guidance runs AGAIN -> should see reasoning_critique

    BUG DETECTED: If reasoning_critique is not in KaiState TypedDict, LangGraph
    may not propagate it to subsequent tools, causing:
    - Second reasoning call has should_replace=false instead of true
    - Two markdown cells created instead of one being replaced
    """

    @pytest.fixture
    def reasoning_tool(self):
        mock_llm = MagicMock()
        return ReasoningResponseWithGuidanceTool(mock_llm)

    @pytest.fixture
    def code_gen_tool(self):
        mock_llm = MagicMock()
        return CodeGenerationWithGuidanceTool(mock_llm)

    def test_reasoning_critique_must_be_in_kai_state(self):
        """
        CRITICAL: reasoning_critique must be defined in KaiState TypedDict.

        If missing, LangGraph won't propagate it between tool calls, causing
        the second reasoning call after critique to have should_replace=false.
        """
        from kai.core.state import KaiState
        import typing

        # Get all fields from KaiState TypedDict
        kai_state_fields = set(typing.get_type_hints(KaiState).keys())

        assert "reasoning_critique" in kai_state_fields, \
            "reasoning_critique MUST be in KaiState TypedDict for LangGraph to propagate it between tools"

    def test_all_critique_output_fields_in_kai_state(self):
        """
        All fields output by ReasoningCritiqueTool must be in KaiState.

        ReasoningCritiqueTool outputs:
        - reasoning_approval
        - reasoning_critique
        - critique_iteration
        - reasoning_response (set to None when not approved)
        """
        from kai.core.state import KaiState
        import typing

        kai_state_fields = set(typing.get_type_hints(KaiState).keys())

        required_fields = [
            "reasoning_approval",
            "reasoning_critique",
            "critique_iteration",
            "reasoning_response",
        ]

        for field in required_fields:
            assert field in kai_state_fields, \
                f"{field} must be in KaiState for critique flow to work"

    @pytest.mark.asyncio
    async def test_simulated_critique_flow_state_propagation(self, reasoning_tool):
        """
        Simulate the full critique flow and verify state propagates correctly.

        This is the EXACT scenario from the bug:
        1. First reasoning call: should_replace=false (correct)
        2. Critique returns MODIFY with critique text
        3. Second reasoning call: should_replace=true (BUG: was false)

        The bug occurs because reasoning_critique isn't in KaiState.
        """
        # Step 1: First reasoning call (new task)
        state_step1 = {
            "positioning_info": {"target_cell": 9},
            "is_reasoning_task": True,
            "reasoning_critique": None,
            "reasoning_approval": None,
            "critique_iteration": 0,
        }

        exec_context_1 = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=state_step1,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result_1 = await reasoning_tool._process_response(
            "## First Analysis\nInitial reasoning.",
            exec_context_1
        )

        assert result_1.output_ui["should_replace_code"] == "false", \
            "First reasoning call should INSERT (should_replace=false)"

        # Step 2: Simulate critique tool output (MODIFY)
        # This is what ReasoningCritiqueTool.output_workflow returns
        critique_output = {
            "reasoning_approval": "MODIFY",
            "reasoning_critique": "Add more detail about the methodology",
            "critique_iteration": 1,
            "reasoning_response": None,  # Cleared to trigger regeneration
        }

        # Step 3: Second reasoning call - state should include critique output
        # This simulates LangGraph merging critique_output into state
        state_step3 = {
            **state_step1,
            **critique_output,
            # positioning_info stays the same (same cell position)
        }

        exec_context_3 = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=state_step3,
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        result_3 = await reasoning_tool._process_response(
            "## Improved Analysis\nMore detailed reasoning.",
            exec_context_3
        )

        assert result_3.output_ui["should_replace_code"] == "true", \
            "Second reasoning call after critique should REPLACE (should_replace=true). " \
            "BUG: Two markdown cells are created instead of replacing the first one."

    @pytest.mark.asyncio
    async def test_cell_position_consistency_during_critique_loop(self, reasoning_tool):
        """
        Verify cell position stays the same during critique iterations.

        BUG SCENARIO: If should_replace is wrong, cells get added at wrong positions:
        - Cell 9: First markdown (correct)
        - Cell 9: Second markdown ADDED instead of replacing (BUG)
        - Cell 10: Code for next task inserted BETWEEN the two markdown cells
        """
        target_cell = 9

        # First call
        state_1 = {
            "positioning_info": {"target_cell": target_cell},
            "is_reasoning_task": True,
            "reasoning_critique": None,
        }
        exec_context_1 = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=state_1,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result_1 = await reasoning_tool._process_response("## First", exec_context_1)

        # Second call after critique
        state_2 = {
            "positioning_info": {"target_cell": target_cell},  # SAME position
            "is_reasoning_task": True,
            "reasoning_critique": "Improve this",  # Critique present
        }
        exec_context_2 = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=state_2,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result_2 = await reasoning_tool._process_response("## Second", exec_context_2)

        # Both should target the SAME cell
        assert result_1.output_ui["positioning_info"]["target_cell"] == target_cell
        assert result_2.output_ui["positioning_info"]["target_cell"] == target_cell

        # First should INSERT, second should REPLACE
        assert result_1.output_ui["should_replace_code"] == "false", \
            "First reasoning should insert"
        assert result_2.output_ui["should_replace_code"] == "true", \
            "Second reasoning should replace at SAME position"

    @pytest.mark.asyncio
    async def test_next_code_task_after_reasoning_critique_loop(
        self, reasoning_tool, code_gen_tool
    ):
        """
        Test that code task after reasoning critique loop gets correct position.

        BUG SCENARIO from logs:
        1. Markdown cell added at position 9 (correct)
        2. Critique runs, says MODIFY
        3. Second markdown ADDED at position 9 (BUG - should replace!)
        4. Code task runs, gets position 10 (inserted BETWEEN the two markdowns)

        CORRECT behavior:
        1. Markdown cell added at position 9
        2. Critique runs, says MODIFY
        3. Markdown cell REPLACED at position 9
        4. Code task runs at position 10 (AFTER the single markdown)
        """
        # Simulate the full flow

        # Step 1: First reasoning at cell 9
        state_reason_1 = {
            "positioning_info": {"target_cell": 9},
            "is_reasoning_task": True,
            "reasoning_critique": None,
            "last_cell_modified_in_auto_mode": 8,
        }
        exec_ctx_1 = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=state_reason_1,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result_reason_1 = await reasoning_tool._process_response("## First", exec_ctx_1)

        assert result_reason_1.output_ui["should_replace_code"] == "false"
        # After this: cell 9 is markdown, last_cell = 9

        # Step 2: Critique returns MODIFY (simulated)
        # ReasoningCritiqueTool outputs: reasoning_critique, reasoning_approval, etc.

        # Step 3: Second reasoning at cell 9 (should REPLACE)
        state_reason_2 = {
            "positioning_info": {"target_cell": 9},
            "is_reasoning_task": True,
            "reasoning_critique": "Add more detail",  # FROM critique tool
            "reasoning_approval": "MODIFY",
            "last_cell_modified_in_auto_mode": 9,
        }
        exec_ctx_2 = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=state_reason_2,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result_reason_2 = await reasoning_tool._process_response("## Improved", exec_ctx_2)

        assert result_reason_2.output_ui["should_replace_code"] == "true", \
            "BUG: Second reasoning added new cell instead of replacing. " \
            "This causes code task to be inserted between two markdown cells!"
        # After this (correct): cell 9 is STILL markdown (replaced), last_cell = 9

        # Step 4: Code task at cell 10 (should be AFTER the single markdown)
        state_code = {
            "positioning_info": {"target_cell": 10},  # After reasoning cell
            "is_reasoning_task": False,
            "autonomous_mode": True,
            "last_cell_modified_in_auto_mode": 9,
        }
        exec_ctx_code = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None, context=state_code,
                task_list={"tasks": []}, user_query="", excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )
        result_code = await code_gen_tool._process_response(
            "```python\nprint('code')\n```", exec_ctx_code
        )

        # Code should be at position 10, AFTER the reasoning cell at 9
        assert result_code.output_ui["positioning_info"]["target_cell"] == 10
        assert result_code.output_ui["should_replace_code"] == "false"  # New code inserts


class TestKaiStateCompleteness:
    """
    Verify KaiState has all required fields for proper state propagation.

    LangGraph TypedDict state may not propagate fields that aren't defined,
    even with total=False. This can cause subtle bugs where tool outputs
    are silently dropped.
    """

    def test_reasoning_critique_in_kai_state(self):
        """reasoning_critique must be defined for critique flow to work."""
        from kai.core.state import KaiState
        import typing

        hints = typing.get_type_hints(KaiState)
        assert "reasoning_critique" in hints, \
            "reasoning_critique missing from KaiState - critique output won't propagate!"

    def test_task_list_critique_in_kai_state(self):
        """task_list_critique must be defined for task critique flow."""
        from kai.core.state import KaiState
        import typing

        hints = typing.get_type_hints(KaiState)
        assert "task_list_critique" in hints, \
            "task_list_critique missing from KaiState"

    def test_autonomous_update_critique_in_kai_state(self):
        """autonomous_update_critique must be defined for update critique flow."""
        from kai.core.state import KaiState
        import typing

        hints = typing.get_type_hints(KaiState)
        assert "autonomous_update_critique" in hints, \
            "autonomous_update_critique missing from KaiState"


class TestAutonomousUpdateTasksInfiniteLoopPrevention:
    """
    Tests to prevent infinite loop in autonomous_update_tasks.

    BUG SCENARIO from logs:
    1. Reasoning task completes
    2. autonomous_mark_completion runs
    3. autonomous_update_tasks runs → sets tasks_updated=True, task_list_update_rule="UPDATE"
    4. autonomous_update_critique runs → sets autonomous_update_approval="MODIFY" (not approved)
    5. Router sees: tasks_updated=True, update_rule="UPDATE", not approved, critique_iter=1
       → routes back to autonomous_update_tasks (line 853)
    6. autonomous_update_tasks runs again BUT doesn't clear autonomous_update_approval
    7. Router still sees old autonomous_update_approval="MODIFY" from step 4
    8. INFINITE LOOP: keeps routing to autonomous_update_tasks

    FIX: autonomous_update_tasks must clear autonomous_update_approval in output_workflow
    when regenerating after critique failure.
    """

    @pytest.fixture
    def update_tasks_tool(self):
        mock_llm = MagicMock()
        from kai.core.orchestration.prompt_tools import AutonomousUpdateTasksTool
        return AutonomousUpdateTasksTool(mock_llm)

    def test_autonomous_update_tasks_clears_approval_on_regeneration(self, update_tasks_tool):
        """
        When autonomous_update_tasks regenerates after critique failure,
        it MUST clear autonomous_update_approval to prevent infinite loop.
        """
        # Check that output_workflow includes clearing autonomous_update_approval
        # This is a structural test - we check the tool's output schema
        from kai.core.orchestration.prompt_tools import AutonomousUpdateTasksTool

        # The fix requires autonomous_update_approval=None in output_workflow
        # We verify this by checking the _process_structured_result method
        import inspect
        source = inspect.getsource(AutonomousUpdateTasksTool._process_structured_result)

        assert "autonomous_update_approval" in source, \
            "autonomous_update_tasks must set autonomous_update_approval in output_workflow to prevent infinite loop"

    def test_router_infinite_loop_scenario(self):
        """
        Simulate the exact router conditions that cause infinite loop.

        After autonomous_update_critique returns MODIFY:
        - tasks_updated=True
        - task_list_update_rule="UPDATE"
        - update_approved=False
        - autonomous_update_approval="MODIFY"
        - critique_iteration=1

        Router should route to autonomous_update_tasks.
        After autonomous_update_tasks runs again:
        - autonomous_update_approval should be None (cleared)
        - critique_iteration should be preserved (for max iteration check)
        """
        from kai.core.orchestration.langgraph_orchestrator import (
            LangGraphOrchestrator,
            TRANSIENT_STATE_FIELDS
        )

        # Verify the fields involved are understood
        assert "autonomous_update_approval" in TRANSIENT_STATE_FIELDS, \
            "autonomous_update_approval should be transient"

        # The key insight: transient fields are only cleared at iteration START,
        # not within a graph execution. So autonomous_update_tasks must clear
        # autonomous_update_approval explicitly.

    @pytest.mark.asyncio
    async def test_update_tasks_output_workflow_structure(self, update_tasks_tool):
        """
        Verify autonomous_update_tasks output_workflow clears critique state.

        When the tool regenerates a task list (after critique failure),
        it must clear:
        - autonomous_update_approval (to allow fresh critique)
        - autonomous_update_critique (the old critique text)
        """
        # Create a mock structured result
        from unittest.mock import MagicMock, AsyncMock
        from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs

        # Simulate state AFTER a critique failure (approval=MODIFY)
        context = {
            "task_list": {"tasks": [{"id": 1, "task": "Test", "status": "completed"}]},
            "tasks_updated": True,
            "autonomous_update_approval": "MODIFY",  # Previous critique said MODIFY
            "autonomous_update_critique": "Add more detail",
            "critique_iteration": 1,
            "rag_enabled": False,
            "last_execution_failed": False,
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context=context,
                task_list=context["task_list"],
                user_query="test",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        # Create mock structured result
        mock_result = MagicMock()
        mock_result.update_rule = "UPDATE"
        mock_result.update_rationale = "Test rationale"
        mock_result.updated_task_list = {"tasks": [{"id": 1, "task": "Updated test", "status": "pending"}]}
        mock_result.retrieval_queries = []

        # Call the processing method
        result = update_tasks_tool._process_structured_result(mock_result, exec_context)

        # Verify output_workflow clears the critique state
        assert result.output_workflow is not None, "output_workflow should not be None"
        assert "autonomous_update_approval" in result.output_workflow, \
            "output_workflow must include autonomous_update_approval to clear it"
        assert result.output_workflow["autonomous_update_approval"] is None, \
            "autonomous_update_approval must be set to None to prevent infinite loop"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
