"""Test that first execution flow correctly assembles prompts with task objective.

This tests the complete flow:
1. Router routes to mark_next_task_active (when no active task)
2. mark_next_task_active sets active_task_objective in state
3. Router routes to code_generation_with_guidance
4. Prompt is assembled with the actual task text (not None)
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
from kai.core.orchestration.deterministic_tools import MarkNextTaskActiveTool
from kai.core.prompt_manager import PromptManager, PromptScenario


class TestFirstExecutionPromptAssembly:
    """Test that the first execution iteration correctly assembles prompts with task objective."""

    @pytest.mark.asyncio
    async def test_mark_next_task_active_sets_objective_from_task_text(self):
        """Verify mark_next_task_active extracts task text correctly."""
        tool = MarkNextTaskActiveTool()

        task_list = {
            "tasks": [
                {"id": 1, "task": "Load the scRNA-seq dataset and verify structure", "status": "pending"},
                {"id": 2, "task": "Perform quality control filtering", "status": "pending"},
            ]
        }

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context={},
                task_list=task_list,
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={}
        )

        result = await tool.execute(exec_context)

        # The objective should be the actual task text
        assert result.output_workflow["active_task_objective"] == "Load the scRNA-seq dataset and verify structure"
        assert result.output_workflow["next_pending_task_objective"] == "Perform quality control filtering"

    def test_prompt_manager_builds_active_vs_next_with_task_text(self):
        """Verify prompt manager builds the active_vs_next section with actual task text."""
        pm = PromptManager()

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context={
                    "active_task_objective": "Load the scRNA-seq dataset and verify structure",
                    "next_pending_task_objective": "Perform quality control filtering",
                    "current_cell": "",
                    "execution_history": [],
                    "conversation_history": [],
                    "notebook_cells": [],
                    "notebook_structure": {"totalCells": 0, "allCells": []},
                },
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        section = pm._build_active_vs_next_section(exec_context)

        # Should contain the actual task text
        assert "Load the scRNA-seq dataset and verify structure" in section
        assert "Perform quality control filtering" in section
        assert "None" not in section

    def test_prompt_manager_handles_none_objective_gracefully(self):
        """If active_task_objective is None, section should be empty (not contain 'None')."""
        pm = PromptManager()

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context={
                    # Note: active_task_objective not set
                    "current_cell": "",
                    "execution_history": [],
                    "conversation_history": [],
                    "notebook_cells": [],
                    "notebook_structure": {"totalCells": 0, "allCells": []},
                },
                task_list={"tasks": []},
                user_query="",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        section = pm._build_active_vs_next_section(exec_context)

        # Should be empty, not "None"
        assert section == ""
        assert "None" not in section

    def test_full_prompt_contains_task_objective(self):
        """Test that CODE_GENERATION_WITH_GUIDANCE prompt contains task objective."""
        pm = PromptManager()

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context={
                    "active_task_objective": "Normalize the matrix",
                    "next_pending_task_objective": "Run PCA",
                    "current_cell": "",
                    "execution_history": [],
                    "conversation_history": [],
                    "notebook_cells": [],
                    "notebook_structure": {"totalCells": 0, "allCells": []},
                    "last_execution_failed": False,
                    "error_message": "",
                    "positioning_info": {"target_cell": 0},
                },
                task_list={"tasks": []},
                user_query="Analyze my single-cell data",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        system_prompt, user_prompt = pm.generate_prompt(
            exec_context=exec_context,
            scenario=PromptScenario.CODE_GENERATION_WITH_GUIDANCE
        )

        # User prompt should contain the task objectives
        assert "Normalize the matrix" in user_prompt
        assert "Run PCA" in user_prompt


class TestRouterFirstExecutionFlow:
    """Test that the router correctly routes to mark_next_task_active in first execution."""

    def test_route_first_execution_no_active_task(self):
        """Router should route to mark_next_task_active when no task is active."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "First task", "status": "pending"},
                    {"id": 2, "task": "Second task", "status": "pending"},
                ]
            },
            "auto_mode_first_execution_done": False,
            "positioning_info": None,
            "generated_code": None,
        }

        result = orch._route_first_execution(state)

        assert result == "mark_next_task_active", \
            f"Should route to mark_next_task_active when no active task, got {result}"

    def test_route_first_execution_has_active_task_no_positioning(self):
        """After task is active, should route to cell_positioning."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "First task", "status": "active"},
                    {"id": 2, "task": "Second task", "status": "pending"},
                ]
            },
            "auto_mode_first_execution_done": False,
            "positioning_info": None,
            "generated_code": None,
            "active_task_objective": "First task",
        }

        result = orch._route_first_execution(state)

        assert result == "cell_positioning", \
            f"Should route to cell_positioning after task is active, got {result}"

    def test_route_first_execution_has_positioning_no_code(self):
        """After positioning, should route to code_generation_with_guidance for CODE tasks."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "First task", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 0},
            "generated_code": None,
            "active_task_objective": "First task",
            "is_reasoning_task": False,  # Explicit: code task
        }

        result = orch._route_first_execution(state)

        assert result == "code_generation_with_guidance", \
            f"Should route to code_generation_with_guidance after positioning, got {result}"

    def test_route_first_execution_reasoning_task_routes_to_reasoning(self):
        """For REASONING tasks, should route to reasoning_response_with_guidance (not code_generation)."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Assess batch effects", "status": "active"},
                    {"id": 2, "task": "Compute PCA", "status": "pending"},
                ]
            },
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 0},
            "generated_code": None,
            "reasoning_response": None,
            "active_task_objective": "[reasoning] Assess batch effects",
            "is_reasoning_task": True,  # This is the key flag
        }

        result = orch._route_first_execution(state)

        assert result == "reasoning_response_with_guidance", \
            f"Should route to reasoning_response_with_guidance for reasoning task, got {result}"

    def test_route_first_execution_reasoning_task_with_response_routes_to_critique(self):
        """After reasoning generated, should route to reasoning_critique."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Assess batch effects", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 0},
            "generated_code": None,
            "reasoning_response": "## Batch Effect Analysis\n...",  # Reasoning was generated
            "reasoning_approval": None,  # Not yet approved
            "critique_iteration": 0,
            "active_task_objective": "[reasoning] Assess batch effects",
            "is_reasoning_task": True,
        }

        result = orch._route_first_execution(state)

        assert result == "reasoning_critique", \
            f"Should route to reasoning_critique after reasoning generated, got {result}"

    def test_route_first_execution_reasoning_approved_marks_completed(self):
        """After reasoning approved, should mark reasoning completed (which also marks first exec done)."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        state = {
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "[reasoning] Assess batch effects", "status": "active"},
                ]
            },
            "auto_mode_first_execution_done": False,
            "positioning_info": {"target_cell": 0},
            "generated_code": None,
            "reasoning_response": "## Batch Effect Analysis\n...",
            "reasoning_approval": "APPROVED",  # Approved!
            "critique_iteration": 1,
            "active_task_objective": "[reasoning] Assess batch effects",
            "is_reasoning_task": True,
        }

        result = orch._route_first_execution(state)

        # After reasoning approved, route to mark_reasoning_completed which marks task complete
        # and also sets first execution done
        assert result == "mark_reasoning_completed", \
            f"Should route to mark_reasoning_completed after reasoning approved, got {result}"


class TestEndToEndFirstExecution:
    """Integration test for the full first execution flow."""

    @pytest.mark.asyncio
    async def test_first_execution_flow_sets_objective_before_code_gen(self):
        """
        Simulate the first execution flow and verify active_task_objective is set
        before code generation would run.
        """
        from langgraph.graph import StateGraph, END
        from kai.core.state import KaiState
        from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType

        # Track state at each step
        state_snapshots = []

        class MockMarkNextTaskActive(BaseTool):
            def __init__(self):
                super().__init__("mark_next_task_active")

            async def execute(self, exec_context, **kwargs):
                # Get the first pending task
                task_list = exec_context.inputs.task_list
                tasks = task_list.get("tasks", [])
                first_pending = next((t for t in tasks if t.get("status") == "pending"), None)

                task_objective = first_pending["task"] if first_pending else None

                # Mark it as active
                if first_pending:
                    first_pending["status"] = "active"

                return ToolResult(
                    output_ui="Task activated",
                    output_type=ToolOutputType.TASK_LIST_DISPLAY,
                    output_workflow={
                        "active_task_objective": task_objective,
                        "active_task": first_pending,
                        "task_list": task_list,
                        "next_task_activated": True,
                    }
                )

        class MockCellPositioning(BaseTool):
            def __init__(self):
                super().__init__("cell_positioning")

            async def execute(self, exec_context, **kwargs):
                state_snapshots.append({
                    "tool": "cell_positioning",
                    "active_task_objective": exec_context.inputs.context.get("active_task_objective")
                })
                return ToolResult(
                    output_ui="Positioned",
                    output_type=ToolOutputType.RESPONSE,
                    output_workflow={"positioning_info": {"target_cell": 0}}
                )

        class MockCodeGeneration(BaseTool):
            def __init__(self):
                super().__init__("code_generation_with_guidance")

            async def execute(self, exec_context, **kwargs):
                state_snapshots.append({
                    "tool": "code_generation_with_guidance",
                    "active_task_objective": exec_context.inputs.context.get("active_task_objective")
                })
                return ToolResult(
                    output_ui={"code": "print('hello')"},
                    output_type=ToolOutputType.EXECUTE_ONLY,
                    output_workflow={"generated_code": "print('hello')"}
                )

        # Build graph that simulates first execution
        graph = StateGraph(KaiState)

        mark_tool = MockMarkNextTaskActive()
        pos_tool = MockCellPositioning()
        code_tool = MockCodeGeneration()

        graph.add_node("mark_next_task_active", mark_tool.as_graph_node())
        graph.add_node("cell_positioning", pos_tool.as_graph_node())
        graph.add_node("code_generation_with_guidance", code_tool.as_graph_node())

        # Linear flow: mark -> positioning -> code_gen
        graph.set_entry_point("mark_next_task_active")
        graph.add_edge("mark_next_task_active", "cell_positioning")
        graph.add_edge("cell_positioning", "code_generation_with_guidance")
        graph.add_edge("code_generation_with_guidance", END)

        compiled = graph.compile()

        initial_state = {
            "active_task_objective": None,  # NOT set initially
            "task_list": {
                "tasks": [
                    {"id": 1, "task": "Load and preprocess the dataset", "status": "pending"},
                    {"id": 2, "task": "Run clustering analysis", "status": "pending"},
                ]
            },
            "user_query": "Analyze my data",
        }

        async for output in compiled.astream(initial_state):
            pass

        # Verify cell_positioning saw the objective
        pos_snapshot = next(s for s in state_snapshots if s["tool"] == "cell_positioning")
        assert pos_snapshot["active_task_objective"] == "Load and preprocess the dataset", \
            f"cell_positioning should see task objective, got: {pos_snapshot['active_task_objective']}"

        # Verify code_generation saw the objective
        code_snapshot = next(s for s in state_snapshots if s["tool"] == "code_generation_with_guidance")
        assert code_snapshot["active_task_objective"] == "Load and preprocess the dataset", \
            f"code_generation should see task objective, got: {code_snapshot['active_task_objective']}"

    @pytest.mark.asyncio
    async def test_prompt_built_with_objective_in_code_generation(self):
        """
        Test that when code_generation_with_guidance runs, the prompt it would
        build contains the active task objective (not None).
        """
        from kai.core.prompt_manager import get_prompt_manager

        # Create context with active_task_objective set
        # (as it would be after mark_next_task_active)
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context={
                    "active_task_objective": "Perform UMAP reduction",
                    "next_pending_task_objective": "Cluster cells",
                    "current_cell": "",
                    "execution_history": [],
                    "conversation_history": [],
                    "notebook_cells": [],
                    "notebook_structure": {"totalCells": 0, "allCells": []},
                    "positioning_info": {"target_cell": 0},
                    "last_execution_failed": False,
                    "error_message": "",
                },
                task_list={
                    "tasks": [
                        {"id": 1, "task": "Perform UMAP reduction",
                         "status": "active"},
                        {"id": 2, "task": "Cluster cells",
                         "status": "pending"},
                    ]
                },
                user_query="Analyze my scRNA-seq data",
                excluded_workflows=[]
            ),
            session_metadata={"autonomous_mode": True}
        )

        # Build the prompt that code_generation_with_guidance would use
        pm = get_prompt_manager()
        system_prompt, user_prompt = pm.generate_prompt(
            exec_context=exec_context,
            scenario=PromptScenario.CODE_GENERATION_WITH_GUIDANCE
        )

        # Verify the task objectives appear in the prompt
        assert "Perform UMAP reduction" in user_prompt, \
            "Active task objective should appear in prompt"
        assert "Cluster cells" in user_prompt, \
            "Next task objective should appear in prompt for context"

        # Verify "None" does not appear where task objectives should be
        # (This catches the bug where active_task_objective was None)
        lines_with_task = [
            line for line in user_prompt.split('\n')
            if 'task' in line.lower()
        ]
        for line in lines_with_task:
            assert "None" not in line, \
                f"Found 'None' in task-related line: {line}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
