"""Test that active_task_objective flows correctly from mark_next_task_active to code_generation."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator, PERSISTENT_STATE_FIELDS
from kai.core.orchestration.deterministic_tools import MarkNextTaskActiveTool
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs


class TestActiveTaskObjectiveFlow:
    """Test active_task_objective propagation through the workflow."""

    def test_active_task_objective_in_persistent_fields(self):
        """Verify active_task_objective is in PERSISTENT_STATE_FIELDS."""
        assert "active_task_objective" in PERSISTENT_STATE_FIELDS, \
            "active_task_objective must be in PERSISTENT_STATE_FIELDS to persist across iterations"
        assert "active_task" in PERSISTENT_STATE_FIELDS
        assert "is_reasoning_task" in PERSISTENT_STATE_FIELDS
        assert "next_pending_task_objective" in PERSISTENT_STATE_FIELDS

    @pytest.mark.asyncio
    async def test_mark_next_task_active_sets_objective(self):
        """Verify mark_next_task_active correctly sets active_task_objective."""
        tool = MarkNextTaskActiveTool()

        # Create a task list with pending tasks
        task_list = {
            "tasks": [
                {"id": 1, "task": "First task description", "status": "pending"},
                {"id": 2, "task": "Second task description", "status": "pending"},
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

        # Check workflow output has the task objective
        assert result.output_workflow is not None, "Tool should return workflow output"
        assert "active_task_objective" in result.output_workflow, \
            "output_workflow should contain active_task_objective"
        assert result.output_workflow["active_task_objective"] == "First task description", \
            f"Expected 'First task description', got '{result.output_workflow['active_task_objective']}'"
        assert result.output_workflow["next_pending_task_objective"] == "Second task description"

    @pytest.mark.asyncio
    async def test_mark_next_task_active_objective_not_none(self):
        """Verify mark_next_task_active never returns None for active_task_objective."""
        tool = MarkNextTaskActiveTool()

        # Create a task list with pending tasks
        task_list = {
            "tasks": [
                {"id": 1, "task": "Test task", "status": "pending"},
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

        assert result.output_workflow["active_task_objective"] is not None, \
            "active_task_objective should not be None"
        assert result.output_workflow["active_task_objective"] != "", \
            "active_task_objective should not be empty string"

    def test_execution_context_from_dict_preserves_active_task(self):
        """Verify ExecutionContext.from_dict preserves active_task_objective."""
        state = {
            "active_task_objective": "My specific task",
            "active_task": {"id": 1, "task": "My specific task", "status": "active"},
            "is_reasoning_task": False,
            "next_pending_task_objective": "Next task",
            "task_list": {"tasks": []},
            "user_query": "",
        }

        exec_context = ExecutionContext.from_dict(state)

        # Check that active_task_objective is accessible via context
        assert exec_context.inputs.context.get("active_task_objective") == "My specific task", \
            f"Expected 'My specific task', got '{exec_context.inputs.context.get('active_task_objective')}'"

    def test_prompt_builder_gets_active_task_objective(self):
        """Verify prompt builder can access active_task_objective from context."""
        from kai.core.prompt_manager import PromptManager

        pm = PromptManager()

        # Create exec_context with active_task_objective
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                backtracking_context=None,
                context={
                    "active_task_objective": "Verify data structure",
                    "next_pending_task_objective": "Perform clustering",
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

        # Build the active_vs_next section
        section = pm._build_active_vs_next_section(exec_context)

        assert "Verify data structure" in section, \
            f"Section should contain task objective. Got: {section}"
        assert "Perform clustering" in section, \
            f"Section should contain next task. Got: {section}"
        assert "None" not in section, \
            f"Section should not contain 'None'. Got: {section}"


class TestStateFlowBetweenNodes:
    """Test that state flows correctly between LangGraph nodes."""

    @pytest.mark.asyncio
    async def test_tool_output_updates_state(self):
        """Verify that tool output_workflow updates are merged into state."""
        from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType

        class MockTool(BaseTool):
            def __init__(self):
                super().__init__("mock_tool")

            async def execute(self, exec_context, **kwargs):
                return ToolResult(
                    output_ui="Test output",
                    output_type=ToolOutputType.RESPONSE,
                    output_workflow={
                        "active_task_objective": "Set by tool",
                        "some_other_field": "value"
                    }
                )

        tool = MockTool()
        node_func = tool.as_graph_node()

        # Call node function with initial state
        initial_state = {
            "active_task_objective": None,  # Initially None
            "task_list": {"tasks": []},
            "user_query": "",
        }

        result = await node_func(initial_state)

        # Check that output_workflow fields are in result
        assert "active_task_objective" in result, \
            "Tool output should include active_task_objective"
        assert result["active_task_objective"] == "Set by tool", \
            f"Expected 'Set by tool', got '{result['active_task_objective']}'"


class TestLangGraphStateFlow:
    """Test that LangGraph properly merges state between nodes."""

    @pytest.mark.asyncio
    async def test_state_flows_between_sequential_nodes(self):
        """
        Test that when mark_next_task_active sets active_task_objective,
        it's available to the next node (code_generation_with_guidance).

        This simulates the actual flow:
        1. mark_next_task_active sets active_task_objective
        2. code_generation_with_guidance should see it
        """
        from langgraph.graph import StateGraph, END
        from kai.core.state import KaiState
        from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType
        from kai.core.orchestration.execution_context import ExecutionContext

        # Track what code_generation sees
        seen_objective = {"value": None}

        class MockMarkNextTaskActive(BaseTool):
            def __init__(self):
                super().__init__("mark_next_task_active")

            async def execute(self, exec_context, **kwargs):
                return ToolResult(
                    output_ui="Task activated",
                    output_type=ToolOutputType.TASK_LIST_DISPLAY,
                    output_workflow={
                        "active_task_objective": "Test task objective",
                        "active_task": {"id": 1, "task": "Test task objective"},
                        "is_reasoning_task": False,
                        "next_pending_task_objective": "Next task",
                        "next_task_activated": True,
                    }
                )

        class MockCodeGeneration(BaseTool):
            def __init__(self):
                super().__init__("code_generation_with_guidance")

            async def execute(self, exec_context, **kwargs):
                # Record what we see
                seen_objective["value"] = exec_context.inputs.context.get("active_task_objective")
                return ToolResult(
                    output_ui="Generated code",
                    output_type=ToolOutputType.EXECUTE_ONLY,
                    output_workflow={"generated_code": "print('hello')"}
                )

        # Build a simple graph
        graph = StateGraph(KaiState)

        mark_tool = MockMarkNextTaskActive()
        code_tool = MockCodeGeneration()

        graph.add_node("mark_next_task_active", mark_tool.as_graph_node())
        graph.add_node("code_generation", code_tool.as_graph_node())

        graph.set_entry_point("mark_next_task_active")
        graph.add_edge("mark_next_task_active", "code_generation")
        graph.add_edge("code_generation", END)

        compiled = graph.compile()

        # Run the graph
        initial_state = {
            "active_task_objective": None,  # Initially None
            "task_list": {"tasks": [{"id": 1, "task": "Test task", "status": "pending"}]},
            "user_query": "",
        }

        async for output in compiled.astream(initial_state):
            pass  # Let graph complete

        # Check what code_generation saw
        assert seen_objective["value"] == "Test task objective", \
            f"code_generation should see 'Test task objective' but saw '{seen_objective['value']}'"


    @pytest.mark.asyncio
    async def test_state_flows_through_router(self):
        """
        Test state flow when router is between nodes (like the real orchestrator).

        Flow: router -> mark_next_task_active -> router -> code_generation
        """
        from langgraph.graph import StateGraph, END
        from kai.core.state import KaiState
        from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType

        seen_objective = {"value": None}
        router_calls = []

        class MockMarkNextTaskActive(BaseTool):
            def __init__(self):
                super().__init__("mark_next_task_active")

            async def execute(self, exec_context, **kwargs):
                return ToolResult(
                    output_ui="Task activated",
                    output_type=ToolOutputType.TASK_LIST_DISPLAY,
                    output_workflow={
                        "active_task_objective": "Test task objective",
                        "active_task": {"id": 1, "task": "Test task objective"},
                        "is_reasoning_task": False,
                        "next_task_activated": True,
                    }
                )

        class MockCodeGeneration(BaseTool):
            def __init__(self):
                super().__init__("code_generation")

            async def execute(self, exec_context, **kwargs):
                seen_objective["value"] = exec_context.inputs.context.get(
                    "active_task_objective"
                )
                return ToolResult(
                    output_ui="Generated",
                    output_type=ToolOutputType.EXECUTE_ONLY,
                    output_workflow={"generated_code": "print('hello')"}
                )

        def router(state: dict) -> str:
            router_calls.append(state.get("active_task_objective"))
            if not state.get("next_task_activated"):
                return "mark_next_task_active"
            elif not state.get("generated_code"):
                return "code_generation"
            return "complete"

        graph = StateGraph(KaiState)

        mark_tool = MockMarkNextTaskActive()
        code_tool = MockCodeGeneration()

        graph.add_node("router", lambda s: {})  # Router doesn't modify state
        graph.add_node("mark_next_task_active", mark_tool.as_graph_node())
        graph.add_node("code_generation", code_tool.as_graph_node())

        graph.set_entry_point("router")

        graph.add_conditional_edges(
            "router",
            router,
            {
                "mark_next_task_active": "mark_next_task_active",
                "code_generation": "code_generation",
                "complete": END,
            }
        )

        graph.add_edge("mark_next_task_active", "router")
        graph.add_edge("code_generation", "router")

        compiled = graph.compile()

        initial_state = {
            "active_task_objective": None,
            "next_task_activated": False,
            "generated_code": None,
            "task_list": {"tasks": []},
            "user_query": "",
        }

        async for output in compiled.astream(initial_state):
            pass

        # Router should have seen the objective after mark_next_task_active
        assert len(router_calls) >= 2, f"Router should be called multiple times"
        assert router_calls[1] == "Test task objective", \
            f"Router call 2 should see objective. Calls: {router_calls}"

        # code_generation should see it
        assert seen_objective["value"] == "Test task objective", \
            f"code_generation saw '{seen_objective['value']}'"


    @pytest.mark.asyncio
    async def test_checkpoint_preserves_state_across_invocations(self):
        """
        Test that checkpointer preserves state across separate graph invocations.

        This is the VSCode scenario:
        1. First invocation: mark_next_task_active sets active_task_objective
        2. Graph exits (complete)
        3. Second invocation: code_generation should see active_task_objective

        CRITICAL: If initial_state has active_task_objective=None, does it
        override the checkpointed value?
        """
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.memory import MemorySaver
        from kai.core.state import KaiState
        from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType

        seen_objective = {"value": None}
        invocation_count = {"count": 0}

        class MockMarkNextTaskActive(BaseTool):
            def __init__(self):
                super().__init__("mark_next_task_active")

            async def execute(self, exec_context, **kwargs):
                return ToolResult(
                    output_ui="Task activated",
                    output_type=ToolOutputType.TASK_LIST_DISPLAY,
                    output_workflow={
                        "active_task_objective": "Persisted task objective",
                        "next_task_activated": True,
                    }
                )

        class MockCodeGeneration(BaseTool):
            def __init__(self):
                super().__init__("code_generation")

            async def execute(self, exec_context, **kwargs):
                seen_objective["value"] = exec_context.inputs.context.get(
                    "active_task_objective"
                )
                return ToolResult(
                    output_ui="Generated",
                    output_type=ToolOutputType.EXECUTE_ONLY,
                    output_workflow={"generated_code": "print('hello')"}
                )

        def router(state: dict) -> str:
            invocation_count["count"] += 1
            inv = invocation_count["count"]

            # First invocation: activate task, then exit
            if inv == 1:
                if not state.get("next_task_activated"):
                    return "mark_next_task_active"
                return "complete"

            # Second invocation: generate code
            if inv == 2:
                if not state.get("generated_code"):
                    return "code_generation"
                return "complete"

            return "complete"

        graph = StateGraph(KaiState)
        memory = MemorySaver()

        mark_tool = MockMarkNextTaskActive()
        code_tool = MockCodeGeneration()

        graph.add_node("router", lambda s: {})
        graph.add_node("mark_next_task_active", mark_tool.as_graph_node())
        graph.add_node("code_generation", code_tool.as_graph_node())

        graph.set_entry_point("router")
        graph.add_conditional_edges(
            "router",
            router,
            {
                "mark_next_task_active": "mark_next_task_active",
                "code_generation": "code_generation",
                "complete": END,
            }
        )
        graph.add_edge("mark_next_task_active", "router")
        graph.add_edge("code_generation", "router")

        compiled = graph.compile(checkpointer=memory)
        thread_id = "test_thread"
        config = {"configurable": {"thread_id": thread_id}}

        # FIRST INVOCATION - activates task
        initial_state_1 = {
            "active_task_objective": None,  # Initially None
            "next_task_activated": False,
            "task_list": {"tasks": []},
            "user_query": "",
        }

        async for output in compiled.astream(initial_state_1, config):
            pass

        # SECOND INVOCATION - should see the persisted objective
        # THIS IS THE BUG: if we pass active_task_objective=None again,
        # it might override the checkpoint!
        initial_state_2 = {
            "active_task_objective": None,  # <-- Problem: this might override!
            "task_list": {"tasks": []},
            "user_query": "",
        }

        async for output in compiled.astream(initial_state_2, config):
            pass

        # Did code_generation see the persisted value?
        assert seen_objective["value"] == "Persisted task objective", \
            f"code_generation should see persisted value but saw: {seen_objective['value']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
