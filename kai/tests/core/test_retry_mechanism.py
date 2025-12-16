"""Unit tests for retry mechanism with mocked LLM failures.

These tests FORCE JSON errors to verify retry logic executes correctly.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType
from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
import json


class TestRetryMechanism:
    """Test retry logic with forced failures."""

    @pytest.mark.asyncio
    async def test_retry_logic_executes_on_json_error(self):
        """Verify that JSON parsing errors trigger retry with format reminders."""

        # Create a mock tool that fails first 2 attempts, succeeds on 3rd
        call_count = 0

        class MockTool(BaseTool):
            def __init__(self):
                super().__init__("test_tool")
                self.description = "Test tool"

            async def execute(self, exec_context, **kwargs):
                nonlocal call_count
                call_count += 1

                if call_count <= 2:
                    # Simulate JSON parsing error (like production)
                    error = json.JSONDecodeError("Expecting ':' delimiter", "test", 100)
                    error.raw_output = '{"tasks": [{"name": "test"  "status": "pending"}]}'
                    raise error
                else:
                    # Succeed on 3rd attempt
                    return ToolResult(
                        output_ui=None,
                        output_type=ToolOutputType.NO_OUTPUT,
                        output_workflow={"success": True}
                    )

        tool = MockTool()
        node_func = tool.as_graph_node()

        # Create minimal state
        state = {
            "user_query": "test query",
            "task_list": {},
            "notebook_structure": {"totalCells": 0},
            "execution_history": [],
            "excluded_workflows": []
        }

        # Execute node function (which includes retry logic)
        result = await node_func(state)

        # Verify:
        # 1. Tool was called 3 times (2 failures + 1 success)
        assert call_count == 3, f"Expected 3 attempts, got {call_count}"

        # 2. Result is successful
        assert result.get("success") is True

        print(f"✅ Retry logic worked: {call_count} attempts, final success")

    @pytest.mark.asyncio
    async def test_retry_logic_fails_after_max_attempts(self):
        """Verify that retry logic gives up after 5 attempts."""

        call_count = 0

        class AlwaysFailingTool(BaseTool):
            def __init__(self):
                super().__init__("failing_tool")
                self.description = "Always fails"

            async def execute(self, exec_context, **kwargs):
                nonlocal call_count
                call_count += 1
                # Always fail with JSON error
                error = json.JSONDecodeError("Expecting ':' delimiter", "test", 100)
                error.raw_output = '{"invalid": json}'
                raise error

        tool = AlwaysFailingTool()
        node_func = tool.as_graph_node()

        state = {
            "user_query": "test query",
            "task_list": {},
            "notebook_structure": {"totalCells": 0},
            "execution_history": [],
            "excluded_workflows": []
        }

        # Should raise ValueError after 5 attempts
        with pytest.raises(ValueError) as exc_info:
            await node_func(state)

        # Verify error message mentions attempts
        error_msg = str(exc_info.value)
        assert "after" in error_msg.lower() and "attempts" in error_msg.lower(), \
            f"Error should mention retry attempts: {error_msg}"

        # Verify exactly 5 attempts were made
        assert call_count == 5, f"Expected 5 attempts, got {call_count}"

        print(f"✅ Retry logic correctly failed after {call_count} attempts")

    @pytest.mark.asyncio
    async def test_format_reminder_added_on_retry(self):
        """Verify that format reminders are added to user_query on retries."""

        captured_queries = []

        class QueryCapturingTool(BaseTool):
            def __init__(self):
                super().__init__("capture_tool")
                self.description = "Captures queries"

            async def execute(self, exec_context, **kwargs):
                # Capture the query for inspection
                captured_queries.append(exec_context.inputs.user_query)

                if len(captured_queries) < 3:
                    # Fail first 2 attempts
                    raise json.JSONDecodeError("Test error", "test", 0)
                else:
                    # Succeed on 3rd
                    return ToolResult(
                        output_ui=None,
                        output_type=ToolOutputType.NO_OUTPUT,
                        output_workflow={"success": True}
                    )

        tool = QueryCapturingTool()
        node_func = tool.as_graph_node()

        state = {
            "user_query": "original query",
            "task_list": {},
            "notebook_structure": {"totalCells": 0},
            "execution_history": [],
            "excluded_workflows": []
        }

        await node_func(state)

        # Verify we captured 3 queries
        assert len(captured_queries) == 3

        # First attempt should have original query
        assert captured_queries[0] == "original query"

        # Second attempt should have format reminder
        assert "IMPORTANT" in captured_queries[1]
        assert "attempt #2" in captured_queries[1]
        assert "original query" in captured_queries[1]  # Original preserved

        # Third attempt should also have format reminder
        assert "attempt #3" in captured_queries[2]

        print(f"✅ Format reminders added correctly on retries")
        print(f"   Attempt 1: {len(captured_queries[0])} chars")
        print(f"   Attempt 2: {len(captured_queries[1])} chars (with reminder)")
        print(f"   Attempt 3: {len(captured_queries[2])} chars (with reminder)")

    @pytest.mark.asyncio
    async def test_context_length_escalation(self):
        """Verify that context_length_factor doubles on each retry."""

        captured_factors = []

        class FactorCapturingTool(BaseTool):
            def __init__(self):
                super().__init__("factor_tool")
                self.description = "Captures context factors"

            async def execute(self, exec_context, context_length_factor=1.0, **kwargs):
                captured_factors.append(context_length_factor)

                if len(captured_factors) < 4:
                    raise json.JSONDecodeError("Test error", "test", 0)
                else:
                    return ToolResult(
                        output_ui=None,
                        output_type=ToolOutputType.NO_OUTPUT,
                        output_workflow={"success": True}
                    )

        tool = FactorCapturingTool()
        node_func = tool.as_graph_node()

        state = {
            "user_query": "test",
            "task_list": {},
            "notebook_structure": {"totalCells": 0},
            "execution_history": [],
            "excluded_workflows": []
        }

        await node_func(state)

        # Verify escalation: 1.0, 2.0, 4.0, 8.0
        assert captured_factors[0] == 1.0
        assert captured_factors[1] == 2.0
        assert captured_factors[2] == 4.0
        assert captured_factors[3] == 8.0

        print(f"✅ Context length escalation correct: {captured_factors}")

    @pytest.mark.asyncio
    async def test_raw_output_captured_for_reminder(self):
        """Verify that raw_output from errors is included in format reminders."""

        captured_queries = []

        class RawOutputTool(BaseTool):
            def __init__(self):
                super().__init__("raw_output_tool")
                self.description = "Test raw output"

            async def execute(self, exec_context, **kwargs):
                captured_queries.append(exec_context.inputs.user_query)

                if len(captured_queries) < 2:
                    # Fail with raw output
                    error = json.JSONDecodeError("Missing comma", "test", 0)
                    error.raw_output = '{"bad": "json" "missing": "comma"}'
                    raise error
                else:
                    return ToolResult(
                        output_ui=None,
                        output_type=ToolOutputType.NO_OUTPUT,
                        output_workflow={"success": True}
                    )

        tool = RawOutputTool()
        node_func = tool.as_graph_node()

        state = {
            "user_query": "test",
            "task_list": {},
            "notebook_structure": {"totalCells": 0},
            "execution_history": [],
            "excluded_workflows": []
        }

        await node_func(state)

        # Second attempt should include the failed output
        assert "previous failed output" in captured_queries[1].lower()
        assert "bad" in captured_queries[1] or "json" in captured_queries[1]

        print(f"✅ Raw output included in format reminder")
