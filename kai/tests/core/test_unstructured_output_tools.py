"""Tests for unstructured output tools with real LLM integration.

Tests each unstructured output tool individually using qwen3:0.6b to verify
code generation, reasoning, and question answering functionality.
"""

import pytest
from typing import Dict, Any
from kai.core.agent import KaiAgent
from kai.core.orchestration.execution_context import ExecutionInputs, BacktrackingContext, ExecutionContext
from kai.core.orchestration.prompt_tools import (
    CodeGenerationTool,
    CodeGenerationWithGuidanceTool,
    ReasoningResponseWithGuidanceTool,
    CodeUpdateTool,
    QuestionAnsweringTool
)
from kai.core.orchestration.base_tool import ToolOutputType


class TestUnstructuredOutputTools:
    """Test all unstructured output tools with real LLM."""

    @pytest.fixture
    def llm_interface(self):
        """Create LLM interface using smaller qwen model for testing."""
        agent = KaiAgent(llm_provider='ollama', model="qwen3:0.6b")
        return agent.llm_interface

    def create_basic_context(self, **kwargs) -> ExecutionContext:
        """Create basic execution context for tool testing."""
        context_fields = {
            "current_cell": kwargs.get("current_cell", ""),
            "error_message": kwargs.get("error_message", ""),
            "execution_history": kwargs.get("execution_history", []),
            "conversation_history": kwargs.get("conversation_history", []),
            "notebook_structure": kwargs.get("notebook_structure", {
                "totalCells": 3,
                "allCells": ["import pandas as pd", "df = pd.read_csv('data.csv')", "print(df.head())"]
            }),
            "last_execution_failed": kwargs.get("last_execution_failed", False),
            "autonomous_mode": kwargs.get("autonomous_mode", True),
            "last_cell_modified_in_auto_mode": kwargs.get("last_cell_modified_in_auto_mode", 1),
            "positioning_info": kwargs.get("positioning_info", {"target_cell": 1}),
            "active_task_objective": kwargs.get("active_task_objective", "Load and analyze data"),
        }
        # Allow additional kwargs to override
        context_fields.update({k: v for k, v in kwargs.items() if k not in context_fields})

        exec_inputs = ExecutionInputs(
            user_query=kwargs.get("user_query", "Analyze the data"),
            context=context_fields,
            task_list=kwargs.get("task_list", {
                "tasks": [
                    {"id": 1, "task": "Load data", "status": "completed"},
                    {"id": 2, "task": "Analyze data", "status": "active"}
                ]
            }),
            backtracking_context=kwargs.get("backtracking_context", None)
        )

        return ExecutionContext(
            inputs=exec_inputs,
            session_metadata={"session_id": "test"}
        )

    @pytest.mark.asyncio
    async def test_code_generation_tool(self, llm_interface):
        """Test CodeGenerationTool with real LLM."""
        tool = CodeGenerationTool(llm_interface)
        context = self.create_basic_context(
            user_query="Create a scatter plot of the data",
            autonomous_mode=False  # Manual mode returns raw response
        )

        try:
            result = await tool.execute(context)
            assert result is not None
            assert hasattr(result, 'output_ui')
            assert hasattr(result, 'output_type')

            # In manual mode, output should be a string
            assert isinstance(result.output_ui, str)
            assert len(result.output_ui) > 0
            assert result.output_type == ToolOutputType.RESPONSE

            print(f"CodeGenerationTool result length: {len(result.output_ui)} chars")
        except Exception as e:
            pytest.fail(f"CodeGenerationTool failed with unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_code_generation_with_guidance_tool(self, llm_interface):
        """Test CodeGenerationWithGuidanceTool with real LLM."""
        tool = CodeGenerationWithGuidanceTool(llm_interface)
        context = self.create_basic_context(
            user_query="Analyze single-cell RNA-seq data",
            active_task_objective="Perform quality control on raw counts",
            positioning_info={"target_cell": 2},
            autonomous_mode=True
        )

        try:
            result = await tool.execute(context)
            assert result is not None
            assert hasattr(result, 'output_ui')
            assert hasattr(result, 'output_type')

            # In autonomous mode, output should be a dict for VSCode
            assert isinstance(result.output_ui, dict)
            assert 'code' in result.output_ui
            assert 'positioning_info' in result.output_ui
            assert 'should_replace_code' in result.output_ui
            assert 'cell_type' in result.output_ui

            # Verify code cell format
            assert result.output_ui['cell_type'] == 'code'
            assert result.output_ui['should_replace_code'] == 'false'
            assert isinstance(result.output_ui['code'], str)
            assert len(result.output_ui['code']) > 0
            assert result.output_type == ToolOutputType.EXECUTE_ONLY

            print(f"CodeGenerationWithGuidanceTool generated {len(result.output_ui['code'])} chars of code")
        except Exception as e:
            pytest.fail(f"CodeGenerationWithGuidanceTool failed with unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_reasoning_response_with_guidance_tool(self, llm_interface):
        """Test ReasoningResponseWithGuidanceTool with real LLM."""
        tool = ReasoningResponseWithGuidanceTool(llm_interface)
        context = self.create_basic_context(
            user_query="Interpret the analysis results",
            active_task_objective="[reasoning] Summarize the quality control findings",
            positioning_info={"target_cell": 3},
            execution_history=["Cell 0: Loaded 10000 cells", "Cell 1: Filtered to 8500 cells"],
            autonomous_mode=True
        )

        try:
            result = await tool.execute(context)
            assert result is not None
            assert hasattr(result, 'output_ui')
            assert hasattr(result, 'output_type')

            # Should return markdown cell format
            assert isinstance(result.output_ui, dict)
            assert 'code' in result.output_ui  # VSCode uses 'code' field for markdown content too
            assert 'positioning_info' in result.output_ui
            assert 'should_replace_code' in result.output_ui
            assert 'cell_type' in result.output_ui

            # Verify markdown cell format
            assert result.output_ui['cell_type'] == 'markdown'
            assert result.output_ui['should_replace_code'] == 'false'
            assert isinstance(result.output_ui['code'], str)
            assert len(result.output_ui['code']) > 0
            assert result.output_type == ToolOutputType.EXECUTE_ONLY

            print(f"ReasoningResponseWithGuidanceTool generated {len(result.output_ui['code'])} chars of markdown")
        except Exception as e:
            pytest.fail(f"ReasoningResponseWithGuidanceTool failed with unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_code_update_tool_with_error(self, llm_interface):
        """Test CodeUpdateTool with error fixing scenario."""
        tool = CodeUpdateTool(llm_interface)
        context = self.create_basic_context(
            user_query="Fix the syntax error",
            current_cell="df = pd.read_csv(",  # Incomplete code with error
            error_message="SyntaxError: unexpected EOF while parsing",
            last_execution_failed=True,
            active_task_objective="Load the dataset",
            positioning_info={"target_cell": 1},
            error_recovery_strategy="REPLACE_AND_RETRY",
            autonomous_mode=True
        )

        try:
            result = await tool.execute(context)
            assert result is not None
            assert hasattr(result, 'output_ui')
            assert hasattr(result, 'output_type')

            # In autonomous mode, should return dict
            assert isinstance(result.output_ui, dict)
            assert 'code' in result.output_ui
            assert 'positioning_info' in result.output_ui
            assert 'should_replace_code' in result.output_ui
            assert 'cell_type' in result.output_ui

            # Verify code replacement format
            assert result.output_ui['cell_type'] == 'code'
            assert result.output_ui['should_replace_code'] == 'true'
            assert isinstance(result.output_ui['code'], str)
            assert len(result.output_ui['code']) > 0
            assert result.output_type == ToolOutputType.EXECUTE_ONLY

            print(f"CodeUpdateTool generated fix: {len(result.output_ui['code'])} chars")
        except Exception as e:
            pytest.fail(f"CodeUpdateTool failed with unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_code_update_tool_manual_mode(self, llm_interface):
        """Test CodeUpdateTool in manual mode with retry objective."""
        tool = CodeUpdateTool(llm_interface)
        context = self.create_basic_context(
            user_query="Improve this code to be more efficient",
            current_cell="for i in range(len(df)):\n    print(df.iloc[i])",
            last_execution_failed=False,
            autonomous_mode=False,
            positioning_info={"target_cell": 2},
            retry_objective="Make the code more efficient using vectorization"
        )

        try:
            result = await tool.execute(context)
            assert result is not None

            # In manual mode, output should be raw string
            assert isinstance(result.output_ui, str)
            assert len(result.output_ui) > 0
            assert result.output_type == ToolOutputType.RESPONSE

            print(f"CodeUpdateTool (manual) result length: {len(result.output_ui)} chars")
        except Exception as e:
            pytest.fail(f"CodeUpdateTool (manual mode) failed with unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_question_answering_tool(self, llm_interface):
        """Test QuestionAnsweringTool with real LLM."""
        tool = QuestionAnsweringTool(llm_interface)
        context = self.create_basic_context(
            user_query="What does this code do?",
            current_cell="df = pd.read_csv('data.csv')\nprint(df.head())",
            autonomous_mode=False
        )

        try:
            result = await tool.execute(context)
            assert result is not None
            assert hasattr(result, 'output_ui')
            assert hasattr(result, 'output_type')

            # Should return text response
            assert isinstance(result.output_ui, str)
            assert len(result.output_ui) > 0
            assert result.output_type == ToolOutputType.RESPONSE

            print(f"QuestionAnsweringTool response: {len(result.output_ui)} chars")
        except Exception as e:
            pytest.fail(f"QuestionAnsweringTool failed with unexpected error: {e}")

    @pytest.mark.asyncio
    async def test_code_extraction_failure(self, llm_interface):
        """Test that tools handle code extraction failures gracefully."""
        tool = CodeGenerationWithGuidanceTool(llm_interface)

        # Mock a response that won't have extractable code
        from unittest.mock import patch

        # Create a mock that returns non-code text
        async def mock_generate(*args, **kwargs):
            return "This is just text without any code blocks."

        with patch.object(tool.llm_provider, 'generate', new=mock_generate):
            context = self.create_basic_context(
                positioning_info={"target_cell": 1},
                autonomous_mode=True
            )

            # The tool should catch the error and return an error result
            result = await tool.execute(context)

            # Check that an error response was returned
            assert result is not None
            assert hasattr(result, 'output_ui')
            # Error handling returns string message
            assert isinstance(result.output_ui, str)
            assert "error" in result.output_ui.lower() or "code_generation_with_guidance" in result.output_ui.lower()

            print(f"✓ Tool handled extraction failure gracefully: {result.output_ui[:100]}...")