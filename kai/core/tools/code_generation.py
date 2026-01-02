"""LLM-based code generation tools.

This module provides:
- CodeGenerationTool: Basic code generation from prompts
- CodeGenerationWithGuidanceTool: Code generation with workflow guidance
"""

from typing import TYPE_CHECKING

from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger
from .base import ToolResult, ToolOutputType
from .prompt_base import UnstructuredPromptTool, extract_code_from_response

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.core.llm_interface import LLMInterface

logger = setup_logger(__name__)


class CodeGenerationTool(UnstructuredPromptTool):
    """Tool for generating code.

    **UI Returns:**
    - Autonomous mode: Dict with "code", "positioning_info", "should_replace"
      fields for VSCode execution
    - Manual mode: Raw LLM response string for chat display
    - `output_type`: EXECUTE_ONLY (autonomous) or RESPONSE (manual)

    **Workflow Returns:**
    - None - this tool doesn't propagate workflow data

    **Used by workflows:** Regular request workflow for manual code generation
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__(
            "code_generation",
            PromptScenario.CODE_GENERATION,
            llm_interface
        )

    async def _process_response(
        self, response: str, state: 'KaiState'
    ) -> ToolResult:
        """Process code generation response and format for VSCode."""
        autonomous_mode = state["autonomous_mode"]
        positioning_info = state['positioning_info']

        # Extract clean code from response
        extracted_code = extract_code_from_response(response)

        # If no code was extracted, raise an error to trigger retry
        if extracted_code is None:
            raise ValueError(
                f"CodeGenerationTool could not extract code from response. "
                f"Response: {response}"
            )

        # Create VSCode-ready response - only include fields VSCode uses
        if autonomous_mode:
            vscode_response = {
                "code": extracted_code,
                "positioning_info": positioning_info,
                "should_replace": False,
                "cell_type": "code"
            }
            output_type = ToolOutputType.EXECUTE_ONLY

            # Create workflow output for LangGraph state
            workflow_output = {
                "generated_code": extracted_code,
                "target_cell": positioning_info.get("target_cell_index", 0)
            }
        else:
            vscode_response = response  # Manual mode: return full response
            output_type = ToolOutputType.RESPONSE
            workflow_output = {}

        return ToolResult(
            output_ui=vscode_response,
            output_type=output_type,
            output_workflow=workflow_output
        )


class CodeGenerationWithGuidanceTool(UnstructuredPromptTool):
    """Code generation with workflow guidance for autonomous mode.

    Uses reference workflows and task context to generate better code.
    Always runs in autonomous mode.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__(
            "code_generation_with_guidance",
            PromptScenario.CODE_GENERATION_WITH_GUIDANCE,
            llm_interface
        )

    async def _process_response(
        self, response: str, state: 'KaiState'
    ) -> ToolResult:
        """Process code generation response and format for VSCode."""
        positioning_info = state['positioning_info']

        # Extract clean code from response
        extracted_code = extract_code_from_response(response)

        # If no code was extracted, raise an error to trigger retry
        if extracted_code is None:
            raise ValueError(
                f"CodeGenerationWithGuidanceTool failed to extract code. "
                f"Response length: {len(response)}. This will trigger a retry."
            )

        # Log code generation action
        code_preview = (
            extracted_code[:100].replace('\n', ' ')
            if len(extracted_code) > 100
            else extracted_code.replace('\n', ' ')
        )
        target_cell = positioning_info.get(
            "target_cell", positioning_info.get("target_cell_index", -1)
        )
        logger.info(f"Generated code for cell {target_cell + 1}: {code_preview}...")

        # Check if restart is required (from backtrack_recovery tool)
        restart_required = state.get("restart_required", False)

        # Create VSCode-ready response
        vscode_response = {
            "code": extracted_code,
            "positioning_info": positioning_info,
            "should_replace": False,
            "cell_type": "code",
            "restart_required": restart_required,  # For backtracking with kernel restart
        }

        # Create workflow output for LangGraph state
        workflow_output = {
            "generated_code": extracted_code,
            "target_cell": positioning_info.get("target_cell_index", 0),
            # Clear backtracking/retry state
            "cells_to_delete": None,
            "cells_deleted": None,
            "backtrack_recovery_done": None,
            "recovery_objective": None,
            "retry_objective": None,
            "restart_required": None,  # Clear after use
        }

        return ToolResult(
            output_ui=vscode_response,
            output_workflow=workflow_output,
            output_type=ToolOutputType.EXECUTE_ONLY
        )
