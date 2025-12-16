"""Base tool interface for unified tool architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .execution_context import ExecutionContext


class ToolOutputType(Enum):
    """Types of tool outputs for different handling by VSCode."""
    RESPONSE = "response"                     # Normal chat response
    DISPLAY_ONLY = "display"                  # Show in chat but don't expect user response
    TASK_LIST_DISPLAY = "task_list_display"   # Show in chat but don't expect user response
    EXECUTE_ONLY = "execute_code"             # Execute code but don't show in chat (autonomous mode)
    NO_OUTPUT = "no_output"                   # Is not parsed by VSCode but is added to the job queue
    REFERENCE_WORKFLOWS = "reference_workflows"  # Reference workflow IDs to be stored in VSCode


@dataclass
class ToolResult:
    """Result from tool execution."""
    output_ui: Any
    output_workflow: Optional[Dict[str, Any]] = None  # Context for subsequent workflow tools
    output_type: ToolOutputType = ToolOutputType.RESPONSE
    

class BaseTool(ABC):
    """Base class for all tools in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, exec_context: "ExecutionContext") -> ToolResult:
        """Execute the tool with ExecutionContext."""
        pass

    def can_execute(self, exec_context: "ExecutionContext") -> bool:
        """Check if tool can execute with given ExecutionContext."""
        return True

    def get_required_inputs(self) -> List[str]:
        """Get list of required input keys."""
        return []

    def get_output_schema(self) -> Dict[str, Any]:
        """Get schema describing tool output."""
        return {}

    def as_graph_node(self):
        """
        Convert tool to LangGraph node function with retry logic.

        Returns a node function that:
        1. Converts state dict to ExecutionContext
        2. Executes tool with retry logic
        3. Returns state updates (output_workflow)
        4. Returns ToolResult in state for orchestrator to send to UI
        """
        async def node_function(state: dict) -> dict:
            import time
            from .execution_context import ExecutionContext
            from kai.utils import setup_logger

            logger = setup_logger(__name__)
            tool_start_time = time.time()

            # Convert state dict to ExecutionContext
            exec_context = ExecutionContext.from_dict(state)

            # Execute tool with retry logic
            max_retries = 5
            last_error = None
            last_failed_output = None
            context_length_factor = 1.0

            reasoning_level_reduction = {
                "low": "low",
                "medium": "low",
                "high": "medium"
            }
            reasoning_level = None

            original_user_query = exec_context.inputs.user_query

            for attempt in range(max_retries):
                try:
                    exec_context.inputs.user_query = original_user_query

                    if attempt > 0:
                        format_reminder = f"\n\nIMPORTANT: This is attempt #{attempt + 1}. You failed to format your output correctly last time."

                        if last_error:
                            format_reminder += f"\n\nThe error was:\n{last_error}"

                        from .prompt_tools import StructuredPromptTool
                        if isinstance(self, StructuredPromptTool):
                            format_reminder += "\n\nPlease ensure your response strictly follows the required JSON schema format. Double-check all brackets, quotes, and commas."
                        else:
                            format_reminder += "\n\nPlease ensure your response strictly follows the required format."

                        if last_failed_output:
                            truncated = last_failed_output[:500] + "... [truncated]" if len(last_failed_output) > 500 else last_failed_output
                            format_reminder += f"\n\nYour previous failed output was:\n{truncated}"

                        exec_context.inputs.user_query += format_reminder

                    result = await self.execute(
                        exec_context,
                        context_length_factor=context_length_factor,
                        reasoning_level=reasoning_level
                    )

                    exec_context.inputs.user_query = original_user_query

                    # Log tool completion time
                    tool_duration = time.time() - tool_start_time
                    from .ui_communicator import UICommunicator
                    vscode = UICommunicator()
                    vscode.send_console_message(f"[KAI] Tool {self.name} completed in {tool_duration:.3f}s")

                    # Return both state updates AND the tool result for UI communication
                    state_update = {}
                    if result.output_workflow:
                        state_update.update(result.output_workflow)

                    # Store tool result in state for orchestrator to send to UI
                    state_update['_last_tool_result'] = {
                        'tool_name': self.name,
                        'result': result,
                        'exec_context': exec_context
                    }

                    return state_update

                except Exception as e:
                    error_str = str(e)
                    error_type = type(e).__name__

                    last_error = f"{error_type}: {error_str}"

                    if hasattr(e, 'raw_output'):
                        last_failed_output = e.raw_output

                    # For ValidationError, just show the error type and count, not the full details
                    if error_type == "ValidationError":
                        # Extract error count from pydantic validation error
                        error_count = error_str.split(" validation error")[0] if "validation error" in error_str else "unknown"
                        logger.warning(f"Tool {self.name} failed on attempt {attempt + 1}/{max_retries}: {error_count} validation errors (schema mismatch)")
                    else:
                        # For other errors, show full message
                        logger.warning(f"Tool {self.name} failed on attempt {attempt + 1}/{max_retries}: {error_type}: {error_str}")

                    if attempt == max_retries - 1:
                        logger.error(f"Tool {self.name} failed after {max_retries} attempts")
                        raise ValueError(f"Error in {self.name} after {max_retries} attempts: {str(e)}")

                    context_length_factor *= 2.0

                    if attempt >= max_retries - 3 and hasattr(self, "reasoning_level"):
                        previous_reasoning_level = reasoning_level if reasoning_level else getattr(self, "reasoning_level", "medium")
                        reasoning_level = reasoning_level_reduction.get(previous_reasoning_level, "medium")

        return node_function