"""Base tool interface for unified tool architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from ..orchestration.state import KaiState


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
    async def execute(self, state: "KaiState") -> ToolResult:
        """Execute the tool with KaiState."""
        pass

    def can_execute(self, state: "KaiState") -> bool:
        """Check if tool can execute with given state."""
        return True

    def get_required_inputs(self) -> List[str]:
        """Get list of required input keys."""
        return []

    def get_output_schema(self) -> Dict[str, Any]:
        """Get schema describing tool output."""
        return {}

    def as_graph_node(self):
        """
        Convert tool to LangGraph node function with context reduction retry.

        Retry Strategy (two-layer approach):
        1. Network/timeout retries: Handled by tenacity at LLM provider level
           (see llm_interface.py - exponential backoff for transient errors)
        2. Validation retries: Handled here with context reduction
           (for schema/parsing errors that need different prompt handling)

        Returns a node function that:
        1. Passes state dict directly to tool (no ExecutionContext wrapper)
        2. Executes tool with context reduction retry for validation errors
        3. Returns state updates (output_workflow)
        4. Returns ToolResult in state for orchestrator to send to UI
        """
        async def node_function(state: dict) -> dict:
            import time
            from pydantic import ValidationError as PydanticValidationError
            from ..llm_interface import NonRetriableError
            from kai.utils import setup_logger

            logger = setup_logger(__name__)
            tool_start_time = time.time()

            # Context reduction retry for validation errors
            # (Network retries are handled by tenacity at LLM layer)
            max_validation_retries = 3
            context_length_factor = 1.0
            original_user_query = state.get("user_query", "")

            for attempt in range(max_validation_retries):
                try:
                    state["user_query"] = original_user_query

                    # Add format reminder on retry attempts
                    if attempt > 0:
                        format_reminder = (
                            f"\n\nIMPORTANT: Attempt #{attempt + 1}. "
                            "Previous response had formatting errors. "
                            "Ensure strict JSON schema compliance."
                        )
                        state["user_query"] = original_user_query + format_reminder

                    result = await self.execute(
                        state,
                        context_length_factor=context_length_factor,
                    )

                    state["user_query"] = original_user_query

                    # Log tool completion time
                    tool_duration = time.time() - tool_start_time
                    from ..orchestration.ui_communicator import (
                        UICommunicator,
                        VscodeInputContext,
                    )
                    vscode = UICommunicator()
                    vscode.send_console_message(
                        f"[KAI] Tool {self.name} completed in {tool_duration:.3f}s"
                    )

                    # Send EXECUTE_ONLY results directly from subgraph
                    # (Main graph streaming loop only sees final subgraph output)
                    # Only send execute_code type - other types (task_list_display,
                    # reference_workflows, etc.) are handled by orchestrator at
                    # appropriate times (e.g., after planning completes)
                    if (result.output_ui and
                            result.output_type == ToolOutputType.EXECUTE_ONLY):
                        vscode_context = VscodeInputContext.from_state(state)
                        await vscode.send_tool_result(result, vscode_context)

                    # Return state updates AND tool result for UI
                    state_update = {}
                    if result.output_workflow:
                        state_update.update(result.output_workflow)

                    # Only include _last_tool_result for NON-EXECUTE_ONLY types
                    # EXECUTE_ONLY types are already sent directly above (lines 120-123)
                    # to avoid duplicate messages in VSCode
                    if result.output_type != ToolOutputType.EXECUTE_ONLY:
                        # Only include minimal state fields needed for UI communication
                        # (full state would cause msgpack serialization recursion issues)
                        state_update['_last_tool_result'] = {
                            'tool_name': self.name,
                            'result': result,
                            'state': {
                                'session_id': state.get('session_id', ''),
                                'autonomous_mode': state.get('autonomous_mode', False),
                                'request_id': state.get('request_id', ''),
                            }
                        }

                    return state_update

                except (NonRetriableError, ValueError, PydanticValidationError) as e:
                    # Validation/parsing errors - retry with context reduction
                    error_type = type(e).__name__
                    logger.warning(
                        f"Tool {self.name} validation error "
                        f"(attempt {attempt + 1}/{max_validation_retries}): "
                        f"{error_type}"
                    )

                    if attempt == max_validation_retries - 1:
                        logger.error(
                            f"Tool {self.name} failed after "
                            f"{max_validation_retries} validation retries"
                        )
                        raise ValueError(
                            f"Error in {self.name} after "
                            f"{max_validation_retries} attempts: {e}"
                        )

                    # Increase context reduction factor for next attempt
                    context_length_factor *= 2.0

                except Exception as e:
                    # Unexpected errors - don't retry, propagate immediately
                    logger.error(f"Tool {self.name} unexpected error: {e}")
                    raise

        return node_function
