"""UI Communication Architecture - Dedicated class for UI interactions (VSCode/Jupyter)."""

from dataclasses import dataclass
import json
import sys
from typing import Optional

from .base_tool import ToolResult
from .execution_context import ExecutionInputs


@dataclass
class VscodeInputContext:
    """Minimal context needed for UI communication."""
    session_id: str
    inputs: 'ExecutionInputs'

    @property
    def autonomous_mode(self) -> bool:
        return self.inputs.context["autonomous_mode"]

    @property
    def request_id(self) -> str:
        return self.inputs.context["request_id"]


class UICommunicator:
    """
    Handles real-time communication with UI through stdout JSON messages.

    Dual Communication Architecture:

    1. **Promise-based Requests** (handled by KaiAgentProvider):
       - VSCode sends request → Python processes → Returns {"status": "processed"}
       - Promise fulfillment confirms "request was processed"

    2. **Real-time Messages** (handled by UICommunicator):
       - Tool results, progress updates, workflow completion signals
       - Streams continuously during Python processing
       - Contains actual workflow data and user-facing content

    Message Types:
    - `console_log`: Progress updates and debug messages
    - `display`: Tool results that appear in chat immediately
    - `execute_code`: Code generation for silent execution
    - `workflow_result`: Workflow completion signals (auto_loop_update/regular_chat_update)

    Message Flow: Python stdout → KaiAgentProvider.handleResponse → Chat UI
    """

    # Class-level flag to indicate VSCode mode (set by python-subprocess.py)
    _vscode_mode = False

    @classmethod
    def set_vscode_mode(cls, enabled: bool):
        """Set whether we're running in VSCode mode (JSON stdout) or Jupyter mode (logger)."""
        cls._vscode_mode = enabled

    def __init__(self):
        """Initialize VSCode communicator."""
        self._disabled = False

    def disable_communication(self):
        """Disable all VSCode communication (used when stopping autonomous execution)."""
        self._disabled = True

    def enable_communication(self):
        """Re-enable VSCode communication."""
        self._disabled = False

    def send_console_message(self, message: str):
        """Send message to console - VSCode JSON or Jupyter logger depending on mode."""
        if self._disabled:
            return  # Skip sending if communication is disabled

        if UICommunicator._vscode_mode:
            # VSCode mode: send JSON to stdout
            msg = {
                "type": "console_log",
                "message": message
            }
            print(json.dumps(msg))
            sys.stdout.flush()
        else:
            # Jupyter mode: use logger (goes to stderr, formatted nicely)
            import logging
            logger = logging.getLogger("kai.orchestration")
            logger.info(message)
    
    async def send_tool_result(self, result: ToolResult, context: VscodeInputContext):
        """
        Send message to VSCode.
        
        Creates JSON message with type: "display" that gets:
        1. Parsed by Agent Provider (agent-provider.ts)
        2. Routed to Chat Provider via streamUpdateCallback('display', ...)  
        3. Handled by _handleDisplayMessage() in chat-provider.ts
        4. Immediately added to chat via _addMessage()
        
        Args:
            result: tool result object
            context: context dict
            msg_type: message-identifying string categorical that is accepted by 
                KaiAgentProvider.handleResponse in the VScode interface.
        """
        if self._disabled:
            return  # Skip sending if communication is disabled
    
        response_data = {
            "type": result.output_type.value,
            "request_id": context.request_id,
            "response": result.output_ui,
            "session_id": context.session_id
        }
        
        # Add code field if present in output
        if isinstance(result.output_ui, dict) and "code" in result.output_ui:
            response_data["code"] = result.output_ui["code"]
        
        # Send to VSCode via stdout - picked up by Agent Provider
        print(json.dumps(response_data), flush=True)

    async def send_workflow_result(self, field: str, state: str):
        """
        Send workflow completion signal to VSCode.

        This is separate from promise fulfillment - it signals workflow state changes
        for autonomous loops or regular chat completion.

        Args:
            field: "auto_loop_update" (autonomous workflows) or "regular_chat_update" (regular requests)
            state: "LOOP_INCOMPLETE", "LOOP_INCOMPLETE_REQUIRE_FEEDBACK", "LOOP_COMPLETE", "STEP_COMPLETE", "ERROR"
        """
        if self._disabled:
            return  # Skip sending if communication is disabled

        response_data = {
            "type": "workflow_result",
            field: state
        }

        print(json.dumps(response_data), flush=True)
