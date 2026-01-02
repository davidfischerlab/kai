"""UI Communication Architecture - Dedicated class for UI interactions (VSCode/Jupyter)."""

from dataclasses import dataclass
import json
import sys
import threading
from typing import Optional, Dict, Any

from kai.core.tools.base import ToolResult


@dataclass
class VscodeInputContext:
    """Minimal context needed for UI communication."""
    session_id: str
    autonomous_mode: bool
    request_id: str

    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> "VscodeInputContext":
        """Create VscodeInputContext from state dict."""
        return cls(
            session_id=state.get("session_id", ""),
            autonomous_mode=state.get("autonomous_mode", False),
            request_id=state.get("request_id", "")
        )


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
    # Class-level lock to prevent concurrent stdout writes (prevents JSON concatenation)
    _stdout_lock = threading.Lock()
    # Class-level hook for capturing tool results (used by Jupyter interface)
    # When set, all instances call this instead of printing to stdout
    _tool_result_hook = None
    # Class-level hook for capturing workflow results (used by Jupyter interface)
    _workflow_result_hook = None

    @classmethod
    def set_vscode_mode(cls, enabled: bool):
        """Set whether we're running in VSCode mode (JSON stdout) or Jupyter mode (logger)."""
        cls._vscode_mode = enabled

    @classmethod
    def set_tool_result_hook(cls, hook):
        """Set a hook function to capture all tool results (used by Jupyter interface).

        Args:
            hook: Async function(result, context) to call for each tool result,
                  or None to clear the hook.
        """
        cls._tool_result_hook = hook

    @classmethod
    def set_workflow_result_hook(cls, hook):
        """Set a hook function to capture all workflow results (used by Jupyter interface).

        Args:
            hook: Async function(field, state) to call for each workflow result,
                  or None to clear the hook.
        """
        cls._workflow_result_hook = hook

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
            with UICommunicator._stdout_lock:
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

        # Check for Jupyter hook first - this allows Jupyter interface to capture
        # tool results from ALL UICommunicator instances (including ones created
        # inside tools via base.py)
        if UICommunicator._tool_result_hook is not None:
            await UICommunicator._tool_result_hook(result, context)
            return

        # Only send JSON to stdout in VSCode mode
        # In Jupyter mode, tool results are handled by JupyterInterface directly
        if not UICommunicator._vscode_mode:
            return

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
        with UICommunicator._stdout_lock:
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
        # CRITICAL: Check for Jupyter hook FIRST, BEFORE disabled check!
        # In Jupyter mode, _disabled=True to suppress VSCode stdout, but we still
        # need to call the hook so Jupyter can receive LOOP_COMPLETE signals.
        # Without this, the autonomous loop never terminates in Jupyter.
        if UICommunicator._workflow_result_hook is not None:
            await UICommunicator._workflow_result_hook(field, state)
            return

        if self._disabled:
            return  # Skip sending if communication is disabled (and no hook set)

        response_data = {
            "type": "workflow_result",
            field: state
        }

        with UICommunicator._stdout_lock:
            print(json.dumps(response_data), flush=True)

    async def send_task_list_update(self, task_list: Dict[str, Any]):
        """
        Send task list update to VSCode UI.

        Used by nodes (like mark_reasoning_completed) that modify task list
        but aren't tools with full ToolResult infrastructure.

        Args:
            task_list: The updated task list dict with tasks array
        """
        if self._disabled:
            return  # Skip sending if communication is disabled

        # Only send JSON to stdout in VSCode mode
        if not UICommunicator._vscode_mode:
            return

        response_data = {
            "type": "task_list_display",
            "response": {
                "text": json.dumps(task_list)
            }
        }

        with UICommunicator._stdout_lock:
            print(json.dumps(response_data), flush=True)
