"""Base classes for LLM-based tools that use prompts for execution.

This module provides:
- Helper functions for extracting code and validating task lists
- BasePromptTool: Base class with shared prompt logging functionality
- StructuredPromptTool: For tools using structured output via Pydantic schemas
- UnstructuredPromptTool: For tools using standard prompt scenarios
"""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union

from kai.utils import setup_logger
from kai.core.prompt_manager import PromptManager, PromptScenario
from ..llm_interface import BaseLLMProvider, LLMInterface
from ..utils import format_task_list
from .base import BaseTool, ToolResult, ToolOutputType
from kai.config.paths import get_debug_prompts_dir

if TYPE_CHECKING:
    from ..orchestration.state import KaiState

logger = setup_logger(__name__)


def extract_code_from_response(response: str) -> Optional[str]:
    """
    Extract code from LLM response containing markdown code blocks.

    Args:
        response: Full LLM response text containing markdown

    Returns:
        Clean code string or None if no code block found
    """
    # Match code blocks with optional language specification
    code_block_pattern = r'```(?:python|py)?\s*\n([\s\S]*?)\n```'
    match = re.search(code_block_pattern, response, re.IGNORECASE)

    if match:
        code = match.group(1).strip()
        logger.debug(f"Extracted code block: {len(code)} characters")
        return code

    # Fallback: if no code blocks, try to extract code from start of response
    lines = response.strip().split('\n')
    if lines and (lines[0].startswith('import ') or
                  lines[0].startswith('from ') or
                  any(keyword in lines[0] for keyword in
                      ['def ', 'class ', '# ', 'print(', 'plt.'])):
        logger.debug(
            "No code block markers found, treating entire response as code")
        return response.strip()

    logger.warning("No code found in response")
    return None


def validate_task_list_format(tasks: List[Dict[str, Any]], tool_name: str = "") -> None:
    """
    Validate that task list items conform to the expected format.

    Expected format for each task:
    {"id": int, "task": str, "status": "pending"|"active"|"completed"}

    Args:
        tasks: List of task dictionaries to validate
        tool_name: Name of the tool for error messages

    Raises:
        ValueError: If any task doesn't conform to the expected format
    """
    if not isinstance(tasks, list):
        raise ValueError(f"{tool_name}: tasks must be a list, got {type(tasks)}")

    if len(tasks) == 0:
        raise ValueError(f"{tool_name}: task list cannot be empty")

    valid_statuses = {"pending", "active", "completed"}

    for i, task in enumerate(tasks):
        if not isinstance(task, dict):
            raise ValueError(f"{tool_name}: task {i} must be a dict, got {type(task)}")

        # Check required fields
        required_fields = {"id", "task", "status"}
        missing_fields = required_fields - set(task.keys())
        if missing_fields:
            raise ValueError(f"{tool_name}: task {i} missing required fields: {missing_fields}")

        # Validate field types and values
        if not isinstance(task["id"], int):
            raise ValueError(f"{tool_name}: task {i} 'id' must be an integer, got {type(task['id'])}")

        if not isinstance(task["task"], str) or not task["task"].strip():
            raise ValueError(f"{tool_name}: task {i} 'task' must be a non-empty string")

        if task["status"] not in valid_statuses:
            raise ValueError(f"{tool_name}: task {i} 'status' must be one of {valid_statuses}, got '{task['status']}'")

    # Check for duplicate IDs
    task_ids = [t.get("id") for t in tasks if isinstance(t, dict)]
    if len(task_ids) != len(set(task_ids)):
        duplicates = [id for id in task_ids if task_ids.count(id) > 1]
        raise ValueError(f"{tool_name}: duplicate task IDs found: {duplicates}")


class BasePromptTool(BaseTool):
    """Base class for all LLM-based tools with shared prompt logging functionality."""
    llm_provider: BaseLLMProvider

    def __init__(self, name: str, scenario: PromptScenario, llm_interface: LLMInterface):
        super().__init__(name)
        self.scenario = scenario
        self.llm_provider = llm_interface.get_llm_for_tool(self)
        self.reasoning_level = llm_interface.get_reasoning_for_tool(self)
        self.prompt_manager = PromptManager()

    def _get_task_type(self) -> str:
        """Get task type for LLM interface."""
        return ("code_generation" if "code" in self.scenario.value
                else "general")

    def _log_prompt(self, system_prompt: str, user_prompt: str,
                    state: 'KaiState') -> str:
        """
        Log prompts to debug folder if DEBUG_PROMPTS is enabled.

        Data flow for session metadata:
        agent.py: session_metadata = {session_timestamp, notebook_uri,
                                     iteration_timestamp, iteration_counter}
        → orchestrator: state fields directly accessible
        → prompt_tools: Extract for debug folder naming

        Debug folder structure:
        ~/.kai_agent/prompt_debugging/{notebook_name}/
        session_{session_timestamp}_{auto|manual}/iteration_{iteration_timestamp}/
        """
        from kai.config.settings import settings
        from kai.config.paths import get_debug_prompts_dir
        from datetime import datetime

        if not settings.DEBUG_PROMPTS:
            return

        session_id = state["session_id"]
        session_timestamp = state["session_timestamp"]
        iteration_timestamp = state["iteration_timestamp"]
        is_autonomous = state["autonomous_mode"]
        notebook_uri = state["notebook_uri"]

        # Create notebook-specific identifier from URI
        notebook_identifier = "default_notebook"
        if notebook_uri:
            try:
                # Convert URI to safe folder name:
                # file:///path/to/notebook.ipynb -> notebook_ipynb
                import urllib.parse
                parsed_uri = urllib.parse.urlparse(notebook_uri)
                if parsed_uri.path:
                    path_parts = parsed_uri.path.split('/')
                    notebook_name = path_parts[-1]  # Get filename
                    if notebook_name:  # Ensure we have a valid filename
                        # Add full_agent_test prefix if this is a full_agent_test notebook
                        if 'full_agent_test' in path_parts:
                            notebook_identifier = f"full_agent_test/{notebook_name}"
                        else:
                            notebook_identifier = notebook_name
                        notebook_identifier = (notebook_identifier
                                                .replace('.', '_')
                                                .replace(' ', '_'))
            except Exception as parse_error:
                # Log parsing error but continue with default
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to parse notebook URI '{notebook_uri}': "
                    f"{parse_error}")
                notebook_identifier = "default_notebook"

        # Create session identifier with date prefix
        # (based on session timestamp)
        session_type = "auto" if is_autonomous else "manual"
        session_identifier = f"{session_timestamp}_{session_type}_{session_id}"

        # Create notebook-specific debug directory: notebook/session/
        # Date is already in the session identifier,
        # so we don't need a separate date folder
        notebook_debug_dir = (get_debug_prompts_dir() /
                                notebook_identifier /
                                session_identifier)
        notebook_debug_dir.mkdir(parents=True, exist_ok=True)
        # Create iteration subdirectory within the session folder
        iteration_identifier = f"{iteration_timestamp}"
        debug_dir = notebook_debug_dir / iteration_identifier
        # Create directory:
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp and tool name (milliseconds for fast calls)
        # Current timestamp for file naming
        now = datetime.now()
        ms_str = f"{now.microsecond // 1000:03d}"
        timestamp_str = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}-{ms_str}"
        filename = f"{timestamp_str}_{self.name}_prompt.txt"
        filepath = debug_dir / filename

        # Write prompt information to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Tool: {self.name}\n")
            f.write(f"Scenario: {self.scenario.value}\n")
            f.write(f"Timestamp: {timestamp_str}\n")
            f.write(f"Notebook: {notebook_identifier}\n")
            f.write(f"Notebook URI: {notebook_uri or 'N/A'}\n")
            f.write(f"Session ID: {session_id}\n")
            f.write(f"Session Init Time: {session_timestamp}\n")
            f.write(f"Iteration Time: {iteration_timestamp}\n")
            f.write(f"Autonomous Mode: {is_autonomous}\n")
            f.write(f"LLM Provider: {self.llm_provider.provider_name}\n")
            f.write(f"LLM Model: {self.llm_provider.model}\n")
            f.write("=" * 80 + "\n")
            f.write("SYSTEM PROMPT:\n")
            f.write("=" * 80 + "\n")
            f.write(system_prompt)
            f.write("\n" + "=" * 80 + "\n")
            f.write("USER PROMPT:\n")
            f.write("=" * 80 + "\n")
            f.write(user_prompt)
            f.write("\n" + "=" * 80 + "\n")
        return str(filepath)


class StructuredPromptTool(BasePromptTool):
    """Base class for LLM tools that use structured output via Pydantic schemas."""

    def __init__(self, name: str, scenario: PromptScenario,
                 llm_interface: 'LLMInterface'):
        super().__init__(name, scenario, llm_interface)

        # Lazy import to avoid circular dependency
        from kai.core.tools.schema_registry import SCHEMA_REGISTRY

        # Get the schema for this tool from the registry
        self.schema = SCHEMA_REGISTRY.get(name)
        if not self.schema:
            raise ValueError(f"No schema found for tool: {name}")

        # Verify that prompt manager can provide JSON format instruction for this scenario
        from kai.core.prompt_manager import PromptManager
        json_instruction = PromptManager()._get_json_format_instruction(scenario)
        if not json_instruction:
            raise ValueError(
                f"No schema mapping found for scenario {scenario.value}. "
                f"Add mapping in PromptManager.scenario_to_schema for tool '{name}'"
            )

    def _log_result(self, structured_result: ToolResult, fn: str):
        from kai.config.settings import settings

        if not settings.DEBUG_PROMPTS:
            return

        filepath = str(fn).split("_prompt.txt")[0] + "_result.txt"

        if isinstance(structured_result.output_workflow, dict):
            str_workflow = ""
            for k, v in structured_result.output_workflow.items():
                str_workflow += f"{k}\n"
                if isinstance(v, str):
                    str_workflow += f"{v}\n"
                elif isinstance(v, dict):
                    str_workflow += f"{json.dumps(v)}\n"
                elif isinstance(v, list):
                    # Handle list of any type (strings, dicts, ints, etc.)
                    if all(isinstance(item, (int, float)) for item in v):
                        # For lists of numbers, write as JSON array
                        str_workflow += f"{json.dumps(v)}\n"
                    else:
                        # For lists of strings/dicts, write each item
                        for vv in v:
                            if isinstance(vv, str):
                                str_workflow += f"{vv}\n"
                            elif isinstance(vv, dict):
                                str_workflow += f"{json.dumps(vv)}\n"
                            else:
                                str_workflow += f"{str(vv)}\n"
                elif isinstance(v, (int, float, bool)):
                    # Handle primitive types
                    str_workflow += f"{v}\n"
        else:
            str_workflow = ""
        if isinstance(structured_result.output_ui, dict):
            str_ui = json.dumps(structured_result.output_ui)
        elif isinstance(structured_result.output_ui, str):
            str_ui = structured_result.output_ui
        else:
            str_ui = ""

        # Write prompt information to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Tool: {self.name}\n")
            f.write(f"Scenario: {self.scenario.value}\n")
            f.write(f"LLM Provider: {self.llm_provider.provider_name}\n")
            f.write(f"LLM Model: {self.llm_provider.model}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Output type: {structured_result.output_type}\n")
            f.write("=" * 80 + "\n")
            f.write("Worklfow result:\n")
            f.write("=" * 80 + "\n")
            f.write(str_workflow)
            f.write("\n" + "=" * 80 + "\n")
            f.write("UI result:\n")
            f.write("=" * 80 + "\n")
            f.write(str_ui)
        return

    def _log_task_list_updates(self, updates: Dict[str, Any],
                               state: 'KaiState'):
        from kai.config.settings import settings

        if not settings.DEBUG_PROMPTS:
            return

        session_id = state["session_id"]
        session_timestamp = state["session_timestamp"]
        iteration_timestamp = state["iteration_timestamp"]
        is_autonomous = state["autonomous_mode"]
        notebook_uri = state["notebook_uri"]

        # Create notebook-specific identifier from URI
        notebook_identifier = "default_notebook"
        if notebook_uri:
            try:
                # Convert URI to safe folder name:
                # file:///path/to/notebook.ipynb -> notebook_ipynb
                import urllib.parse
                parsed_uri = urllib.parse.urlparse(notebook_uri)
                if parsed_uri.path:
                    path_parts = parsed_uri.path.split('/')
                    notebook_name = path_parts[-1]  # Get filename
                    if notebook_name:  # Ensure we have a valid filename
                        # Add full_agent_test prefix if this is a full_agent_test notebook
                        if 'full_agent_test' in path_parts:
                            notebook_identifier = f"full_agent_test/{notebook_name}"
                        else:
                            notebook_identifier = notebook_name
                        notebook_identifier = (notebook_identifier
                                                .replace('.', '_')
                                                .replace(' ', '_'))
            except Exception as parse_error:
                # Log parsing error but continue with default
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to parse notebook URI '{notebook_uri}': "
                    f"{parse_error}")
                notebook_identifier = "default_notebook"

        # Create session identifier with date prefix
        # (based on session timestamp)
        session_type = "auto" if is_autonomous else "manual"
        session_identifier = f"{session_timestamp}_{session_type}_{session_id}"

        # Create notebook-specific debug directory: notebook/session/
        # Date is already in the session identifier,
        # so we don't need a separate date folder
        notebook_debug_dir = (get_debug_prompts_dir() /
                                notebook_identifier /
                                session_identifier)
        notebook_debug_dir.mkdir(parents=True, exist_ok=True)
        # Create iteration subdirectory within the session folder
        iteration_identifier = f"{iteration_timestamp}"
        debug_dir = notebook_debug_dir / "task_list_updates"
        # Create directory:
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp and tool name (milliseconds for fast calls)
        # Current timestamp for file naming
        now = datetime.now()
        ms_str = f"{now.microsecond // 1000:03d}"
        timestamp_str = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}-{ms_str}"
        filename = f"{iteration_identifier}_{timestamp_str}_{self.name}.txt"
        filepath = debug_dir / filename

        # Write prompt information to file
        if len(updates.keys()) == 0:
            return
        else:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Tool: {self.name}\n")
                for k, v in updates.items():
                    f.write(f"Output type: {k}\n")
                    if k == "text":
                        # Revert json formating:
                        v = json.loads(v)
                        # Format task list to readable format:
                        if "tasks" in v.keys():
                            v["tasks"] = format_task_list(v)
                        for kk, vv in v.items():
                            f.write(f"* {kk}\n")
                            f.write(vv)
                            f.write("\n\n")
                        f.write("\n" + "=" * 80 + "\n")
                    else:
                        f.write(v)
                        f.write("\n" + "=" * 80 + "\n")
            return

    def _modify_user_query(self, state: 'KaiState') -> None:
        """Modify user query in state before prompt generation.

        Override in subclasses.
        """
        pass

    async def execute(
            self,
            state: 'KaiState',
            reasoning_level=None,
            **kwargs
    ) -> ToolResult:
        """Execute the tool using configured structured output method.

        Uses native or JSON prompting.
        """
        # Allow tool-specific user query modifications
        self._modify_user_query(state)

        # Use JSON prompting instructions if needed,
        # otherwise rely on LLM interface
        use_json_prompting = not self.llm_provider.use_structured_output
        system_prompt, user_prompt = self.prompt_manager.generate_prompt(
            state,
            self.scenario,
            structured_output=not use_json_prompting,
            reasoning_level=reasoning_level if reasoning_level else self.reasoning_level
        )

        # Log prompt if DEBUG_PROMPTS is enabled
        log_fn = self._log_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            state=state)

        # Generate structured response using LLM interface
        # (handles both paths internally)
        structured_result = await self.llm_provider.generate_structured(
            prompt=user_prompt,
            schema=self.schema,
            system_prompt=system_prompt,
            task_type=self._get_task_type(),
            tool_name=self.name,
            **kwargs
        )
        structured_result = self._process_structured_result(
            structured_result, state)
        self._log_result(
            structured_result=structured_result,
            fn=log_fn)
        return structured_result

    def _process_structured_result(
            self, structured_result,
            state: Union[None, 'KaiState'] = None) -> ToolResult:
        """Process the structured response into a ToolResult.

        Override in subclasses.
        """
        return ToolResult(
            output_ui=structured_result.model_dump(),
            output_type=ToolOutputType.RESPONSE
        )


class UnstructuredPromptTool(BasePromptTool):
    """Base class for LLM-based tools that use standard prompt scenarios."""

    def __init__(self, name: str, scenario: PromptScenario,
                 llm_interface: 'LLMInterface'):
        super().__init__(name, scenario, llm_interface)

    def _modify_user_query(self, state: 'KaiState') -> None:
        """Modify user query in state before prompt generation.

        Override in subclasses.
        """
        pass

    def _log_result(self, response: str, state: 'KaiState'):
        from kai.config.settings import settings
        from kai.config.paths import get_debug_prompts_dir
        from datetime import datetime

        if not settings.DEBUG_PROMPTS:
            return

        session_id = state["session_id"]
        session_timestamp = state["session_timestamp"]
        iteration_timestamp = state["iteration_timestamp"]
        is_autonomous = state["autonomous_mode"]
        notebook_uri = state["notebook_uri"]

        # Create notebook-specific identifier from URI
        notebook_identifier = "default_notebook"
        if notebook_uri:
            try:
                # Convert URI to safe folder name:
                # file:///path/to/notebook.ipynb -> notebook_ipynb
                import urllib.parse
                parsed_uri = urllib.parse.urlparse(notebook_uri)
                if parsed_uri.path:
                    path_parts = parsed_uri.path.split('/')
                    notebook_name = path_parts[-1]  # Get filename
                    if notebook_name:  # Ensure we have a valid filename
                        # Add full_agent_test prefix if this is a full_agent_test notebook
                        if 'full_agent_test' in path_parts:
                            notebook_identifier = f"full_agent_test/{notebook_name}"
                        else:
                            notebook_identifier = notebook_name
                        notebook_identifier = (notebook_identifier
                                                .replace('.', '_')
                                                .replace(' ', '_'))
            except Exception as parse_error:
                # Log parsing error but continue with default
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to parse notebook URI '{notebook_uri}': "
                    f"{parse_error}")
                notebook_identifier = "default_notebook"

        # Create session identifier with date prefix
        # (based on session timestamp)
        session_type = "auto" if is_autonomous else "manual"
        session_identifier = f"{session_timestamp}_{session_type}_{session_id}"

        # Create notebook-specific debug directory: notebook/session/
        # Date is already in the session identifier,
        # so we don't need a separate date folder
        notebook_debug_dir = (get_debug_prompts_dir() /
                                notebook_identifier /
                                session_identifier)
        notebook_debug_dir.mkdir(parents=True, exist_ok=True)
        # Create iteration subdirectory within the session folder
        iteration_identifier = f"{iteration_timestamp}"
        debug_dir = notebook_debug_dir / iteration_identifier
        # Create directory:
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp and tool name (milliseconds for fast calls)
        # Current timestamp for file naming
        now = datetime.now()
        ms_str = f"{now.microsecond // 1000:03d}"
        timestamp_str = f"{now.strftime('%Y-%m-%d_%H-%M-%S')}-{ms_str}"
        filename = f"{timestamp_str}_{self.name}_result.txt"
        filepath = debug_dir / filename

        # Write prompt information to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"Tool: {self.name}\n")
            f.write(f"Scenario: {self.scenario.value}\n")
            f.write(f"LLM Provider: {self.llm_provider.provider_name}\n")
            f.write(f"LLM Model: {self.llm_provider.model}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Output:\n")
            f.write("=" * 80 + "\n")
            f.write(response)
        return

    async def execute(self, state: 'KaiState', reasoning_level=None, **kwargs) -> ToolResult:
        """Execute the prompt-based tool."""
        # Allow tool-specific user query modifications
        self._modify_user_query(state)

        # Generate prompt using prompt manager
        system_prompt, user_prompt = self.prompt_manager.generate_prompt(
            state, self.scenario, reasoning_level=reasoning_level if reasoning_level else self.reasoning_level)

        # Log prompt if DEBUG_PROMPTS is enabled
        _ = self._log_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            state=state)

        response = await self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            task_type=self._get_task_type(),
            **kwargs
        )
        self._log_result(response, state)

        # Parse and format response
        result = await self._process_response(
            response, state)

        return result

    async def _process_response(self, response: str,
                                state: Union[None, 'KaiState'] = None) -> ToolResult:
        """Process LLM response into ToolResult.

        Override in subclasses.
        """
        return ToolResult(
            output_ui=response,
            output_type=ToolOutputType.RESPONSE
        )
