"""LLM-based tools that use prompts for execution."""

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING, Union
from kai.core.orchestration.execution_context import ExecutionContext
from kai.core.orchestration.schemas import ReferenceWorkflowSelection, ReferenceWorkflowSelectionOnly
from kai.utils import setup_logger
from kai.retrieval.workflow_summaries.notebook_selector import NotebookSelector
from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag

from kai.core.prompt_manager import PromptManager, PromptScenario
from ..llm_interface import BaseLLMProvider, LLMInterface
from ..utils import format_task_list
from .base_tool import BaseTool, ToolResult, ToolOutputType
from .schemas import SCHEMA_REGISTRY
from kai.config.paths import get_debug_prompts_dir

if TYPE_CHECKING:
    from .execution_context import ExecutionContext, ExecutionInputs

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
                    exec_context: 'ExecutionContext') -> str:
        """
        Log prompts to debug folder if DEBUG_PROMPTS is enabled.

        Data flow for session metadata:
        agent.py: session_metadata = {session_timestamp, notebook_uri,
                                     iteration_timestamp, iteration_counter}
        → orchestrator: inputs['context']['session_metadata'] = session_metadata
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

        session_id = exec_context.session_metadata["session_id"]
        session_timestamp = exec_context.session_metadata["session_timestamp"]
        iteration_timestamp = exec_context.session_metadata["iteration_timestamp"]
        is_autonomous = exec_context.inputs.context["autonomous_mode"]
        notebook_uri = exec_context.session_metadata["notebook_uri"]

        # Create notebook-specific identifier from URI
        notebook_identifier = "default_notebook"
        if notebook_uri:
            try:
                # Convert URI to safe folder name:
                # file:///path/to/notebook.ipynb -> notebook_ipynb
                import urllib.parse
                parsed_uri = urllib.parse.urlparse(notebook_uri)
                if parsed_uri.path:
                    notebook_name = parsed_uri.path.split('/')[-1]  # Get filename
                    if notebook_name:  # Ensure we have a valid filename
                        notebook_identifier = (notebook_name
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

        # Create filename with timestamp and tool name
        # Current timestamp for file naming
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
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
        return filepath


class StructuredPromptTool(BasePromptTool):
    """Base class for LLM tools that use structured output via Pydantic schemas."""

    def __init__(self, name: str, scenario: PromptScenario,
                 llm_interface: 'LLMInterface'):
        super().__init__(name, scenario, llm_interface)

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
                               exec_context: 'ExecutionContext'):
        from kai.config.settings import settings

        if not settings.DEBUG_PROMPTS:
            return        

        session_id = exec_context.session_metadata["session_id"]
        session_timestamp = exec_context.session_metadata["session_timestamp"]
        iteration_timestamp = exec_context.session_metadata["iteration_timestamp"]
        is_autonomous = exec_context.inputs.context["autonomous_mode"]
        notebook_uri = exec_context.session_metadata["notebook_uri"]

        # Create notebook-specific identifier from URI
        notebook_identifier = "default_notebook"
        if notebook_uri:
            try:
                # Convert URI to safe folder name:
                # file:///path/to/notebook.ipynb -> notebook_ipynb
                import urllib.parse
                parsed_uri = urllib.parse.urlparse(notebook_uri)
                if parsed_uri.path:
                    notebook_name = parsed_uri.path.split('/')[-1]  # Get filename
                    if notebook_name:  # Ensure we have a valid filename
                        notebook_identifier = (notebook_name
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

        # Create filename with timestamp and tool name
        # Current timestamp for file naming
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
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

    def _modify_user_query(self, exec_context: 'ExecutionContext') -> None:
        """Modify user query in exec_context.inputs before prompt generation.

        Override in subclasses.
        """
        pass

    async def execute(
            self, 
            exec_context: 'ExecutionContext', 
            reasoning_level=None, 
            **kwargs
    ) -> ToolResult:
        """Execute the tool using configured structured output method.

        Uses native or JSON prompting.
        """
        # Allow tool-specific user query modifications
        self._modify_user_query(exec_context)

        # Use JSON prompting instructions if needed,
        # otherwise rely on LLM interface
        use_json_prompting = not self.llm_provider.use_structured_output
        system_prompt, user_prompt = self.prompt_manager.generate_prompt(
            exec_context,
            self.scenario,
            structured_output=not use_json_prompting,
            reasoning_level=reasoning_level if reasoning_level else self.reasoning_level
        )

        # Log prompt if DEBUG_PROMPTS is enabled
        log_fn = self._log_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            exec_context=exec_context)

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
            structured_result, exec_context)
        self._log_result(
            structured_result=structured_result,
            fn=log_fn)
        return structured_result

    def _process_structured_result(
            self, structured_result,
            exec_context: Union[None, 'ExecutionContext'] = None) -> ToolResult:
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

    def _modify_user_query(self, exec_context: 'ExecutionContext') -> None:
        """Modify user query in exec_context.inputs before prompt generation.

        Override in subclasses.
        """
        pass

    def _log_result(self, response: str, exec_context: 'ExecutionContext'):
        from kai.config.settings import settings
        from kai.config.paths import get_debug_prompts_dir
        from datetime import datetime

        if not settings.DEBUG_PROMPTS:
            return

        session_id = exec_context.session_metadata["session_id"]
        session_timestamp = exec_context.session_metadata["session_timestamp"]
        iteration_timestamp = exec_context.session_metadata["iteration_timestamp"]
        is_autonomous = exec_context.inputs.context["autonomous_mode"]
        notebook_uri = exec_context.session_metadata["notebook_uri"]

        # Create notebook-specific identifier from URI
        notebook_identifier = "default_notebook"
        if notebook_uri:
            try:
                # Convert URI to safe folder name:
                # file:///path/to/notebook.ipynb -> notebook_ipynb
                import urllib.parse
                parsed_uri = urllib.parse.urlparse(notebook_uri)
                if parsed_uri.path:
                    notebook_name = parsed_uri.path.split('/')[-1]  # Get filename
                    if notebook_name:  # Ensure we have a valid filename
                        notebook_identifier = (notebook_name
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

        # Create filename with timestamp and tool name
        # Current timestamp for file naming
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S")
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

    async def execute(self, exec_context: 'ExecutionContext', reasoning_level=None, **kwargs) -> ToolResult:
        """Execute the prompt-based tool."""
        # Allow tool-specific user query modifications
        self._modify_user_query(exec_context)

        # Generate prompt using prompt manager
        system_prompt, user_prompt = self.prompt_manager.generate_prompt(
            exec_context, self.scenario, reasoning_level=reasoning_level if reasoning_level else self.reasoning_level)

        # Log prompt if DEBUG_PROMPTS is enabled
        _ = self._log_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            exec_context=exec_context)

        response = await self.llm_provider.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            task_type=self._get_task_type(),
            **kwargs
        )
        self._log_result(response, exec_context)

        # Parse and format response
        result = await self._process_response(
            response, exec_context)

        return result

    async def _process_response(self, response: str,
                                exec_context: Union[None, 'ExecutionContext'] = None) -> ToolResult:
        """Process LLM response into ToolResult.

        Override in subclasses.
        """
        return ToolResult(
            output_ui=response,
            output_type=ToolOutputType.RESPONSE
        )


class TaskListGenerationTool(StructuredPromptTool):
    """Tool for generating task lists in autonomous mode.

    **UI Returns:**
    - `output_ui`: Dict with "text" field containing JSON task list for VSCode display
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_list`: Complete structured task list with tasks array

    **Used by workflows:** Autonomous initiation workflow to create initial task breakdown
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("task_list_generation", PromptScenario.TASK_LIST_GENERATION, llm_interface)

    def _process_structured_result(self, structured_result, exec_context: 'ExecutionContext') -> ToolResult:
        """Process structured task list result for VSCode."""
        # Create JSON-embedded text format for consistent frontend parsing
        task_list = {"tasks": [task.model_dump() for task in structured_result.tasks]}
        if len(task_list["tasks"]) == 0:
            raise Exception("Generated task list did not include any tasks.")

        # Validate task list format
        validate_task_list_format(task_list["tasks"], "TaskListGenerationTool")

        json_text = json.dumps(task_list)

        output_workflow = {
            "task_list": task_list,
            "retrieval_queries": structured_result.retrieval_queries
        }
        
        # Create VSCode-ready response for task list display - only include fields VSCode uses
        vscode_response = {
            "text": json_text,
        }
        if structured_result.retrieval_queries and len(structured_result.retrieval_queries) > 0:
            vscode_response["agent_notification"] = "\n".join(["Reading up on:"] + structured_result.retrieval_queries)
        self._log_task_list_updates(vscode_response, exec_context)

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=output_workflow
        )


class CodeGenerationTool(UnstructuredPromptTool):
    """Tool for generating code.

    **UI Returns:**
    - Autonomous mode: Dict with "code", "positioning_info", "should_replace_code" fields for VSCode execution
    - Manual mode: Raw LLM response string for chat display
    - `output_type`: EXECUTE_ONLY (autonomous) or RESPONSE (manual)

    **Workflow Returns:**
    - None - this tool doesn't propagate workflow data

    **Used by workflows:** Regular request workflow for manual code generation
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("code_generation", PromptScenario.CODE_GENERATION, llm_interface)

    async def _process_response(self, response: str, exec_context: 'ExecutionContext') -> ToolResult:
        """Process code generation response and format for VSCode."""
        autonomous_mode = exec_context.inputs.context["autonomous_mode"]
        positioning_info = exec_context.inputs.context['positioning_info']

        # Extract clean code from response
        extracted_code = extract_code_from_response(response)

        # If no code was extracted, raise an error to trigger retry in orchestration loop
        if extracted_code is None:
            raise ValueError(f"CodeFixingTool could not extract code from response. Response: {response}")

        # Create VSCode-ready response - only include fields VSCode uses
        if autonomous_mode:
            vscode_response = {
                "code": extracted_code,
                "positioning_info": positioning_info,
                "should_replace_code": "false",
                "cell_type": "code"
            }
            output_type = ToolOutputType.EXECUTE_ONLY
        else:
            vscode_response = response  # Manual mode: return full response as string
            output_type = ToolOutputType.RESPONSE

        return ToolResult(
            output_ui=vscode_response,
            output_type=output_type
        )


class CodeGenerationWithGuidanceTool(UnstructuredPromptTool):
    
    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("code_generation_with_guidance", PromptScenario.CODE_GENERATION_WITH_GUIDANCE, llm_interface)

    async def _process_response(self, response: str, exec_context: 'ExecutionContext') -> ToolResult:
        """Process code generation response and format for VSCode (always autonomous mode)."""
        positioning_info = exec_context.inputs.context['positioning_info']

        # Extract clean code from response
        extracted_code = extract_code_from_response(response)

        # If no code was extracted, raise an error to trigger retry in orchestration loop
        if extracted_code is None:
            raise ValueError(f"TaskStepCodeGenerationTool failed to extract code from LLM response. Response length: {len(response)}. This will trigger a retry.")
        
        # Create VSCode-ready response - only include fields VSCode uses
        vscode_response = {
            "code": extracted_code,
            "positioning_info": positioning_info,
            "should_replace_code": "false",
            "cell_type": "code"
        }

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.EXECUTE_ONLY
        )


class ReasoningResponseWithGuidanceTool(UnstructuredPromptTool):

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("reasoning_response_with_guidance", PromptScenario.REASONING_RESPONSE_WITH_GUIDANCE, llm_interface)

    async def _process_response(self, response: str, exec_context: 'ExecutionContext') -> ToolResult:
        """Process code generation response and format for VSCode (always autonomous mode)."""
        positioning_info = exec_context.inputs.context['positioning_info']

        # Check if this is a re-generation (after critique) - if so, replace the previous reasoning cell
        # 1) replace in critique iteration
        # 2) replace if this is a retry of a reasoning task (as marked by the markcompletiontool)
        should_replace = (
            "reasoning_critique" in exec_context.inputs.context or
            "retry_objective" in exec_context.inputs.context
        )

        # Create VSCode-ready response - only include fields VSCode uses
        vscode_response = {
            "code": response,
            "positioning_info": positioning_info,
            "should_replace_code": "true" if should_replace else "false",
            "cell_type": "markdown"
        }
        # Make reasoning available for potential critiques:
        output_workflow = {"reasoning_response": response}

        return ToolResult(
            output_type=ToolOutputType.EXECUTE_ONLY,
            output_ui=vscode_response,
            output_workflow=output_workflow,
        )


class CodeUpdateTool(UnstructuredPromptTool):
    """Tool for generating updated code after error analysis or feedback.

    **UI Returns:**
    - Autonomous mode: Dict with "code", "should_replace_code", "error_recovery_strategy", "positioning_info" for VSCode
    - Manual mode: Raw LLM response string for chat display
    - `output_type`: EXECUTE_ONLY (autonomous) or RESPONSE (manual)

    **Workflow Returns:**
    - `error_recovery_strategy`: Strategy used for fixing (from ErrorRecoveryTool output)

    **Used by workflows:** Error recovery workflow after ErrorRecoveryTool determines strategy

    **Special behavior:** Dynamically switches between CODE_FIXING, CODE_FIXING_WITH_GUIDANCE, CODE_UPDATE_WITH_GUIDANCE scenarios
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("code_update", PromptScenario.CODE_FIXING, llm_interface)

    async def execute(self, exec_context: 'ExecutionContext', **kwargs) -> ToolResult:
        """Execute code fixing with active task guidance if available."""
        # Choose scenario based on whether we have active task information
        has_error = exec_context.inputs.context['last_execution_failed']
        active_task_objective = exec_context.inputs.context.get("active_task_objective")
        retry_objective = exec_context.inputs.context.get("retry_objective")
        if has_error:
            if active_task_objective:
                self.scenario = PromptScenario.CODE_FIXING_WITH_GUIDANCE
            else:
                # Use regular code fixing scenario
                self.scenario = PromptScenario.CODE_FIXING
        elif retry_objective is not None:
            # Use feedback-centric code update
            self.scenario = PromptScenario.CODE_UPDATE_WITH_GUIDANCE
        else:
            raise ValueError((has_error, active_task_objective, retry_objective))
        result = await super().execute(exec_context, **kwargs)
        # Reset default scenario:
        self.scenario = PromptScenario.CODE_FIXING
        return result

    async def _process_response(self, response: str, exec_context: 'ExecutionContext') -> ToolResult:
        """Process code fixing response and format for VSCode execution."""
        autonomous_mode = exec_context.inputs.context["autonomous_mode"]
        output_type = ToolOutputType.EXECUTE_ONLY if autonomous_mode else ToolOutputType.RESPONSE

        # Extract context for code fixing first
        positioning_info = exec_context.inputs.context["positioning_info"]
        error_recovery_strategy = exec_context.inputs.context.get("error_recovery_strategy")

        # Extract clean code from response
        extracted_code = extract_code_from_response(response)

        # If no code was extracted, raise an error to trigger retry in orchestration loop
        if extracted_code is None:
            raise ValueError(f"CodeUpdateTool could not extract code from response. Response: {response}")

        # Create VSCode-ready response - only include fields VSCode uses
        if autonomous_mode:
            vscode_response = {
                "code": extracted_code,
                "should_replace_code": "true",
                "error_recovery_strategy": error_recovery_strategy,
                "positioning_info": positioning_info,
                "cell_type": "code"
            }
            output_type = ToolOutputType.EXECUTE_ONLY
        else:
            vscode_response = response  # Manual mode: return full response as string
            output_type = ToolOutputType.RESPONSE

        return ToolResult(
            output_ui=vscode_response,
            output_type=output_type,
            output_workflow={}
        )


class ErrorRecoveryTool(StructuredPromptTool):
    """Tool for analyzing errors and determining recovery strategy.

    **UI Returns:**
    - `output_ui`: String intent value ("code_fixing", "replace_and_restart", etc.)
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `error_recovery_strategy`: Recovery strategy intent for CodeFixingTool to use

    **Used by workflows:** Error recovery workflow to analyze errors and determine fixing approach

    **Special behavior:** Modifies user_query with structured error context and failed code
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("error_recovery", PromptScenario.ERROR_RECOVERY, llm_interface)

    def _process_structured_result(self, result, exec_context: Optional['ExecutionContext'] = None) -> ToolResult:
        """Process structured error recovery result."""
        return ToolResult(
            output_ui=result.intent,
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "error_recovery_strategy": result.intent
            }
        )


class BacktrackRecoveryTool(StructuredPromptTool):
    """Tool for determining if notebook restart is needed for backtracking recovery.

    **UI Returns:**
    - `output_ui`: Boolean indicating if kernel restart is required
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `restart_required`: Boolean flag for workflow orchestration decisions

    **Used by workflows:** Backtracking recovery workflow to determine if restart is needed

    **Special behavior:** Analyzes deleted tasks and error context to make restart decision
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("backtrack_recovery", PromptScenario.BACKTRACK_RECOVERY, llm_interface)

    def _modify_user_query(self, exec_context: 'ExecutionContext') -> None:
        """Create PromptContext with recovery objective and error details."""
        # Build context for backtrack recovery decision
        context_parts = []

        backtracking_context = exec_context.inputs.backtracking_context
        if backtracking_context and backtracking_context.is_active:
            context_parts.append(f"## Recovery Objective:\n{backtracking_context.recovery_objective}")
            context_parts.append("")

        error_details = exec_context.inputs.context.get("error_details", "")
        if error_details:
            context_parts.append(f"## Observed Errors:\n{error_details}")
            context_parts.append("")

        reset_tasks = exec_context.inputs.context["reset_tasks"]
        if reset_tasks:
            context_parts.append("## Tasks Being Reset (corresponding to deleted code):")
            for task in reset_tasks:
                task_desc = task["task"]
                context_parts.append(f"- **Task {task['id']}**: {task_desc}")
            context_parts.append("")

        user_query = "\n".join(context_parts)

        # Update the user query in exec_context
        exec_context.inputs.user_query = user_query

    def _process_structured_result(self, result, exec_context: Optional['ExecutionContext'] = None) -> ToolResult:
        """Process structured backtrack recovery result."""
        return ToolResult(
            output_ui=result.restart_required,
            output_type=ToolOutputType.NO_OUTPUT
        )


class ExecutionMonitorTool(StructuredPromptTool):
    """Tool for monitoring long-running cell execution and deciding whether to continue or terminate.

    **UI Returns:**
    - `output_ui`: Action decision ("continue" or "terminate")
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `action`: "continue" or "terminate" - decision for execution control
    - `feedback`: String detailing suggeseted changes

    **Used by workflows:** Execution progress check workflow to analyze stuck cells

    **Special behavior:** Analyzes cell code, elapsed time, and partial outputs to detect stuck execution
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("execution_monitor", PromptScenario.EXECUTION_MONITOR, llm_interface)

    def _process_structured_result(self, result, exec_context: Optional['ExecutionContext'] = None) -> ToolResult:
        """Process structured execution monitor result."""
        return ToolResult(
            output_ui=result.action,
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "action": result.action,
                "feedback": result.feedback
            }
        )


class AutonomousMarkCompletionTool(StructuredPromptTool):
    """Tool for analyzing task completion status in autonomous mode - handles success, error, and backtracking cases.

    **UI Returns:**
    - `output_ui`: Dict with "text" field containing JSON task list for VSCode display
    - `output_type`: TASK_LIST_DISPLAY - always shows updated task list in chat

    **Workflow Returns:**
    - `recovery_objective`: Description of recovery needed (if backtracking detected)
    - `backtrack_to_task`: Task object to backtrack to (if backtracking detected)

    **Used by workflows:** Autonomous continuation workflow to update task statuses and detect backtracking

    **Special behavior:** Detects when backtracking is needed and provides recovery context.
    Always sends task list updates to chat to keep UI synchronized.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autonomous_mark_completion", PromptScenario.AUTONOMOUS_MARK_COMPLETION, llm_interface)

    def _process_structured_result(self, structured_result, exec_context: Union[None, 'ExecutionContext'] = None) -> ToolResult:
        """Process the structured TaskCompletionUpdate response with status updates and backtracking support."""
        # Get the original task list structure
        original_task_list = exec_context.inputs.task_list

        # Extract tasks from original structure
        original_tasks = original_task_list['tasks'].copy()

        # Apply status updates to existing tasks
        status_updates = {update.id: update.status for update in structured_result.status_updates}

        updated_tasks = []
        for task in original_tasks:
            assert isinstance(task, dict), task
            # Check that updates do not affect pending tasks:
            if task['status'] == "pending" and task['id'] in status_updates.keys() and status_updates[task['id']] != "pending":
                raise Exception(f"Tried to set task {task['id']} from pending to {status_updates[task['id']]}.")
            task_copy = task.copy()
            if task.get('id') in status_updates:
                task_copy['status'] = status_updates[task['id']]
            updated_tasks.append(task_copy)
        updated_task_list = {'tasks': updated_tasks}

        # Validate task list format
        validate_task_list_format(updated_task_list["tasks"], "AutonomousMarkCompletionTool")

        # Always provide task_list in output_workflow for state propagation
        output_workflow = {
            "task_list": updated_task_list
        }

        # Handle backtracking if detected - add backtracking context
        if structured_result.backtrack_detected:
            first_pending_task = None
            for task in updated_tasks:
                if task['status'] == 'pending':
                    first_pending_task = task
                    break

            # Add backtracking info for workflow orchestration
            output_workflow["recovery_objective"] = structured_result.recovery_objective
            output_workflow["backtrack_to_task"] = first_pending_task or {}

        # Handle retry if detected
        if structured_result.retry_objective:
            output_workflow["retry_objective"] = structured_result.retry_objective

        # Always send task list update to UI
        import json
        updated_task_json = json.dumps(updated_task_list)
        vscode_response = {"text": updated_task_json}
        if structured_result.retry_objective:
            vscode_response["agent_notification"] = structured_result.retry_objective
        self._log_task_list_updates(vscode_response, exec_context)

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=output_workflow
        )


class AutonomousUpdateTasksTool(StructuredPromptTool):
    """Tool for updating tasks in autonomous mode (no decision-making).

    **UI Returns:**
    - `output_ui`: Dict with "text" field containing JSON updated task list for VSCode display
    - `output_type`: TASK_LIST_DISPLAY - shows formatted updated task list in chat

    **Workflow Returns:**
    - `task_list`: Complete updated task list structure

    **Used by workflows:** Feedback continuation workflow when user requests task modifications

    **Special behavior:** Updates task list based on user feedback without making autonomous decisions
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autonomous_update_tasks", PromptScenario.AUTONOMOUS_UPDATE_TASKS, llm_interface)

    def _modify_user_query(self, exec_context: 'ExecutionContext') -> None:
        """Create PromptContext from tool inputs, including current task list."""
        # Include user query and task list
        user_query = exec_context.inputs.user_query
        backtracking_context = exec_context.inputs.backtracking_context

        query_parts = []
        if backtracking_context and backtracking_context.is_active:
            query_parts.append(f"recovery_objective: {backtracking_context.recovery_objective}")

        # Only modify user query if we have backtracking context
        if query_parts:
            user_query = "\n\n".join(query_parts)
            # Update the user query in exec_context
            exec_context.inputs.user_query = user_query

    def _process_structured_result(self, structured_result, exec_context: 'ExecutionContext') -> ToolResult:
        """Process the structured AutonomousTaskUpdate response."""
        original_tasks = exec_context.inputs.task_list["tasks"].copy()
        # Check if update was requested:
        if structured_result.update_rule == "UPDATE":
            # Get the original task list structure
            # Use the structured result directly (includes analysis_type and tasks)
            updated_tasks = [task.model_dump() for task in structured_result.tasks]
            # Extract completed tasks from original structure
            new_tasks = []
            completed_task_ids = set()
            for task in original_tasks:
                if task["status"] != "completed":
                    break
                new_tasks.append(task)
                completed_task_ids.add(task["id"])
            # Add updated tasks, skipping any that are already in completed section
            for task in updated_tasks:
                if task["id"] not in completed_task_ids:
                    new_tasks.append(task)
        else:
            assert structured_result.update_rule == "KEEP"
            new_tasks = original_tasks
        # Create JSON-embedded text format for consistent frontend parsing
        updated_task_list = {"tasks": new_tasks}

        # Validate task list format
        validate_task_list_format(updated_task_list["tasks"], "AutonomousTaskUpdateTool") 
        updated_task_json = json.dumps(updated_task_list)
        output_workflow = {
            "task_list": updated_task_list,
            "task_list_update_rule": structured_result.update_rule,
            "task_list_update_rationale": structured_result.update_rationale,
        }
            
        # Extract optional rag queries
        rag_enabled = exec_context.inputs.context["rag_enabled"]
        has_error = exec_context.inputs.context['last_execution_failed']
        if (rag_enabled and not has_error) and structured_result.retrieval_queries:
            output_workflow["snippet_retrieval_query"] = structured_result.retrieval_queries
        
        # Create display response with updated task list - only include fields VSCode uses
        # Use update_rationale string for display
        agent_notification = structured_result.update_rationale or ""

        vscode_response = {
            "text": updated_task_json,
            "agent_notification": agent_notification,
        }
        self._log_task_list_updates(vscode_response, exec_context)

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=output_workflow
        )


class AutoLoopIntentClassificationTool(StructuredPromptTool):
    """Tool for classifying user input during autonomous mode.

    **UI Returns:**
    - `output_ui`: Dict with "intent", "target_tasks", "modification_description" fields
    - `output_type`: RESPONSE - internal classification result

    **Workflow Returns:**
    - None - UI output used directly by workflow orchestration for routing decisions

    **Used by workflows:** Feedback continuation workflow to classify feedback type and route appropriately

    **Possible intents:** TASK_LIST_MODIFICATION, CODE_IMPLEMENTATION_FEEDBACK, CONTINUE_WITH_FEEDBACK
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autoloop_intent_classification", PromptScenario.AUTOLOOP_INTENT_CLASSIFICATION, llm_interface)

    def _process_structured_result(self, result, exec_context: Optional['ExecutionContext'] = None) -> ToolResult:
        """Process feedback intent classification result."""
        return ToolResult(
            output_ui={},
            output_workflow=result.model_dump(),
            output_type=ToolOutputType.NO_OUTPUT
        )


class QuestionAnsweringTool(UnstructuredPromptTool):
    """Tool for answering questions about code/analysis.

    **UI Returns:**
    - `output_ui`: Raw LLM response string containing the answer
    - `output_type`: RESPONSE - displayed in chat for user to read

    **Workflow Returns:**
    - None - this tool doesn't propagate workflow data

    **Used by workflows:** Regular request workflow for question_about_code intent after retrieval
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("question_answering", PromptScenario.QUESTION_ANSWERING, llm_interface)

    async def _process_response(self, response: str, inputs: 'ExecutionInputs') -> ToolResult:
        """Process question answering response."""
        return ToolResult(
            output_ui=response,
            output_type=ToolOutputType.RESPONSE
        )


class IntentClassificationTool(StructuredPromptTool):
    """Tool for classifying user intents using structured output.

    **UI Returns:**
    - `output_ui`: Dict with "intent" field containing classification result
    - `output_type`: RESPONSE - internal classification result

    **Workflow Returns:**
    - None - UI output used directly by workflow orchestration for routing decisions

    **Used by workflows:** Regular request workflow to determine how to handle user requests

    **Possible intents:** generate_code, question_about_code, generate_code_in_place, remove_code
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("intent_classification", PromptScenario.INTENT_CLASSIFICATION, llm_interface)

    def _process_structured_result(self, result, exec_context: Optional['ExecutionContext'] = None) -> ToolResult:
        """Process structured intent classification result."""
        # Return schema output directly - orchestration handles categorical classification
        return ToolResult(
            output_ui={},
            output_workflow=result.model_dump(),
            output_type=ToolOutputType.NO_OUTPUT
        )


class CellPositioningTool(StructuredPromptTool):
    """Tool for determining cell positioning using LLM with addition/replacement logic.

    **UI Returns:**
    - `output_ui`: Dict with "target_cell" field containing selected cell index
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - `positioning_info.target_cell`: Cell index for code generation tools to use

    **Used by workflows:** Multiple workflows before code generation to determine cell placement

    **Special behavior:** Dynamically switches scenarios (ADDITION vs REPLACEMENT) based on context
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        # Start with ADDITION scenario, will be determined dynamically
        super().__init__("cell_positioning", PromptScenario.CELL_SELECTION_ADDITION, llm_interface)

    async def execute(self, exec_context: 'ExecutionContext', **kwargs) -> ToolResult:
        """Execute cell positioning with proper scenario selection."""
        # Check for backtracking context (from autonomous workflow)
        backtracking_context = exec_context.inputs.backtracking_context

        # Check if this is error recovery context
        error_recovery = exec_context.inputs.context.get("error_recovery", False)

        if backtracking_context and backtracking_context.is_active:
            self.scenario = PromptScenario.CELL_SELECTION_REPLACEMENT
        elif error_recovery:
            self.scenario = PromptScenario.CELL_SELECTION_REPLACEMENT
        else:
            self.scenario = PromptScenario.CELL_SELECTION_ADDITION

        return await super().execute(exec_context, **kwargs)

    def _modify_user_query(self, exec_context: 'ExecutionContext') -> None:
        """Create PromptContext with backtracking information if available."""
        # Check for backtracking context (autonomous continue workflow)
        backtracking_context = exec_context.inputs.backtracking_context

        if backtracking_context and backtracking_context.is_active:
            # In backtracking mode - build structured context
            context_parts = []

            # Recovery objective
            recovery_objective = backtracking_context.recovery_objective
            if recovery_objective:
                context_parts.append(f"## Recovery Objective:\n{recovery_objective}")
                context_parts.append(
                    "You are selecting a position in a notebook at which to add new code to start a recovery of a failed analysis attempt. "
                    "This failed atttempt involved deletion of parts to the analysis. "
                    "You are given the positions of cell deletions of failed tasks " \
                    "and a description of the last valid completed task as a reference point. "
                    "Use both to determine where to position the new code to be added as part of the recovery.")
                context_parts.append("")

            # Cell deletion info - convert to cleaned indices
            deleted_cells = backtracking_context.deleted_cells
            index_translation = backtracking_context.index_translation

            if deleted_cells and index_translation:
                # Find the gaps in the current notebook where cells were deleted
                current_notebook_gaps = self._find_deletion_gaps(deleted_cells, index_translation)
                if current_notebook_gaps:
                    context_parts.append("## Cells Removed:")
                    context_parts.append(f"Cells were deleted. The content now at indices {current_notebook_gaps} immediately preceded deleted cells.")
                    context_parts.append("")

            user_query = "\n".join(context_parts)

        else:
            user_query = exec_context.inputs.user_query

        # Update the user query in exec_context
        exec_context.inputs.user_query = user_query

    def _process_structured_result(self, result, exec_context: Optional['ExecutionContext'] = None) -> ToolResult:
        """Convert LLM positioning result to format expected by code generation tools."""
        positioning_info = {
            "target_cell": result.target_cell
        }

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "positioning_info": positioning_info
            }
        )

    def _find_deletion_gaps(self, deleted_cells: List[int], index_translation: Dict[int, int]) -> List[int]:
        """Find cells that preceded deletions - these are safe insertion points.

        Args:
            deleted_cells: Original cell indices that were deleted (e.g., [5, 6, 7])
            index_translation: Mapping from original -> current indices

        Returns:
            List of current notebook indices that preceded the deleted sections
        """
        if not deleted_cells or not index_translation:
            return []

        preceding_cells = []
        deleted_set = set(deleted_cells)

        # Find cells that came right before the deleted cells
        for original_idx in sorted(index_translation.keys()):
            # Check if this cell came right before a deletion
            if original_idx + 1 in deleted_set:
                current_idx = index_translation[original_idx]
                preceding_cells.append(current_idx)

        return sorted(list(set(preceding_cells)))


class CellSelectionDeletionTool(StructuredPromptTool):
    """Tool for selecting cells to delete during backtracking.

    **UI Returns:**
    - `output_ui`: String describing selected cells and reasoning
    - `output_type`: NO_OUTPUT - internal tool, not displayed to user

    **Workflow Returns:**
    - None - cells_to_delete passed via context to CellDeletionTool

    **Used by workflows:** Backtracking workflow to intelligently select cells for deletion

    **Special behavior:** Modifies user_query with reset tasks and recovery objective context
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("cell_selection_deletion", PromptScenario.CELL_SELECTION_DELETION_FOR_BACKTRACKING, llm_interface)

    def _process_structured_result(self, structured_result, exec_context: Union[None, 'ExecutionContext'] = None) -> ToolResult:
        """Process structured CellDeletionSelection response."""
        cells_to_delete = structured_result.cells_to_delete

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "cells_to_delete": cells_to_delete
            }
        )


class SectionCodeReviewTool(StructuredPromptTool):
    """Tool for reviewing and fixing code sections that encounter errors during execution.

    **UI Returns:**
    - `output_ui`: Structured result with "operation" and "target_cells" fields
    - `output_type`: NO_OUTPUT - internal tool used by RunSectionWorkflow

    **Workflow Returns:**
    - `operation`: Fix operation to perform ("fix", "skip", "restart")
    - `target_cells`: List of cell indices that need attention

    **Used by workflows:** RunSectionWorkflow for intelligent section execution with error recovery

    **Special behavior:** Analyzes section code, errors, and previous fix attempts to recommend actions
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("section_code_review", PromptScenario.SECTION_CODE_REVIEW, llm_interface)

    def _modify_user_query(self, exec_context: 'ExecutionContext') -> None:
        """Create PromptContext with section code, error details, and fix history."""
        context_parts = []

        # Add section code being executed
        section_code = exec_context.inputs.context["section_code"]
        if section_code:
            context_parts.append("## Code Section Being Executed:")
            for i, cell_code in enumerate(section_code):
                context_parts.append(f"**Cell {i}:**")
                context_parts.append(f"```python\n{cell_code}\n```")
            context_parts.append("")

        # Add error details
        last_execution_failed = exec_context.inputs.context["last_execution_failed"]
        error_cell = exec_context.inputs.context["error_cell"]
        error_message = exec_context.inputs.context["error_message"]
        if last_execution_failed:
            context_parts.append("## Error Encountered:")
            if error_cell is not None:
                context_parts.append(f"**Error in Cell {error_cell}:**")
            context_parts.append(f"```\n{error_message}\n```")
            context_parts.append("")

        # Add previous fix attempts if any
        fix_attempts = exec_context.inputs.context["fix_attempts"]
        if fix_attempts:
            context_parts.append("## Previous Fix Attempts:")
            for i, attempt in enumerate(fix_attempts):
                context_parts.append(f"**Attempt {i+1}:** {attempt['operation']} on cells {attempt['target_cells']}")
                context_parts.append(f"Reasoning: {attempt['reasoning']}")
            context_parts.append("")

        # Add current objective
        context_parts.append("## Objective:")
        context_parts.append("Fix the error with minimal changes to make this code section run successfully.")

        user_query = "\n".join(context_parts)

        # Update the user query in exec_context
        exec_context.inputs.user_query = user_query

    def _process_structured_result(self, result, inputs: Dict[str, Any]) -> ToolResult:
        """Process structured section code review result."""
        return ToolResult(
            output_ui=result.model_dump(),
            output_type=ToolOutputType.NO_OUTPUT  # Internal only - used by RunSectionWorkflow
        )


class RestartAndRerunTool(BaseTool):
    """Shared tool for restarting notebook and running cells up to target position.

    **UI Returns:**
    - Success: Dict with "text" describing successful execution
    - Failure: Dict with "text" describing section execution failure
    - Simple mode: Dict with "text" and "vscode_commands" for basic restart+rerun
    - `output_type`: EXECUTE_ONLY - performs restart and execution operations

    **Workflow Returns:**
    - None - this tool doesn't propagate workflow data

    **Used by workflows:** Error recovery workflow when RESTART_AND_RETRY strategy is selected

    **Special behavior:** Delegates to RunSectionWorkflow for intelligent execution with error recovery,
    falls back to simple restart+rerun if section workflow unavailable
    """

    def __init__(self, llm_interface=None):
        super().__init__("restart_and_rerun")
        self.llm_interface = llm_interface

        # Initialize RunSectionWorkflow for intelligent execution
        if llm_interface:
            from .run_section_workflow import RunSectionWorkflow
            self.section_workflow = RunSectionWorkflow(llm_interface)
        else:
            self.section_workflow = None

    async def execute(self, exec_context: 'ExecutionContext', **kwargs) -> ToolResult:
        """Execute notebook restart and intelligent rerun of cells up to target position."""
        # Get target cell position from positioning info
        positioning_info = exec_context.inputs.context["positioning_info"]
        target_cell = positioning_info['target_cell']

        if target_cell is None:
            return ToolResult(
                output_ui="No target cell specified for restart and rerun",
                output_type=ToolOutputType.NO_OUTPUT,
            )

        # Step 1: Restart kernel)
        restart_commands = [{"command": "restartKernel"}]

        # Step 2: If we have section workflow capability, use intelligent execution
        if self.section_workflow and target_cell > 0:
            # Get section code from inputs or notebook context (may not be available)
            section_code = exec_context.inputs.context.get("section_code")
            conversation_history = exec_context.inputs.context.get("conversation_history")
            execution_history = exec_context.inputs.context.get("execution_history")

            # If no section code provided, fall back to simple execution
            if not section_code:
                return await self._simple_restart_and_rerun(target_cell, restart_commands)

            # Use RunSectionWorkflow for intelligent execution with error recovery
            section_result = await self.section_workflow.execute_section(
                start_cell=0,
                end_cell=target_cell - 1,  # Run up to (but not including) target
                section_code=section_code,
                conversation_history=conversation_history,
                execution_history=execution_history
            )
            if isinstance(section_result.output_ui, str) and "successfully" in section_result.output_ui:
                return ToolResult(
                    output_ui=f"Kernel restarted and section 0-{target_cell-1} executed successfully with intelligent error recovery",
                    output_type=ToolOutputType.EXECUTE_ONLY,
                )
            else:
                # Section execution failed, report the failure
                return ToolResult(
                    output_ui=f"Kernel restarted but section execution failed: {section_result.output_ui}",
                    output_type=ToolOutputType.EXECUTE_ONLY,
                )
        else:
            # Fall back to simple restart and rerun
            return await self._simple_restart_and_rerun(target_cell, restart_commands)

    async def _simple_restart_and_rerun(self, target_cell: int, restart_commands: List[Dict]) -> ToolResult:
        """Simple restart and rerun without intelligent error recovery."""
        # Create VSCode commands for restart and simple rerun
        vscode_commands = restart_commands + [
            {"command": "runCellsUpTo", "targetCell": target_cell - 1}  # Run up to (but not including) target
        ]

        return ToolResult(
            output_ui={
                "text": f"Restarting kernel and running cells 0 to {target_cell - 1} (simple mode)",
                "vscode_commands": vscode_commands
            },
            output_type=ToolOutputType.EXECUTE_ONLY,
        )


class ReferenceWorkflowSelectionTool(StructuredPromptTool):
    """Selects reference workflows from putative summaries and generates new retrieval queries.

    This tool is used in initial planning to select workflows and optionally generate queries
    for iterative refinement. It processes summaries from ReferenceWorkflowQueryPreparationTool.

    ID Handling:
        - LLM sees full IDs in summaries: "scverse/scanpy-tutorials/pbmc3k.ipynb"
        - LLM returns full IDs in selected_notebooks
        - Tool converts full IDs → internal IDs for storage operations
        - Supports both formats for backward compatibility

    Context Inputs:
        - putative_reference_workflow_summaries: Candidate workflows from semantic search

    Context Updates:
        - reference_workflow_ids: Comma-separated full IDs for UI display
        - reference_workflow_internal_ids: List of internal IDs for storage
        - reference_workflow_content: Formatted notebook content (not cell-filtered yet)
        - retrieval_queries: New queries for next iteration (if iterative mode)

    UI Output:
        - Comma-separated list of selected workflow full IDs
    """

    def __init__(self, scenario: PromptScenario, llm_interface: 'LLMInterface',
                 notebook_selector: NotebookSelector):
        """Initialize with standard prompt tool parameters and notebook components.

        Args:
            scenario: Prompt scenario for this tool
            llm_interface: LLM interface for prompt execution
            notebook_selector: Notebook selection system
        """
        super().__init__(
            name="reference_workflow_selection",
            scenario=scenario,
            llm_interface=llm_interface
        )
        self.selector = notebook_selector

    def _process_structured_result(self, structured_result: ReferenceWorkflowSelection, exec_context: 'ExecutionContext') -> ToolResult:
        # Convert LLM's full IDs to internal IDs
        # LLM sees: "scverse/scanpy-tutorials/pbmc3k.ipynb"
        # Storage needs: "scverse_scanpy_tutorials_pbmc3k"
        internal_ids = []
        for notebook_id in structured_result.selected_notebooks:
            # Check if already internal format (no slashes/dots) or needs conversion
            if "/" in notebook_id or ".ipynb" in notebook_id:
                # Convert full path to internal ID
                internal_id = notebook_id.replace("/", "_").replace("-", "_").replace(".ipynb", "")
                internal_ids.append(internal_id)
            else:
                # Already internal format
                internal_ids.append(notebook_id)

        # Convert structured result to expected format
        selection_data = {"selected_notebooks": internal_ids}

        # Add notebook content to selection data
        selected_notebooks = self.selector.get_selected_notebook_content(
            internal_ids
        )
        selection_data["notebook_content"] = selected_notebooks

        # Format as dict {internal_id: content_string}
        rag_summary_dict = self.selector.format_notebook_context_dict(selection_data)

        # Get selected notebook IDs as full paths (org/repo/filename.ipynb)
        full_ids = []
        for notebook_id, notebook_data in selected_notebooks.items():
            metadata = notebook_data.get("metadata", {})
            full_id = f"{metadata.get('source_repository', 'unknown')}/{metadata.get('workflow_filename', notebook_id)}"
            full_ids.append(full_id)
        selected_notebook_ids = ", ".join(full_ids)

        vscode_response = {"text": selected_notebook_ids}
        if structured_result.retrieval_queries and len(structured_result.retrieval_queries) > 0:
            vscode_response["agent_notification"] = "\n".join(
                ["Reading up on:"] + structured_result.retrieval_queries
            )

        return ToolResult(
            output_ui=vscode_response,
            output_workflow={
                "reference_workflow_ids": selected_notebook_ids,
                "reference_workflow_content": rag_summary_dict,
                "retrieval_queries": structured_result.retrieval_queries,
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        )


class ReferenceWorkflowSelectionOnlyTool(StructuredPromptTool):
    """Updates workflow selection without generating new retrieval queries.

    Used in task iteration loops where retrieval queries come from task generation.
    Updates the workflow selection based on new candidates while preserving workflows
    cited in the task list.

    ID Handling:
        - Same as ReferenceWorkflowSelectionTool
        - Automatic conversion: full IDs ↔ internal IDs

    Context Inputs:
        - putative_reference_workflow_summaries: Candidates from query preparation
        - task_list: Current tasks (workflows cited here are protected from removal)
        - reference_workflow_content: Current workflows (shown to LLM for context)

    Context Updates:
        - reference_workflow_ids: Updated comma-separated full IDs
        - reference_workflow_internal_ids: Updated list of internal IDs
        - reference_workflow_content: Formatted content for new selection (not cell-filtered)

    UI Output:
        - None (cell selection will show final list with percentages)
    """

    def __init__(self, scenario: PromptScenario, llm_interface: 'LLMInterface',
                 notebook_selector: NotebookSelector):
        super().__init__(
            name="reference_workflow_selection_only",
            scenario=scenario,
            llm_interface=llm_interface
        )
        self.selector = notebook_selector

    def _extract_cited_workflows(self, exec_context: 'ExecutionContext') -> set:
        """Extract internal IDs of workflows cited in task list.

        Parses task descriptions looking for citations in format:
        [adapted from: 'org/repo/file.ipynb', cells: X-Y]

        Returns set of internal IDs (e.g., 'org_repo_file')
        """
        import re

        cited_workflows = set()
        task_list = exec_context.inputs.task_list

        if not task_list or "tasks" not in task_list:
            return cited_workflows

        # Pattern matches: [adapted from: 'path/to/file.ipynb', cells: ...]
        # or [custom from: 'path/to/file.ipynb']
        citation_pattern = r"\[(?:adapted|custom) from: ['\"]([^'\"]+\.ipynb)['\"]"

        for task in task_list["tasks"]:
            task_text = task.get("task", "")
            matches = re.findall(citation_pattern, task_text)

            for full_path in matches:
                # Convert FULL ID to INTERNAL ID
                # "org/repo-name/file.ipynb" -> "org_repo_name_file"
                internal_id = full_path.replace("/", "_").replace("-", "_").replace(".ipynb", "")
                cited_workflows.add(internal_id)

        return cited_workflows

    def _process_structured_result(self, structured_result: ReferenceWorkflowSelectionOnly, exec_context: 'ExecutionContext') -> ToolResult:
        # Convert LLM's full IDs to internal IDs
        # LLM sees: "scverse/scanpy-tutorials/pbmc3k.ipynb"
        # Storage needs: "scverse_scanpy_tutorials_pbmc3k"
        internal_ids = set()
        for notebook_id in structured_result.selected_notebooks:
            # Check if already internal format (no slashes/dots) or needs conversion
            if "/" in notebook_id or ".ipynb" in notebook_id:
                # Convert full path to internal ID
                internal_id = notebook_id.replace("/", "_").replace("-", "_").replace(".ipynb", "")
                internal_ids.add(internal_id)
            else:
                # Already internal format
                internal_ids.add(notebook_id)

        # Defensive filtering: ensure workflows cited in task list are never removed
        # Extract cited workflows from task list (already in internal format)
        cited_workflows = self._extract_cited_workflows(exec_context)

        # Merge LLM selection with cited workflows (cited workflows take priority)
        missing_cited = cited_workflows - internal_ids
        if missing_cited:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"LLM removed {len(missing_cited)} cited workflow(s) from selection. Re-adding: {missing_cited}")
        internal_ids.update(cited_workflows)  # Add any cited workflows LLM forgot

        # Convert structured result to expected format
        selection_data = {"selected_notebooks": list(internal_ids)}

        # Add notebook content to selection data
        selected_notebooks = self.selector.get_selected_notebook_content(
            list(internal_ids)
        )
        selection_data["notebook_content"] = selected_notebooks

        # Format as dict {internal_id: content_string}
        rag_summary_dict = self.selector.format_notebook_context_dict(selection_data)

        # Get selected notebook IDs as full paths (org/repo/filename.ipynb)
        full_ids = []
        for notebook_id, notebook_data in selected_notebooks.items():
            metadata = notebook_data.get("metadata", {})
            full_id = f"{metadata.get('source_repository', 'unknown')}/{metadata.get('workflow_filename', notebook_id)}"
            full_ids.append(full_id)
        selected_notebook_ids = ", ".join(full_ids)

        # No UI output - cell selection will show the final list with percentages
        return ToolResult(
            output_ui={},
            output_workflow={
                "reference_workflow_ids": selected_notebook_ids,
                "reference_workflow_content": rag_summary_dict,  # Dict format
            },
            output_type=ToolOutputType.NO_OUTPUT
        )


class ReferenceWorkflowCellSelectionTool(StructuredPromptTool):
    """Tool for selecting relevant cells from reference workflows.

    Optimized to reuse filtered content for unchanged workflow IDs, running LLM only on new ones.

    Storage Format:
        - reference_workflow_content: Dict {internal_id: filtered_content_string}
        - Selection tools set unfiltered content, this tool filters cells
        - Kept IDs: reuse previous filtered content from dict
        - New IDs: run LLM cell selection, add to dict

    Change Detection:
        - Derives previous IDs from context["reference_workflow_percentages"]
        - Calculates: kept_ids (reuse content), new_ids (run LLM)
        - If no new IDs: returns UI message with existing percentages (replaces loading)

    Processing Flow:
        1. Get current IDs from context
        2. Derive previous IDs from percentages
        3. Calculate kept vs new
        4. If no new: return UI with existing percentages
        5. If new: run LLM on new IDs only
        6. Merge: kept content + new filtered content

    Performance:
        - O(new_workflows) LLM calls instead of O(total_workflows)
        - Example: 5 existing + 1 new = 1 LLM call instead of 6 (83% reduction)
        - Dict lookup O(1) for reusing kept content

    Context Updates:
        - reference_workflow_content: {internal_id: filtered_string} for all current IDs
        - reference_workflow_percentages: {full_id: percentage} for all workflows
        - excluded_workflows: Internal IDs of workflows with 0 cells selected

    UI Output:
        - Always sends message to replace loading state
        - Format: "📚 org/repo/file.ipynb (50% of file)" for each workflow
    """

    def __init__(self, scenario: PromptScenario, llm_interface: 'LLMInterface',
                 notebook_selector: NotebookSelector):
        super().__init__(
            name="reference_workflow_cell_selection",
            scenario=scenario,
            llm_interface=llm_interface
        )
        self.selector = notebook_selector

    async def execute(self, exec_context: 'ExecutionContext', **kwargs) -> ToolResult:
        """Execute cell selection for each selected notebook.

        Reuses filtered content for kept workflow IDs, runs LLM only on new IDs.
        """
        # Get current content dict
        current_content_dict = exec_context.inputs.context.get("reference_workflow_content", {})
        if not current_content_dict:
            return ToolResult(
                output_ui={},
                output_workflow={},
                output_type=ToolOutputType.NO_OUTPUT
            )

        # Derive current internal IDs from dict keys
        current_ids = set(current_content_dict.keys())
        previous_percentages = exec_context.inputs.context.get("reference_workflow_percentages", {})

        # Build mapping from internal ID to full ID for current workflows
        all_notebooks = self.selector.get_selected_notebook_content(list(current_ids))
        internal_to_full = {}
        for internal_id, notebook_data in all_notebooks.items():
            metadata = notebook_data.get("metadata", {})
            full_id = f"{metadata.get('source_repository')}/{metadata.get('workflow_filename')}"
            internal_to_full[internal_id] = full_id

        # Derive previous IDs from percentages
        full_to_internal = {v: k for k, v in internal_to_full.items()}
        previous_ids = set()
        for full_id in previous_percentages.keys():
            internal_id = full_to_internal.get(full_id)
            if internal_id:
                previous_ids.add(internal_id)

        # Calculate changes
        kept_ids = current_ids & previous_ids  # Workflows to reuse
        new_ids = current_ids - previous_ids   # Workflows needing LLM cell selection

        # If no new workflows, just send UI to replace loading state
        if not new_ids:
            results = []
            for full_id, percentage in previous_percentages.items():
                results.append((full_id, percentage))

            if results:
                results.sort(key=lambda x: x[0])
                bullet_list = "\n".join([f"📚 {full_id} (considering {percentage:.0f}% of file)" for full_id, percentage in results])
                return ToolResult(
                    output_ui={"text": bullet_list},
                    output_workflow={},  # No updates
                    output_type=ToolOutputType.REFERENCE_WORKFLOWS
                )
            else:
                return ToolResult(
                    output_ui={},
                    output_workflow={},
                    output_type=ToolOutputType.NO_OUTPUT
                )

        # Run LLM cell selection ONLY on NEW workflows
        selected_ranges = {}
        new_notebooks = {nid: all_notebooks[nid] for nid in new_ids if nid in all_notebooks}

        for notebook_id, notebook_data in new_notebooks.items():
            # Temporarily store this single notebook in context for the prompt
            exec_context.inputs.context["current_notebook_for_cell_selection"] = {
                "notebook_id": notebook_id,
                "notebook_data": notebook_data
            }

            # Use the parent's LLM call mechanism
            structured_result = await self._call_llm_structured(exec_context, **kwargs)

            # Validate selected cells
            actual_cell_indices = {cell.get("order") for cell in notebook_data.get("cells", [])}
            valid_cells = [idx for idx in structured_result.selected_cells if idx in actual_cell_indices]
            selected_ranges[notebook_id] = sorted(set(valid_cells))

        # Clean up temporary context
        exec_context.inputs.context.pop("current_notebook_for_cell_selection", None)

        # Format NEW workflows with selected cell ranges
        new_content_dict = {}
        if new_notebooks:
            selection_data = {
                "selected_notebooks": list(new_notebooks.keys()),
                "notebook_content": new_notebooks,
            }
            new_content_dict = self.selector.format_notebook_context_dict(selection_data, selected_ranges=selected_ranges)

        # Merge: kept content from previous + new content from LLM
        merged_content_dict = {}
        for internal_id in kept_ids:
            if internal_id in current_content_dict:
                merged_content_dict[internal_id] = current_content_dict[internal_id]
        merged_content_dict.update(new_content_dict)

        # Build percentages dict combining kept and new
        percentages_dict = {}
        results = []

        # Add kept workflows with their previous percentages
        for internal_id in kept_ids:
            full_id = internal_to_full.get(internal_id)
            if full_id and full_id in previous_percentages:
                percentage = previous_percentages[full_id]
                percentages_dict[full_id] = percentage
                results.append((full_id, percentage))

        # Add new workflows with calculated percentages
        excluded_workflows = []
        for internal_id in new_ids:
            if internal_id in all_notebooks:
                full_id = internal_to_full.get(internal_id)
                notebook_data = all_notebooks[internal_id]
                total_cells = len(notebook_data.get("cells", []))
                selected_cells = len(selected_ranges.get(internal_id, []))

                if total_cells > 0:
                    percentage = min((selected_cells / total_cells * 100), 100)
                else:
                    percentage = 0

                percentages_dict[full_id] = percentage
                results.append((full_id, percentage))

                # Track empty selections
                if selected_cells == 0:
                    excluded_workflows.append(internal_id)

        # Sort and format UI message
        results.sort(key=lambda x: x[0])
        bullet_list = "\n".join([f"📚 {full_id} (considering {percentage:.0f}% of file)" for full_id, percentage in results])

        return ToolResult(
            output_ui={"text": bullet_list},
            output_workflow={
                "reference_workflow_content": merged_content_dict,  # Dict format
                "reference_workflow_percentages": percentages_dict,
                "excluded_workflows": excluded_workflows,
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        )

    async def _call_llm_structured(self, exec_context: 'ExecutionContext', **kwargs) -> 'ReferenceWorkflowCellSelection':
        """Call LLM and parse structured output with logging."""
        from .schemas import ReferenceWorkflowCellSelection

        # Get notebook ID for logging context
        notebook_info = exec_context.inputs.context.get("current_notebook_for_cell_selection", {})
        notebook_id = notebook_info.get("notebook_id", "unknown")

        # Build prompt using parent's mechanism
        use_json_prompting = not self.llm_provider.use_structured_output
        system_prompt, user_prompt = self.prompt_manager.generate_prompt(
            exec_context,
            self.scenario,
            structured_output=not use_json_prompting
        )

        # Log prompt (once per notebook)
        log_filename = self._log_prompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            exec_context=exec_context
        )

        # Call LLM with structured output
        response = await self.llm_provider.generate_structured(
            prompt=user_prompt,
            schema=ReferenceWorkflowCellSelection,
            system_prompt=system_prompt,
            task_type=self._get_task_type(),
            tool_name=f"{self.name}_{notebook_id}",
            **kwargs
        )

        # Log result (once per notebook)
        if log_filename:
            # Create a minimal ToolResult for logging
            from .base_tool import ToolResult, ToolOutputType
            temp_result = ToolResult(
                output_ui={"notebook_id": notebook_id},
                output_workflow={
                    "selected_cells": response.selected_cells,
                    "cell_count": len(response.selected_cells)
                },
                output_type=ToolOutputType.NO_OUTPUT
            )
            self._log_result(temp_result, log_filename)

        return response

    def _process_structured_result(self, structured_result: 'ReferenceWorkflowCellSelection',
                                   exec_context: 'ExecutionContext') -> ToolResult:
        """Not used - we override execute() to handle multiple notebooks."""
        raise NotImplementedError("This method should not be called")


## Critique tools


class AutonomousUpdateCritiqueTool(StructuredPromptTool):
    """Tool for critiquing task list updates in autonomous mode.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_list`: Complete structured task list with tasks array

    **Used by workflows:** Autonomous plannign workflow to create initial task breakdown
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("autonomous_update_critique", PromptScenario.AUTONOMOUS_UPDATE_CRITIQUE, llm_interface)

    def _process_structured_result(self, structured_result, exec_context: 'ExecutionContext') -> ToolResult:
        approval = structured_result.approval.strip()
        critique = structured_result.critique.strip()

        # Validate logic of output:
        if approval == "MODIFY" and not critique:
            # If MODIFY but no critique, force a retry with better instruction
            raise ValueError("MODIFY approval requires a critique explaining what needs to be changed")

        # Send critique to VSCode for display if we have one
        vscode_response = {}
        if critique:
            vscode_response = {"critique": critique}
        self._log_task_list_updates(vscode_response, exec_context)

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow={
                "autonomous_update_approval": approval,
                "autonomous_update_critique": critique,
            },
            output_type=ToolOutputType.TASK_LIST_DISPLAY if critique else ToolOutputType.NO_OUTPUT,
        )

        return result
    

class ReasoningCritiqueTool(StructuredPromptTool):

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("reasoning_critique", PromptScenario.REASONING_CRITIQUE, llm_interface)

    def _process_structured_result(self, structured_result, exec_context: 'ExecutionContext') -> ToolResult:
        approval = structured_result.approval.strip()
        critique = structured_result.critique.strip()

        # Validate logic of output:
        if approval == "MODIFY" and not critique:
            # If MODIFY but no critique, force a retry with better instruction
            raise ValueError("MODIFY approval requires a critique explaining what needs to be changed")

        # Send critique to VSCode for display if we have one
        vscode_response = {}
        if critique:
            vscode_response = {"critique": critique}
        self._log_task_list_updates(vscode_response, exec_context)

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow={
                "reasoning_approval": approval,
                "reasoning_critique": critique,
            },
            output_type=ToolOutputType.TASK_LIST_DISPLAY if critique else ToolOutputType.NO_OUTPUT,
        )

        return result
    

class TaskListCritiqueTool(StructuredPromptTool):
    """Tool for critiquing task lists in autonomous mode.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows formatted task list in chat

    **Workflow Returns:**
    - `task_list`: Complete structured task list with tasks array

    **Used by workflows:** Autonomous initiation workflow to create initial task breakdown
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("task_list_critique", PromptScenario.TASK_LIST_CRITIQUE, llm_interface)

    def _process_structured_result(self, structured_result, exec_context: 'ExecutionContext') -> ToolResult:
        approval = structured_result.approval.strip()
        critique = structured_result.critique.strip()

        # Validate logic of output:
        if approval == "MODIFY" and not critique:
            # If MODIFY but no critique, force a retry with better instruction
            raise ValueError("MODIFY approval requires a critique explaining what needs to be changed")

        # Send critique to VSCode for display if we have one
        vscode_response = {}
        if critique:
            vscode_response = {"critique": critique}
        self._log_task_list_updates(vscode_response, exec_context)

        result = ToolResult(
            output_ui=vscode_response,
            output_workflow={
                "task_list_approval": approval,
                "task_list_critique": critique,
            },
            output_type=ToolOutputType.TASK_LIST_DISPLAY if critique else ToolOutputType.NO_OUTPUT,
        )

        return result
