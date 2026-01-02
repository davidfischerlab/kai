"""
Main Jupyter interface for Kai.

Provides autonomous and interactive modes for running Kai without VSCode.
Mirrors the functionality of the VSCode extension's autonomous execution.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell

from kai.core.agent import KaiAgent
from kai.utils import setup_logger

from .notebook_executor import NotebookExecutor, ExecutionResult
from .context_builder import ContextBuilder

logger = setup_logger(__name__)


class JupyterInterface:
    """
    Main interface between Jupyter notebooks and Kai agent.

    Provides two modes:
    1. Autonomous mode: Agent iteratively works on tasks until completion
    2. Interactive mode: REPL-style interaction with agent

    Mirrors VSCode's AutonomousExecution functionality but uses pure Python
    jupyter_client instead of VSCode API.

    Example:
        interface = JupyterInterface('analysis.ipynb')
        await interface.run_autonomous("Load and analyze data from data.csv")
        interface.save('analysis_completed.ipynb')
    """

    def __init__(
        self,
        notebook_path: str,
        notebook_python: str,
        llm_provider: str = 'ollama',
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        rag_enabled: bool = True,
        turbo_enabled: bool = False,
        max_task_planning_iterations: Optional[int] = None,
        max_workflow_retrieval_iterations: Optional[int] = None
    ):
        """
        Initialize Jupyter interface.

        Args:
            notebook_path: Path to Jupyter notebook file
            notebook_python: REQUIRED. Python executable or conda/mamba environment for notebook kernel.
                           Examples:
                           - '/Users/user/mamba/envs/myenv/bin/python' (direct Python)
                           - '/Users/user/mamba/envs/myenv' (conda env, will use bin/python)
            llm_provider: LLM provider for agent (default: 'ollama')
            model: Specific model to use (default: None = use provider default)
            api_key: API key for LLM provider (default: None = use environment)
            rag_enabled: Enable retrieval-augmented generation (default: True)
            turbo_enabled: Enable turbo mode for faster iteration (default: False)
            max_task_planning_iterations: Max planning iterations (None = use orchestrator default)
            max_workflow_retrieval_iterations: Max workflow retrieval iterations (None = use orchestrator default)
        """
        self.notebook_path = Path(notebook_path)

        # Load notebook
        if self.notebook_path.exists():
            logger.info(f"Loading notebook from {self.notebook_path}")
            with open(self.notebook_path, 'r', encoding='utf-8') as f:
                self.notebook = nbformat.read(f, as_version=4)
        else:
            logger.info(f"Creating new notebook at {self.notebook_path}")
            self.notebook = nbformat.v4.new_notebook()

        # Initialize components
        self.executor = NotebookExecutor(notebook_python=notebook_python)
        self.context_builder = ContextBuilder(self.notebook, str(self.notebook_path))

        # Initialize agent with VSCode messages suppressed
        logger.info(f"Initializing Kai agent with {llm_provider} provider")
        self.agent = KaiAgent(
            llm_provider=llm_provider,
            model=model,
            api_key=api_key,
            suppress_vscode_messages=True,  # Suppress all VSCode JSON output for Jupyter interface
            max_task_planning_iterations=max_task_planning_iterations,
            max_workflow_retrieval_iterations=max_workflow_retrieval_iterations
        )

        # Configuration
        self.rag_enabled = rag_enabled
        self.turbo_enabled = turbo_enabled

        # State tracking for autonomous mode
        self.autonomous_active = False
        self.session_id: Optional[str] = None

        # Message handler for capturing tool outputs
        self.pending_tool_messages: List[Dict[str, Any]] = []

        # Iteration tracking for summary logging
        self.current_iteration_actions: List[Dict[str, Any]] = []

    async def run_autonomous(
        self,
        initial_message: str,
        max_iterations: int = 100,
        graph_recursion_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run agent in autonomous mode.

        Mirrors VSCode's runAutonomousLoop functionality:
        1. Send initial message to agent (planning)
        2. Agent creates task list and plans approach
        3. Agent iteratively executes tasks
        4. Loop continues until all tasks complete or max iterations reached

        Args:
            initial_message: User's initial task description
            max_iterations: Maximum number of autonomous iterations (default: 100)
            graph_recursion_limit: Maximum graph steps per iteration (default: 100, tests can lower to 20)

        Returns:
            Dictionary with execution summary
        """
        logger.info(f"Starting autonomous mode: {initial_message}")
        self.autonomous_active = True
        iteration = 0

        # Set graph recursion limit if provided (for testing)
        if graph_recursion_limit is not None:
            self.agent.orchestrator.set_graph_recursion_limit(graph_recursion_limit)
            logger.info(f"Set graph recursion limit to {graph_recursion_limit}")

        # Set up debug file logging next to output notebook
        debug_log_path = self.notebook_path.with_suffix('.debug.log')
        debug_handler = logging.FileHandler(debug_log_path, mode='w', encoding='utf-8')
        debug_handler.setLevel(logging.DEBUG)
        debug_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        debug_handler.setFormatter(debug_formatter)

        # DON'T add any filters - we want ALL debug logs
        # (QuietModeFilter is added to console handlers, not this file handler)

        # Add to root logger to capture all modules (kai, UI, httpx, etc.)
        root_logger = logging.getLogger()
        original_root_level = root_logger.level
        root_logger.setLevel(logging.DEBUG)

        # Set root logger's console handlers to INFO
        # (DEBUG messages go to debug file only)
        original_root_handler_levels = {}
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler):
                handler_key = f"root:{id(handler)}"
                original_root_handler_levels[handler_key] = handler.level
                handler.setLevel(logging.INFO)

        # CRITICAL: Set ALL existing kai/UI loggers to DEBUG level AND enable propagation
        # kai loggers have propagate=False by default, which prevents logs reaching root handler
        original_logger_levels = {}
        original_logger_propagate = {}
        original_handler_levels = {}
        original_handlers = {}
        for logger_name in list(logging.Logger.manager.loggerDict.keys()):
            if logger_name.startswith('kai') or logger_name.startswith('UI'):
                child_logger = logging.getLogger(logger_name)
                # Save original propagate setting
                original_logger_propagate[logger_name] = child_logger.propagate
                # Enable propagation so logs reach root logger's debug file handler
                child_logger.propagate = True
                # Only change loggers that have an explicit level set (not NOTSET)
                if child_logger.level != logging.NOTSET:
                    original_logger_levels[logger_name] = child_logger.level
                    child_logger.setLevel(logging.DEBUG)
                # REMOVE console handlers to prevent duplicate messages
                # (logs will propagate to root logger's handlers instead)
                original_handlers[logger_name] = child_logger.handlers.copy()
                for handler in child_logger.handlers[:]:  # Iterate over copy to allow removal
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        child_logger.removeHandler(handler)

        root_logger.addHandler(debug_handler)

        logger.info(f"Debug logging to: {debug_log_path}")

        # Log pre-execution directly to root logger to ensure it appears in console
        # (before we start the main loop where planning/execution logs appear)
        root_logger.info("[PRE-EXECUTION]")

        try:
            # Execute all existing cells before agent starts
            pre_exec_result = self.execute_existing_cells()

            if not pre_exec_result['success']:
                root_logger.error("Pre-execution failed, cannot start autonomous mode")
                return {
                    'success': False,
                    'error': 'Pre-execution of existing cells failed',
                    'failed_cells': pre_exec_result['failed'],
                    'iterations': 0
                }

            root_logger.info(f"  Executed {pre_exec_result['executed']} cells")

            # Add initial message to conversation BEFORE getting context
            # so conversation history is populated when context is built
            self.context_builder.add_user_message(initial_message)

            # Build initial context
            context = self.context_builder.get_context(
                autonomous_mode=True,
                autonomous_mode_continue=False,
                rag_enabled=self.rag_enabled,
                turbo_enabled=self.turbo_enabled
            )

            # Set up message capture from UICommunicator
            self._setup_message_capture()

            current_message = initial_message

            # Main autonomous loop
            # Loop iteration 1 = planning phase (may also include first execution if confirm_plan=False)
            # Loop iteration 2+ = execution iterations
            execution_iteration = 0
            planning_done = False
            while self.autonomous_active and iteration < max_iterations:
                iteration += 1

                # Log execution iteration start
                # After planning is done, each loop iteration is an execution iteration
                if planning_done:
                    execution_iteration += 1
                    root_logger.info(f"[EXECUTION ITERATION {execution_iteration}/{max_iterations}]")

                # Clear pending tool messages and iteration tracking
                self.pending_tool_messages = []
                self.current_iteration_actions = []

                # Send message to agent
                response, self.session_id = await self.agent.chat(
                    user_input=current_message,
                    session_id=self.session_id,
                    user_id="jupyter_user",
                    context=context
                )

                # Check if autonomous mode completed
                if not self.agent.is_autonomous_active(self.session_id):
                    logger.info("Autonomous mode completed by agent")
                    break

                # Process any tool messages that were captured
                # Returns True if LOOP_COMPLETE signal was received
                loop_complete = await self._process_tool_messages()

                # Mark planning as done after first iteration (before checking loop_complete)
                # This ensures execution iterations are counted even if planning completes in one iteration
                if iteration == 1:
                    planning_done = True

                # Break if all tasks are complete (mirrors VSCode autonomous-execution.ts line 221-223)
                if loop_complete:
                    break

                # Log iteration summary (only for execution iterations after planning)
                if iteration > 1 or context.get('autonomousModeContinue'):
                    self._log_iteration_summary(iteration, max_iterations)

                # Update context for next iteration
                # NOTE: task_list is now managed by orchestrator's checkpointer
                # We don't pass it in context - orchestrator loads from checkpoint
                context = self.context_builder.get_context(
                    autonomous_mode=True,
                    autonomous_mode_continue=True,  # Continue mode (no user message)
                    rag_enabled=self.rag_enabled,
                    turbo_enabled=self.turbo_enabled
                )

                # Continue with empty message (execution mode)
                current_message = ""

            # Autonomous loop completed
            logger.info(f"Autonomous mode finished after {iteration} iterations")

            # Clear checkpoints if in TRANSIENT mode (successful completion)
            if self.session_id and iteration < max_iterations:
                await self.agent.orchestrator.clear_session_on_completion(
                    self.session_id
                )

            return {
                'success': True,
                'iterations': iteration,
                'final_state': 'completed' if iteration < max_iterations else 'max_iterations_reached'
            }

        except Exception as e:
            logger.error(f"Error in autonomous mode: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'iterations': iteration
            }

        finally:
            self.autonomous_active = False

            # Remove debug file handler and restore all logger levels
            root_logger.removeHandler(debug_handler)
            root_logger.setLevel(original_root_level)

            # Restore root logger's handler levels
            for handler in root_logger.handlers:
                if isinstance(handler, logging.StreamHandler):
                    handler_key = f"root:{id(handler)}"
                    if handler_key in original_root_handler_levels:
                        handler.setLevel(original_root_handler_levels[handler_key])

            # Restore all kai/UI logger levels and propagate settings
            for logger_name, original_level in original_logger_levels.items():
                logging.getLogger(logger_name).setLevel(original_level)
            for logger_name, original_propagate in original_logger_propagate.items():
                logging.getLogger(logger_name).propagate = original_propagate

            # Restore kai/UI handlers that were removed
            for logger_name, handlers in original_handlers.items():
                child_logger = logging.getLogger(logger_name)
                # Re-add handlers that were removed
                for handler in handlers:
                    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                        if handler not in child_logger.handlers:
                            child_logger.addHandler(handler)

            debug_handler.close()

    async def run_interactive(self, initial_message: str):
        """
        Run agent in interactive mode (REPL-style).

        Args:
            initial_message: Initial user message
        """
        logger.info("Starting interactive mode")

        # Add to conversation BEFORE getting context
        self.context_builder.add_user_message(initial_message)

        # Build context
        context = self.context_builder.get_context(
            autonomous_mode=False,
            rag_enabled=self.rag_enabled,
            turbo_enabled=self.turbo_enabled
        )

        # Send to agent
        response, self.session_id = await self.agent.chat(
            user_input=initial_message,
            session_id=self.session_id,
            user_id="jupyter_user",
            context=context
        )

        # Process tool messages
        await self._process_tool_messages()

        logger.info("Interactive mode message processed")

        return response

    def _setup_message_capture(self):
        """
        Set up message capture from UICommunicator.
        """
        logger.info("[MESSAGE_CAPTURE] Setting up message capture for Jupyter interface")

        # Import UICommunicator to access class methods
        from kai.core.orchestration.ui_communicator import UICommunicator

        # Create capture functions that store messages for processing
        async def capture_tool_result(result, context):
            """Capture tool results instead of sending to stdout."""
            logger.info(f"[MESSAGE_CAPTURE] Intercepted tool result: type={result.output_type.value}, has_ui={bool(result.output_ui)}, has_workflow={bool(result.output_workflow)}")

            # Store the tool result for processing (including workflow data for task list updates)
            self.pending_tool_messages.append({
                'type': result.output_type.value,
                'data': result.output_ui,
                'workflow': result.output_workflow  # Critical for task list updates!
            })


        async def capture_workflow_result(field, state):
            """Capture workflow results (e.g., LOOP_COMPLETE)."""
            logger.info(f"[MESSAGE_CAPTURE] Intercepted workflow result: field={field}, state={state}")
            self.pending_tool_messages.append({
                'type': 'workflow_result',
                'data': {field: state},
                'workflow': {}
            })
            # Don't send to stdout - we're capturing for Jupyter processing

        # Set class-level hooks - these capture messages from ALL UICommunicator instances
        UICommunicator.set_tool_result_hook(capture_tool_result)
        UICommunicator.set_workflow_result_hook(capture_workflow_result)
        logger.info("[MESSAGE_CAPTURE] Set class-level hooks on UICommunicator")

        # Note: We don't need to suppress console messages here
        # because suppress_vscode_messages=True in KaiAgent already sets _disabled=True
        # which prevents all VSCode communication methods from outputting to stdout

    def _log_iteration_summary(self, iteration: int, max_iterations: int):
        """
        Log high-level summary of what happened in this iteration.

        This is shown in quiet mode instead of detailed action logs.
        """
        if not self.current_iteration_actions:
            logger.info("  No actions taken")
            return

        # Summarize actions
        for action in self.current_iteration_actions:
            action_type = action['type']

            if action_type == 'add_cell':
                cell_type = action.get('cell_type', 'code')
                code_preview = action.get('code', '')[:150].replace('\n', ' ')
                logger.info(f"  Added {cell_type} cell: {code_preview}...")

            elif action_type == 'replace_cell':
                cell_index = action.get('cell_index', '?')
                code_preview = action.get('code', '')[:150].replace('\n', ' ')
                logger.info(f"  Replaced cell {cell_index}: {code_preview}...")

            elif action_type == 'delete_cell':
                cell_index = action.get('cell_index', '?')
                logger.info(f"  Deleted cell {cell_index}")

            elif action_type == 'execute':
                cell_index = action.get('cell_index', '?')
                success = action.get('success', False)
                status = 'SUCCESS' if success else 'FAILED'
                logger.info(f"  Executed cell {cell_index} - {status}")

    async def _process_tool_messages(self) -> bool:
        """
        Process captured tool messages.

        Handles execute_code, display, task_list_display, workflow_result, etc.
        Mirrors VSCode's handleAutonomousCodeExecution functionality.

        Returns:
            True if LOOP_COMPLETE signal was received, False otherwise
        """
        loop_complete = False

        logger.info(f"[PROCESS_MSGS] Processing {len(self.pending_tool_messages)} tool messages")

        # Log all message types we're about to process
        message_types = [msg['type'] for msg in self.pending_tool_messages]
        logger.info(f"[PROCESS_MSGS] Message types: {message_types}")

        for msg in self.pending_tool_messages:
            msg_type = msg['type']
            data = msg['data']
            workflow = msg.get('workflow', {})

            logger.info(f"[PROCESS_MSGS] Processing message type: {msg_type}")

            # NOTE: task_list and other persistent fields are now managed by LangGraph's checkpointer
            # We no longer need to extract and save them from workflow messages
            # The orchestrator persists them automatically across iterations via checkpoint
            if workflow:
                logger.debug(f"[PROCESS_MSGS] Workflow data received. Keys: {list(workflow.keys())}")

            if msg_type == 'execute_code':
                logger.info(f"[PROCESS_MSGS] Executing code from message")
                await self._handle_execute_code(data)

            elif msg_type == 'display':
                # Display messages (logging only in Jupyter interface)
                logger.info(f"Agent message: {data.get('text', data)}")

            elif msg_type == 'task_list_display':
                # Just log task list updates for visibility
                # No need to save - orchestrator's checkpointer manages persistence
                if 'text' in data:
                    try:
                        task_list_json = json.loads(data['text'])
                        logger.info(f"[TASK UPDATE] Task list display: {len(task_list_json.get('tasks', []))} tasks")
                    except (json.JSONDecodeError, KeyError):
                        logger.warning("Failed to parse task_list from task_list_display message")

                # Only log full details if DEBUG level is enabled
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Task list update: {json.dumps(data, indent=2)}")

            elif msg_type == 'workflow_result':
                # Check for completion signal (mirrors VSCode autonomous-execution.ts line 221)
                if data.get('auto_loop_update') == 'LOOP_COMPLETE':
                    logger.info("Autonomous loop completed (all tasks finished)")
                    loop_complete = True

        return loop_complete

    async def _handle_execute_code(self, code_response: Dict[str, Any]):
        """
        Handle code execution request from agent.

        Mirrors VSCode's handleAutonomousCodeExecution:
        - Extracts code and positioning info
        - Adds/replaces cells in notebook
        - Executes code via kernel
        - Captures outputs and updates context

        Args:
            code_response: Dictionary with code, positioning_info, etc.
        """
        # Check for cell deletion commands
        if 'vscode_commands' in code_response:
            commands = code_response['vscode_commands']
            for cmd in commands:
                if cmd['command'] == 'deleteCell':
                    cell_index = cmd['cellIndex']
                    logger.info(f"Deleting cell {cell_index}")
                    self._delete_cell(cell_index)
                    # Track action
                    self.current_iteration_actions.append({
                        'type': 'delete_cell',
                        'cell_index': cell_index
                    })
            return

        # Extract code execution parameters
        code = code_response.get('code', '').strip()
        cell_type = code_response.get('cell_type', 'code')
        positioning_info = code_response.get('positioning_info', {})
        target_cell = positioning_info.get('target_cell', -1)
        should_replace = code_response.get('should_replace') is True
        recovery_strategy = code_response.get('error_recovery_strategy')

        logger.info(f"Execute code request: cell_type={cell_type}, target={target_cell}, replace={should_replace}")

        # Handle markdown cells (no execution)
        if cell_type == 'markdown':
            actual_cell_index = None
            if should_replace:
                actual_cell_index = self._replace_markdown(code, target_cell)
                # Track action
                self.current_iteration_actions.append({
                    'type': 'replace_cell',
                    'cell_type': 'markdown',
                    'cell_index': actual_cell_index,
                    'code': code
                })
            else:
                actual_cell_index = self._add_markdown(code, target_cell)
                # Track action
                self.current_iteration_actions.append({
                    'type': 'add_cell',
                    'cell_type': 'markdown',
                    'cell_index': actual_cell_index,
                    'code': code
                })

            # Markdown cells don't execute, mark as success
            self.context_builder.update_execution_state(
                success=True,
                output="",
                cell_index=None
            )

            # Update last modified cell (mirrors VSCode behavior)
            self.context_builder.last_cell_modified_in_auto_mode = actual_cell_index

            # Add markdown cell to execution history so agent sees it was processed
            # (VSCode includes markdown cells in execution history)
            mock_exec_result = ExecutionResult(
                success=True,
                outputs=[],
                error=None,
                terminated=False,
                duration=0.0
            )
            self.context_builder.add_to_execution_history(actual_cell_index, mock_exec_result, code)

            self.save()
            return

        # Handle code cells
        cell_index = None

        if should_replace:
            cell_index = self._replace_code(code, target_cell)
            # Track action
            self.current_iteration_actions.append({
                'type': 'replace_cell',
                'cell_index': cell_index,
                'code': code
            })

            # Handle REPLACE_AND_RESTART strategy
            if recovery_strategy == 'REPLACE_AND_RESTART':
                await self._handle_restart_and_rerun(target_cell)
                # Update last modified cell even for restart strategy (mirrors VSCode)
                self.context_builder.last_cell_modified_in_auto_mode = cell_index
                self.save()
                return  # Restart handles execution

        else:
            cell_index = self._add_code(code, target_cell)
            # Track action
            self.current_iteration_actions.append({
                'type': 'add_cell',
                'cell_type': 'code',
                'cell_index': cell_index,
                'code': code
            })

        # Execute the cell
        logger.info(f"Executing cell {cell_index}")

        # Create progress callback for monitoring
        async def progress_callback(elapsed_time, partial_outputs):
            """Check with agent if long-running cell should continue."""
            try:
                result = await self.agent.orchestrator._handle_execution_progress_check(
                    context={
                        'current_cell': code,
                        'elapsed_time': elapsed_time,
                        'partial_outputs': partial_outputs,
                        'active_task': "Executing cell"
                    },
                    session_metadata={}
                )
                return result.get('action') == 'continue'
            except Exception as e:
                logger.error(f"Progress check failed: {e}")
                return True  # Default to continue on error

        # Execute cell without progress monitoring
        # (Progress monitoring requires async callback which execute_cell doesn't support)
        exec_result = self.executor.execute_cell(
            code=code,
            timeout=1800,  # 30 minutes
            progress_check_callback=None  # Disable to avoid async/sync mismatch
        )

        # Update cell outputs in notebook (so they show in VSCode/Jupyter)
        self._update_cell_outputs(cell_index, exec_result)

        # Add to execution history
        self.context_builder.add_to_execution_history(cell_index, exec_result, code)

        # Log execution (use root_logger to ensure visibility in autonomous mode)
        status = 'SUCCESS' if exec_result.success else 'FAILED'
        logging.getLogger().info(f"Executed cell {cell_index} - {status}")

        # Track execution
        self.current_iteration_actions.append({
            'type': 'execute',
            'cell_index': cell_index,
            'success': exec_result.success
        })

        # Update execution state for error recovery
        if exec_result.terminated:
            # Cell was terminated by monitoring
            output = f"[EXECUTION TERMINATED BY MONITORING AGENT]\n{exec_result.error.get('evalue', '')}\n\nPartial outputs:\n{self._format_execution_output(exec_result)}"
            self.context_builder.update_execution_state(
                success=False,
                output=output,
                cell_index=cell_index
            )
        elif exec_result.success:
            # Execution succeeded
            output = self._format_execution_output(exec_result)
            self.context_builder.update_execution_state(
                success=True,
                output=output,
                cell_index=None
            )
        else:
            # Execution failed
            output = self._format_execution_output(exec_result)
            self.context_builder.update_execution_state(
                success=False,
                output=output,
                cell_index=cell_index
            )

        # Update last modified cell
        self.context_builder.last_cell_modified_in_auto_mode = cell_index

        # Auto-save notebook after modification
        self.save()
        
    def _add_code(self, code: str, after_cell: int) -> int:
        """
        Add new code cell to notebook.

        Mirrors VSCode behavior with position clamping.

        Args:
            code: Code content
            after_cell: Insert after this cell index (-1 for start)

        Returns:
            Index of new cell
        """
        # Mirror VSCode: clamp to valid range to prevent inserting beyond end
        insert_position = 0 if after_cell == -1 else min(after_cell + 1, len(self.notebook.cells))

        new_cell = new_code_cell(source=code)
        self.notebook.cells.insert(insert_position, new_cell)

        # Track modification
        self.context_builder.track_modification(
            cell_index=insert_position,
            modification_type='created',
            content=code
        )

        logger.info(f"Added code cell at position {insert_position}")
        return insert_position

    def _replace_code(self, code: str, cell_index: int) -> int:
        """
        Replace cell with code content.

        Mirrors VSCode behavior: if target cell is not code, replace it with a new code cell.

        Args:
            code: New code content
            cell_index: Cell to replace

        Returns:
            Cell index
        """
        if 0 <= cell_index < len(self.notebook.cells):
            current_cell = self.notebook.cells[cell_index]
            old_content = current_cell.source

            if current_cell.cell_type == 'code':
                # Simple case: just update content and clear outputs
                current_cell.source = code
                current_cell.outputs = []
                logging.getLogger().info(f"Replaced code in cell {cell_index}")
            else:
                # Cell type mismatch: replace entire cell (VSCode behavior)
                logging.getLogger().info(f"Cell {cell_index} is {current_cell.cell_type}, converting to code")
                new_cell = new_code_cell(source=code)
                self.notebook.cells[cell_index] = new_cell
                logging.getLogger().info(f"Replaced cell {cell_index} with code cell")

            # Track modification
            self.context_builder.track_modification(
                cell_index=cell_index,
                modification_type='modified',
                content=code,
                old_content=old_content
            )
        else:
            logger.warning(f"Invalid cell index {cell_index}, adding at end")
            cell_index = len(self.notebook.cells)
            self._add_code(code, cell_index - 1)

        return cell_index

    def _add_markdown(self, content: str, after_cell: int) -> int:
        """
        Add markdown cell to notebook.

        Mirrors VSCode behavior with position clamping.

        Args:
            content: Markdown content
            after_cell: Insert after this cell index (-1 for start)

        Returns:
            Index of the new cell
        """
        # Mirror VSCode: clamp to valid range to prevent inserting beyond end
        insert_position = 0 if after_cell == -1 else min(after_cell + 1, len(self.notebook.cells))

        new_cell = new_markdown_cell(source=content)
        self.notebook.cells.insert(insert_position, new_cell)

        # Track modification
        self.context_builder.track_modification(
            cell_index=insert_position,
            modification_type='created',
            content=content
        )

        logger.info(f"Added markdown cell at position {insert_position}")
        return insert_position

    def _replace_markdown(self, content: str, cell_index: int) -> int:
        """
        Replace cell with markdown content.

        Mirrors VSCode behavior: if target cell is not markdown, replace it with a new markdown cell.
        This is what VSCode's NotebookEdit.replaceCells() does.

        Args:
            content: Markdown content
            cell_index: Cell to replace

        Returns:
            Index of the replaced cell (same as input)
        """
        if 0 <= cell_index < len(self.notebook.cells):
            current_cell = self.notebook.cells[cell_index]
            old_content = current_cell.source

            if current_cell.cell_type == 'markdown':
                # Simple case: just update content
                current_cell.source = content
                logging.getLogger().info(f"Replaced markdown content in cell {cell_index}")
            else:
                # Cell type mismatch: replace entire cell (VSCode behavior)
                logging.getLogger().info(f"Cell {cell_index} is {current_cell.cell_type}, converting to markdown")
                new_cell = new_markdown_cell(source=content)
                self.notebook.cells[cell_index] = new_cell
                logging.getLogger().info(f"Replaced cell {cell_index} with markdown cell")

            # Track modification
            self.context_builder.track_modification(
                cell_index=cell_index,
                modification_type='modified',
                content=content,
                old_content=old_content
            )

        return cell_index

    def _delete_cell(self, cell_index: int):
        """Delete cell from notebook."""
        if 0 <= cell_index < len(self.notebook.cells):
            # Store content before deletion
            old_content = self.notebook.cells[cell_index].source

            del self.notebook.cells[cell_index]
            logger.info(f"Deleted cell {cell_index}")

            # Track modification
            self.context_builder.track_modification(
                cell_index=cell_index,
                modification_type='deleted',
                content='',
                old_content=old_content
            )

            # Update context tracking
            if self.context_builder.last_cell_modified_in_auto_mode >= cell_index:
                self.context_builder.last_cell_modified_in_auto_mode = max(
                    -1,
                    self.context_builder.last_cell_modified_in_auto_mode - 1
                )
            self.save()

    async def _handle_restart_and_rerun(self, error_cell_index: int):
        """
        Handle REPLACE_AND_RESTART error recovery strategy.

        1. Restart kernel
        2. Re-execute all cells up to and including error cell

        Args:
            error_cell_index: Index of cell that had error
        """
        logger.info(f"Executing REPLACE_AND_RESTART strategy for cell {error_cell_index}")

        # Restart kernel
        self.executor.restart_kernel()

        # Re-execute cells from 0 to error_cell_index
        for i in range(min(error_cell_index + 1, len(self.notebook.cells))):
            cell = self.notebook.cells[i]

            if cell.cell_type == 'code':
                logger.info(f"Re-executing cell {i} after restart")

                exec_result = self.executor.execute_cell(
                    code=cell.source,
                    timeout=1800
                )

                # Update cell outputs in notebook
                self._update_cell_outputs(i, exec_result)

                # Add to execution history
                self.context_builder.add_to_execution_history(i, exec_result, cell.source)

                # Update context if this is the error cell
                if i == error_cell_index:
                    output = self._format_execution_output(exec_result)
                    self.context_builder.update_execution_state(
                        success=exec_result.success,
                        output=output,
                        cell_index=None if exec_result.success else i
                    )

    def _update_cell_outputs(self, cell_index: int, exec_result: ExecutionResult):
        """
        Update cell outputs in notebook with execution results.

        Converts ExecutionResult outputs to nbformat output objects
        so they display properly in Jupyter/VSCode with ALL content
        including images, HTML, stderr, execution counts, etc.

        Args:
            cell_index: Index of cell to update
            exec_result: Execution result from NotebookExecutor
        """
        from nbformat.v4 import new_output

        self.notebook.cells[cell_index].outputs = []
        if exec_result.outputs:
            for output_type, output_data in exec_result.outputs:
                if output_type == 'stream':
                    # Stream output (stdout/stderr)
                    self.notebook.cells[cell_index].outputs.append(
                        new_output(output_type, **output_data)
                    )
                elif output_type == 'display_data':
                    # Display data (plots, dataframes, images, HTML, etc.)
                    self.notebook.cells[cell_index].outputs.append(
                        new_output(output_type, **output_data)
                    )
                elif output_type == 'execute_result':
                    # Execution result with return value
                    self.notebook.cells[cell_index].outputs.append(
                        new_output(output_type, **output_data)
                    )
                elif output_type == 'error':
                    # Error output
                    self.notebook.cells[cell_index].outputs.append(
                        new_output(output_type, **output_data)
                    )

    def _format_execution_output(self, exec_result: ExecutionResult) -> str:
        """
        Format execution result for context.

        Args:
            exec_result: ExecutionResult from NotebookExecutor

        Returns:
            Formatted output string
        """
        parts = []

        # Add outputs
        if exec_result.outputs:
            output_str = self.context_builder._format_outputs(exec_result.outputs)
            parts.append(output_str)

        # Add error if present
        if exec_result.error:
            error_str = self.context_builder._format_error(exec_result.error)
            parts.append(f">>> Error output\n{error_str}")

        return '\n\n'.join(parts) if parts else '[No output]'

    def execute_existing_cells(self):
        """
        Execute all existing cells in the notebook before agent starts.
        
        This is important for base notebooks that set up the environment,
        load data, and prepare the workspace for the agent.
        
        Returns:
            Dictionary with execution summary
        """
        logger.info(f"Executing {len(self.notebook.cells)} existing cells...")
        
        executed_count = 0
        failed_cells = []
        
        for idx, cell in enumerate(self.notebook.cells):
            if cell.cell_type == 'code':
                logger.info(f"Executing existing cell {idx}")
                
                exec_result = self.executor.execute_cell(
                    code=cell.source,
                    timeout=1800  # 30 minutes per cell
                )
                
                # Update cell outputs in notebook
                self._update_cell_outputs(idx, exec_result)
                
                # Add to execution history
                self.context_builder.add_to_execution_history(idx, exec_result, cell.source)
                
                executed_count += 1
                
                if not exec_result.success:
                    logger.error(f"Cell {idx} failed: {exec_result.error}")
                    failed_cells.append({
                        'index': idx,
                        'error': exec_result.error
                    })
                    # Stop on first error
                    break
        
        # Save notebook with outputs
        self.save()
        
        logger.info(f"Executed {executed_count} cells, {len(failed_cells)} failed")
        
        return {
            'executed': executed_count,
            'failed': failed_cells,
            'success': len(failed_cells) == 0
        }

    def save(self, output_path: Optional[str] = None):
        """
        Save notebook to file.

        Args:
            output_path: Path to save to (default: overwrite original)
        """
        save_path = Path(output_path) if output_path else self.notebook_path

        logger.info(f"Saving notebook to {save_path}")
        with open(save_path, 'w', encoding='utf-8') as f:
            nbformat.write(self.notebook, f)

        logger.info("Notebook saved successfully")

    # =========================================================================
    # Session Restart / Resume (SqliteSaver persistence)
    # =========================================================================

    async def list_resumable_sessions(self) -> List[Dict[str, Any]]:
        """List sessions that can be resumed from SqliteSaver checkpoints.

        Only works when SqliteSaver is configured (CHECKPOINT_DB_PATH set).
        MemorySaver sessions are lost when the process exits.

        Returns:
            List of session info dicts with keys:
            - session_id: Thread ID for resuming
            - notebook_uri: Path to notebook (if stored in state)
            - last_cell_modified: Last cell modified by agent
            - timestamp: Last checkpoint timestamp
            - task_count: Number of tasks in task list
            - completed_count: Number of completed tasks
        """
        from kai.core.persistence.checkpointer import is_sqlite_checkpointer

        checkpointer = self.agent.orchestrator.checkpointer

        if not is_sqlite_checkpointer(checkpointer):
            logger.warning("Session resume requires SqliteSaver. Set CHECKPOINT_DB_PATH in settings.")
            return []

        sessions = []
        try:
            # SqliteSaver supports list(None) to get all threads
            threads = list(checkpointer.list(None))

            for thread_info in threads:
                thread_id = thread_info.get("thread_id", "")
                if not thread_id:
                    continue

                # Get latest state for this thread
                config = {"configurable": {"thread_id": thread_id}}
                try:
                    state = await self.agent.orchestrator.main_graph.aget_state(config)
                    if state and hasattr(state, 'values') and state.values:
                        values = state.values
                        task_list = values.get("task_list", {})
                        tasks = task_list.get("tasks", [])

                        sessions.append({
                            "session_id": thread_id,
                            "notebook_uri": values.get("notebook_uri"),
                            "last_cell_modified": values.get("last_cell_modified_in_auto_mode"),
                            "timestamp": state.metadata.get("ts") if state.metadata else None,
                            "task_count": len(tasks),
                            "completed_count": sum(1 for t in tasks if t.get("status") == "completed"),
                        })
                except Exception as e:
                    logger.debug(f"Could not get state for thread {thread_id}: {e}")

        except Exception as e:
            logger.error(f"Failed to list sessions: {e}")

        return sessions

    async def restart_session(
        self,
        session_id: str,
        max_iterations: int = 100,
    ) -> Dict[str, Any]:
        """Restart a previously interrupted session.

        Iteration Model (see kai.core.orchestration.graphs.main):
        ==========================================================
        LangGraph checkpoints capture state after each node, but Jupyter
        cell execution happens as a side effect after code_generation.

        On restart:
        1. Kernel is restarted (fresh Python state)
        2. All cells up to last_cell_modified are re-executed
        3. This brings notebook to consistent state matching checkpoint
        4. LangGraph resumes from checkpoint, continuing with next iteration

        LangGraph doesn't track fine-grained Jupyter state (inserted vs executed).
        The notebook save + kernel re-run pattern handles state reconciliation.

        Args:
            session_id: Session ID to restart (from list_resumable_sessions)
            max_iterations: Maximum iterations for continued autonomous execution

        Returns:
            Dict with restart summary (success, iterations, etc.)
        """
        from kai.core.persistence.checkpointer import is_sqlite_checkpointer

        checkpointer = self.agent.orchestrator.checkpointer

        if not is_sqlite_checkpointer(checkpointer):
            return {
                'success': False,
                'error': 'Session restart requires SqliteSaver. Set CHECKPOINT_DB_PATH in settings.',
            }

        # Get checkpoint state
        config = {"configurable": {"thread_id": session_id}}
        try:
            state = await self.agent.orchestrator.main_graph.aget_state(config)
            if not state or not hasattr(state, 'values') or not state.values:
                return {
                    'success': False,
                    'error': f'No checkpoint found for session {session_id}',
                }
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to load checkpoint: {e}',
            }

        values = state.values

        # Get notebook path from checkpoint
        notebook_uri = values.get("notebook_uri")
        if not notebook_uri:
            return {
                'success': False,
                'error': 'Checkpoint does not contain notebook_uri. Cannot restart.',
            }

        # Verify notebook exists
        notebook_path = Path(notebook_uri)
        if not notebook_path.exists():
            return {
                'success': False,
                'error': f'Notebook not found: {notebook_path}',
            }

        # Get last cell modified (for re-run target)
        last_cell_modified = values.get("last_cell_modified_in_auto_mode", -1)
        if last_cell_modified is None:
            last_cell_modified = -1

        logger.info(f"Restarting session {session_id}")
        logger.info(f"  Notebook: {notebook_path}")
        logger.info(f"  Last cell modified: {last_cell_modified}")

        # Load the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            self.notebook = nbformat.read(f, as_version=4)
        self.notebook_path = notebook_path
        self.context_builder = ContextBuilder(self.notebook, str(notebook_path))

        # Restart kernel
        logger.info("Restarting kernel...")
        self.executor.restart_kernel()

        # Re-run cells up to and including last_cell_modified
        # This brings the notebook to the state it was in when interrupted
        rerun_target = last_cell_modified if last_cell_modified >= 0 else len(self.notebook.cells) - 1

        logger.info(f"Re-running cells 0 to {rerun_target}...")
        for idx in range(min(rerun_target + 1, len(self.notebook.cells))):
            cell = self.notebook.cells[idx]
            if cell.cell_type == 'code':
                logger.info(f"  Re-executing cell {idx}")
                exec_result = self.executor.execute_cell(
                    code=cell.source,
                    timeout=1800  # 30 minutes
                )
                self._update_cell_outputs(idx, exec_result)
                self.context_builder.add_to_execution_history(idx, exec_result, cell.source)

                if not exec_result.success:
                    logger.error(f"Cell {idx} failed during restart: {exec_result.error}")
                    return {
                        'success': False,
                        'error': f'Cell {idx} failed during re-execution',
                        'cell_index': idx,
                        'cell_error': exec_result.error,
                    }

        self.save()
        logger.info("Notebook cells re-executed successfully")

        # Set up for autonomous continuation
        self.autonomous_active = True
        self.session_id = session_id
        self._setup_message_capture()

        # Continue autonomous loop from checkpoint
        # The checkpoint contains task_list, so LangGraph will continue from where it left off
        iteration = 0

        try:
            while self.autonomous_active and iteration < max_iterations:
                iteration += 1
                logger.info(f"[RESTART] Iteration {iteration}/{max_iterations}")

                self.pending_tool_messages = []
                self.current_iteration_actions = []

                # Build context from current notebook state
                context = self.context_builder.get_context(
                    autonomous_mode=True,
                    autonomous_mode_continue=True,  # Continue mode
                    rag_enabled=self.rag_enabled,
                    turbo_enabled=self.turbo_enabled
                )

                # Send empty message (continue mode)
                response, self.session_id = await self.agent.chat(
                    user_input="",
                    session_id=self.session_id,
                    user_id="jupyter_user",
                    context=context
                )

                # Check completion
                if not self.agent.is_autonomous_active(self.session_id):
                    logger.info("Autonomous mode completed")
                    break

                loop_complete = await self._process_tool_messages()
                if loop_complete:
                    break

                self._log_iteration_summary(iteration, max_iterations)

            # Clear checkpoints if in TRANSIENT mode (successful completion)
            if iteration < max_iterations:
                await self.agent.orchestrator.clear_session_on_completion(
                    session_id
                )

            return {
                'success': True,
                'iterations': iteration,
                'final_state': 'completed' if iteration < max_iterations else 'max_iterations_reached',
                'session_id': session_id,
            }

        except Exception as e:
            logger.error(f"Error during restart continuation: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'iterations': iteration,
            }

        finally:
            self.autonomous_active = False

    def shutdown(self):
        """Shutdown kernel and cleanup resources."""
        logger.info("Shutting down Jupyter interface")
        self.executor.shutdown()

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.shutdown()
