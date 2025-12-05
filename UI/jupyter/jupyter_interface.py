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
from kai.core.orchestration.vscode_communicator import VSCodeCommunicator
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
        turbo_enabled: bool = False
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
            suppress_vscode_messages=True  # Suppress all VSCode JSON output for Jupyter interface
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
        max_iterations: int = 50
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
            max_iterations: Maximum number of autonomous iterations (default: 50)

        Returns:
            Dictionary with execution summary
        """
        logger.info(f"Starting autonomous mode: {initial_message}")
        self.autonomous_active = True
        iteration = 0

        try:
            # Execute all existing cells before agent starts
            logger.info("Pre-executing all existing notebook cells...")
            pre_exec_result = self.execute_existing_cells()
            
            if not pre_exec_result['success']:
                logger.error("Pre-execution failed, cannot start autonomous mode")
                return {
                    'success': False,
                    'error': 'Pre-execution of existing cells failed',
                    'failed_cells': pre_exec_result['failed'],
                    'iterations': 0
                }
            
            logger.info(f"Pre-execution complete: {pre_exec_result['executed']} cells executed")
            
            # Build initial context
            context = self.context_builder.get_context(
                autonomous_mode=True,
                autonomous_mode_continue=False,
                rag_enabled=self.rag_enabled,
                turbo_enabled=self.turbo_enabled
            )

            # Add initial message to conversation
            self.context_builder.add_user_message(initial_message)

            # Set up message capture from VSCodeCommunicator
            self._setup_message_capture()

            current_message = initial_message

            # Main autonomous loop
            while self.autonomous_active and iteration < max_iterations:
                iteration += 1

                # Clear pending tool messages and iteration tracking
                self.pending_tool_messages = []
                self.current_iteration_actions = []

                # Send message to agent
                logger.info(f"Sending message to agent: {current_message[:100]}...")
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

                # Break if all tasks are complete (mirrors VSCode autonomous-execution.ts line 221-223)
                if loop_complete:
                    break

                # Log iteration summary
                self._log_iteration_summary(iteration, max_iterations)

                # Update context for next iteration
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

    async def run_interactive(self, initial_message: str):
        """
        Run agent in interactive mode (REPL-style).

        Args:
            initial_message: Initial user message
        """
        logger.info("Starting interactive mode")

        # Build context
        context = self.context_builder.get_context(
            autonomous_mode=False,
            rag_enabled=self.rag_enabled,
            turbo_enabled=self.turbo_enabled
        )

        # Add to conversation
        self.context_builder.add_user_message(initial_message)

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
        Set up message capture from VSCodeCommunicator.

        Intercepts messages that would normally go to VSCode stdout
        and captures them for processing.
        """
        # Create a custom message handler that captures tool outputs
        original_send = self.agent.vscode.send_tool_result

        async def capture_tool_result(result, context):
            """Capture tool results instead of sending to stdout."""
            # Store the tool result for processing (including workflow data for task list updates)
            self.pending_tool_messages.append({
                'type': result.output_type.value,
                'data': result.output_ui,
                'workflow': result.output_workflow  # Critical for task list updates!
            })

            # Don't call original - we don't want JSON printed to stdout in Jupyter mode

        # Monkey patch the send method to capture tool results
        self.agent.vscode.send_tool_result = capture_tool_result

        # Also capture workflow results (for LOOP_COMPLETE detection)
        original_send_workflow = self.agent.vscode.send_workflow_result

        async def capture_workflow_result(field, state):
            """Capture workflow results (e.g., LOOP_COMPLETE)."""
            self.pending_tool_messages.append({
                'type': 'workflow_result',
                'data': {field: state},
                'workflow': {}
            })
            # Don't call original - already suppressed by suppress_vscode_messages=True

        self.agent.vscode.send_workflow_result = capture_workflow_result

        # Note: We don't need to suppress console messages here
        # because suppress_vscode_messages=True in KaiAgent already sets _disabled=True
        # which prevents all VSCode communication methods from outputting to stdout

    def _log_iteration_summary(self, iteration: int, max_iterations: int):
        """
        Log high-level summary of what happened in this iteration.

        This is shown in quiet mode instead of detailed action logs.
        """
        if not self.current_iteration_actions:
            logger.info(f"[ITERATION {iteration}/{max_iterations}] No actions taken")
            return

        # Summarize actions
        for action in self.current_iteration_actions:
            action_type = action['type']

            if action_type == 'add_cell':
                cell_type = action.get('cell_type', 'code')
                code_preview = action.get('code', '')[:150].replace('\n', ' ')
                logger.info(f"[ITERATION {iteration}/{max_iterations}] Added {cell_type} cell: {code_preview}...")

            elif action_type == 'replace_cell':
                cell_index = action.get('cell_index', '?')
                code_preview = action.get('code', '')[:150].replace('\n', ' ')
                logger.info(f"[ITERATION {iteration}/{max_iterations}] Replaced cell {cell_index}: {code_preview}...")

            elif action_type == 'delete_cell':
                cell_index = action.get('cell_index', '?')
                logger.info(f"[ITERATION {iteration}/{max_iterations}] Deleted cell {cell_index}")

            elif action_type == 'execute':
                cell_index = action.get('cell_index', '?')
                success = action.get('success', False)
                status = 'SUCCESS' if success else 'FAILED'
                logger.info(f"[ITERATION {iteration}/{max_iterations}] Executed cell {cell_index} - {status}")

    async def _process_tool_messages(self) -> bool:
        """
        Process captured tool messages.

        Handles execute_code, display, task_list_display, workflow_result, etc.
        Mirrors VSCode's handleAutonomousCodeExecution functionality.

        Returns:
            True if LOOP_COMPLETE signal was received, False otherwise
        """
        loop_complete = False

        for msg in self.pending_tool_messages:
            msg_type = msg['type']
            data = msg['data']
            workflow = msg.get('workflow', {})

            # Update task list from workflow data (critical for task progression!)
            # Tools like AutonomousMarkCompletionTool return updated task_list in output_workflow
            if workflow and 'task_list' in workflow:
                self.context_builder.task_list = workflow['task_list']
                logger.debug(f"Updated task list from workflow: {workflow['task_list']}")

            if msg_type == 'execute_code':
                await self._handle_execute_code(data)

            elif msg_type == 'display':
                # Display messages (logging only in Jupyter interface)
                logger.info(f"Agent message: {data.get('text', data)}")

            elif msg_type == 'task_list_display':
                # Task list updates (suppress in quiet mode - too verbose)
                # Only log if DEBUG level is enabled
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
        should_replace = code_response.get('should_replace_code') == "true"
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
                result = await self.agent.workflow_orchestrator._handle_execution_progress_check(
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
                logger.info(f"Replaced code in cell {cell_index}")
            else:
                # Cell type mismatch: replace entire cell (VSCode behavior)
                logger.info(f"Cell {cell_index} is {current_cell.cell_type}, converting to code")
                new_cell = new_code_cell(source=code)
                self.notebook.cells[cell_index] = new_cell
                logger.info(f"Replaced cell {cell_index} with code cell")

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
                logger.info(f"Replaced markdown content in cell {cell_index}")
            else:
                # Cell type mismatch: replace entire cell (VSCode behavior)
                logger.info(f"Cell {cell_index} is {current_cell.cell_type}, converting to markdown")
                new_cell = new_markdown_cell(source=content)
                self.notebook.cells[cell_index] = new_cell
                logger.info(f"Replaced cell {cell_index} with markdown cell")

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
