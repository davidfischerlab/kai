"""
Context builder for Kai Jupyter interface.

Constructs the agent context dictionary that matches the VSCode extension format.
This ensures the core agent code receives identical input regardless of interface.
"""

import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional
import nbformat
from kai.utils import setup_logger

logger = setup_logger(__name__)


class ContextBuilder:
    """
    Builds agent context from notebook state.

    Mirrors the context construction in VSCode's ChatCore class to ensure
    the agent receives identical context format regardless of interface.

    Tracks:
    - Execution history: Recent cell executions with outputs
    - Conversation history: User messages and agent responses
    - Notebook structure: All cells with content and metadata
    - Current state: Active cell, last execution results, errors
    """

    def __init__(self, notebook: nbformat.NotebookNode, notebook_uri: str):
        """
        Initialize context builder.

        Args:
            notebook: nbformat NotebookNode object
            notebook_uri: Path to notebook file (for session identification)
        """
        self.notebook = notebook
        self.notebook_uri = notebook_uri

        # Tracking state
        self.execution_history: List[str] = []  # Formatted cell execution strings
        self.conversation_history: List[Dict[str, str]] = []  # [{'role': 'user'|'assistant', 'content': '...'}]
        self.modification_history: List[Dict[str, Any]] = []  # Cell modifications (created/modified/deleted/moved)
        self.current_cell_index: int = -1
        self.last_execution_failed: bool = False
        self.error_cell_index: int = -1
        self.last_execution_output: str = ""
        self.last_cell_modified_in_auto_mode: int = -1
        self.task_list: Optional[Dict[str, Any]] = None  # Task list for autonomous mode
        self.just_executed: bool = False  # Set to True after cell execution, cleared after get_context()

        # Cell content hash cache for change detection
        self._cell_content_hashes: Dict[int, str] = {}

        # Constants
        self.MAX_EXECUTION_HISTORY = 5
        self.MAX_MODIFICATION_HISTORY = 5

    def get_context(
        self,
        autonomous_mode: bool = False,
        autonomous_mode_continue: bool = False,
        rag_enabled: bool = True,
        turbo_enabled: bool = False
    ) -> Dict[str, Any]:
        """
        Build complete context dictionary for agent.

        Matches the format from VSCode's ChatCore._getContextForMessage().

        Args:
            autonomous_mode: Whether autonomous mode is active
            autonomous_mode_continue: Whether this is a continue iteration (no user message)
            rag_enabled: Whether RAG is enabled
            turbo_enabled: Whether turbo mode is enabled

        Returns:
            Context dictionary with all fields expected by KaiAgent
        """
        context = {
            # Request metadata
            'request_id': self._generate_request_id(),

            # Execution context
            'executionHistory': self.execution_history.copy(),
            'conversationHistory': self.conversation_history.copy(),
            'modificationHistory': self.modification_history.copy(),
            'notebookStructure': self._get_notebook_structure(),

            # Current state
            'currentCell': self._get_current_cell_content(),
            'currentCellIndex': self.current_cell_index,

            # Error information
            'errorCellIndex': self.error_cell_index,
            'executionResult': self.last_execution_output,
            'lastExecutionFailed': self.last_execution_failed,
            'justExecuted': self.just_executed,  # True immediately after execution

            # Autonomous mode flags
            'autonomousMode': autonomous_mode,
            'autonomousModeContinue': autonomous_mode_continue,
            'autonomousModeTermination': False,  # Set to True to stop autonomous mode
            'lastCellModifiedInAutoMode': self.last_cell_modified_in_auto_mode,
            'confirmPlan': False,  # Jupyter: don't pause after planning, continue directly to execution

            # Backend details
            'turboEnabled': turbo_enabled,
            'ragEnabled': rag_enabled,

            # Notebook URI for session tracking
            'notebookUri': self.notebook_uri
        }

        # Add task list if it exists (for autonomous mode)
        if self.task_list is not None:
            context['taskList'] = self.task_list

        # Clear just_executed AFTER including it in context
        # This ensures it's only True for ONE get_context() call
        # The orchestrator will see it once and run completion analysis
        self.just_executed = False

        return context

    def add_to_execution_history(self, cell_index: int, execution_result: 'ExecutionResult', code: str):
        """
        Add cell execution to history.

        Args:
            cell_index: Index of executed cell
            execution_result: ExecutionResult from NotebookExecutor
            code: Code that was executed
        """
        # Format similar to VSCode's formatCellToString
        timestamp = datetime.now().strftime('%H:%M:%S')
        status = 'SUCCESS' if execution_result.success else 'FAILED'
        duration = f'{execution_result.duration:.3f}s'

        entry = f"> CELL at index {cell_index}: {status}\n"
        entry += f"Executed at {timestamp}, took {duration}\n"
        entry += f">>Content of cell at index {cell_index}:\n{code.strip()}\n"

        # Add outputs if available
        if execution_result.outputs:
            output_str = self._format_outputs(execution_result.outputs)
            entry += f">> Outputs of cell at index {cell_index}:\n{output_str}"

        # Add error if present
        if execution_result.error:
            error_str = self._format_error(execution_result.error)
            entry += f"\n>> Error output:\n{error_str}"

        # Add to history (most recent first)
        self.execution_history.insert(0, entry)

        # Trim to max size
        while len(self.execution_history) > self.MAX_EXECUTION_HISTORY:
            self.execution_history.pop()

    def add_user_message(self, message: str):
        """Add user message to conversation history."""
        self.conversation_history.append({
            'role': 'user',
            'content': message
        })

    def add_assistant_message(self, message: str):
        """Add assistant message to conversation history."""
        self.conversation_history.append({
            'role': 'assistant',
            'content': message
        })

    def update_execution_state(
        self,
        success: bool,
        output: str,
        cell_index: Optional[int] = None
    ):
        """
        Update execution state after cell execution.

        Args:
            success: Whether execution succeeded
            output: Formatted output string
            cell_index: Index of executed cell (None if no error)
        """
        self.last_execution_failed = not success
        self.last_execution_output = output
        self.just_executed = True  # Set flag so orchestrator knows to analyze completion

        if not success and cell_index is not None:
            self.error_cell_index = cell_index
        else:
            self.error_cell_index = -1

    def _get_notebook_structure(self) -> Dict[str, Any]:
        """
        Get complete notebook structure.

        Returns:
            Dictionary with totalCells and allCells (formatted strings)
        """
        all_cells = []

        for idx, cell in enumerate(self.notebook.cells):
            cell_str = self._format_cell(idx, cell)
            all_cells.append(cell_str)

        return {
            'totalCells': len(self.notebook.cells),
            'allCells': all_cells
        }

    def _get_current_cell_content(self) -> Optional[str]:
        """Get content of current cell."""
        if 0 <= self.current_cell_index < len(self.notebook.cells):
            return self.notebook.cells[self.current_cell_index].source
        return None

    def _format_cell(self, index: int, cell: nbformat.NotebookNode) -> str:
        """
        Format cell for inclusion in notebook structure.

        Args:
            index: Cell index
            cell: NotebookNode cell

        Returns:
            Formatted string representation
        """
        if cell.cell_type == 'markdown':
            return f"> MARKDOWN CELL at index {index}\n>>Content:\n{cell.source.strip()}"

        # Code cell
        content = cell.source.strip()

        # Check if cell has outputs (from previous execution)
        if hasattr(cell, 'outputs') and cell.outputs:
            # Format outputs
            outputs_str = self._format_cell_outputs(cell.outputs)
            return f"> CODE CELL at index {index}\n>>Code:\n{content}\n>>Outputs:\n{outputs_str}"

        return f"> CODE CELL at index {index}\n>>Code:\n{content}"

    def _format_cell_outputs(self, outputs: List) -> str:
        """Format cell outputs from nbformat cell."""
        formatted = []

        for output in outputs:
            output_type = output.output_type

            if output_type == 'stream':
                # stdout or stderr
                stream_name = output.get('name', 'stdout')
                text = output.get('text', '')
                if isinstance(text, list):
                    text = ''.join(text)
                formatted.append(f">>> {stream_name.capitalize()} output\n{text}")

            elif output_type in ['execute_result', 'display_data']:
                # Display data (text, plots, etc.)
                data = output.get('data', {})

                if 'text/plain' in data:
                    text = data['text/plain']
                    if isinstance(text, list):
                        text = ''.join(text)
                    formatted.append(f">>> Text output\n{text}")

                if 'image/png' in data or 'image/jpeg' in data:
                    formatted.append(">>> Plot output\n[Image output]")

            elif output_type == 'error':
                # Error output
                error = {
                    'ename': output.get('ename', 'Error'),
                    'evalue': output.get('evalue', 'Unknown error'),
                    'traceback': output.get('traceback', [])
                }
                error_str = self._format_error(error)
                formatted.append(f">>> Error output\n{error_str}")

        return '\n\n'.join(formatted) if formatted else '[No outputs]'

    def _format_outputs(self, outputs: List[tuple]) -> str:
        """
        Format execution outputs for Kai context.

        Args:
            outputs: List of (output_type, output_data) tuples from ExecutionResult

        Returns:
            Formatted string for agent context
        """
        formatted = []
        output_counts = {'text': 0, 'error': 0, 'plot': 0}

        # Count outputs by type
        for output_type, output_data in outputs:
            if output_type == 'stream':
                output_counts['text'] += 1
            elif output_type in ['execute_result', 'display_data']:
                # Check what's in the data dict
                if isinstance(output_data, dict) and 'data' in output_data:
                    data = output_data['data']
                    if 'text/plain' in data or 'text/html' in data:
                        output_counts['text'] += 1
                    if 'image/png' in data or 'image/jpeg' in data:
                        output_counts['plot'] += 1

        total_outputs = sum(output_counts.values())
        current_num = 0

        for output_type, output_data in outputs:
            if output_type == 'stream':
                # Stream output (stdout/stderr)
                if isinstance(output_data, dict):
                    stream_name = output_data.get('name', 'stdout')
                    text = output_data.get('text', '')
                    if stream_name == 'stdout':
                        current_num += 1
                        formatted.append(f">>> Text output {current_num}/{total_outputs}\n{text}")
                    elif stream_name == 'stderr':
                        formatted.append(f">>> Warning output\n{text}")

            elif output_type in ['execute_result', 'display_data']:
                # Display/result data
                if isinstance(output_data, dict) and 'data' in output_data:
                    data = output_data['data']

                    # Text output
                    if 'text/plain' in data:
                        current_num += 1
                        formatted.append(f">>> Text output {current_num}/{total_outputs}\n{data['text/plain']}")

                    # HTML output (e.g., DataFrames)
                    elif 'text/html' in data:
                        current_num += 1
                        formatted.append(f">>> HTML output {current_num}/{total_outputs}\n{data['text/html']}")

                    # Image output
                    if 'image/png' in data or 'image/jpeg' in data:
                        current_num += 1
                        formatted.append(f">>> Plot output {current_num}/{total_outputs}\n[Image output]")

        return '\n\n'.join(formatted) if formatted else '[No outputs]'

    def _format_error(self, error: Dict[str, Any]) -> str:
        """
        Format error output.

        Args:
            error: Error dictionary with ename, evalue, traceback

        Returns:
            Formatted error string
        """
        ename = error.get('ename', 'Error')
        evalue = error.get('evalue', 'Unknown error')
        traceback = error.get('traceback', [])

        result = f"{ename}: {evalue}"

        if traceback:
            # Join traceback lines (may contain ANSI codes)
            if isinstance(traceback, list):
                traceback_str = '\n'.join(traceback)
            else:
                traceback_str = str(traceback)

            # Remove ANSI escape codes
            import re
            traceback_str = re.sub(r'\x1b\[[0-9;]*m', '', traceback_str)

            result += f"\n{traceback_str}"

        return result

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:8]

    def _hash_content(self, content: str) -> str:
        """Generate hash of cell content for change detection."""
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def track_modification(
        self,
        cell_index: int,
        modification_type: str,
        content: str,
        old_content: Optional[str] = None
    ):
        """
        Track cell modification in history.

        Mirrors VSCode's NotebookOperations.trackStructuralChange().

        Args:
            cell_index: Index of modified cell
            modification_type: 'created' | 'modified' | 'deleted' | 'moved'
            content: New/current cell content
            old_content: Previous content (for 'modified' type)
        """
        timestamp = datetime.now().strftime('%H:%M:%S')

        # Generate content hashes
        new_hash = self._hash_content(content) if content else None
        old_hash = self._hash_content(old_content) if old_content else None

        # Create modification entry
        entry = {
            'cell_index': cell_index,
            'timestamp': timestamp,
            'modification_type': modification_type,
            'content_preview': content[:50].replace('\n', ' ') if content else ''
        }

        # Add hashes if applicable
        if modification_type == 'modified':
            entry['old_content_hash'] = old_hash
            entry['new_content_hash'] = new_hash
        elif modification_type == 'created':
            entry['new_content_hash'] = new_hash
        elif modification_type == 'deleted':
            entry['old_content_hash'] = old_hash

        # Add to history (most recent first)
        self.modification_history.insert(0, entry)

        # Update cache
        if new_hash and modification_type in ['created', 'modified']:
            self._cell_content_hashes[cell_index] = new_hash
        elif modification_type == 'deleted' and cell_index in self._cell_content_hashes:
            del self._cell_content_hashes[cell_index]

        # Trim to max size
        if len(self.modification_history) > self.MAX_MODIFICATION_HISTORY:
            self.modification_history = self.modification_history[:self.MAX_MODIFICATION_HISTORY]
