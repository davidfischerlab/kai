"""
Notebook execution engine for Kai Jupyter interface.

Handles kernel management, cell execution, and output capture using jupyter_client.
Mirrors the execution behavior of VSCode's NotebookOperations class.
"""

import time
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from jupyter_client import KernelManager
from kai.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExecutionResult:
    """Result of cell execution."""
    success: bool
    outputs: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)  # [(output_type, output_dict), ...]
    error: Optional[Dict[str, Any]] = None
    terminated: bool = False  # Set to True if execution was interrupted by monitoring
    duration: float = 0.0  # Execution duration in seconds


class NotebookExecutor:
    """
    Manages Jupyter kernel and executes notebook cells.

    Provides the same functionality as VSCode's NotebookOperations:
    - Executes cells and waits for completion
    - Captures all output types (stdout, stderr, errors, display_data)
    - Supports long-running cell monitoring with interruption
    - Handles kernel restart and re-execution

    Attributes:
        km: KernelManager instance
        kc: KernelClient for communication
        kernel_name: Name of kernel to use (default: 'python3')
    """

    def __init__(self, notebook_python: str):
        """
        Initialize notebook executor with kernel.

        Args:
            notebook_python: REQUIRED. Path to Python executable OR conda/mamba environment.
                            Examples:
                            - '/Users/user/mamba/envs/myenv/bin/python' (direct Python path)
                            - '/Users/user/mamba/envs/myenv' (conda env path, will append /bin/python)
        """
        if notebook_python is None:
            raise ValueError(
                "notebook_python is required. Provide either:\n"
                "  - Python executable path: /path/to/env/bin/python\n"
                "  - Conda/mamba environment: /path/to/env (will use /path/to/env/bin/python)"
            )

        # Resolve Python executable from environment path or direct path
        python_path = Path(notebook_python)
        if python_path.is_dir():
            # It's a conda/mamba environment directory
            python_executable = python_path / 'bin' / 'python'
            if not python_executable.exists():
                raise ValueError(
                    f"Python not found at {python_executable}. "
                    f"Is {python_path} a valid conda/mamba environment?"
                )
        else:
            # It's a direct Python executable path
            python_executable = python_path
            if not python_executable.exists():
                raise ValueError(f"Python executable not found: {python_executable}")

        logger.info(f"Using Python for notebook kernel: {python_executable}")

        # Store Python path for kernel environment
        self.python_executable = python_executable

        # Always use python3 kernel spec
        self.km = KernelManager(kernel_name='python3')

        # Prepare environment with correct Python in PATH
        # This ensures the kernel uses the specified Python interpreter
        import os
        env = os.environ.copy()
        python_dir = str(python_executable.parent)
        env['PATH'] = f"{python_dir}:{env.get('PATH', '')}"

        # Start kernel with custom environment (pass env to start_kernel, not kernel_spec)
        self.km.start_kernel(env=env)
        self.kc = self.km.client()
        self.kc.start_channels()

        # Wait for kernel to be ready
        self.kc.wait_for_ready(timeout=60)
        logger.info(f"Kernel started and ready")

    def execute_cell(
        self,
        code: str,
        timeout: int = 1800,
        progress_check_callback: Optional[callable] = None
    ) -> ExecutionResult:
        """
        Execute code in kernel and wait for completion.

        Mirrors VSCode's executeCell behavior:
        - Executes code and waits for completion (up to timeout)
        - Captures all outputs (stdout, stderr, errors, display_data)
        - Supports progress monitoring for long-running cells
        - Can interrupt execution if monitoring callback returns False

        Args:
            code: Python code to execute
            timeout: Maximum execution time in seconds (default: 1800 = 30 minutes)
            progress_check_callback: Optional function(elapsed_time, partial_outputs) -> bool
                                   Called every 5 minutes to check if execution should continue

        Returns:
            ExecutionResult with success status, outputs, and errors
        """
        logger.info(f"Executing cell (timeout={timeout}s)")

        # Execute code in kernel
        msg_id = self.kc.execute(code, silent=False, store_history=True)

        outputs = []
        error = None
        start_time = time.time()
        last_progress_check = 0
        progress_check_interval = 300  # 5 minutes

        # Poll for messages until execution completes
        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > timeout:
                logger.warning(f"Cell execution timeout after {timeout}s")
                self.km.interrupt_kernel()
                return ExecutionResult(
                    success=False,
                    outputs=outputs,
                    error={'ename': 'TimeoutError', 'evalue': f'Execution exceeded {timeout}s'},
                    terminated=True,
                    duration=elapsed
                )

            # Progress monitoring (every 5 minutes for long-running cells)
            if progress_check_callback and elapsed - last_progress_check >= progress_check_interval:
                last_progress_check = elapsed

                # Build partial outputs string
                partial_outputs_str = self._format_outputs(outputs)

                # Check if execution should continue
                should_continue = progress_check_callback(elapsed, partial_outputs_str)

                if not should_continue:
                    logger.info(f"Execution monitoring requested termination after {elapsed:.1f}s")
                    self.km.interrupt_kernel()
                    return ExecutionResult(
                        success=False,
                        outputs=outputs,
                        error={'ename': 'MonitorTermination', 'evalue': 'Terminated by execution monitor'},
                        terminated=True,
                        duration=elapsed
                    )

            # Get message from kernel (with 1s timeout to check for progress/timeout)
            try:
                msg = self.kc.get_iopub_msg(timeout=1)
            except queue.Empty:
                continue

            # Only process messages from this execution
            if msg['parent_header'].get('msg_id') != msg_id:
                continue

            msg_type = msg['msg_type']
            content = msg.get('content', {})

            # Process different message types
            if msg_type == 'stream':
                # stdout or stderr
                stream_name = content.get('name', 'stdout')
                text = content.get('text', '')
                outputs.append(('stream', {'name': stream_name, 'text': text}))

            elif msg_type == 'display_data':
                # Display outputs (plots, dataframes, images, HTML, etc.)
                data = content.get('data', {})
                metadata = content.get('metadata', {})
                outputs.append(('display_data', {'data': data, 'metadata': metadata}))

            elif msg_type == 'execute_result':
                # Execution result (e.g., return values)
                data = content.get('data', {})
                metadata = content.get('metadata', {})
                execution_count = content.get('execution_count', None)
                outputs.append(('execute_result', {
                    'data': data,
                    'metadata': metadata,
                    'execution_count': execution_count
                }))

            elif msg_type == 'error':
                # Execution error
                error = {
                    'ename': content.get('ename', 'Error'),
                    'evalue': content.get('evalue', 'Unknown error'),
                    'traceback': content.get('traceback', [])
                }

            elif msg_type == 'status':
                # Check if execution is complete
                execution_state = content.get('execution_state')
                if execution_state == 'idle':
                    # Execution complete
                    duration = time.time() - start_time
                    success = error is None

                    logger.info(f"Cell execution {'succeeded' if success else 'failed'} in {duration:.2f}s")

                    return ExecutionResult(
                        success=success,
                        outputs=outputs,
                        error=error,
                        terminated=False,
                        duration=duration
                    )

    def restart_kernel(self):
        """
        Restart the Jupyter kernel.

        Implements the REPLACE_AND_RESTART error recovery strategy.
        """
        logger.info("Restarting kernel...")
        self.km.restart_kernel()
        self.kc.wait_for_ready(timeout=60)
        logger.info("Kernel restarted and ready")

    def shutdown(self):
        """Shutdown kernel and cleanup resources."""
        logger.info("Shutting down kernel...")
        try:
            # Stop channels first
            if hasattr(self, 'kc') and self.kc:
                self.kc.stop_channels()
                # Clean up client reference
                del self.kc

            # Shutdown kernel
            if hasattr(self, 'km') and self.km:
                self.km.shutdown_kernel(now=True)
                # Clean up manager reference
                del self.km

            # Give system time to close ZMQ sockets and file descriptors
            import time
            time.sleep(1.0)

            logger.info("Kernel shutdown complete")
        except Exception as e:
            logger.warning(f"Error during kernel shutdown: {e}")

    def _format_outputs(self, outputs: List[Tuple[str, str]]) -> str:
        """
        Format outputs for display or monitoring.

        Args:
            outputs: List of (output_type, content) tuples

        Returns:
            Formatted string representation
        """
        formatted = []
        for output_type, content in outputs:
            if output_type == 'stdout':
                formatted.append(f"[stdout] {content}")
            elif output_type == 'stderr':
                formatted.append(f"[stderr] {content}")
            elif output_type in ['image/png', 'image/jpeg']:
                formatted.append(f"[{output_type}]")
            else:
                formatted.append(content)

        return '\n'.join(formatted)

    def __enter__(self):
        """Context manager support."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup."""
        self.shutdown()
