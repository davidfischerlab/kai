"""
Jupyter interface for Kai - enables running Kai without VSCode.

This module provides a Python-native interface to run Kai's autonomous agent
directly with Jupyter notebooks, without requiring an active VSCode window.

Main components:
- JupyterInterface: Main entry point for autonomous/interactive modes
- NotebookExecutor: Kernel management and cell execution
- ContextBuilder: Builds agent context from notebook state

Note: This package is installed as UI.jupyter (not kai.ui.jupyter) due to
the directory structure. Use: from UI.jupyter import JupyterInterface
"""

from .jupyter_interface import JupyterInterface
