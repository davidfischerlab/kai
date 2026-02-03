"""Learning explanation tool for explaining analysis steps in learning mode.

This module contains the LearningExplanationTool, an UnstructuredPromptTool
that generates educational explanations AFTER each successful cell execution.

The explanation runs in a separate learning graph (not the main execution graph)
and has access to the execution results. It explains the JUST-COMPLETED task,
not the upcoming task.
"""

import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger
from kai.config.paths import BIOINFORMATICS_CACHE_DIR
from .base import ToolResult, ToolOutputType
from .prompt_base import UnstructuredPromptTool

if TYPE_CHECKING:
    from kai.core.llm_interface import LLMInterface
    from kai.core.orchestration.state import KaiState

logger = setup_logger(__name__)


class LearningExplanationTool(UnstructuredPromptTool):
    """Tool for generating learning explanations after cell execution.

    This tool runs in a separate learning graph AFTER the main execution graph
    completes and cell execution succeeds. It explains the just-completed task
    with access to execution results.

    **UI Returns:**
    - `output_ui`: Dict with explanation text and learning mode flag
    - `output_type`: DISPLAY_ONLY - shown in chat as educational content

    **Workflow Returns:**
    - None (no workflow state changes needed - runs in separate graph)

    **Used by workflows:** Learning mode after each successful cell execution
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__(
            "learning_explanation",
            PromptScenario.LEARNING_EXPLANATION,
            llm_interface
        )
        self._cache_dir = BIOINFORMATICS_CACHE_DIR

    def _extract_task_info(
        self, state: 'KaiState'
    ) -> Tuple[Optional[str], Optional[str], Optional[Dict[str, Any]]]:
        """Extract task title, step number, and reference from the just-executed task.

        The explanation runs AFTER code execution in a separate learning graph.
        The task that was just executed is the ACTIVE task (not yet marked completed,
        since autonomous_mark_completion runs at the start of the NEXT iteration).

        Returns:
            Tuple of (step_number, task_description, reference_info)
            where reference_info is a dict with notebook_id, cells, and source_path
        """
        task_list = state.get("task_list", {})
        tasks = task_list.get("tasks", [])

        # Find the currently ACTIVE task - this is the one that was just executed
        # Task completion happens at the START of the next iteration, so after
        # execution the task is still marked as "active"
        target_task = None
        step_number = None
        for i, task in enumerate(tasks):
            if task.get("status") == "active":
                target_task = task
                step_number = str(i + 1)  # 1-indexed
                break

        if not target_task:
            # Fallback: if no active task, find the most recently completed task
            # This handles edge cases where the task was already marked completed
            for i in range(len(tasks) - 1, -1, -1):
                task = tasks[i]
                if task.get("status") == "completed":
                    target_task = task
                    step_number = str(i + 1)  # 1-indexed
                    break

        if not target_task:
            return None, None, None

        task_text = target_task.get("task", "")

        # Extract reference: [adapted from: 'notebook_id', cells: ...]
        # Pattern handles both: cells: [3,4,5] (bracketed) and cells: 5 (single)
        reference_pattern = r"\[adapted from: '([^']+)', cells: \[?([^\]]+?)\]?\]"
        match = re.search(reference_pattern, task_text)

        if match:
            notebook_id = match.group(1)
            cells = match.group(2)
            # Remove reference from task description
            task_description = re.sub(reference_pattern, "", task_text).strip()

            # Try to find the source_path from reference_workflow_content
            source_path = self._find_source_path(notebook_id, state)

            reference_info = {
                "notebook_id": notebook_id,
                "cells": cells,
                "source_path": source_path
            }
        else:
            reference_info = None
            task_description = task_text

        return step_number, task_description, reference_info

    def _find_source_path(self, notebook_id: str, state: 'KaiState') -> Optional[str]:
        """Find the source_path for a notebook.

        Tries to construct the path from the notebook_id and cache directory.

        Args:
            notebook_id: The full notebook ID (e.g., 'teichlab/celltypist/celltypist_tutorial.ipynb')
            state: Current workflow state

        Returns:
            Absolute path to the notebook file, or None if not found
        """
        # Try to find notebook file directly from notebook_id
        # notebook_id format: 'repo/path/to/notebook.ipynb'
        if notebook_id:
            candidate_path = Path(self._cache_dir) / notebook_id
            if candidate_path.exists():
                return str(candidate_path)

        return None

    async def _process_response(
        self, response: str, state: 'KaiState'
    ) -> ToolResult:
        """Process learning explanation response."""
        # Extract task info for the title
        step_number, task_description, reference_info = self._extract_task_info(state)

        # Build the title
        if step_number and task_description:
            title = f"**Step {step_number}. {task_description}**"
            if reference_info:
                # Format reference line for display
                notebook_id = reference_info["notebook_id"]
                cells = reference_info["cells"]
                title += f"\n*[adapted from: '{notebook_id}', cells: [{cells}]]*"
        else:
            # Fallback if no task info available
            title = "**Step Explanation**"

        explanation_text = f"{title}\n\n{response}"

        # Build output_ui with reference info for clickable link
        output_ui: Dict[str, Any] = {
            "text": explanation_text,
            "isLearningExplanation": True,  # Signal to UI for learning pause
        }

        # Include reference info for UI to make notebook link clickable
        if reference_info:
            output_ui["referenceNotebook"] = {
                "notebookId": reference_info["notebook_id"],
                "cells": reference_info["cells"],
                "sourcePath": reference_info["source_path"]  # May be None
            }

        return ToolResult(
            output_ui=output_ui,
            output_type=ToolOutputType.DISPLAY_ONLY,
            # No workflow output needed - this runs in a separate learning graph
            # and doesn't need to prevent re-running (orchestrator handles that)
            output_workflow=None
        )
