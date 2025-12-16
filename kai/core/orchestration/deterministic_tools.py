"""Deterministic tools that don't require LLM calls."""

import json
import sys
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
from kai.core.orchestration.execution_context import ExecutionContext
from kai.utils import setup_logger

from .base_tool import BaseTool, ToolResult, ToolOutputType
from kai.retrieval import ChromaDbManager

if TYPE_CHECKING:
    from .execution_context import ExecutionContext
    from kai.retrieval.knowledge_base import KnowledgeBase

logger = setup_logger(__name__)


class CodeRetrievalTool(BaseTool):
    """Tool for retrieving relevant documentation via RAG."""
    knowledge_base: 'ChromaDbManager'
    
    def __init__(self, knowledge_base: 'ChromaDbManager' = None):
        super().__init__("rag_retrieval")
        self.knowledge_base = knowledge_base
    
    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """Retrieve relevant knowledge via RAG."""
        # Retrieval query is a list of query strings
        queries = exec_context.inputs.context.get("snippet_retrieval_query")

        # If no queries specified, return empty (RAG disabled or not needed)
        if not queries:
            logger.debug("[RAG] No retrieval queries specified, skipping RAG")
            return ToolResult(
                output_ui=None,
                output_type=ToolOutputType.NO_OUTPUT,
                output_workflow={"rag_retrieval": ""}  # Use rag_retrieval field so router knows RAG ran
            )

        # Log the queries being searched
        if isinstance(queries, list):
            logger.info(f"[RAG] Searching with {len(queries)} queries: {queries[0][:100]}..." if queries else "[RAG] Empty query list")
        else:
            logger.info(f"[RAG] Searching with query: {str(queries)[:100]}...")

        try:
            # Wait for background initialization to complete for optimal performance
            # This ensures collection embeddings are cached before the search
            if hasattr(self.knowledge_base, 'wait_for_background_initialization'):
                await self.knowledge_base.wait_for_background_initialization(timeout=5.0)

            # Perform RAG retrieval
            results = await self.knowledge_base.search(queries, n_results=10)

            if results:
                # Log which notebooks/collections were retrieved
                metadata = results.get("metadata", [])
                if metadata:
                    sources = set()
                    for item in metadata:
                        # Extract notebook/collection identifiers
                        if isinstance(item, dict):
                            source = item.get("source") or item.get("collection") or item.get("notebook_id")
                            if source:
                                sources.add(source)
                    if sources:
                        logger.info(f"[RAG] Retrieved {len(metadata)} snippets from {len(sources)} sources: {', '.join(sorted(sources)[:5])}")
                    else:
                        logger.info(f"[RAG] Retrieved {len(metadata)} snippets")
                else:
                    logger.info(f"[RAG] Retrieved snippets (no metadata)")

                # Log successful RAG results - pass the actual content string
                self._log_rag_query_if_enabled(queries, exec_context, "success", results["content"])

                return ToolResult(
                    output_workflow={"rag_retrieval": results["content"]},
                    output_ui={},  # TODO could output tool usage from result here in the future
                    output_type=ToolOutputType.NO_OUTPUT
                )
            else:
                # Log empty results
                self._log_rag_query_if_enabled(queries, exec_context, "no_results")
                
                return ToolResult(
                    output_workflow={},
                    output_ui={},
                    output_type=ToolOutputType.NO_OUTPUT
                )
                
        except Exception as e:
            self._log_rag_query_if_enabled(queries, exec_context, "error", error=e)
            return ToolResult(
                output_workflow={},
                output_ui={},
                output_type=ToolOutputType.NO_OUTPUT
            )
    
    def _log_rag_query_if_enabled(self, queries: List[str], exec_context: "ExecutionContext", status: str, results: Optional[str] = None, error: Optional[Exception] = None):
        """Log RAG query to debug folder if DEBUG_PROMPTS is enabled."""
        from kai.config.settings import settings
        from kai.config.paths import get_debug_prompts_dir
        from datetime import datetime
        
        if not settings.DEBUG_PROMPTS:
            return
            
        try:
            session_id = exec_context.session_metadata["session_id"]
            session_timestamp = exec_context.session_metadata["session_timestamp"]
            iteration_timestamp = exec_context.session_metadata["iteration_timestamp"]
            is_autonomous = exec_context.inputs.context["autonomous_mode"]
            notebook_uri = exec_context.session_metadata["notebook_uri"]
            # Concatenate queries to one string:
            if isinstance(queries, list):
                query = ", ".join(queries)
            else:
                query = str(queries)

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
            filename = f"{timestamp_str}_{self.name}.txt"
            filepath = debug_dir / filename
            
            # Write RAG query information to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"Tool: {self.name}\n")
                f.write(f"Type: Retrieval\n")
                f.write(f"Timestamp: {timestamp_str}\n")
                f.write(f"Notebook: {notebook_identifier}\n")
                f.write(f"Notebook URI: {notebook_uri or 'N/A'}\n")
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Session Init Time: {session_timestamp}\n")
                f.write(f"Iteration Time: {iteration_timestamp}\n")
                f.write(f"Autonomous Mode: {is_autonomous}\n")
                f.write("=" * 80 + "\n")
                f.write("Retrieval Query:\n")
                f.write("=" * 80 + "\n")
                f.write(query or "N/A")
                f.write("\n\n")
                
                if status == "success" and results:
                    f.write("=" * 80 + "\n")
                    f.write("RAG RESULTS:\n")
                    f.write("=" * 80 + "\n")
                    f.write(results)
                    f.write("\n\n")
                    
                elif status == "error" and error:
                    f.write("=" * 80 + "\n")
                    f.write("ERROR:\n")
                    f.write("=" * 80 + "\n")
                    f.write(str(error))
                    f.write("\n\n")
                    
                elif status == "no_results":
                    f.write("=" * 80 + "\n")
                    f.write("RESULT: No matching documents found\n")
                    f.write("=" * 80 + "\n")
                
                # Add knowledge base metadata
                if self.knowledge_base:
                    f.write("=" * 80 + "\n")
                    f.write("KNOWLEDGE BASE INFO:\n")
                    f.write("=" * 80 + "\n")
                    try:
                        # Try to get knowledge base info if available
                        if hasattr(self.knowledge_base, 'get_collections'):
                            collections = self.knowledge_base.get_collections()
                            f.write(f"Available Collections: {collections}\n")
                        if hasattr(self.knowledge_base, 'total_documents'):
                            total_docs = self.knowledge_base.total_documents()
                            f.write(f"Total Documents: {total_docs}\n")
                    except Exception as kb_error:
                        f.write(f"Could not retrieve KB info: {kb_error}\n")
                    
            logger.debug(f"Retrieval query logged to: {filepath}")
            
        except Exception as log_error:
            logger.error(f"Failed to log retrieval query: {log_error}")


class MarkNextTaskActiveTool(BaseTool):
    """Deterministic tool for marking the next pending task as active and sending UI update.

    **UI Returns:**
    - `output_type`: TASK_LIST_DISPLAY - shows updated task list with new active task

    **Workflow Returns:**
    - `task_list`: Updated task list with next task marked as active
    - `active_task_objective`: Description of the newly active task

    **Used by workflows:** Autonomous execution workflows to advance to the next task
    """

    def __init__(self):
        super().__init__("mark_next_task_active")

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """Mark the next pending task as active and send updated task list to UI."""
        if not exec_context.inputs.task_list:
            return ToolResult(
                output_ui="No task list available for task marking",
                output_type=ToolOutputType.RESPONSE,
                output_workflow={}
            )

        # Find the first non-completed task and mark it as active
        # Ignore cases in which all are completed / active
        active_task_description = ""
        active_task_dict = None

        i = 0
        for i, task in enumerate(exec_context.inputs.task_list['tasks']):
            if task.get('status') == 'active':
                # There is already an active task
                active_task_description = task.get('task', '')
                active_task_dict = task
                break
            if task.get('status') == 'pending':
                # Update the task status
                task['status'] = 'active'
                active_task_description = task.get('task', '')
                active_task_dict = task
                logger.info(f"Marking task {task.get('id')} as active: {active_task_description[:75]}")
                break
        if i < len(exec_context.inputs.task_list['tasks']) - 1:
            next_pending_task_objective = exec_context.inputs.task_list['tasks'][i + 1]["task"]
        else:
            next_pending_task_objective = ""
        is_reasoning_task = "[reasoning]" in active_task_description

        # Create task list display
        import json
        task_list_json = json.dumps(exec_context.inputs.task_list)
        vscode_response = {
            "text": task_list_json,
        }

        # Prepare workflow output
        workflow_output = {
            "task_list": exec_context.inputs.task_list,
            "active_task": active_task_dict,  # Full task dict (for router)
            "active_task_objective": active_task_description,  # String description (legacy)
            "is_reasoning_task": is_reasoning_task,
            "next_pending_task_objective": next_pending_task_objective,
            "next_task_activated": True  # For deterministic router phase tracking
        }

        return ToolResult(
            output_ui=vscode_response,
            output_type=ToolOutputType.TASK_LIST_DISPLAY,
            output_workflow=workflow_output
        )


class SetPositioningFromLastCellTool(BaseTool):
    """Set positioning_info from last_cell_modified_in_auto_mode.

    Matches kai_dev behavior: in standard continuation and error recovery,
    positioning is determined by the last modified cell, NOT the LLM.
    This ensures we add/replace at the correct position after cells have been inserted.

    Use cases:
    - Standard continue (success): Position at last modified cell to add after it
    - Standard retry (error): Position at last modified cell to replace it
    - NOT used for: First execution (no last_cell yet) or backtracking (indices changed)

    **UI Returns:**
    - `output_type`: NO_OUTPUT - internal positioning tool

    **Workflow Returns:**
    - `positioning_info`: Dict with target_cell from last_cell_modified_in_auto_mode
    """

    def __init__(self):
        super().__init__("set_positioning_from_last_cell")

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """Set positioning from last cell modified in auto mode."""
        last_cell = exec_context.inputs.context.get("last_cell_modified_in_auto_mode")

        if last_cell is None:
            # Fallback to error cell if available (for retry scenarios)
            error_cell = exec_context.inputs.context.get("error_cell_index", -1)
            if error_cell >= 0:
                last_cell = error_cell
                logger.info(f"[SET_POSITIONING] Using error_cell_index as fallback: {error_cell}")
            else:
                # This shouldn't happen in normal flow - log warning
                logger.warning("[SET_POSITIONING] No last_cell_modified_in_auto_mode or error_cell_index found")
                # Ultimate fallback - use last cell in notebook
                notebook_structure = exec_context.inputs.context.get("notebook_structure", {})
                total_cells = notebook_structure.get("totalCells", 0)
                last_cell = max(0, total_cells - 1)
                logger.info(f"[SET_POSITIONING] Using notebook last cell as fallback: {last_cell}")

        positioning_info = {"target_cell": last_cell}
        logger.info(f"[SET_POSITIONING] Set positioning to cell {last_cell}")

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={"positioning_info": positioning_info}
        )


class IncrementPositioningTool(BaseTool):
    """Increment positioning_info target_cell by 1.

    Used after adding a new cell (e.g., reasoning cell) so that subsequent
    operations (like critique regeneration) target the NEW cell, not the original.

    Matches kai_dev lines 497-498:
        exec_context.inputs.context["positioning_info"] = {
            "target_cell": exec_context.inputs.context["positioning_info"]["target_cell"] + 1
        }

    **UI Returns:**
    - `output_type`: NO_OUTPUT - internal positioning tool

    **Workflow Returns:**
    - `positioning_info`: Dict with target_cell incremented by 1
    """

    def __init__(self):
        super().__init__("increment_positioning")

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """Increment positioning target_cell by 1."""
        positioning_info = exec_context.inputs.context.get("positioning_info", {})
        current_target = positioning_info.get("target_cell", 0)
        new_target = current_target + 1

        new_positioning = {"target_cell": new_target}
        logger.info(
            f"[INCREMENT_POSITIONING] Incremented positioning "
            f"from {current_target} to {new_target}"
        )

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow={
                "positioning_info": new_positioning,
                "reasoning_positioning_incremented": True  # Prevent double-increment
            }
        )


class ReferenceWorkflowQueryPreparationTool(BaseTool):
    """Deterministic tool for preparing reference workflow queries.

    Extracts queries from execution context and performs semantic search
    to populate putative_reference_workflow_summaries for the LLM prompt.
    This replaces the _modify_user_query functionality in ReferenceWorkflowSelectionTool.

    Note: retrieval_queries can be reset each iteration - cumulalation of selected reference
    candidates happens here at the level of putative_reference_workflow_summaries.
    """

    def __init__(self, summary_search=None):
        super().__init__("reference_workflow_query_preparation")
        self.summary_search = summary_search

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """Prepare semantic search results for reference workflow selection."""
        # Extract queries from context
        has_specific_retrieval_query = (
            "retrieval_queries" in exec_context.inputs.context.keys() and
            len(exec_context.inputs.context.get("retrieval_queries", [])) > 0
        )

        if has_specific_retrieval_query:
            queries = exec_context.inputs.context["retrieval_queries"]
            # If semantic search is available, get top candidates
            candidates = []
            if self.summary_search:
                for q in queries:
                    # Top 10 hits for each query:
                    candidates.extend(
                        self.summary_search.search_summaries(q, n_results=10)
                    )

            existing_summaries = exec_context.inputs.context.get(
                "putative_reference_workflow_summaries", ""
            )
            # Build enhanced prompt with semantic candidates
            summary_list = []
            for result in candidates:
                notebook_id = result["notebook_id"]
                # Check that this notebook ID is not already in existing summaries to avoid duplicates:
                if notebook_id not in existing_summaries:
                    summary = result["summary"]
                    metadata = result["metadata"]
                    score = result["similarity_score"]
                    # Use full notebook ID if available, otherwise fall back to internal ID
                    display_id = metadata.get('full_notebook_id', notebook_id)

                    summary_list.append(
                        f"> Notebook ID: '{display_id}' (similarity: {score:.2f})\n"
                        f"Repository: {metadata.get('source_repository', 'unknown')}\n"
                        f">> Summary:\n{summary}\n\n"
                    )
            summary_str = "".join(summary_list)
            # Update execution context with preselection results
            output_workflow = {
                "putative_reference_workflow_summaries": existing_summaries + summary_str
            }
        else:
            output_workflow = {}

        return ToolResult(
            output_ui={},
            output_type=ToolOutputType.NO_OUTPUT,
            output_workflow=output_workflow
        )


class FilterUnusedReferenceWorkflowsTool(BaseTool):
    """Filters reference workflows to only keep those mentioned in the task list.

    Removes any selected reference workflow IDs and their content that are not
    explicitly cited in task descriptions via hard ID matching.
    Should run at the end of initial planning to clean up unused references.
    """

    def __init__(self):
        super().__init__("filter_unused_reference_workflows")

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        """Filter reference workflows to only those mentioned in task list."""
        # Get current reference workflow IDs and content
        reference_workflow_ids = exec_context.inputs.context.get("reference_workflow_ids", "")
        reference_workflow_content = exec_context.inputs.context.get("reference_workflow_content", {})
        percentages = exec_context.inputs.context.get("reference_workflow_percentages", {})

        # Get task list
        task_list = exec_context.inputs.task_list

        if not reference_workflow_ids or not task_list:
            # Nothing to filter
            return ToolResult(
                output_ui={},
                output_type=ToolOutputType.NO_OUTPUT,
                output_workflow={}
            )

        # Build mapping: internal_id -> full_id from content dict
        internal_to_full = {}
        for internal_id, content in reference_workflow_content.items():
            # Extract full ID from first line: "> Notebook ID: full_id"
            first_line = content.split('\n')[0] if content else ""
            if first_line.startswith("> Notebook ID:"):
                full_id = first_line.replace("> Notebook ID:", "").strip()
                internal_to_full[internal_id] = full_id

        # Parse full notebook IDs from reference_workflow_ids string
        full_notebook_ids = [id.strip() for id in reference_workflow_ids.split(",")]

        # Extract all notebook IDs mentioned in task descriptions
        mentioned_full_ids = set()
        for task in task_list.get("tasks", []):
            task_desc = task.get("task", "")
            # Look for citations like: [adapted from: 'notebook_id', cells: ...]
            for full_id in full_notebook_ids:
                if full_id in task_desc:
                    mentioned_full_ids.add(full_id)

        # If all notebooks are mentioned, no filtering needed
        if len(mentioned_full_ids) == len(full_notebook_ids):
            return ToolResult(
                output_ui={},
                output_type=ToolOutputType.NO_OUTPUT,
                output_workflow={}
            )

        # Build reverse mapping
        full_to_internal = {v: k for k, v in internal_to_full.items()}

        # Filter content dict - keep only mentioned workflows
        filtered_content_dict = {}
        filtered_full_ids = []
        for full_id in mentioned_full_ids:
            internal_id = full_to_internal.get(full_id)
            if internal_id and internal_id in reference_workflow_content:
                filtered_content_dict[internal_id] = reference_workflow_content[internal_id]
                filtered_full_ids.append(full_id)

        # Sort for consistent ordering
        filtered_full_ids.sort()

        # Rebuild reference_workflow_ids string
        filtered_ids_str = ", ".join(filtered_full_ids)

        # Build display message with filtered notebooks
        bullet_list_items = []
        for full_id in filtered_full_ids:
            # Get percentage from stored dict
            percentage = percentages.get(full_id, 0)
            if percentage > 0:
                bullet_list_items.append(f"📚 {full_id} (considering {percentage:.0f}% of file)")
            else:
                bullet_list_items.append(f"📚 {full_id}")

        # Format display message
        bullet_list = "\n".join(bullet_list_items)

        # Return filtered data with updated display
        return ToolResult(
            output_ui={"text": bullet_list},
            output_workflow={
                "reference_workflow_ids": filtered_ids_str,
                "reference_workflow_content": filtered_content_dict
            },
            output_type=ToolOutputType.REFERENCE_WORKFLOWS
        )


class CellDeletionTool(BaseTool):
    """Tool for actually deleting selected cells and translating indices.

    **UI Returns:**
    - `output_ui`: Dict with "text", "vscode_commands", "deleted_cells", "index_translation" for VSCode
    - `output_type`: EXECUTE_ONLY - executes cell deletions without chat display

    **Workflow Returns:**
    - `deleted_cells`: List of deleted cell indices for backtracking context
    - `index_translation`: Mapping of original->new indices after deletion

    **Used by workflows:** Backtracking workflow after CellSelectionDeletionTool selects cells

    **Special behavior:** Creates VSCode deletion commands and calculates index translation mapping
    """

    def __init__(self):
        super().__init__("cell_deletion")

    async def execute(self, exec_context: 'ExecutionContext', **kwargs) -> ToolResult:
        """Execute cell deletion and index translation."""
        # Get cells to delete from previous tool
        cells_to_delete = exec_context.inputs.context["cells_to_delete"]

        if not cells_to_delete:
            return ToolResult(
                output_ui="No cells selected for deletion",
                output_type=ToolOutputType.NO_OUTPUT,
            )

        # Sort in descending order to delete from end to beginning (preserves indices)
        cells_to_delete = sorted(cells_to_delete, reverse=True)

        # Create VSCode commands for cell deletion)
        vscode_commands = []
        for cell_num in cells_to_delete:
            vscode_commands.append({
                "command": "deleteCell",
                "cellIndex": cell_num
            })

        # Calculate index translation mapping for remaining cells
        index_translation = self._calculate_index_translation(cells_to_delete)

        # Log cell deletion
        deleted_list = sorted(cells_to_delete)
        cells_str = ", ".join(str(c) for c in deleted_list)
        logger.info(f"Deleted {len(deleted_list)} cells: {cells_str}")

        # Create output dict with all necessary data for VSCode
        output_data = {
            "text": f"Deleted cells: {sorted(cells_to_delete)}",
            "vscode_commands": vscode_commands,
            "deleted_cells": sorted(cells_to_delete),
            "index_translation": index_translation
        }

        return ToolResult(
            output_ui=output_data,
            output_type=ToolOutputType.EXECUTE_ONLY,
            output_workflow={
                "deleted_cells": sorted(cells_to_delete),
                "index_translation": index_translation,
                "cells_deleted": True,  # For deterministic router phase tracking
            }
        )

    def _calculate_index_translation(self, deleted_cells: List[int]) -> Dict[int, int]:
        """Calculate how original cell indices map to new indices after deletion.

        Args:
            deleted_cells: List of cell indices that were deleted (sorted)

        Returns:
            Dict mapping original_index -> new_index for remaining cells
        """
        if not deleted_cells:
            return {}

        deleted_set = set(deleted_cells)
        translation = {}
        new_index = 0

        # Assume we have cells up to max deleted cell + some buffer
        max_cell = max(deleted_cells) + 20  # Buffer for cells after deletions

        for original_index in range(max_cell):
            if original_index not in deleted_set:
                translation[original_index] = new_index
                new_index += 1

        return translation
