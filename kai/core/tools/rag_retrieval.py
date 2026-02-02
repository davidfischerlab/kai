"""RAG retrieval tool for code documentation and examples.

This module provides a deterministic tool for retrieving relevant documentation
via semantic search across code snippet collections.
"""

from typing import List, Optional, TYPE_CHECKING

from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType
from kai.utils import setup_logger

if TYPE_CHECKING:
    from kai.core.orchestration.state import KaiState
    from kai.retrieval import ChromaDbManager

logger = setup_logger(__name__)


class CodeRetrievalTool(BaseTool):
    """Tool for retrieving relevant documentation via RAG.

    Performs semantic search across code snippet collections to find relevant
    examples and documentation for the current task.

    **UI Returns:**
    - `output_type`: NO_OUTPUT - internal retrieval tool

    **Workflow Returns:**
    - `rag_retrieval`: Retrieved documentation content string (or empty string if no results)

    **Used by workflows:** Execution workflows when retrieval queries are specified

    **Special behavior:**
    - Waits for background initialization of knowledge base
    - Logs retrieval queries and results if DEBUG_PROMPTS is enabled
    - Returns empty rag_retrieval field if no queries specified (RAG disabled)
    """

    def __init__(self, knowledge_base: Optional['ChromaDbManager'] = None):
        super().__init__("rag_retrieval")
        self.knowledge_base = knowledge_base

    async def execute(self, state: 'KaiState', **kwargs) -> ToolResult:
        """Retrieve relevant knowledge via RAG."""
        # Retrieval query is a list of query strings
        queries = state.get("snippet_retrieval_query")

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
                self._log_rag_query_if_enabled(queries, state, "success", results["content"])

                return ToolResult(
                    output_workflow={"rag_retrieval": results["content"]},
                    output_ui={},  # TODO could output tool usage from result here in the future
                    output_type=ToolOutputType.NO_OUTPUT
                )
            else:
                # Log empty results
                self._log_rag_query_if_enabled(queries, state, "no_results")

                return ToolResult(
                    output_workflow={},
                    output_ui={},
                    output_type=ToolOutputType.NO_OUTPUT
                )

        except Exception as e:
            self._log_rag_query_if_enabled(queries, state, "error", error=e)
            return ToolResult(
                output_workflow={},
                output_ui={},
                output_type=ToolOutputType.NO_OUTPUT
            )

    def _log_rag_query_if_enabled(self, queries: List[str], state: 'KaiState', status: str, results: Optional[str] = None, error: Optional[Exception] = None):
        """Log RAG query to debug folder if DEBUG_PROMPTS is enabled."""
        from kai.config.settings import settings
        from kai.config.paths import get_debug_prompts_dir
        from datetime import datetime

        if not settings.DEBUG_PROMPTS:
            return

        try:
            session_id = state["session_id"]
            session_timestamp = state["session_timestamp"]
            iteration_timestamp = state["iteration_timestamp"]
            is_autonomous = state["autonomous_mode"]
            notebook_uri = state["notebook_uri"]
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
