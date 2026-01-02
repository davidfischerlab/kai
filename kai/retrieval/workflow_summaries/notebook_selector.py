"""Data processing for notebook selection - LLM interaction handled by RAG tool."""

import json
from typing import Dict, List, Any, Optional

from .notebook_storage import NotebookStorage
from kai.utils import setup_logger

logger = setup_logger(__name__)


class NotebookSelector:
    """Data processing for notebook selection - no direct LLM interaction."""

    def __init__(self, notebook_storage: NotebookStorage):
        """Initialize with storage.

        Args:
            notebook_storage: Storage system for notebooks and summaries
        """
        self.storage = notebook_storage

    def create_selection_prompt(self, query: str, max_notebooks: int = 3) -> str:
        """Create selection prompt with query and all summaries.

        Args:
            query: User query
            max_notebooks: Maximum notebooks to select

        Returns:
            Formatted selection prompt for LLM
        """
        # Get all available summaries
        summaries = self.storage.get_all_summaries()

        if not summaries:
            return f"No notebook summaries available for query: {query}"

        # Build list of available notebooks with summaries
        notebook_list = []
        for i, (notebook_id, summary) in enumerate(summaries.items(), 1):
            # Truncate very long summaries for prompt efficiency
            truncated_summary = summary[:800] + "..." if len(summary) > 800 else summary
            notebook_list.append(f"{i}. **{notebook_id}**:\n{truncated_summary}\n")

        notebooks_text = "\n".join(notebook_list)

        prompt = f"""User query: "{query}"

Available workflow notebooks:

{notebooks_text}

Select the {max_notebooks} most relevant notebooks for this user query. Consider:
- Direct relevance to the biological question or analysis type mentioned
- Appropriateness for the data types mentioned or implied
- Completeness and practical applicability of the workflow
- How well the workflow matches the user's apparent needs

Respond in JSON format:
{{
  "selected_notebooks": ["notebook_id_1", "notebook_id_2", "notebook_id_3"],
  "reasoning": "Detailed explanation of why these notebooks were selected and how they address the user's query...",
  "relevance_summary": "Summary of what specifically in these notebooks will help the user..."
}}

JSON Response:"""

        return prompt

    def parse_selection_response(self, response: str, query: str) -> Dict[str, Any]:
        """Parse LLM selection response.

        Args:
            response: Raw LLM response
            query: Original user query

        Returns:
            Parsed selection result
        """
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed_response = json.loads(json_str)

                # Validate required fields
                required_fields = ["selected_notebooks", "reasoning", "relevance_summary"]
                for field in required_fields:
                    if field not in parsed_response:
                        parsed_response[field] = f"Missing {field}"

                # Ensure selected_notebooks is a list
                if not isinstance(parsed_response["selected_notebooks"], list):
                    parsed_response["selected_notebooks"] = []

                return parsed_response

            else:
                logger.error("No JSON found in LLM response")
                raise ValueError("Could not find valid JSON in LLM selection response")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise ValueError(f"Could not parse LLM selection response as JSON: {e}")


    def get_selected_notebook_content(self, selected_ids: List[str]) -> Dict[str, Any]:
        """Retrieve full content for selected notebooks.

        Args:
            selected_ids: List of selected notebook IDs

        Returns:
            Dictionary of notebook content {notebook_id: notebook_data}
        """
        selected_notebooks = {}

        for notebook_id in selected_ids:
            try:
                notebook_data = self.storage.get_notebook_content(notebook_id)
                if notebook_data:
                    selected_notebooks[notebook_id] = notebook_data
                else:
                    logger.debug(f"Could not retrieve content for notebook: {notebook_id}")

            except Exception as e:
                logger.error(f"Error retrieving notebook {notebook_id}: {e}")
                continue

        return selected_notebooks

    def format_notebook_context_dict(self, selection_result: Dict[str, Any],
                                      selected_ranges: Optional[Dict[str, List[int]]] = None) -> Dict[str, str]:
        """Format notebook content as dictionary mapping internal_id -> formatted_string.

        Args:
            selection_result: Result from select_relevant_notebooks
            selected_ranges: Optional dict mapping notebook IDs to lists of cell indices to include.
                           If provided, only cells at the specified indices will be included.
                           Example: {"notebook1": [0, 2, 5], "notebook2": [1, 3]}

        Returns:
            Dictionary {internal_id: formatted_content_string}
        """
        notebook_content = selection_result.get("notebook_content", {})
        if not notebook_content:
            return {}

        # Filter out notebooks with empty cell selections if selected_ranges is provided
        if selected_ranges:
            notebook_content = {
                nb_id: nb_data
                for nb_id, nb_data in notebook_content.items()
                if selected_ranges.get(nb_id) and len(selected_ranges.get(nb_id, [])) > 0
            }

        if not notebook_content:
            return {}

        # Format each notebook individually
        result_dict = {}
        for notebook_id, notebook_data in notebook_content.items():
            metadata = notebook_data.get("metadata", {})
            readable_notebook_title = metadata.get('title', notebook_id)
            full_notebook_id = f"{metadata.get('source_repository')}/{metadata.get('workflow_filename')}"

            # Build individual notebook section
            parts = []
            parts.append(f"> Notebook ID: {full_notebook_id}")
            parts.append(f"Title: {readable_notebook_title}")
            parts.append("")

            # Add notebook content
            cell_range = selected_ranges.get(notebook_id) if selected_ranges else None
            notebook_text = self._format_notebook_content(notebook_data, full_notebook_id, cell_range)
            parts.append(notebook_text)

            result_dict[notebook_id] = "\n".join(parts)

        return result_dict

    def _format_notebook_content(self, notebook_data: Dict[str, Any], notebook_id: str,
                                  cell_range: Optional[List[int]] = None) -> str:
        """Format individual notebook content for display.

        Args:
            notebook_data: Notebook data structure
            notebook_id: ID of the notebook
            cell_range: Optional list of cell indices to include. If None, includes all cells.

        Returns:
            Formatted notebook text
        """
        content_parts = []
        cells = notebook_data.get("cells", [])

        # If cell_range is specified and not empty, filter cells to only those indices
        if cell_range is not None and len(cell_range) > 0:
            # Create a set for O(1) lookup
            selected_indices = set(cell_range)
            # Filter cells where their order (index) is in the selected range
            cells = [cell for cell in cells if cell.get("order") in selected_indices]

        for cell in cells:
            content = cell.get("content", "")
            idx = cell.get("order", "")  # index of cell in notebook
            content_parts.append(f">> Cell at index {idx} in {notebook_id}")
            content_parts.append(content)

        return "\n".join(content_parts)