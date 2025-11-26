"""Generate comprehensive summaries using LLM."""

import os
from typing import Dict, Any, Optional
import asyncio

from kai.core.llm_interface import LLMInterface
from kai.config.settings import Settings
from kai.utils import setup_logger

logger = setup_logger(__name__)


class SummaryGenerator:
    """Generate comprehensive summaries of workflow notebooks using LLM."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize with settings for LLM creation.

        Args:
            settings: Application settings
        """
        self.settings = settings or Settings.from_env()

        # Get API key from environment variable
        api_key = os.getenv('OLLAMA_API_KEY')
        if not api_key:
            raise ValueError("OLLAMA_API_KEY environment variable is required for summary generation")

        # Create LLM interface for ollama-turbo (oss120B for comprehensive summaries)
        self.llm = LLMInterface(
            provider="ollama-turbo",
            model="gpt-oss:20b",
            settings=self.settings,
            api_key=api_key
        )

    async def generate_notebook_summary(self, notebook_data: Dict[str, Any]) -> str:
        """Generate comprehensive summary of notebook.

        Args:
            notebook_data: Complete notebook data structure

        Returns:
            Generated summary text
        """
        try:
            # Convert notebook to text format for analysis
            notebook_text = self._notebook_to_text(notebook_data)

            # Create comprehensive summary prompt
            prompt = self._create_summary_prompt(notebook_text, notebook_data)

            # Generate summary using ollama-turbo provider
            provider = self.llm.provider_large   # Get the large LLM provider
            summary = await provider.generate(
                prompt,
                system_prompt="You are an expert bioinformatics analyst that summarises jupyter notebooks for future querying.",
                reasoning_level="low",
            )

            logger.debug(
                f"Generated summary ({len(summary)} chars) for "
                f"{notebook_data.get('notebook_id', 'unknown')}"
            )

            return summary.strip()

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

    def _notebook_to_text(self, notebook_data: Dict[str, Any]) -> str:
        """Convert notebook structure to readable text.

        Args:
            notebook_data: Notebook data dictionary

        Returns:
            Text representation of the notebook
        """
        text_parts = []

        # Add metadata information
        metadata = notebook_data.get("metadata", {})
        text_parts.append(f"Tool: {metadata.get('tool', 'unknown')}")
        text_parts.append(f"Title: {metadata.get('title', 'Untitled')}")
        text_parts.append("")

        # Add cell content
        cells = notebook_data.get("cells", [])
        current_section = None

        for cell in cells:
            cell_type = cell.get("cell_type", "unknown")
            content = cell.get("content", "")
            section = cell.get("section", "")

            # Add section headers
            if section and section != current_section:
                text_parts.append(f"\n## Section: {section}\n")
                current_section = section

            if cell_type == "markdown":
                text_parts.append(content)
                text_parts.append("")
            elif cell_type == "code":
                text_parts.append("```python")
                text_parts.append(content)
                text_parts.append("```")
                text_parts.append("")

        return "\n".join(text_parts)

    def _create_summary_prompt(self, notebook_text: str, notebook_data: Dict[str, Any]) -> str:
        """Create comprehensive summary prompt.

        Args:
            notebook_text: Text representation of notebook
            notebook_data: Notebook data structure

        Returns:
            Formatted prompt for LLM
        """
        metadata = notebook_data.get("metadata", {})

        prompt = f"""You are analyzing a bioinformatics workflow notebook. Generate a comprehensive summary that will help an AI system decide if this workflow is relevant for a user's analysis task.

Notebook Metadata:
- Tool: {metadata.get('tool', 'unknown')}
- Title: {metadata.get('title', 'Untitled')}
- Repository: {metadata.get('source_repository', 'unknown')}

Notebook Content:
{notebook_text}

Generate a 100 word summary covering:

1. **Overarching objective**: What type of problem or question does this workflow address?
2. **Worflow summary**: What conceptual analysis steps are implemented in this workflow? Abstracted away from tool names only.
3. **Tools described**: What tools and packages are the focus of this workflow?
4. **Expected Outputs**: What outputs and visualizations are produced?
5. **Use Cases**: When should a researcher use this workflow?

"""

        return prompt

    async def generate_all_summaries(self, notebooks: Dict[str, Dict], storage=None) -> Dict[str, str]:
        """Generate summaries for all notebooks with progress tracking.

        Args:
            notebooks: Dictionary of notebook data {notebook_id: notebook_data}
            storage: Optional storage instance to save summaries immediately

        Returns:
            Dictionary of summaries {notebook_id: summary}
        """
        summaries = {}
        total_notebooks = len(notebooks)

        logger.info(f"Generating summaries for {total_notebooks} notebooks...")

        # Process notebooks with progress tracking
        completed = 0

        for notebook_id, notebook_data in notebooks.items():
            try:
                logger.info(f"Generating summary for {notebook_id} ({completed + 1}/{total_notebooks})")

                summary = await self.generate_notebook_summary(notebook_data)
                summaries[notebook_id] = summary

                # Save immediately if storage provided
                if storage:
                    storage.store_notebook(notebook_id, notebook_data, summary)
                    logger.debug(f"Saved summary for {notebook_id}")

                completed += 1
                logger.info(f"Completed {completed}/{total_notebooks} summaries")

                # Add small delay to avoid overwhelming the LLM
                await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Failed to generate summary for {notebook_id}: {e}")
                continue

        logger.info(f"Summary generation complete: {len(summaries)}/{total_notebooks} successful")
        return summaries

    async def regenerate_summary(self, notebook_id: str, notebook_data: Dict[str, Any]) -> str:
        """Regenerate summary for a specific notebook.

        Args:
            notebook_id: Notebook identifier
            notebook_data: Notebook data structure

        Returns:
            Generated summary text
        """
        logger.info(f"Regenerating summary for {notebook_id}")
        return await self.generate_notebook_summary(notebook_data)
