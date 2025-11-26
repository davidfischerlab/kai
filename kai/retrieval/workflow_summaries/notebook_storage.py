"""File-based storage system for notebooks and summaries."""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from kai.utils import setup_logger

logger = setup_logger(__name__)


class NotebookStorage:
    """File-based storage for notebook content and summaries."""

    def __init__(self, storage_path: Path):
        """Initialize storage with base path.

        Args:
            storage_path: Base path for notebook storage
        """
        self.storage_path = Path(storage_path)
        self.summaries_dir = self.storage_path / "summaries"
        self.notebooks_dir = self.storage_path / "notebooks"

        # Create directories if they don't exist
        self.summaries_dir.mkdir(parents=True, exist_ok=True)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Initialized notebook storage at {storage_path}")

    def store_notebook(self, notebook_id: str, notebook_data: Dict[str, Any], summary: str = "") -> bool:
        """Store notebook content and summary in organized subdirectories.

        Args:
            notebook_id: Unique identifier for the notebook
            notebook_data: Complete notebook data structure
            summary: Text summary of the notebook

        Returns:
            True if storage successful, False otherwise
        """
        try:
            # Extract organization and repo from metadata
            metadata = notebook_data.get("metadata", {})
            source_repo = metadata.get("source_repository", "unknown/unknown")

            # Parse org/repo from source_repository
            if "/" in source_repo:
                org_name, repo_name = source_repo.split("/", 1)
            else:
                org_name, repo_name = "unknown", source_repo

            # Create organized directory structure
            org_notebooks_dir = self.notebooks_dir / org_name / repo_name
            org_summaries_dir = self.summaries_dir / org_name / repo_name

            # Create directories if they don't exist
            org_notebooks_dir.mkdir(parents=True, exist_ok=True)
            org_summaries_dir.mkdir(parents=True, exist_ok=True)

            # Store notebook as JSON in organized subfolder
            notebook_path = org_notebooks_dir / f"{notebook_id}.json"
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook_data, f, indent=2, ensure_ascii=False)

            # Store summary as text file in organized subfolder (if provided)
            if summary:
                summary_path = org_summaries_dir / f"{notebook_id}.txt"
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(summary)

            logger.debug(f"Stored notebook and summary for {notebook_id} in {org_name}/{repo_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to store notebook {notebook_id}: {e}")
            return False

    def get_notebook_content(self, notebook_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve full notebook content from organized subdirectories.

        Args:
            notebook_id: Unique identifier for the notebook

        Returns:
            Notebook data dictionary or None if not found
        """
        try:
            # Search for the notebook in all org/repo subdirectories
            for org_dir in self.notebooks_dir.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    notebook_path = repo_dir / f"{notebook_id}.json"
                    if notebook_path.exists():
                        with open(notebook_path, 'r', encoding='utf-8') as f:
                            notebook_data = json.load(f)
                        return notebook_data

            logger.warning(f"Notebook {notebook_id} not found in any subdirectory")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve notebook {notebook_id}: {e}")
            return None

    def get_summary(self, notebook_id: str) -> Optional[str]:
        """Retrieve summary for a specific notebook from organized subdirectories.

        Args:
            notebook_id: Unique identifier for the notebook

        Returns:
            Summary text or None if not found
        """
        try:
            # Search for the summary in all org/repo subdirectories
            for org_dir in self.summaries_dir.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    summary_path = repo_dir / f"{notebook_id}.txt"
                    if summary_path.exists():
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            summary = f.read().strip()
                        return summary

            logger.warning(f"Summary for {notebook_id} not found in any subdirectory")
            return None

        except Exception as e:
            logger.error(f"Failed to retrieve summary for {notebook_id}: {e}")
            return None

    def get_all_summaries(self) -> Dict[str, str]:
        """Get all notebook summaries for LLM selection from organized subdirectories.

        Returns:
            Dictionary mapping notebook_id to summary text
        """
        summaries = {}

        try:
            # Check if summaries directory exists and has content
            if not self.summaries_dir.exists():
                logger.info("Summaries directory does not exist yet")
                return summaries

            # Read all .txt files from organized subdirectories
            for org_dir in self.summaries_dir.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    for summary_file in repo_dir.glob("*.txt"):
                        notebook_id = summary_file.stem  # filename without extension

                        try:
                            with open(summary_file, 'r', encoding='utf-8') as f:
                                summary = f.read().strip()

                            if summary:
                                summaries[notebook_id] = summary

                        except Exception as e:
                            logger.warning(f"Failed to read summary file {summary_file}: {e}")
                            continue

            logger.info(f"Retrieved {len(summaries)} notebook summaries")
            return summaries

        except Exception as e:
            logger.error(f"Failed to retrieve summaries: {e}")
            return {}

    def get_all_notebook_ids(self) -> list[str]:
        """Get list of all stored notebook IDs from organized subdirectories.

        Returns:
            List of notebook IDs
        """
        try:
            notebook_ids = []

            for org_dir in self.notebooks_dir.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    for notebook_file in repo_dir.glob("*.json"):
                        notebook_id = notebook_file.stem
                        notebook_ids.append(notebook_id)

            return sorted(notebook_ids)

        except Exception as e:
            logger.error(f"Failed to get notebook IDs: {e}")
            return []

    def has_notebook(self, notebook_id: str) -> bool:
        """Check if a notebook exists in organized subdirectories.

        Args:
            notebook_id: Unique identifier for the notebook

        Returns:
            True if notebook exists, False otherwise
        """
        try:
            for org_dir in self.notebooks_dir.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    notebook_path = repo_dir / f"{notebook_id}.json"
                    if notebook_path.exists():
                        return True

            return False

        except Exception:
            return False

    def has_summary(self, notebook_id: str) -> bool:
        """Check if a summary exists for a notebook in organized subdirectories.

        Args:
            notebook_id: Unique identifier for the notebook

        Returns:
            True if summary exists, False otherwise
        """
        try:
            for org_dir in self.summaries_dir.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    summary_path = repo_dir / f"{notebook_id}.txt"
                    if summary_path.exists():
                        return True

            return False

        except Exception:
            return False

    def delete_notebook(self, notebook_id: str) -> bool:
        """Delete notebook and its summary.

        Args:
            notebook_id: Unique identifier for the notebook

        Returns:
            True if deletion successful, False otherwise
        """
        try:
            notebook_path = self.notebooks_dir / f"{notebook_id}.json"
            summary_path = self.summaries_dir / f"{notebook_id}.txt"

            success = True

            if notebook_path.exists():
                notebook_path.unlink()
                logger.debug(f"Deleted notebook file for {notebook_id}")

            if summary_path.exists():
                summary_path.unlink()
                logger.debug(f"Deleted summary file for {notebook_id}")

            return success

        except Exception as e:
            logger.error(f"Failed to delete notebook {notebook_id}: {e}")
            return False

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics from organized subdirectories.

        Returns:
            Dictionary with storage statistics
        """
        try:
            notebook_count = 0
            summary_count = 0
            total_notebook_size = 0
            total_summary_size = 0
            org_stats = {}

            # Count files in organized subdirectories
            for org_dir in self.notebooks_dir.iterdir():
                if not org_dir.is_dir():
                    continue

                org_name = org_dir.name
                org_stats[org_name] = {"repos": 0, "notebooks": 0, "summaries": 0}

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    org_stats[org_name]["repos"] += 1

                    # Count notebooks
                    notebook_files = list(repo_dir.glob("*.json"))
                    notebook_count += len(notebook_files)
                    org_stats[org_name]["notebooks"] += len(notebook_files)

                    # Sum notebook sizes
                    for f in notebook_files:
                        total_notebook_size += f.stat().st_size

                    # Count summaries
                    summary_dir = self.summaries_dir / org_name / repo_dir.name
                    if summary_dir.exists():
                        summary_files = list(summary_dir.glob("*.txt"))
                        summary_count += len(summary_files)
                        org_stats[org_name]["summaries"] += len(summary_files)

                        # Sum summary sizes
                        for f in summary_files:
                            total_summary_size += f.stat().st_size

            return {
                "notebook_count": notebook_count,
                "summary_count": summary_count,
                "total_notebook_size_mb": total_notebook_size / (1024 * 1024),
                "total_summary_size_mb": total_summary_size / (1024 * 1024),
                "storage_path": str(self.storage_path),
                "organization_stats": org_stats
            }

        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}