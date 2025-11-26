"""Direct file-based workflow extractor that bypasses ChromaDB entirely."""

import json
from pathlib import Path
from typing import Dict, List, Any
import re
from datetime import datetime

from kai.utils import setup_logger

logger = setup_logger(__name__)


class DirectFileExtractor:
    """Extract workflows directly from cached repository files."""

    def __init__(self, cache_base_path: Path):
        """Initialize with cache base path.

        Args:
            cache_base_path: Base path to bioinformatics cache directory
        """
        self.cache_base_path = Path(cache_base_path)

        # Workflow file patterns (from GitHub extractor config)
        self.workflow_patterns = ["*.ipynb"]

    def _get_organizations(self) -> List[str]:
        """Dynamically discover all organizations in the cache directory."""
        organizations = []
        if self.cache_base_path.exists():
            for item in self.cache_base_path.iterdir():
                if item.is_dir() and not item.name.startswith('.') and item.name != 'chromadb':
                    # Check if it has a repos subdirectory (organization structure)
                    repos_path = item / "repos"
                    if repos_path.exists() and repos_path.is_dir():
                        organizations.append(item.name)
        return sorted(organizations)

    def extract_all_workflows(self) -> Dict[str, Dict]:
        """Extract all workflow notebooks from cached repositories.

        Returns:
            Dictionary of {notebook_id: notebook_data} for all workflows
        """
        all_notebooks = {}
        total_files = 0

        logger.info("Starting direct file extraction from cached repositories...")

        organizations = self._get_organizations()
        logger.info(f"Found {len(organizations)} organizations: {organizations}")

        for org_name in organizations:
            org_path = self.cache_base_path / org_name / "repos"

            if not org_path.exists():
                logger.warning(f"Organization path not found: {org_path}")
                continue

            logger.info(f"Processing organization: {org_name}")

            # Get all repository directories
            repo_dirs = [d for d in org_path.iterdir() if d.is_dir()]
            logger.info(f"Found {len(repo_dirs)} repositories in {org_name}")

            for i, repo_dir in enumerate(repo_dirs):
                try:
                    repo_notebooks = self._extract_notebooks_from_repo(repo_dir, org_name)
                    all_notebooks.update(repo_notebooks)
                    total_files += len(repo_notebooks)

                    if repo_notebooks:
                        logger.debug(f"Extracted {len(repo_notebooks)} notebooks from {repo_dir.name}")

                except Exception as e:
                    logger.error(f"Error processing repository {repo_dir.name}: {e}")
                    continue

        logger.info(f"Successfully extracted {total_files} workflow notebooks from {len(organizations)} organizations")
        print(f"✅ Extraction complete: {total_files} notebooks extracted")
        return all_notebooks

    def _extract_notebooks_from_repo(self, repo_path: Path, org_name: str) -> Dict[str, Dict]:
        """Extract notebooks from a single repository.

        Args:
            repo_path: Path to repository directory
            org_name: Organization name

        Returns:
            Dictionary of notebooks from this repo
        """
        repo_name = repo_path.name
        notebooks = {}

        # Find all workflow files (notebooks) in the repository
        workflow_files = []
        for pattern in self.workflow_patterns:
            workflow_files.extend(repo_path.rglob(pattern))

        # Filter to only .ipynb files that actually exist (not broken symlinks)
        notebook_files = [f for f in workflow_files if f.suffix == '.ipynb' and f.exists() and f.is_file()]

        # Further filter to exclude test files, benchmarks, and simple examples
        filtered_notebooks = self._filter_meaningful_notebooks(notebook_files)

        for notebook_file in filtered_notebooks:
            try:
                notebook_data = self._parse_notebook(notebook_file, repo_name, org_name)
                if notebook_data:
                    notebook_id = notebook_data["notebook_id"]
                    notebooks[notebook_id] = notebook_data

            except Exception as e:
                logger.warning(f"Error parsing notebook {notebook_file}: {e}")
                continue

        return notebooks

    def _parse_notebook(self, notebook_path: Path, repo_name: str, org_name: str) -> Dict[str, Any]:
        """Parse a Jupyter notebook into structured format.

        Args:
            notebook_path: Path to the notebook file
            repo_name: Repository name
            org_name: Organization name

        Returns:
            Structured notebook data
        """
        try:
            # Check if this is a Git LFS pointer file
            with open(notebook_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if first_line.startswith('version https://git-lfs.github.com'):
                    logger.debug(f"Skipping Git LFS pointer file: {notebook_path}")
                    return None

                # Reset file pointer and read as JSON
                f.seek(0)
                notebook_content = json.load(f)

        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Could not parse notebook {notebook_path}: {e}")
            return None

        # Extract cells
        cells = []
        raw_cells = notebook_content.get('cells', [])

        for i, cell in enumerate(raw_cells):
            cell_type = cell.get('cell_type', 'unknown')

            # Extract cell content
            if 'source' in cell:
                if isinstance(cell['source'], list):
                    content = ''.join(cell['source'])
                else:
                    content = str(cell['source'])
            else:
                content = ""

            # Skip empty cells
            if not content.strip():
                continue

            cells.append({
                "cell_type": cell_type,
                "content": content.strip(),
                "section": self._determine_section(content, i),
                "order": i
            })

        # Generate notebook ID
        notebook_id = self._generate_notebook_id(notebook_path, repo_name, org_name)

        # Extract title
        title = self._extract_title(notebook_content, notebook_path)

        # Create metadata
        metadata = {
            "tool": self._determine_tool(repo_name),
            "collection_name": notebook_id,
            "title": title,
            "source_repository": f"{org_name}/{repo_name}",
            "workflow_filename": notebook_path.name,
            "source_path": str(notebook_path.relative_to(self.cache_base_path)),
            "extraction_date": datetime.now().isoformat()
        }

        return {
            "notebook_id": notebook_id,
            "metadata": metadata,
            "cells": cells,
            "cell_count": len(cells)
        }

    def _generate_notebook_id(self, notebook_path: Path, repo_name: str, org_name: str) -> str:
        """Generate unique notebook ID."""
        # Remove .ipynb extension and clean up the name
        notebook_name = notebook_path.stem

        # Create unique ID: org_repo_notebookname
        notebook_id = f"{org_name}_{repo_name}_{notebook_name}"

        # Clean up the ID (remove special characters, make lowercase)
        notebook_id = re.sub(r'[^a-zA-Z0-9_]', '_', notebook_id)
        notebook_id = re.sub(r'_+', '_', notebook_id)  # Collapse multiple underscores
        notebook_id = notebook_id.lower()

        return notebook_id

    def _extract_title(self, notebook_content: Dict, notebook_path: Path) -> str:
        """Extract title from notebook content or filename."""
        # Try to get title from first markdown cell
        cells = notebook_content.get('cells', [])

        for cell in cells:
            if cell.get('cell_type') == 'markdown':
                source = cell.get('source', [])
                if isinstance(source, list):
                    text = ''.join(source)
                else:
                    text = str(source)

                # Look for markdown headers
                lines = text.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('#'):
                        # Extract title from first header
                        title = re.sub(r'^#+\s*', '', line).strip()
                        if title:
                            return title

        # Fallback to filename
        return notebook_path.stem.replace('_', ' ').replace('-', ' ').title()

    def _determine_tool(self, repo_name: str) -> str:
        """Determine the primary tool based on repository name."""
        repo_lower = repo_name.lower()

        if 'scanpy' in repo_lower:
            return 'scanpy'
        elif 'scvi' in repo_lower:
            return 'scvi-tools'
        elif 'squidpy' in repo_lower:
            return 'squidpy'
        elif 'scirpy' in repo_lower:
            return 'scirpy'
        elif 'decoupler' in repo_lower:
            return 'decoupler'
        elif 'mudata' in repo_lower:
            return 'mudata'
        elif 'anndata' in repo_lower:
            return 'anndata'
        else:
            return 'python'  # Default fallback

    def _determine_section(self, content: str, cell_index: int) -> str:
        """Determine section based on content and position."""
        content_lower = content.lower()

        # Look for common section indicators
        if any(word in content_lower for word in ['import', 'loading', 'setup']):
            return 'setup'
        elif any(word in content_lower for word in ['preprocessing', 'quality control', 'qc']):
            return 'preprocessing'
        elif any(word in content_lower for word in ['analysis', 'clustering', 'dimension reduction']):
            return 'analysis'
        elif any(word in content_lower for word in ['visualization', 'plotting', 'plot']):
            return 'visualization'
        elif any(word in content_lower for word in ['conclusion', 'summary', 'results']):
            return 'results'
        else:
            return 'main'

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about available repositories and files."""
        stats = {
            "organizations": {},
            "total_repos": 0,
            "total_notebooks": 0
        }

        organizations = self._get_organizations()

        for org_name in organizations:
            org_path = self.cache_base_path / org_name / "repos"

            if not org_path.exists():
                stats["organizations"][org_name] = {"repos": 0, "notebooks": 0}
                continue

            repo_dirs = [d for d in org_path.iterdir() if d.is_dir()]

            org_notebook_count = 0
            for repo_dir in repo_dirs:
                workflow_files = []
                for pattern in self.workflow_patterns:
                    workflow_files.extend(repo_dir.rglob(pattern))

                notebook_files = [f for f in workflow_files if f.suffix == '.ipynb']
                meaningful_notebooks = self._filter_meaningful_notebooks(notebook_files)
                org_notebook_count += len(meaningful_notebooks)

            stats["organizations"][org_name] = {
                "repos": len(repo_dirs),
                "notebooks": org_notebook_count
            }
            stats["total_repos"] += len(repo_dirs)
            stats["total_notebooks"] += org_notebook_count

        return stats

    def _filter_meaningful_notebooks(self, notebook_files: List[Path]) -> List[Path]:
        """Filter to only meaningful workflow notebooks, excluding tests and simple examples.

        Args:
            notebook_files: List of all notebook files found

        Returns:
            Filtered list of meaningful workflow notebooks
        """
        meaningful_notebooks = []

        for notebook_file in notebook_files:
            # Convert path to lowercase for case-insensitive matching
            path_str = str(notebook_file).lower()

            # Only skip test directories and files
            if any(exclude in path_str for exclude in [
                '/test/', '/tests/', '_test.ipynb', 'test_',
                '/scratch/', '/tmp/', '/temp/'
            ]):
                continue

            # Keep all other notebooks (docs/, examples/, tutorials/, etc.)
            meaningful_notebooks.append(notebook_file)

        return meaningful_notebooks