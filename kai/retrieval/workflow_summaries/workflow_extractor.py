"""Extract workflow notebooks from ChromaDB collections."""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import re

from kai.utils import setup_logger

logger = setup_logger(__name__)


class WorkflowExtractor:
    """Extract workflow notebooks from existing ChromaDB collections."""

    def __init__(self, chromadb_manager):
        """Initialize with ChromaDB manager."""
        self.chromadb = chromadb_manager

    def identify_workflow_collections(self) -> List[str]:
        """Find collections that contain workflows/notebooks.

        Returns:
            List of collection names that contain workflow content
        """
        workflow_collections = []

        for collection_name in self.chromadb.tool_registries.keys():
            # Look for workflow/notebook/tutorial indicators
            if any(indicator in collection_name.lower() for indicator in
                   ['workflow', 'notebook', 'tutorial', 'example']):
                workflow_collections.append(collection_name)

        logger.info(f"Identified {len(workflow_collections)} workflow collections")
        return workflow_collections

    def extract_notebook_from_collection(self, collection_name: str) -> Dict[str, Any]:
        """Extract complete notebook structure from ChromaDB collection.

        Args:
            collection_name: Name of the ChromaDB collection

        Returns:
            Dictionary containing structured notebook data
        """
        try:
            collection = self.chromadb.client.get_collection(collection_name)

            # Get all documents with metadata
            results = collection.get(include=["documents", "metadatas"])
            documents = results.get("documents", [])
            metadatas = results.get("metadatas", [])

            if not documents:
                logger.warning(f"No documents found in collection {collection_name}")
                return {}

            # Extract notebook metadata from first document
            notebook_metadata = self._extract_notebook_metadata(collection_name, metadatas)

            # Group documents by notebook and reconstruct structure
            notebooks = self._reconstruct_notebooks(documents, metadatas)

            # For now, assume one main notebook per collection
            # In future, could handle multiple notebooks per collection
            main_notebook_id = list(notebooks.keys())[0] if notebooks else collection_name

            if main_notebook_id in notebooks:
                notebook_data = notebooks[main_notebook_id]
                notebook_data["metadata"] = notebook_metadata
                return notebook_data
            else:
                logger.warning(f"No structured notebook found in {collection_name}")
                return {}

        except Exception as e:
            logger.error(f"Error extracting notebook from {collection_name}: {e}")
            return {}

    def _extract_notebook_metadata(self, collection_name: str, metadatas: List[Dict]) -> Dict[str, Any]:
        """Extract notebook metadata from collection.

        Args:
            collection_name: Name of the collection
            metadatas: List of metadata dictionaries

        Returns:
            Notebook metadata dictionary
        """
        # Find the first metadata entry with useful information
        for meta in metadatas:
            if meta:
                tool = meta.get('tool', '')
                repo = meta.get('repo', '')

                # Extract source repository and filename from metadata
                source_repository = self._extract_repository_name(repo, tool)
                workflow_filename = self._extract_workflow_filename(meta)

                # Extract title from chunk_text or chunk_id
                title = self._extract_title(meta)

                return {
                    "tool": tool,
                    "collection_name": collection_name,
                    "title": title,
                    "source_repository": source_repository,
                    "workflow_filename": workflow_filename,
                    "source_path": meta.get('source_path', '')
                }

        # Fallback metadata
        return {
            "tool": self._extract_tool_from_collection_name(collection_name),
            "collection_name": collection_name,
            "title": self._generate_title_from_collection(collection_name),
            "source_repository": "unknown",
            "workflow_filename": "unknown",
            "source_path": ""
        }

    def _extract_repository_name(self, repo: str, tool: str) -> str:
        """Extract repository name in org/repo format."""
        if repo and repo != 'unknown':
            # If repo contains organization info, use it
            if '/' in repo:
                return repo
            else:
                # Try to infer organization from known patterns
                org_mappings = {
                    'scanpy': 'scverse/scanpy',
                    'scirpy': 'scverse/scirpy',
                    'squidpy': 'scverse/squidpy',
                    'anndata': 'scverse/anndata',
                    'scvi': 'scverse/scvi-tools'
                }
                return org_mappings.get(tool.lower(), f"{repo}/{tool}")

        # Fallback to tool-based inference
        return f"unknown/{tool}" if tool else "unknown/unknown"

    def _extract_workflow_filename(self, meta: Dict) -> str:
        """Extract workflow filename from metadata."""
        chunk_id = meta.get('chunk_id', '')

        # Look for notebook-like patterns
        if '.ipynb' in chunk_id or 'notebook' in chunk_id.lower():
            return f"{chunk_id}.ipynb"
        elif 'tutorial' in chunk_id.lower():
            return f"tutorials/{chunk_id}.ipynb"
        else:
            return f"workflows/{chunk_id}.ipynb"

    def _extract_title(self, meta: Dict) -> str:
        """Extract title from metadata."""
        chunk_text = meta.get('chunk_text', '')

        # Look for title in chunk_text
        if chunk_text:
            lines = chunk_text.split('\n')
            for line in lines[:3]:  # Check first few lines
                if line.strip().startswith('Tutorial:'):
                    return line.replace('Tutorial:', '').strip()
                elif line.strip().startswith('#'):
                    return line.replace('#', '').strip()

        # Fallback to chunk_id
        chunk_id = meta.get('chunk_id', '')
        if chunk_id:
            # Convert snake_case to title case
            return chunk_id.replace('_', ' ').title()

        return "Untitled Workflow"

    def _extract_tool_from_collection_name(self, collection_name: str) -> str:
        """Extract tool name from collection name."""
        # Remove version suffixes and common patterns
        base_name = collection_name.split('_v')[0]  # Remove version
        base_name = base_name.split('_workflow')[0]  # Remove workflow suffix
        base_name = base_name.split('_notebook')[0]  # Remove notebook suffix

        return base_name

    def _generate_title_from_collection(self, collection_name: str) -> str:
        """Generate title from collection name."""
        # Convert to readable format
        title = collection_name.replace('_', ' ').title()
        title = re.sub(r'\bV\w+', '', title)  # Remove version info
        return title.strip()

    def _reconstruct_notebooks(self, documents: List[str], metadatas: List[Dict]) -> Dict[str, Dict]:
        """Reconstruct notebook structure from documents and metadata.

        Args:
            documents: List of document content
            metadatas: List of metadata dictionaries

        Returns:
            Dictionary of notebooks {notebook_id: notebook_data}
        """
        notebooks = {}

        # Group by notebook (usually one per collection)
        for doc, meta in zip(documents, metadatas):
            if not meta:
                continue

            # Determine notebook ID (could be chunk_id root or tool name)
            notebook_id = self._get_notebook_id(meta)

            if notebook_id not in notebooks:
                notebooks[notebook_id] = {
                    "notebook_id": notebook_id,
                    "cells": []
                }

            # Convert document to cell
            cell = self._document_to_cell(doc, meta)
            if cell:
                notebooks[notebook_id]["cells"].append(cell)

        # Sort cells by order within each notebook
        for notebook in notebooks.values():
            notebook["cells"].sort(key=lambda x: x.get("order", 0))

        return notebooks

    def _get_notebook_id(self, meta: Dict) -> str:
        """Extract notebook ID from metadata."""
        chunk_id = meta.get('chunk_id', '')

        # Remove section/cell suffixes to get base notebook ID
        base_id = chunk_id.split('_section_')[0]
        base_id = base_id.split('_code_')[0]

        return base_id if base_id else meta.get('tool', 'unknown')

    def _document_to_cell(self, content: str, meta: Dict) -> Optional[Dict]:
        """Convert document content to notebook cell.

        Args:
            content: Document content
            meta: Document metadata

        Returns:
            Cell dictionary or None if conversion fails
        """
        chunk_level = meta.get('chunk_level', 'unknown')
        chunk_index = meta.get('chunk_index', 0)

        # Determine cell type and content
        if chunk_level == 'code_cell' or 'Code:' in content:
            # Extract code content
            if 'Code:' in content:
                code_content = content.split('Code:', 1)[1].strip()
            else:
                code_content = content

            return {
                "cell_type": "code",
                "content": code_content,
                "section": meta.get('parent_id', 'main'),
                "order": chunk_index
            }
        else:
            # Markdown cell
            return {
                "cell_type": "markdown",
                "content": content,
                "section": meta.get('parent_id', 'main'),
                "order": chunk_index
            }

    def extract_all_workflows(self) -> Dict[str, Dict]:
        """Extract all workflow notebooks from ChromaDB with batching.

        Returns:
            Dictionary of {notebook_id: notebook_data} for all workflows
        """
        import gc
        import time

        workflow_collections = self.identify_workflow_collections()
        all_notebooks = {}

        logger.info(f"Extracting notebooks from {len(workflow_collections)} collections...")

        # Process collections in batches to manage file handles
        batch_size = 10
        for i in range(0, len(workflow_collections), batch_size):
            batch = workflow_collections[i:i + batch_size]

            for collection_name in batch:
                try:
                    notebook_data = self.extract_notebook_from_collection(collection_name)
                    if notebook_data:
                        notebook_id = notebook_data.get("notebook_id", collection_name)
                        all_notebooks[notebook_id] = notebook_data
                        logger.debug(f"Extracted notebook: {notebook_id}")
                    else:
                        logger.warning(f"Failed to extract notebook from {collection_name}")

                except Exception as e:
                    logger.error(f"Error processing collection {collection_name}: {e}")
                    continue

            # Force garbage collection and small delay after each batch
            gc.collect()
            time.sleep(0.1)
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(workflow_collections) + batch_size - 1)//batch_size}")

        logger.info(f"Successfully extracted {len(all_notebooks)} notebooks")
        return all_notebooks