"""Tests for workflow summary system.

Tests the notebook storage, semantic search, and workflow selection components
that provide RAG-enabled reference workflow retrieval for autonomous execution.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import json

from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag
from kai.retrieval.workflow_summaries.notebook_selector import NotebookSelector


class TestNotebookStorage:
    """Test NotebookStorage functionality."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def sample_notebook_data(self):
        """Sample notebook data for testing."""
        return {
            "test_notebook_1": {
                "notebook_id": "test_notebook_1",
                "summary": "This notebook demonstrates single-cell RNA-seq preprocessing with scanpy.",
                "metadata": {
                    "title": "Single-cell preprocessing tutorial",
                    "source_repository": "test/repository",
                    "summary_length": 65
                }
            },
            "test_notebook_2": {
                "notebook_id": "test_notebook_2",
                "summary": "Spatial transcriptomics analysis using squidpy for tissue visualization.",
                "metadata": {
                    "title": "Spatial transcriptomics analysis",
                    "source_repository": "spatial/analysis",
                    "summary_length": 68
                }
            }
        }

    def test_notebook_storage_initialization(self, temp_storage_path):
        """Test NotebookStorage initialization."""
        storage = NotebookStorage(temp_storage_path)
        assert storage.storage_path == temp_storage_path

    def test_save_and_load_notebook(self, temp_storage_path, sample_notebook_data):
        """Test storing and loading notebook data."""
        storage = NotebookStorage(temp_storage_path)

        # Store a notebook with complete data structure
        notebook_id = "test_notebook_1"
        notebook_data = {
            "notebook_id": notebook_id,
            "metadata": sample_notebook_data[notebook_id]["metadata"],
            "cells": [],  # Empty cells for testing
            "cell_count": 0
        }

        # Store notebook
        success = storage.store_notebook(notebook_id, notebook_data, sample_notebook_data[notebook_id]["summary"])
        assert success

        # Load the notebook
        loaded_data = storage.get_notebook_content(notebook_id)
        assert loaded_data is not None
        assert loaded_data["notebook_id"] == notebook_id
        assert loaded_data["metadata"]["title"] == notebook_data["metadata"]["title"]

    def test_get_all_summaries(self, temp_storage_path, sample_notebook_data):
        """Test retrieving all notebook summaries."""
        storage = NotebookStorage(temp_storage_path)

        # Store multiple notebooks
        for notebook_id, data in sample_notebook_data.items():
            notebook_data = {
                "notebook_id": notebook_id,
                "metadata": data["metadata"],
                "cells": [],
                "cell_count": 0
            }
            storage.store_notebook(notebook_id, notebook_data, data["summary"])

        # Get all summaries (returns dict of notebook_id -> summary_text)
        all_summaries = storage.get_all_summaries()
        assert len(all_summaries) == 2

        # Check that both notebook IDs are present
        assert "test_notebook_1" in all_summaries
        assert "test_notebook_2" in all_summaries

        # Check that summaries contain expected content
        assert "scanpy" in all_summaries["test_notebook_1"]
        assert "squidpy" in all_summaries["test_notebook_2"]


class TestWorkflowSummaryRag:
    """Test WorkflowSummaryRag semantic search functionality."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def populated_storage(self, temp_storage_path):
        """Create storage with sample notebooks."""
        storage = NotebookStorage(temp_storage_path)

        # Sample notebooks with different analysis types
        notebooks = [
            {
                "notebook_id": "scanpy_preprocessing",
                "summary": "Complete single-cell RNA sequencing preprocessing workflow using scanpy. Includes quality control, normalization, and filtering steps for cell and gene analysis.",
                "metadata": {
                    "title": "Scanpy preprocessing workflow",
                    "source_repository": "scverse/scanpy-tutorials"
                }
            },
            {
                "notebook_id": "spatial_analysis",
                "summary": "Spatial transcriptomics data analysis with squidpy. Visualization of tissue sections and spatial gene expression patterns.",
                "metadata": {
                    "title": "Spatial transcriptomics with squidpy",
                    "source_repository": "scverse/squidpy-tutorials"
                }
            },
            {
                "notebook_id": "tcr_analysis",
                "summary": "T cell receptor repertoire analysis using scirpy. TCR sequence analysis and clonotype identification.",
                "metadata": {
                    "title": "TCR analysis with scirpy",
                    "source_repository": "scverse/scirpy-tutorials"
                }
            },
            {
                "notebook_id": "differential_expression",
                "summary": "Differential gene expression analysis between cell populations. Statistical testing and visualization of results.",
                "metadata": {
                    "title": "Differential expression analysis",
                    "source_repository": "analysis/differential"
                }
            }
        ]

        for notebook in notebooks:
            notebook_data = {
                "notebook_id": notebook["notebook_id"],
                "metadata": notebook["metadata"],
                "cells": [],
                "cell_count": 0
            }
            storage.store_notebook(notebook["notebook_id"], notebook_data, notebook["summary"])

        return storage

    def test_summary_rag_initialization(self, temp_storage_path):
        """Test WorkflowSummaryRag initialization."""
        rag = WorkflowSummaryRag(temp_storage_path)
        assert rag.collection is not None
        assert rag.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"

    def test_index_summaries(self, temp_storage_path, populated_storage):
        """Test indexing notebook summaries for semantic search."""
        rag = WorkflowSummaryRag(temp_storage_path)

        # Index all summaries
        indexed_count = rag.index_all_summaries(populated_storage)
        assert indexed_count == 4

        # Verify collection stats
        stats = rag.get_collection_stats()
        assert stats["total_summaries"] == 4
        assert stats["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_semantic_search_single_cell(self, temp_storage_path, populated_storage):
        """Test semantic search for single-cell analysis workflows."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(populated_storage)

        # Search for single-cell preprocessing
        results = rag.search_summaries("single cell RNA sequencing preprocessing")

        assert len(results) > 0
        # Should find scanpy preprocessing as top result
        top_result = results[0]
        assert "scanpy_preprocessing" in top_result["notebook_id"]
        assert top_result["similarity_score"] > 0.5

    def test_semantic_search_spatial(self, temp_storage_path, populated_storage):
        """Test semantic search for spatial transcriptomics workflows."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(populated_storage)

        # Search for spatial analysis
        results = rag.search_summaries("spatial transcriptomics tissue visualization")

        assert len(results) > 0
        # Should find spatial analysis as top result
        top_result = results[0]
        assert "spatial_analysis" in top_result["notebook_id"]
        assert top_result["similarity_score"] > 0.5

    def test_semantic_search_tcr(self, temp_storage_path, populated_storage):
        """Test semantic search for TCR analysis workflows."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(populated_storage)

        # Search for TCR analysis
        results = rag.search_summaries("T cell receptor repertoire clonotype")

        assert len(results) > 0
        # Should find TCR analysis as top result
        top_result = results[0]
        assert "tcr_analysis" in top_result["notebook_id"]
        assert top_result["similarity_score"] > 0.5

    def test_search_results_structure(self, temp_storage_path, populated_storage):
        """Test structure of search results."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(populated_storage)

        results = rag.search_summaries("gene expression analysis", n_results=2)

        assert len(results) <= 2
        for result in results:
            # Check required fields
            assert "notebook_id" in result
            assert "summary" in result
            assert "similarity_score" in result
            assert "metadata" in result

            # Check metadata structure (without tool field)
            metadata = result["metadata"]
            assert "title" in metadata
            assert "source_repository" in metadata
            assert "summary_length" in metadata
            assert "notebook_id" in metadata

            # Ensure tool field is not present
            assert "tool" not in metadata

            # Check data types
            assert isinstance(result["similarity_score"], float)
            assert isinstance(metadata["summary_length"], int)

    def test_empty_collection_stats(self, temp_storage_path):
        """Test stats for empty collection."""
        rag = WorkflowSummaryRag(temp_storage_path)

        stats = rag.get_collection_stats()
        assert stats["total_summaries"] == 0
        assert stats["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"

    def test_search_with_no_results(self, temp_storage_path, populated_storage):
        """Test search behavior with very specific query that has no matches."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(populated_storage)

        # Search for very specific unrelated query
        results = rag.search_summaries("quantum physics molecular dynamics simulation")

        # Should still return results (semantic search always returns something)
        # but with lower similarity scores
        assert len(results) > 0
        for result in results:
            assert isinstance(result["similarity_score"], float)


class TestNotebookSelector:
    """Test NotebookSelector workflow selection functionality."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def populated_system(self, temp_storage_path):
        """Create complete workflow summary system with sample data."""
        storage = NotebookStorage(temp_storage_path)

        # Create notebooks with realistic bioinformatics content
        notebooks = [
            {
                "notebook_id": "scanpy_pbmc_3k",
                "summary": "Analysis of 3k PBMCs from 10x Genomics. Quality control with scanpy including doublet detection, normalization with scran, clustering with leiden algorithm, and UMAP visualization.",
                "metadata": {
                    "title": "PBMC 3k analysis",
                    "source_repository": "scverse/scanpy-tutorials"
                }
            },
            {
                "notebook_id": "spatial_brain_analysis",
                "summary": "Spatial transcriptomics of mouse brain tissue using Visium technology. Spatial clustering, gene expression mapping, and tissue architecture analysis with squidpy.",
                "metadata": {
                    "title": "Brain spatial transcriptomics",
                    "source_repository": "scverse/squidpy-tutorials"
                }
            },
            {
                "notebook_id": "scvi_integration",
                "summary": "Integration of multiple single-cell datasets using scVI variational inference. Batch effect removal and joint embedding of datasets from different conditions.",
                "metadata": {
                    "title": "Multi-dataset integration with scVI",
                    "source_repository": "scverse/scvi-tools-tutorials"
                }
            }
        ]

        for notebook in notebooks:
            notebook_data = {
                "notebook_id": notebook["notebook_id"],
                "metadata": notebook["metadata"],
                "cells": [],
                "cell_count": 0
            }
            storage.store_notebook(notebook["notebook_id"], notebook_data, notebook["summary"])

        # Initialize RAG system
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(storage)

        # Initialize selector
        selector = NotebookSelector(storage)

        return storage, rag, selector

    def test_notebook_selector_initialization(self, temp_storage_path):
        """Test NotebookSelector initialization."""
        storage = NotebookStorage(temp_storage_path)
        selector = NotebookSelector(storage)
        assert selector.storage == storage

    def test_get_notebook_content(self, populated_system):
        """Test getting full notebook content."""
        storage, rag, selector = populated_system

        # Get notebook content using the selector's method
        selected_content = selector.get_selected_notebook_content(["scanpy_pbmc_3k"])

        assert selected_content is not None
        assert "scanpy_pbmc_3k" in selected_content
        notebook_data = selected_content["scanpy_pbmc_3k"]
        assert notebook_data["notebook_id"] == "scanpy_pbmc_3k"

    def test_get_multiple_notebooks(self, populated_system):
        """Test retrieving multiple notebooks."""
        storage, rag, selector = populated_system

        notebook_ids = ["scanpy_pbmc_3k", "spatial_brain_analysis"]
        notebooks = selector.get_selected_notebook_content(notebook_ids)

        assert len(notebooks) == 2
        retrieved_ids = set(notebooks.keys())
        assert retrieved_ids == set(notebook_ids)

    def test_get_nonexistent_notebook(self, populated_system):
        """Test handling of nonexistent notebook requests."""
        storage, rag, selector = populated_system

        content = selector.get_selected_notebook_content(["nonexistent_notebook"])
        # Should return empty dict or dict with None value for nonexistent notebook
        assert content == {} or content.get("nonexistent_notebook") is None

    def test_workflow_selection_integration(self, populated_system):
        """Test integration between search and selection."""
        storage, rag, selector = populated_system

        # Search for relevant workflows
        search_results = rag.search_summaries("single cell PBMC clustering", n_results=2)

        # Get top notebook using selector
        top_notebook_id = search_results[0]["notebook_id"]
        notebook_content = selector.get_selected_notebook_content([top_notebook_id])

        assert notebook_content is not None
        assert top_notebook_id in notebook_content
        notebook_data = notebook_content[top_notebook_id]
        assert notebook_data["notebook_id"] == top_notebook_id
        # Should be the PBMC analysis
        assert "pbmc" in notebook_data["notebook_id"].lower()

    def test_create_selection_prompt(self, populated_system):
        """Test creation of LLM selection prompts."""
        storage, rag, selector = populated_system

        # Create selection prompt
        prompt = selector.create_selection_prompt("single cell analysis")

        # Should contain the query
        assert "single cell analysis" in prompt

        # Should contain notebook IDs
        assert "scanpy_pbmc_3k" in prompt
        assert "spatial_brain_analysis" in prompt
        assert "scvi_integration" in prompt


class TestWorkflowSummarySystemIntegration:
    """Integration tests for the complete workflow summary system."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create temporary storage directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_end_to_end_workflow_retrieval(self, temp_storage_path):
        """Test complete end-to-end workflow retrieval process."""
        # Initialize components
        storage = NotebookStorage(temp_storage_path)

        # Add realistic workflow data
        notebook_data = {
            "notebook_id": "comprehensive_scrnaseq_analysis",
            "metadata": {
                "title": "Comprehensive scRNA-seq analysis pipeline",
                "source_repository": "scverse/comprehensive-tutorials"
            },
            "cells": [],
            "cell_count": 0
        }
        storage.store_notebook(
            "comprehensive_scrnaseq_analysis",
            notebook_data,
            "Complete single-cell RNA-seq analysis pipeline from raw count matrix to cell type annotation. Includes quality control, doublet removal, normalization, batch correction, dimensionality reduction, clustering, differential expression, and pathway analysis."
        )

        # Initialize and index
        rag = WorkflowSummaryRag(temp_storage_path)
        indexed_count = rag.index_all_summaries(storage)
        assert indexed_count == 1

        # Search for relevant workflow
        results = rag.search_summaries("single cell analysis quality control clustering")
        assert len(results) > 0

        # Verify result quality
        top_result = results[0]
        assert top_result["similarity_score"] > 0.5  # Good relevance expected
        assert "scrnaseq" in top_result["notebook_id"]

        # Get full notebook content
        selector = NotebookSelector(storage)
        full_content = selector.get_selected_notebook_content([top_result["notebook_id"]])

        assert full_content is not None
        notebook_data = full_content[top_result["notebook_id"]]
        assert notebook_data is not None

    def test_metadata_consistency(self, temp_storage_path):
        """Test metadata consistency across storage and search."""
        storage = NotebookStorage(temp_storage_path)

        # Save notebook with complete metadata
        metadata = {
            "title": "Test workflow analysis",
            "source_repository": "test/repository",
            "custom_field": "custom_value"
        }

        notebook_data = {
            "notebook_id": "test_workflow",
            "metadata": metadata,
            "cells": [],
            "cell_count": 0
        }
        storage.store_notebook(
            "test_workflow",
            notebook_data,
            "Test workflow summary for metadata validation"
        )

        # Index and search
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(storage)

        results = rag.search_summaries("test workflow")
        assert len(results) > 0

        # Check metadata structure in search results
        result_metadata = results[0]["metadata"]
        assert result_metadata["title"] == metadata["title"]
        assert result_metadata["source_repository"] == metadata["source_repository"]
        # Custom fields should be preserved in storage but not necessarily in search metadata

        # Verify storage preserves all metadata
        stored_content = storage.get_notebook_content("test_workflow")
        stored_metadata = stored_content["metadata"]
        assert stored_metadata["custom_field"] == metadata["custom_field"]

    def test_large_scale_search_performance(self, temp_storage_path):
        """Test search performance with multiple notebooks."""
        storage = NotebookStorage(temp_storage_path)

        # Create multiple notebooks
        notebook_count = 20
        for i in range(notebook_count):
            notebook_data = {
                "notebook_id": f"workflow_{i}",
                "metadata": {
                    "title": f"Workflow {i}",
                    "source_repository": f"test/repo_{i % 5}"
                },
                "cells": [],
                "cell_count": 0
            }
            storage.store_notebook(
                f"workflow_{i}",
                notebook_data,
                f"Workflow {i} for testing search performance. Analysis type {i % 3} with different parameters."
            )

        # Index all notebooks
        rag = WorkflowSummaryRag(temp_storage_path)
        indexed_count = rag.index_all_summaries(storage)
        assert indexed_count == notebook_count

        # Test search performance
        import time
        start_time = time.time()
        results = rag.search_summaries("workflow analysis testing", n_results=5)
        search_time = time.time() - start_time

        assert len(results) == 5
        assert search_time < 5.0  # Should be fast

        # Verify all results have proper structure
        for result in results:
            assert "notebook_id" in result
            assert "similarity_score" in result
            assert result["similarity_score"] >= 0.0