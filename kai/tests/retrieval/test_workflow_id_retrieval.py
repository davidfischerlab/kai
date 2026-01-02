"""Integration test for workflow ID retrieval.

This test verifies that workflows can be retrieved by ID mentions in queries,
testing the dual embedding (content + ID) search behavior.
"""

import pytest
from pathlib import Path
import tempfile
import shutil

from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag


@pytest.fixture
def temp_storage_path():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def workflow_storage(temp_storage_path):
    """Create storage with test workflows."""
    storage = NotebookStorage(temp_storage_path)

    # Primary workflow that should be found by ID
    primary_data = {
        "metadata": {
            "source_repository": "org_a/analysis_tools",
            "workflow_filename": "rna_sc.ipynb",
            "title": "RNA-seq Single Cell Tutorial"
        },
        "cells": []
    }
    primary_summary = """**Overarching objective** – Apply enrichment-based inference to single-cell RNA-seq data to generate cell-type, transcription-factor, and pathway activity scores that support automated annotation and biological interpretation.

**Workflow summary** – Load and preprocess the dataset, retrieve predefined gene sets (markers, GRNs, pathways), compute enrichment scores with a univariate linear model, store the scores in an AnnData object, rank scores by cluster, and visualize results on UMAPs, violin plots, and matrix heatmaps.

**Tools described** – Analysis library (core scoring library), Scanpy/Anndata for data handling, database resources (marker databases, transcription factor networks, pathways).

**Expected outputs** – Activity matrices, ranked lists per cluster, annotated UMAPs, heatmaps, and violin distributions of scores.

**Use cases** – cell-type annotation, TF activity profiling, hypothesis generation."""

    storage.store_notebook("org_a_analysis_tools_rna_sc", primary_data, primary_summary)

    # Competing workflow with similar content
    secondary_data = {
        "metadata": {
            "source_repository": "org_b/workshop_2023",
            "workflow_filename": "bulk_functional_analysis.ipynb",
            "title": "Bulk Functional Analysis"
        },
        "cells": []
    }
    secondary_summary = """The workflow tackles the interpretation of bulk RNA-seq data by inferring pathway and transcription-factor activities, transforming gene-level changes into meaningful signals. Conceptually it performs data import, quality filtering, differential expression, enrichment via gene-set statistics, and activity inference using curated regulatory networks, followed by visual exploration of results. Core tools are Scanpy for data handling, pydeseq2 for differential testing, and analysis libraries for network-based inference, along with standard plotting libraries. It outputs differential tables, volcano plots, activity barplots, and target-scatter plots that highlight pathway/TF activation patterns. Ideal for researchers analyzing bulk RNA-seq cohorts who wish to link gene changes to pathway or TF activity."""

    storage.store_notebook("org_b_workshop_2023_bulk_functional_analysis",
                          secondary_data, secondary_summary)

    # Third competing workflow
    tertiary_data = {
        "metadata": {
            "source_repository": "org_c/tf_activity",
            "workflow_filename": "example.ipynb",
            "title": "Transcription Factor Activity Example"
        },
        "cells": []
    }
    tertiary_summary = """Example workflow demonstrating transcription factor activity analysis. Shows basic TF inference methods."""

    storage.store_notebook("org_c_tf_activity_example", tertiary_data, tertiary_summary)

    return storage


class TestWorkflowIdRetrieval:
    """Test workflow retrieval by ID."""

    def test_query_with_workflow_id(self, temp_storage_path, workflow_storage):
        """Test that queries mentioning workflow ID retrieve the correct workflow.

        With dual embedding (content + ID), ID mentions should boost relevance.
        """
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(workflow_storage)

        # Query mentioning the workflow ID
        query = "tutorial using org_a_analysis_tools_rna_sc with transcription factor database for TF activity inference in scRNA-seq"

        results = rag.search_summaries(query, n_results=10)

        # Verify target workflow is in results
        notebook_ids = [r["notebook_id"] for r in results]
        assert "org_a_analysis_tools_rna_sc" in notebook_ids, (
            f"org_a_analysis_tools_rna_sc not found in results. Got: {notebook_ids}"
        )

        # Find its ranking
        target_idx = notebook_ids.index("org_a_analysis_tools_rna_sc")

        # It should rank in top 3 (ideally #1) due to ID match
        assert target_idx < 3, f"org_a_analysis_tools_rna_sc ranked at {target_idx}, should be in top 3"

        # Verify it has good ID similarity
        target_result = results[target_idx]
        assert target_result["id_similarity"] > 0.4, (
            f"ID similarity too low: {target_result['id_similarity']}"
        )

    def test_semantic_query_without_id(self, temp_storage_path, workflow_storage):
        """Test that semantic search still works when ID is not mentioned."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(workflow_storage)

        # Query with semantic terms only (no ID mention)
        query = "tutorial with transcription factor database for TF activity in single-cell RNA-seq"

        results = rag.search_summaries(query, n_results=10)

        # Should still find target through content similarity
        notebook_ids = [r["notebook_id"] for r in results]
        assert "org_a_analysis_tools_rna_sc" in notebook_ids

        target_result = next(r for r in results if r["notebook_id"] == "org_a_analysis_tools_rna_sc")

        # Content similarity should be primary driver here
        assert target_result["content_similarity"] > 0.3

    def test_internal_id_format_helps_retrieval(self, temp_storage_path, workflow_storage):
        """Test that internal ID format (underscores) helps retrieval."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(workflow_storage)

        # Query using internal ID format
        query = "use org_a_analysis_tools_rna_sc for transcription factor analysis"

        results = rag.search_summaries(query, n_results=10)

        notebook_ids = [r["notebook_id"] for r in results]
        assert "org_a_analysis_tools_rna_sc" in notebook_ids

        # Should rank well due to partial ID match
        target_idx = notebook_ids.index("org_a_analysis_tools_rna_sc")
        assert target_idx < 5, f"Ranked at {target_idx}, should be in top 5 for partial ID match"

    def test_full_path_query(self, temp_storage_path, workflow_storage):
        """Test that full path format (as stored) gives best ID match."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(workflow_storage)

        # Query with exact full path as stored
        query = "org_a/analysis_tools/rna_sc.ipynb tutorial for transcription factors"

        results = rag.search_summaries(query, n_results=10)

        # Should be #1 with very high ID similarity
        top_result = results[0]
        assert "org_a_analysis_tools_rna_sc" in top_result["notebook_id"]
        assert top_result["id_similarity"] > 0.6, (
            f"Expected high ID similarity for exact path match, got {top_result['id_similarity']}"
        )

    def test_id_mention_helps_retrieval(self, temp_storage_path, workflow_storage):
        """Test that mentioning workflow ID helps retrieval."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(workflow_storage)

        # Query mentioning the specific workflow ID
        query = "org_a_analysis_tools_rna_sc tutorial transcription factor single-cell"

        results = rag.search_summaries(query, n_results=10)

        notebook_ids = [r["notebook_id"] for r in results]

        # Target workflow should be found
        assert "org_a_analysis_tools_rna_sc" in notebook_ids, (
            "Workflow should be found when ID is mentioned in query"
        )

        # Should be in top results due to ID match
        target_idx = notebook_ids.index("org_a_analysis_tools_rna_sc")
        assert target_idx < 3, f"Workflow should be in top 3 when ID mentioned, got position {target_idx}"

    def test_max_aggregation_improves_ranking(self, temp_storage_path, workflow_storage):
        """Test that max(content_sim, id_sim) gives better ranking than content alone."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(workflow_storage)

        # Query where ID match is stronger than content match
        query = "org_a/analysis_tools/rna_sc.ipynb workflow"

        results = rag.search_summaries(query, n_results=10)

        target_result = next(r for r in results if r["notebook_id"] == "org_a_analysis_tools_rna_sc")

        # ID similarity should be higher than content similarity for this query
        assert target_result["id_similarity"] > target_result["content_similarity"]

        # Final score should use the higher ID similarity
        assert target_result["similarity_score"] == target_result["id_similarity"]

        # This should give it top ranking
        assert results[0]["notebook_id"] == "org_a_analysis_tools_rna_sc"
