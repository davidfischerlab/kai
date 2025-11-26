"""Integration test for scverse_decoupler_rna_sc retrieval issue.

This test verifies the fix for the issue where the query
"decoupler tutorial using scverse_decoupler_rna_sc with the collectri transcription factor database"
was not retrieving the correct workflow due to the ID not being searchable.
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
def real_world_storage(temp_storage_path):
    """Create storage mimicking the real-world scenario."""
    storage = NotebookStorage(temp_storage_path)

    # The actual scverse decoupler workflow that should be found
    scverse_decoupler_data = {
        "metadata": {
            "source_repository": "scverse/decoupler",
            "workflow_filename": "rna_sc.ipynb",
            "title": "Decoupler RNA-seq Single Cell Tutorial"
        },
        "cells": []
    }
    # Real summary from the actual file
    scverse_decoupler_summary = """**Overarching objective** – Apply enrichment‑based inference to single‑cell RNA‑seq data to generate cell‑type, transcription‑factor, and pathway activity scores that support automated annotation and biological interpretation.

**Workflow summary** – Load and preprocess the dataset, retrieve predefined gene sets (markers, GRNs, pathways) via OmniPath, compute enrichment scores with a univariate linear model, store the scores in an AnnData object, rank scores by cluster, and visualize results on UMAPs, violin plots, and matrix heatmaps.

**Tools described** – decoupler (core scoring library), Scanpy/Anndata for data handling, OmniPath resources (PanglaoDB, CollecTRI, PROGENy, Hallmark).

**Expected outputs** – Activity matrices, ranked lists per cluster, annotated UMAPs, heatmaps, and violin distributions of scores.

**Use cases** – cell‑type annotation, TF activity profiling, hypothesis generation."""

    storage.store_notebook("scverse_decoupler_rna_sc", scverse_decoupler_data, scverse_decoupler_summary)

    # Add competing workflows that were ranked higher in the bug
    saezlab_decoupler_data = {
        "metadata": {
            "source_repository": "saezlab/PerMedCoE_summer_school_2023",
            "workflow_filename": "decoupler_bulk_functional_analysis.ipynb",
            "title": "Decoupler Bulk Functional Analysis"
        },
        "cells": []
    }
    saezlab_decoupler_summary = """The workflow tackles the interpretation of bulk RNA‑seq data by inferring pathway and transcription‑factor activities, transforming gene‑level changes into meaningful signals. Conceptually it performs data import, quality filtering, differential expression, enrichment via gene‑set statistics, and activity inference using curated regulatory networks, followed by visual exploration of results. Core tools are Scanpy for data handling, pydeseq2 for differential testing, and Decoupler for network‑based inference, along with standard plotting libraries. It outputs differential tables, volcano plots, activity barplots, and target‑scatter plots that highlight pathway/TF activation patterns. Ideal for researchers analyzing bulk RNA‑seq cohorts who wish to link gene changes to pathway or TF activity."""

    storage.store_notebook("saezlab_permedcoe_summer_school_2023_decoupler_bulk_functional_analysis",
                          saezlab_decoupler_data, saezlab_decoupler_summary)

    # Add theislab TF activity workflow that was actually selected
    theislab_tf_data = {
        "metadata": {
            "source_repository": "theislab/transcription_factor_activity",
            "workflow_filename": "example.ipynb",
            "title": "Transcription Factor Activity Example"
        },
        "cells": []
    }
    theislab_tf_summary = """Example workflow demonstrating transcription factor activity analysis. Shows basic TF inference methods."""

    storage.store_notebook("theislab_transcription_factor_activity_example", theislab_tf_data, theislab_tf_summary)

    return storage


class TestScverseDecouplerRetrieval:
    """Test the real-world bug scenario."""

    def test_original_failing_query(self, temp_storage_path, real_world_storage):
        """Test the exact query that was failing in production.

        Original issue: Query "decoupler tutorial using scverse_decoupler_rna_sc with the collectri..."
        was NOT retrieving scverse_decoupler_rna_sc in top 49 results.

        With dual embedding (content + ID), this should now work.
        """
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(real_world_storage)

        # The exact query from the production bug
        query = "decoupler tutorial using scverse_decoupler_rna_sc with the collectri transcription factor database for TF activity inference in scRNA‑seq"

        results = rag.search_summaries(query, n_results=10)

        # Verify scverse_decoupler_rna_sc is in results
        notebook_ids = [r["notebook_id"] for r in results]
        assert "scverse_decoupler_rna_sc" in notebook_ids, (
            f"scverse_decoupler_rna_sc not found in results. Got: {notebook_ids}"
        )

        # Find its ranking
        scverse_idx = notebook_ids.index("scverse_decoupler_rna_sc")

        # It should rank in top 3 (ideally #1) due to ID match
        assert scverse_idx < 3, f"scverse_decoupler_rna_sc ranked at {scverse_idx}, should be in top 3"

        # Verify it has good ID similarity
        scverse_result = results[scverse_idx]
        assert scverse_result["id_similarity"] > 0.4, (
            f"ID similarity too low: {scverse_result['id_similarity']}"
        )

    def test_semantic_query_without_id(self, temp_storage_path, real_world_storage):
        """Test that semantic search still works when ID is not mentioned."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(real_world_storage)

        # Query with semantic terms only (no ID mention)
        query = "decoupler tutorial with CollecTRI transcription factor database for TF activity in single-cell RNA-seq"

        results = rag.search_summaries(query, n_results=10)

        # Should still find scverse_decoupler_rna_sc through content similarity
        notebook_ids = [r["notebook_id"] for r in results]
        assert "scverse_decoupler_rna_sc" in notebook_ids

        scverse_result = next(r for r in results if r["notebook_id"] == "scverse_decoupler_rna_sc")

        # Content similarity should be primary driver here
        assert scverse_result["content_similarity"] > 0.3

    def test_wrong_id_format_still_works(self, temp_storage_path, real_world_storage):
        """Test that even internal ID format (underscores) helps retrieval."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(real_world_storage)

        # Query using internal ID format as user mentioned it
        query = "use scverse_decoupler_rna_sc for decoupler transcription factor analysis"

        results = rag.search_summaries(query, n_results=10)

        notebook_ids = [r["notebook_id"] for r in results]
        assert "scverse_decoupler_rna_sc" in notebook_ids

        # Should rank well due to partial ID match
        scverse_idx = notebook_ids.index("scverse_decoupler_rna_sc")
        assert scverse_idx < 5, f"Ranked at {scverse_idx}, should be in top 5 for partial ID match"

    def test_full_path_query(self, temp_storage_path, real_world_storage):
        """Test that full path format (as stored) gives best ID match."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(real_world_storage)

        # Query with exact full path as stored
        query = "scverse/decoupler/rna_sc.ipynb tutorial for transcription factors"

        results = rag.search_summaries(query, n_results=10)

        # Should be #1 with very high ID similarity
        top_result = results[0]
        assert "scverse_decoupler_rna_sc" in top_result["notebook_id"]
        assert top_result["id_similarity"] > 0.6, (
            f"Expected high ID similarity for exact path match, got {top_result['id_similarity']}"
        )

    def test_competing_workflows_ranked_correctly(self, temp_storage_path, real_world_storage):
        """Test that when ID is mentioned, correct workflow ranks higher than competitors."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(real_world_storage)

        # Query mentioning scverse specifically
        query = "scverse decoupler tutorial CollecTRI transcription factor single-cell"

        results = rag.search_summaries(query, n_results=10)

        notebook_ids = [r["notebook_id"] for r in results]

        # scverse_decoupler should rank higher than saezlab_decoupler and theislab
        scverse_idx = notebook_ids.index("scverse_decoupler_rna_sc")

        if "saezlab_permedcoe_summer_school_2023_decoupler_bulk_functional_analysis" in notebook_ids:
            saezlab_idx = notebook_ids.index("saezlab_permedcoe_summer_school_2023_decoupler_bulk_functional_analysis")
            assert scverse_idx < saezlab_idx, "scverse should rank higher than saezlab when 'scverse' in query"

        if "theislab_transcription_factor_activity_example" in notebook_ids:
            theislab_idx = notebook_ids.index("theislab_transcription_factor_activity_example")
            assert scverse_idx < theislab_idx, "scverse should rank higher than theislab when 'scverse' in query"

    def test_max_aggregation_improves_ranking(self, temp_storage_path, real_world_storage):
        """Test that max(content_sim, id_sim) gives better ranking than content alone."""
        rag = WorkflowSummaryRag(temp_storage_path)
        rag.index_all_summaries(real_world_storage)

        # Query where ID match is stronger than content match
        query = "scverse/decoupler/rna_sc.ipynb workflow"

        results = rag.search_summaries(query, n_results=10)

        scverse_result = next(r for r in results if r["notebook_id"] == "scverse_decoupler_rna_sc")

        # ID similarity should be higher than content similarity for this query
        assert scverse_result["id_similarity"] > scverse_result["content_similarity"]

        # Final score should use the higher ID similarity
        assert scverse_result["similarity_score"] == scverse_result["id_similarity"]

        # This should give it top ranking
        assert results[0]["notebook_id"] == "scverse_decoupler_rna_sc"
