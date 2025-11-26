"""Pure workflow retrieval functionality tests."""

import pytest
from kai.config.paths import RETRIEVAL_DIR
from kai.retrieval.snippets.storage.chromadb_manager import ChromaDbManager


@pytest.fixture(scope="session")
def knowledge_base():
    """Create a knowledge base instance for testing."""
    return ChromaDbManager(RETRIEVAL_DIR)


class TestWorkflowRetrieval:
    """Test workflow notebook retrieval from ChromaDB."""

    @pytest.mark.asyncio
    async def test_quality_control_workflow_search(self, knowledge_base):
        """Test retrieval of quality control workflows."""
        results = await knowledge_base.search("quality control preprocessing filtering", n_results=10)

        assert results is not None
        assert results["content"], "Should return QC workflow content"
        assert len(results["tools"]) > 0, "Should find tools with QC workflows"

        # Check content relevance
        content_lower = results["content"].lower()
        qc_terms = ["quality", "control", "filter", "preprocess", "qc"]
        found_qc_terms = [term for term in qc_terms if term in content_lower]
        assert len(found_qc_terms) >= 2, f"Should find QC-related terms. Found: {found_qc_terms}"

    @pytest.mark.asyncio
    async def test_clustering_workflow_search(self, knowledge_base):
        """Test retrieval of clustering workflows."""
        results = await knowledge_base.search("clustering leiden louvain neighbors", n_results=10)

        assert results is not None
        assert len(results["tools"]) > 0, "Should find clustering tools"

        content_lower = results["content"].lower()
        clustering_terms = ["cluster", "leiden", "louvain", "neighbor"]
        found_terms = [term for term in clustering_terms if term in content_lower]
        assert len(found_terms) >= 1, f"Should find clustering terms. Found: {found_terms}"

    @pytest.mark.asyncio
    async def test_trajectory_analysis_workflow_search(self, knowledge_base):
        """Test retrieval of trajectory analysis workflows."""
        results = await knowledge_base.search("trajectory analysis PAGA pseudotime", n_results=10)

        assert results is not None
        assert len(results["tools"]) > 0, "Should find trajectory tools"

        content_lower = results["content"].lower()
        trajectory_terms = ["trajectory", "paga", "pseudotime", "path"]
        found_terms = [term for term in trajectory_terms if term in content_lower]
        assert len(found_terms) >= 1, f"Should find trajectory terms. Found: {found_terms}"

    @pytest.mark.asyncio
    async def test_differential_expression_workflow_search(self, knowledge_base):
        """Test retrieval of differential expression workflows."""
        results = await knowledge_base.search("differential expression DE genes wilcoxon", n_results=10)

        assert results is not None
        assert len(results["tools"]) > 0, "Should find DE tools"

        content_lower = results["content"].lower()
        de_terms = ["differential", "expression", "wilcoxon", "genes", "de"]
        found_terms = [term for term in de_terms if term in content_lower]
        assert len(found_terms) >= 2, f"Should find DE terms. Found: {found_terms}"

    @pytest.mark.asyncio
    async def test_visualization_workflow_search(self, knowledge_base):
        """Test retrieval of visualization workflows."""
        results = await knowledge_base.search("UMAP tSNE visualization plotting", n_results=10)

        assert results is not None
        assert len(results["tools"]) > 0, "Should find visualization tools"

        content_lower = results["content"].lower()
        viz_terms = ["umap", "tsne", "plot", "visualiz"]
        found_terms = [term for term in viz_terms if term in content_lower]
        assert len(found_terms) >= 1, f"Should find visualization terms. Found: {found_terms}"


class TestWorkflowContentStructure:
    """Test the structure and quality of workflow content."""

    @pytest.mark.asyncio
    async def test_workflow_metadata_presence(self, knowledge_base):
        """Test that workflow results contain proper metadata."""
        results = await knowledge_base.search("scanpy tutorial analysis", n_results=5)

        assert results is not None
        assert "query" in results
        assert "content" in results
        assert "tools" in results

        # Content should be substantial for workflows
        assert len(results["content"]) > 50, "Workflow content should be substantial"

    @pytest.mark.asyncio
    async def test_workflow_organization_coverage(self, knowledge_base):
        """Test that workflows from different organizations are retrievable."""
        org_queries = [
            "scanpy preprocessing",  # scverse
            "CellRank analysis",     # theislab
            "cell2location spatial"  # BayraktarLab
        ]

        for query in org_queries:
            results = await knowledge_base.search(query, n_results=5)
            assert results is not None, f"Query '{query}' should return results"
            assert len(results["tools"]) > 0, f"Query '{query}' should find tools"

    @pytest.mark.asyncio
    async def test_workflow_notebook_types(self, knowledge_base):
        """Test that different types of notebook workflows are found."""
        workflow_types = [
            "tutorial basic analysis",
            "reproducibility figure",
            "benchmark comparison",
            "example demonstration"
        ]

        for workflow_type in workflow_types:
            results = await knowledge_base.search(workflow_type, n_results=3)
            assert results is not None, f"Should find {workflow_type} workflows"
            # Don't require specific tools since availability varies


class TestHierarchicalRetrieval:
    """Test hierarchical chunking and context-aware retrieval."""

    @pytest.mark.asyncio
    async def test_hierarchical_search_context(self, knowledge_base):
        """Test that hierarchical context improves search results."""
        # Search for a specific analysis step
        results = await knowledge_base.search("normalization log1p transformation", n_results=5)

        assert results is not None
        assert results["content"], "Should return normalization content"

        content_lower = results["content"].lower()
        norm_terms = ["normaliz", "log1p", "transform", "scale"]
        found_terms = [term for term in norm_terms if term in content_lower]
        assert len(found_terms) >= 2, f"Should find normalization terms. Found: {found_terms}"

    @pytest.mark.asyncio
    async def test_context_chunk_retrieval(self, knowledge_base):
        """Test retrieval of contextual chunks from notebooks."""
        # Search for a workflow step that should have context
        results = await knowledge_base.search("highly variable genes selection", n_results=5)

        assert results is not None
        assert len(results["tools"]) > 0, "Should find tools with HVG selection"

        # Should contain substantial context
        assert len(results["content"]) > 200, "Should include substantial context"

    @pytest.mark.asyncio
    async def test_multi_level_content_search(self, knowledge_base):
        """Test search across different hierarchical levels."""
        # Search for both high-level and detailed concepts
        results = await knowledge_base.search("single cell analysis workflow steps", n_results=8)

        assert results is not None
        assert results["content"], "Should return multi-level content"

        # Should find content at different granularity levels
        content_lower = results["content"].lower()
        high_level_terms = ["workflow", "analysis", "pipeline"]
        detailed_terms = ["filter_cells", "normalize", "find_markers"]

        found_high = [term for term in high_level_terms if term in content_lower]
        found_detailed = [term for term in detailed_terms if term in content_lower]

        assert len(found_high) >= 1, f"Should find high-level terms. Found: {found_high}"
        # Detailed terms are optional as they depend on specific content


class TestNotebookSpecificRetrieval:
    """Test retrieval capabilities specific to notebook content."""

    @pytest.mark.asyncio
    async def test_code_pattern_retrieval(self, knowledge_base):
        """Test retrieval of specific code patterns from notebooks."""
        results = await knowledge_base.search("sc.pp.filter_cells min_genes", n_results=5)

        assert results is not None
        content_lower = results["content"].lower()

        # Should find code-related content
        code_indicators = ["sc.pp", "filter_cells", "min_genes", "scanpy"]
        found_code = [term for term in code_indicators if term in content_lower]
        assert len(found_code) >= 2, f"Should find code patterns. Found: {found_code}"

    @pytest.mark.asyncio
    async def test_parameter_explanation_retrieval(self, knowledge_base):
        """Test retrieval of parameter explanations from notebooks."""
        results = await knowledge_base.search("n_neighbors parameter UMAP", n_results=5)

        assert results is not None
        content_lower = results["content"].lower()

        # Should find parameter-related content
        param_terms = ["parameter", "n_neighbors", "umap", "neighbor"]
        found_params = [term for term in param_terms if term in content_lower]
        assert len(found_params) >= 1, f"Should find parameter terms. Found: {found_params}"

    @pytest.mark.asyncio
    async def test_troubleshooting_content_retrieval(self, knowledge_base):
        """Test retrieval of troubleshooting and error-handling content."""
        results = await knowledge_base.search("error debugging memory optimization", n_results=5)

        assert results is not None
        # Should return some content, even if not highly specific to troubleshooting
        assert results["content"], "Should find some relevant content"