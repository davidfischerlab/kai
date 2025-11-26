"""Pure retrieval integration tests for both workflow and API functionality."""

import pytest
from kai.config.paths import RETRIEVAL_DIR
from kai.retrieval.snippets.storage.chromadb_manager import ChromaDbManager


@pytest.fixture(scope="session")
def knowledge_base():
    """Create a knowledge base instance for testing."""
    return ChromaDbManager(RETRIEVAL_DIR)


class TestRetrievalIntegration:
    """Test integration between API and workflow retrieval."""

    @pytest.mark.asyncio
    async def test_combined_api_workflow_search(self, knowledge_base):
        """Test searching across both API docs and workflows."""
        results = await knowledge_base.search("scanpy quality control filter", n_results=10)

        assert results is not None
        assert results["content"], "Should return combined content"
        assert len(results["tools"]) > 0, "Should find multiple tool types"

        # Should find relevant bioinformatics tools
        tools = results["tools"]
        has_scanpy = any("scanpy" in tool.lower() for tool in tools)
        has_bio_tools = any(term in tool.lower() for term in ["scanpy", "anndata", "squidpy", "cellrank"] for tool in tools)

        # At least some relevant tools should be present
        assert has_scanpy or has_bio_tools, f"Should find bioinformatics tools: {tools}"

    @pytest.mark.asyncio
    async def test_context_aware_search(self, knowledge_base):
        """Test that search results are contextually relevant."""
        # Search for a specific bioinformatics workflow
        results = await knowledge_base.search("single cell RNA seq preprocessing", n_results=5)

        assert results is not None
        content_lower = results["content"].lower()

        # Should contain relevant single-cell terms
        relevant_terms = ["single", "cell", "rna", "preprocess", "scanpy", "anndata"]
        found_terms = [term for term in relevant_terms if term in content_lower]

        assert len(found_terms) >= 2, f"Should find relevant terms. Found: {found_terms}"

    @pytest.mark.asyncio
    async def test_search_result_quality(self, knowledge_base):
        """Test that search results have good quality and relevance."""
        results = await knowledge_base.search("differential expression analysis", n_results=5)

        assert results is not None
        assert results["content"], "Should return content"
        assert len(results["tools"]) > 0, "Should return tools"

        # Content should be substantial
        assert len(results["content"]) > 100, "Content should be substantial"

        # Tools should be unique
        tools = results["tools"]
        assert len(tools) == len(set(tools)), "Tools should be unique"

    @pytest.mark.asyncio
    async def test_multi_term_search_accuracy(self, knowledge_base):
        """Test search accuracy with multiple terms."""
        results = await knowledge_base.search("leiden clustering resolution parameter", n_results=5)

        assert results is not None
        content_lower = results["content"].lower()

        # Should find clustering-related content
        clustering_terms = ["leiden", "clustering", "resolution"]
        found_clustering = [term for term in clustering_terms if term in content_lower]

        assert len(found_clustering) >= 1, f"Should find clustering terms. Found: {found_clustering}"


class TestRetrievalPerformance:
    """Test retrieval performance characteristics."""

    @pytest.mark.asyncio
    async def test_search_response_time(self, knowledge_base):
        """Test that search completes within reasonable time."""
        import time

        queries = [
            "scanpy preprocessing",
            "anndata basics",
            "quality control filtering",
            "UMAP visualization",
            "differential expression"
        ]

        for query in queries:
            start = time.time()
            results = await knowledge_base.search(query, n_results=5)
            duration = time.time() - start

            assert results is not None, f"Query '{query}' should return results"
            assert duration < 5.0, f"Query '{query}' took {duration:.2f}s, should be under 5s"

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, knowledge_base):
        """Test that concurrent searches work correctly."""
        import asyncio

        queries = [
            "scanpy filter_cells",
            "anndata read_h5ad",
            "clustering leiden",
            "trajectory analysis"
        ]

        # Run searches concurrently
        tasks = [knowledge_base.search(query, n_results=3) for query in queries]
        results = await asyncio.gather(*tasks)

        # All searches should succeed
        for i, result in enumerate(results):
            assert result is not None, f"Query {i} should return results"
            assert result["content"], f"Query {i} should return content"

    @pytest.mark.asyncio
    async def test_search_consistency(self, knowledge_base):
        """Test that repeated searches return consistent results."""
        query = "scanpy preprocessing"

        # Run the same search multiple times
        results = []
        for _ in range(3):
            result = await knowledge_base.search(query, n_results=5)
            results.append(result)

        # Results should be consistent
        first_tools = set(results[0]["tools"])
        for i, result in enumerate(results[1:], 1):
            current_tools = set(result["tools"])
            overlap = len(first_tools & current_tools) / len(first_tools | current_tools)
            assert overlap > 0.8, f"Search {i} should have >80% tool overlap with first search"


class TestCollectionSpecificRetrieval:
    """Test retrieval from specific collection types."""

    @pytest.mark.asyncio
    async def test_api_specific_search(self, knowledge_base):
        """Test searching specifically in API collections."""
        # Use selected collections that are API-focused
        collections = list(knowledge_base.client.list_collections())
        api_collections = [c.name for c in collections if "api" in c.name.lower()][:5]

        if not api_collections:
            pytest.skip("No API collections found")

        results = await knowledge_base.search_selected_collections(
            "filter_cells parameters", api_collections, n_results=3
        )

        assert results is not None
        assert len(results) > 0, "Should find API-specific results"

    @pytest.mark.asyncio
    async def test_workflow_specific_search(self, knowledge_base):
        """Test searching specifically in workflow collections."""
        collections = list(knowledge_base.client.list_collections())
        workflow_collections = [c.name for c in collections if "workflow" in c.name.lower()][:5]

        if not workflow_collections:
            pytest.skip("No workflow collections found")

        results = await knowledge_base.search_selected_collections(
            "quality control steps", workflow_collections, n_results=3
        )

        assert results is not None
        assert len(results) > 0, "Should find workflow-specific results"

    @pytest.mark.asyncio
    async def test_organization_specific_search(self, knowledge_base):
        """Test searching by organization."""
        collections = list(knowledge_base.client.list_collections())
        scverse_collections = [c.name for c in collections if c.name.startswith("scverse_")][:5]

        if not scverse_collections:
            pytest.skip("No scverse collections found")

        results = await knowledge_base.search_selected_collections(
            "scanpy tutorial", scverse_collections, n_results=5
        )

        assert results is not None
        assert len(results) > 0, "Should find organization-specific results"