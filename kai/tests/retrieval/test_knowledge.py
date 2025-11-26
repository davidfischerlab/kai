"""Test knowledge base functionality with real data."""

from kai.retrieval.snippets.storage.chromadb_manager import ChromaDbManager
from kai.config.paths import RETRIEVAL_DIR
import pytest

@pytest.mark.asyncio
async def test_knowledge_search_real():
    kb = ChromaDbManager(RETRIEVAL_DIR)
    # Test search functionality
    results = await kb.search("Scanpy", n_results=5)
    assert isinstance(results, dict)
    assert "query" in results
    assert "tools" in results  # Changed from "detected_tools" to match actual API
    assert "content" in results  # ChromaDbManager returns content directly
    # Verify we got some results
    assert len(results["tools"]) > 0, "No tools found in search results"
    assert results["content"], "No content found in search results" 