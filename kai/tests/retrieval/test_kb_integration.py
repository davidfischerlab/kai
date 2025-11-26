"""Test knowledge base integration and status checks."""

import pytest
from kai.config.paths import RETRIEVAL_DIR
from kai.retrieval.snippets.storage.chromadb_manager import ChromaDbManager


def test_knowledge_base_initialization():
    """Test that knowledge base initializes correctly."""
    kb = ChromaDbManager(RETRIEVAL_DIR)

    # Should have loaded collection registry
    assert len(kb.tool_registries) > 0, "Should load tool registries"

    # Should have access to ChromaDB client
    assert kb.client is not None, "Should have ChromaDB client"

    # Should be able to list collections
    collections = list(kb.client.list_collections())
    assert len(collections) > 0, "Should have collections"


def test_knowledge_base_status():
    """Test knowledge base status and health."""
    kb = ChromaDbManager(RETRIEVAL_DIR)

    # Check that we have expected collection types
    collections = list(kb.client.list_collections())
    collection_names = [c.name for c in collections]

    api_collections = [name for name in collection_names if 'api' in name.lower()]
    workflow_collections = [name for name in collection_names if 'workflows' in name.lower()]

    assert len(api_collections) > 0, f"Should have API collections. Found: {len(api_collections)}"
    assert len(workflow_collections) > 0, f"Should have workflow collections. Found: {len(workflow_collections)}"

    # Check that collections have content
    sample_collections = collections[:5]
    for collection in sample_collections:
        coll_obj = kb.client.get_collection(collection.name)
        count = coll_obj.count()
        assert count > 0, f"Collection {collection.name} should have documents"


@pytest.mark.asyncio
async def test_knowledge_base_search_functionality():
    """Test core search functionality."""
    kb = ChromaDbManager(RETRIEVAL_DIR)

    # Test basic search
    results = await kb.search("scanpy", n_results=5)

    assert results is not None
    assert "query" in results
    assert "content" in results
    assert "tools" in results
    assert len(results["tools"]) > 0, "Should find scanpy-related tools"


@pytest.mark.asyncio
async def test_knowledge_base_collection_search():
    """Test collection-specific search."""
    kb = ChromaDbManager(RETRIEVAL_DIR)

    # Get some collections to test with
    collections = list(kb.client.list_collections())
    test_collections = [c.name for c in collections[:3]]

    if test_collections:
        results = await kb.search_selected_collections(
            "analysis", test_collections, n_results=3
        )

        assert results is not None
        assert len(results) >= 0  # May be 0 if no matches in selected collections


def test_collection_registry_consistency():
    """Test that collection registry is consistent with actual collections."""
    kb = ChromaDbManager(RETRIEVAL_DIR)

    # Get actual collections
    actual_collections = set(c.name for c in kb.client.list_collections())

    # Get registry collections
    registry_collections = set(kb.tool_registries.keys())

    # Registry should not have more collections than actually exist
    extra_in_registry = registry_collections - actual_collections
    assert len(extra_in_registry) == 0, f"Registry has extra collections: {extra_in_registry}"

    # Most actual collections should be in registry (allow some missing for rebuilds)
    missing_from_registry = actual_collections - registry_collections
    coverage = (len(actual_collections) - len(missing_from_registry)) / len(actual_collections)
    assert coverage > 0.8, f"Registry coverage too low: {coverage:.2f}. Missing: {missing_from_registry}"