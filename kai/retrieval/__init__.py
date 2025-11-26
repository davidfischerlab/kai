"""Retrieval module for RAG-based bioinformatics knowledge extraction and indexing.

This module provides functionality for:
- Extracting documentation from GitHub repositories
- Crawling ReadTheDocs sites
- Hierarchical chunking of API documentation and workflows
- Indexing into ChromaDB for semantic search
- Version-aware knowledge base management

Key components:
- extractors: GitHub, ReadTheDocs, and hierarchical parsers
- storage: ChromaDB manager with integrated indexing
"""

from .snippets.extractors import (
    GitHubDocumentationExtractor,
    ReadTheDocsCrawler,
    HierarchicalWorkflowParser,
)

from .snippets.storage import (
    ChromaDbManager,
)

__all__ = [
    "GitHubDocumentationExtractor",
    "ReadTheDocsCrawler", 
    "HierarchicalWorkflowParser",
    "ChromaDbManager",
]

# Convenience function for quick setup
def create_knowledge_base(knowledge_path, settings=None):
    """Create a version-aware knowledge base with default settings.
    
    Args:
        knowledge_path: Path to store knowledge bases
        settings: Optional settings object
        
    Returns:
        ChromaDbManager instance
    """
    return ChromaDbManager(knowledge_path, settings)
