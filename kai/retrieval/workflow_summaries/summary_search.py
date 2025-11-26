"""Fast semantic search on notebook summaries using ChromaDB."""

import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio

from kai.utils import setup_logger

logger = setup_logger(__name__)


class WorkflowSummaryRag:
    """ChromaDB-based semantic search on notebook summaries."""

    def __init__(self, storage_path: Path, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize summary search index.

        Args:
            storage_path: Path to notebook storage directory
            embedding_model: Sentence transformer model for embeddings
        """
        self.storage_path = storage_path
        self.embedding_model = embedding_model

        # Initialize notebook storage for fetching summaries
        from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
        self.notebook_storage = NotebookStorage(storage_path)

        # Initialize ChromaDB client
        chroma_path = storage_path / "summary_index"
        chroma_path.mkdir(exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(chroma_path))
        # Collection for summary content embeddings
        self.collection = self.client.get_or_create_collection(
            name="notebook_summaries",
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for text
        )
        # Collection for notebook ID embeddings (for direct ID queries)
        self.id_collection = self.client.get_or_create_collection(
            name="notebook_ids",
            metadata={"hnsw:space": "cosine"}
        )

    def index_all_summaries(self, notebook_storage) -> int:
        """Index all existing summaries into ChromaDB.
        Creates two embeddings per notebook:
        1. Summary content embedding (semantic search)
        2. Full notebook ID embedding (for direct ID queries)

        Args:
            notebook_storage: NotebookStorage instance

        Returns:
            Number of summaries indexed
        """
        logger.info("Indexing all summaries for semantic search...")

        # Store reference for later summary lookups
        self.notebook_storage = notebook_storage

        # Get all summaries
        summaries = notebook_storage.get_all_summaries()

        if not summaries:
            logger.warning("No summaries found to index")
            return 0

        # Prepare documents for ChromaDB
        summary_documents = []
        id_documents = []
        metadatas = []
        ids = []

        for notebook_id, summary_text in summaries.items():
            # Get notebook metadata for enhanced search context
            notebook_data = notebook_storage.get_notebook_content(notebook_id)
            metadata = notebook_data.get("metadata", {}) if notebook_data else {}

            # Construct full notebook ID in format: org/repo/filename.ipynb
            full_notebook_id = f"{metadata.get('source_repository', 'unknown')}/{metadata.get('workflow_filename', notebook_id)}"

            # Summary content for semantic search
            summary_documents.append(summary_text)
            # Full notebook ID as searchable text (e.g., "scverse/decoupler/rna_sc.ipynb")
            id_documents.append(full_notebook_id)

            meta = {
                "notebook_id": notebook_id,  # Keep internal ID for lookups
                "full_notebook_id": full_notebook_id,  # Add human-readable ID
                "source_repository": metadata.get("source_repository", "unknown"),
                "title": metadata.get("title", "Untitled"),
                "summary_length": len(summary_text)
            }
            metadatas.append(meta)
            ids.append(notebook_id)

        # Add to ChromaDB collections in batches to avoid size limits
        batch_size = 5000
        total_indexed = 0

        for i in range(0, len(summary_documents), batch_size):
            batch_summary_docs = summary_documents[i:i + batch_size]
            batch_id_docs = id_documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            # Index summary content
            self.collection.add(
                documents=batch_summary_docs,
                metadatas=batch_meta,
                ids=batch_ids
            )

            # Index notebook IDs with suffix to avoid ID collision
            self.id_collection.add(
                documents=batch_id_docs,
                metadatas=batch_meta,
                ids=[f"{id}_id" for id in batch_ids]
            )

            total_indexed += len(batch_summary_docs)
            logger.info(f"Indexed batch {i//batch_size + 1}: {len(batch_summary_docs)} summaries + IDs (total: {total_indexed}/{len(summaries)})")

        logger.info(f"Indexed {len(summaries)} summaries and IDs for semantic search")
        return len(summaries)

    def search_summaries(self, query: str, n_results: int = 20) -> List[Dict[str, Any]]:
        """Fast semantic search on summaries.
        Queries both summary content and notebook IDs, then merges results by taking
        the maximum similarity score per notebook.

        Args:
            query: User query for notebook search
            n_results: Number of top results to return

        Returns:
            List of search results with notebook_id, summary, and metadata
        """
        # Query both collections - get more results initially for merging
        n_query = n_results * 2

        # Search summary content
        summary_results = self.collection.query(
            query_texts=[query],
            n_results=n_query
        )

        # Search notebook IDs
        id_results = self.id_collection.query(
            query_texts=[query],
            n_results=n_query
        )

        # Merge results by notebook_id, taking maximum similarity
        notebook_scores = {}

        # Process summary results
        if summary_results["documents"] and summary_results["documents"][0]:
            for doc, metadata, distance in zip(
                summary_results["documents"][0],
                summary_results["metadatas"][0],
                summary_results["distances"][0]
            ):
                notebook_id = metadata["notebook_id"]
                similarity = 1 - distance
                notebook_scores[notebook_id] = {
                    "notebook_id": notebook_id,
                    "summary": doc,
                    "metadata": metadata,
                    "similarity_score": similarity,
                    "content_similarity": similarity,
                    "id_similarity": 0.0
                }

        # Process ID results - update with max similarity
        if id_results["metadatas"] and id_results["metadatas"][0]:
            for metadata, distance in zip(
                id_results["metadatas"][0],
                id_results["distances"][0]
            ):
                notebook_id = metadata["notebook_id"]
                id_similarity = 1 - distance

                if notebook_id in notebook_scores:
                    # Take maximum of content and ID similarity
                    notebook_scores[notebook_id]["id_similarity"] = id_similarity
                    notebook_scores[notebook_id]["similarity_score"] = max(
                        notebook_scores[notebook_id]["content_similarity"],
                        id_similarity
                    )
                else:
                    # New notebook found via ID search - fetch its summary
                    summary = ""
                    if self.notebook_storage:
                        summary = self.notebook_storage.get_summary(notebook_id) or ""

                    notebook_scores[notebook_id] = {
                        "notebook_id": notebook_id,
                        "summary": summary,
                        "metadata": metadata,
                        "similarity_score": id_similarity,
                        "content_similarity": 0.0,
                        "id_similarity": id_similarity
                    }

        # Sort by similarity score and take top n_results
        search_results = sorted(
            notebook_scores.values(),
            key=lambda x: x["similarity_score"],
            reverse=True
        )[:n_results]

        # Add rank
        for i, result in enumerate(search_results):
            result["rank"] = i + 1

        logger.info(f"Found {len(search_results)} summaries matching query: '{query}'")
        return search_results

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed summaries."""
        count = self.collection.count()

        return {
            "total_summaries": count,
            "embedding_model": self.embedding_model
        }

