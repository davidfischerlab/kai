"""ChromaDB manager for bioinformatics tools and workflows with integrated indexing."""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import importlib.metadata
import json
import numpy as np
import re
import os

# Disable ChromaDB telemetry to prevent file descriptor issues
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_CLIENT_TELEMETRY_ENABLED'] = 'False'
os.environ['CHROMA_SERVER_TELEMETRY_ENABLED'] = 'False'
os.environ['CHROMA_TELEMETRY'] = 'false'
os.environ['CHROMA_TELEMETRY_ENABLED'] = 'false'
os.environ['CHROMA_DISABLE_TELEMETRY'] = 'true'

import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from kai.config.paths import BIOINFORMATICS_CACHE_DIR, get_org_api_cache_file
from kai.config.rag_config import ORGANIZATION_SCORES
from kai.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolKnowledgeBase:
    """Represents a tool-specific knowledge base."""
    tool_name: str
    version: str
    created_at: datetime
    last_updated: datetime
    collection_name: str
    document_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        data["last_updated"] = self.last_updated.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolKnowledgeBase":
        """Create from dictionary."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class ChromaDbManager:
    """
    ChromaDB Manager with Version Management and Integrated JSON Indexing.
    
    This class provides a unified interface for managing bioinformatics knowledge bases
    using ChromaDB as the underlying vector storage. It serves as the central hub for
    document storage, semantic search, and JSON indexing from extracted documentation.
    
    Architecture:
    ============
    
    1. Version Management:
       - Tracks tool versions and manages knowledge base updates automatically
       - Creates version-specific collections (e.g., "scanpy_v1_11_3")  
       - Handles version compatibility checking and migration
       - Maintains persistent registry of tool versions and metadata
    
    2. ChromaDB Integration:
       - Uses ChromaDB's native SentenceTransformerEmbeddingFunction (all-MiniLM-L6-v2)
       - Automatic embedding generation for documents and queries
       - Persistent storage with hierarchical organization
       - Supports multiple collections for different tools and versions
    
    3. Direct Indexing Pipeline:
       - Processes extraction results directly from GitHubDocumentationExtractor
       - Converts structured extraction results into searchable documents
       - Supports both flat and hierarchical document chunking
       - Handles multiple document types (functions, classes, workflows, examples)
    
    4. Multi-Modal Search:
       - Tool-specific search with automatic detection
       - Hierarchical context-aware retrieval
       - Metadata-filtered searches
       - Optimal chunk-level selection for different query types
    
    Storage Structure:
    =================
    
    File Organization:
    - knowledge_path/
      ├── chromadb/           # ChromaDB persistent storage
      │   ├── chroma.sqlite3  # ChromaDB database
      │   └── collections/    # Collection-specific data
      └── collection_registry.json  # Tool version and metadata tracking
    
    File Roles and Access Patterns:
    
    1. chromadb/chroma.sqlite3:
       - Main ChromaDB database file containing collection metadata and schemas
       - Accessed: Automatically by ChromaDB client during all operations
       - Contains: Collection registry, metadata indexes, configuration
       - Size: Relatively small (~MB), doesn't contain actual document embeddings
       - Usage: Never accessed directly by application code
    
    2. chromadb/collections/:
       - Individual collection data folders (one per tool/version)
       - Accessed: Automatically by ChromaDB during search/indexing operations
       - Contains: Actual document embeddings, vector indexes, document content
       - Examples: scanpy_v1_11_3/, anndata_v0_10_4/
       - Usage: Managed entirely by ChromaDB, never touched by application
       - WHY NEEDED: Collections are ChromaDB's way of organizing documents into logical groups.
         You cannot store documents directly in chroma.sqlite3 - it only contains metadata.
         All actual embeddings and documents live in collection-specific folders.
    
    Collections vs Single Database:
    - chroma.sqlite3 is NOT a document store - it's a collection registry
    - Collections are required by ChromaDB architecture - you cannot bypass them
    - Each collection has its own embedding space and can use different embedding models
    - Collections enable: tool-specific search, version management, metadata filtering
    - Without collections: No way to store or search documents in ChromaDB
    
    3. collection_registry.json:
       - Human-readable registry of all indexed tools and their metadata
       - Accessed: Read during ChromaDbManager initialization, written after updates
       - Contains: Tool versions, creation dates, usage stats, collection mappings
       - Size: Small (~KB), one entry per tool version
       - Usage: Loaded into memory at startup, persisted after changes
       - Format: {"tool_version": {tool_name, version, created_at, last_used, collection_name, document_count}}
       - Critical for: Version compatibility checking, collection discovery, cleanup operations
    
    Tool Registry vs ChromaDB Collections:
    
    WHY WE NEED TOOL REGISTRY:
    ChromaDB only knows about collections (e.g., "scanpy_v1_11_3") but doesn't understand:
    - Which tool a collection belongs to
    - What version of the tool it represents
    - When it was created or last used
    - Whether the user's installed version matches the indexed version
    
    TOOL REGISTRY PROVIDES:
    - Business logic mapping: "scanpy" tool → "scanpy_v1_11_3" collection
    - Version management: Check if user has scanpy v1.11.3 installed
    - Usage tracking: When was this tool's knowledge last accessed?
    - Metadata: How many documents, when created, etc.
    - Cleanup logic: Which collections are old and can be deleted?
    
    EXAMPLE WORKFLOW:
    1. User asks: "How do I use scanpy for clustering?"
    2. detect_tools_in_code() identifies "scanpy" from query
    3. check_version_compatibility() uses registry to find user has scanpy v1.11.3
    4. Registry maps "scanpy v1.11.3" → "scanpy_v1_11_3" collection
    5. ChromaDB searches the "scanpy_v1_11_3" collection
    6. Registry updates last_used timestamp for tracking
    
    Without tool registry: We'd have to manually manage collection names and lose all 
    version tracking, usage statistics, and intelligent collection discovery.
    
    Collection Naming Convention:
    - Format: "{tool_name}_v{version_with_underscores}"
    - Examples: "scanpy_v1_11_3", "anndata_v0_10_4"
    - Enables version-specific knowledge bases and migration
    
    Document Types and Metadata:
    
    API DOCUMENTATION (Static Reference):
    - doc_type: "function" - Individual API functions (signatures, parameters, examples)
    - doc_type: "class" - API classes (methods, inheritance)
    - doc_type: "module" - API modules (function/class collections)
    - doc_type: "overview" - Library overviews (high-level summaries)
    - Granularity: Single-level chunks (function-level detail)
    - Search Use: "How do I use sc.pp.filter_cells?" / "What parameters does normalize take?"
    
    WORKFLOW DOCUMENTATION (Dynamic Tutorials):
    - doc_type: "workflow" - Tutorial/workflow documentation
    - doc_type: "example" - Code example documentation
    - Granularity: Multi-level hierarchical chunks via chunk_level metadata:
      * "document" - Full tutorial/workflow overview
      * "section" - Individual sections/steps within workflow  
      * "code_cell" - Individual code blocks/executable steps
    - Search Use: "How do I do single-cell clustering?" / "Show me a complete analysis pipeline"
    
    CURRENT PROBLEMS:
    1. HARDCODED TOOL DETECTION: detect_tools_in_code() has hardcoded regex patterns
    2. INCONSISTENT GRANULARITY: API docs lack hierarchical levels that workflows have
    3. MIXED SEARCH SEMANTICS: Same search interface for different documentation types
    4. NO DYNAMIC DISCOVERY: Tool patterns not based on actual indexed content
    
    PROPOSED SEARCH INTERFACE REDESIGN:
    
    1. DYNAMIC TOOL DETECTION:
       - Build tool patterns from indexed content, not hardcoded regex
       - Extract import patterns, function calls, and aliases from actual documents
       - Auto-discover new tools when they're indexed
       - Example: self._build_dynamic_tool_patterns() → patterns from real usage
    
    2. SEARCH TYPE SPECIALIZATION:
       - search_api(query, tools=None) → API reference search (functions, classes, modules)
       - search_workflows(query, tools=None, granularity="auto") → Tutorial/example search
       - search_mixed(query) → Current unified search for backward compatibility
    
    3. CONSISTENT HIERARCHICAL GRANULARITY:
       - API docs: package → module → class → function
       - Workflows: document → section → code_cell → line
       - Both support granularity-aware search and context expansion
    
    4. INTELLIGENT QUERY ROUTING:
       - Detect query intent: API reference vs workflow/tutorial needs
       - Route to appropriate search type automatically
       - "sc.pp.filter_cells parameters" → search_api()
       - "clustering workflow scanpy" → search_workflows()
    
    5. CODE GENERATION OPTIMIZED INTERFACE:
       - search_for_code_generation(task, context, detected_tools)
       - Combines API reference for specific functions + workflow patterns for structure
       - Returns both exact API signatures AND usage examples from workflows
       - Optimized for RAG-based code generation
    
    Interface with Extraction Pipeline:
    ==================================
    
    GitHubExtractor → Direct Indexing → ChromaDbManager → ChromaDB collections
    
    Direct Indexing Flow:
    1. GitHubDocumentationExtractor calls _index_library_from_data() directly during extraction
    2. _index_library_from_data() processes structured extraction results
    3. _index_api_reference() / _index_workflows() convert to searchable documents
    4. ChromaDB stores documents with automatic embedding generation
    
    
    Search Pipeline:
    1. detect_tools_in_code() identifies relevant tools from query context
    2. Query routing based on tool detection and query classification
    3. Collection-specific search with embedding similarity
    4. Result post-processing and hierarchical context enhancement
    
    Key Methods:
    ===========
    
    Initialization & Configuration:
    - __init__(): Initialize ChromaDB manager with persistent storage
    - _load_registry(): Load tool version registry from persistent storage
    - _save_registry(): Save tool version registry to persistent storage
    - get_installed_version(): Get currently installed version of a tool
    
    Version Management:
    - check_version_compatibility(): Check if knowledge base needs updating
    - ensure_tool_knowledge(): Ensure up-to-date knowledge for a tool
    - cleanup_old_versions(): Clean up outdated knowledge bases
    - _create_tool_knowledge_base(): Create new knowledge base for tool version
    
    Direct Indexing Pipeline (Data → ChromaDB):
    - _index_library_from_data(): Index library from structured data (main entry point)
    - _index_api_reference(): Index API functions, classes, and modules
    - _index_workflows(): Index workflow documentation with hierarchical chunking
    - _index_examples(): Index code examples and usage patterns
    
    PURPOSE: Process extraction results directly from GitHubDocumentationExtractor
    and convert them into searchable ChromaDB documents.
    
    EXAMPLE: extraction_result dict → ChromaDB documents with embeddings
    
    THE FLOW:
    1. Direct Indexing: GitHubDocumentationExtractor → {api_documentation: [{name: "filter_cells", ...}]}
    2. Document Processing: Function dict → "Function: filter_cells\nLibrary: scanpy\nDescription: ..."
    3. ChromaDB Storage: Text document → Vector embedding + metadata → Searchable collection
    
    Search Interface:
    - search(): Main search with automatic tool detection and routing
    - search_hierarchical(): Context-aware hierarchical search
    - search_with_optimal_chunking(): Optimal chunk level selection
    - search_with_metadata(): Metadata-filtered search
    
    Collection Management:
    - add_documents(): Add documents to knowledge base with automatic embedding
    - remove_documents_by_metadata(): Remove documents by metadata filters
    - _search_tool_collection(): Search within specific tool collection
    - _search_all_collections(): Search across all registered collections
    
    Utility & Status:
    - detect_tools_in_code(): Detect bioinformatics tools in code snippets
    - get_tool_status(): Get status of all tool knowledge bases
    - get_system_status(): Get comprehensive system status information
    
    Storage:
    - Persistent ChromaDB storage with automatic collection management
    - Tool registry persistence for version tracking and metadata
    - Hierarchical caching integration for extracted documentation
    """
    
    def __init__(self, knowledge_path: Path, settings=None):
        """Initialize ChromaDB manager.
        
        Args:
            knowledge_path: Path to store knowledge bases
            settings: Application settings
        """
        self.knowledge_path = Path(knowledge_path)
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        self.settings = settings
        
        # Initialize embedding function for ChromaDB with progress bar disabled
        # Create custom wrapper to disable tqdm progress bars
        class SilentSentenceTransformerEmbeddingFunction(embedding_functions.SentenceTransformerEmbeddingFunction):
            def __call__(self, input):
                # Override to match ChromaDB's __call__ but add show_progress_bar=False
                import numpy as np
                embeddings = self._model.encode(
                    list(input),
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                    show_progress_bar=False  # Key addition to suppress tqdm
                )
                return [np.array(embedding, dtype=np.float32) for embedding in embeddings]

        self.embedding_function = SilentSentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Initialize ChromaDB client with telemetry disabled
        self.client = chromadb.PersistentClient(
            path=str(self.knowledge_path / "chromadb"),
            settings=chromadb.config.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        
        # Tool knowledge base registry
        self.tool_registries: Dict[str, ToolKnowledgeBase] = {}
        self._load_registry()

        # Collection embedding cache for performance
        self._collection_embedding_cache: Dict[str, List[float]] = {}
        self._cache_file_path = self.knowledge_path / "collection_embeddings_cache.json"
        self._load_embedding_cache()

        # Background initialization state
        self._background_init_task: Optional[asyncio.Task] = None
        self._background_init_complete = False

        logger.info(f"ChromaDB manager initialized at {knowledge_path}")
    
    def _load_registry(self):
        """Load tool registry from persistent storage.
        
        Reads the collection_registry.json file from the knowledge base directory
        and reconstructs the in-memory registry of tool knowledge bases.
        Creates ToolKnowledgeBase objects from stored data.
        
        The registry tracks which tools have knowledge bases, their versions,
        creation dates, usage statistics, and collection names in ChromaDB.
        
        Handles missing registry files gracefully by starting with empty registry.
        """
        registry_path = self.knowledge_path / "collection_registry.json"
        if registry_path.exists():
            with open(registry_path, 'r') as f:
                data = json.load(f)
                self.tool_registries = {
                    key: ToolKnowledgeBase.from_dict(value)
                    for key, value in data.items()
                }
    
    def _save_registry(self):
        """Save tool registry to persistent storage.
        
        Serializes the current tool registry to collection_registry.json in the
        knowledge base directory. Converts ToolKnowledgeBase objects to
        dictionaries for JSON storage.
        
        Called after any changes to the registry (new tools, version updates,
        usage tracking, cleanup operations) to maintain persistence.
        
        Raises:
            IOError: If unable to write registry file
        """
        registry_path = self.knowledge_path / "collection_registry.json"
        with open(registry_path, 'w') as f:
            json.dump(
                {key: value.to_dict() for key, value in self.tool_registries.items()},
                f,
                indent=2
            )

    def _load_embedding_cache(self):
        """Load collection embedding cache from persistent storage."""
        try:
            if self._cache_file_path.exists():
                with open(self._cache_file_path, 'r') as f:
                    cache_data = json.load(f)
                    self._collection_embedding_cache = cache_data
        except Exception as e:
            logger.warning(f"Failed to load embedding cache: {e}")
            self._collection_embedding_cache = {}

    def _save_embedding_cache(self):
        """Save collection embedding cache to persistent storage."""
        try:
            with open(self._cache_file_path, 'w') as f:
                json.dump(self._collection_embedding_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")

    async def _async_save_cache(self):
        """Asynchronously save embedding cache without blocking."""
        def save():
            self._save_embedding_cache()
        # Run in thread pool to avoid blocking
        await asyncio.get_event_loop().run_in_executor(None, save)

    def start_background_initialization(self):
        """Start background initialization of collection embeddings.

        This method should be called when RAG is activated to precompute
        all collection embeddings in the background, avoiding blocking
        on the first search query.
        """
        try:
            # Check if there's an event loop running
            asyncio.get_running_loop()
            # Only start background task if there's an event loop running
            if self._background_init_task is None or self._background_init_task.done():
                self._background_init_task = asyncio.create_task(self._background_initialize_collection_embeddings())
                logger.info("Started background initialization of collection embeddings")
        except RuntimeError:
            # No event loop running, skip background initialization
            logger.debug("No event loop running, skipping background initialization")

    async def _background_initialize_collection_embeddings(self):
        """Background initialization of all collection embeddings."""
        try:
            logger.info("Starting background initialization of collection embeddings...")

            # Get all tool knowledge bases
            tool_kbs = list(self.tool_registries.values())
            missing_embeddings = [
                tool_kb for tool_kb in tool_kbs
                if tool_kb.collection_name not in self._collection_embedding_cache
            ]

            if not missing_embeddings:
                logger.info("All collection embeddings already cached")
                self._background_init_complete = True
                return

            logger.info(f"Computing embeddings for {len(missing_embeddings)} collections...")

            # Process collections sequentially to avoid "too many open files" error
            # ChromaDB keeps collection handles open and doesn't release them properly
            import gc
            for i, tool_kb in enumerate(missing_embeddings):
                try:
                    await self._get_collection_embedding(tool_kb.collection_name)
                    if (i + 1) % 50 == 0:
                        logger.info(f"Computed {i + 1}/{len(missing_embeddings)} collection embeddings...")
                        # Force garbage collection every 50 collections to release file handles
                        gc.collect()
                except Exception as e:
                    logger.debug(f"Failed to compute embedding for {tool_kb.collection_name}: {e}")

            # Final garbage collection
            gc.collect()

            self._background_init_complete = True
            logger.info(f"Background initialization complete. Cached embeddings for {len(self._collection_embedding_cache)} collections")

        except Exception as e:
            logger.error(f"Background initialization failed: {e}")

    def is_background_initialization_complete(self) -> bool:
        """Check if background initialization is complete."""
        return self._background_init_complete

    async def wait_for_background_initialization(self, timeout: float = 10.0):
        """Wait for background initialization to complete with timeout.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if self._background_init_task and not self._background_init_task.done():
            try:
                await asyncio.wait_for(self._background_init_task, timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Background initialization did not complete within {timeout} seconds")
            except Exception as e:
                logger.error(f"Background initialization failed: {e}")

    def get_installed_version(self, tool_name: str) -> Optional[str]:
        """Get the installed version of a tool from the current Python environment.
        
        Args:
            tool_name: Name of the tool/package
            
        Returns:
            Version string or None if not installed
        """
        try:
            # Try standard package name first
            version = importlib.metadata.version(tool_name)
            return version
        except importlib.metadata.PackageNotFoundError:
            # Try common alternatives
            alternatives = {
                "scvi": "scvi-tools",
                "sc": "scanpy",
                "sq": "squidpy",
            }
            
            if tool_name in alternatives:
                try:
                    return importlib.metadata.version(alternatives[tool_name])
                except importlib.metadata.PackageNotFoundError:
                    pass
            
            logger.debug(f"Package {tool_name} not found in environment")
            return None

    async def _create_tool_knowledge_base(
        self, 
        tool_name: str, 
        version: str, 
        existing_versions: List[str]
    ) -> str:
        """Create a new knowledge base for a specific tool version.
        
        Creates a new ChromaDB collection for the tool version and populates it
        with documentation. Uses existing versions for incremental updates when
        available, otherwise creates from scratch by fetching documentation.
        
        Args:
            tool_name: Name of the tool (e.g., 'scanpy', 'seurat')
            version: Specific version string (e.g., '1.9.3')
            existing_versions: List of version strings for which we already
                have knowledge bases (used for incremental updates)
            
        Returns:
            ChromaDB collection name for the new knowledge base
            
        The collection name follows the pattern: {tool_name}_v{version_sanitized}
        where dots in version are replaced with underscores.
        
        Registers the new knowledge base in the tool registry with metadata
        including creation time, document count, and usage tracking.
        """
        # Sanitize version string for ChromaDB collection name
        # Replace invalid characters: . -> _, + -> plus, - -> minus
        sanitized_version = version.replace('.', '_').replace('+', 'plus').replace('-', 'minus')
        collection_name = f"{tool_name}_v{sanitized_version}"
        
        # Create collection with embedding function
        collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "tool": tool_name,
                "version": version,
                "created_at": datetime.now().isoformat(),
            }
        )
        
        # Create from scratch
        await self._create_from_documentation(tool_name, version, collection)
        
        # Register the new knowledge base
        self.tool_registries[f"{tool_name}_{version}"] = ToolKnowledgeBase(
            tool_name=tool_name,
            version=version,
            created_at=datetime.now(),
            last_used=datetime.now(),
            collection_name=collection_name,
            document_count=collection.count(),
        )
        self._save_registry()
        
        logger.info(f"Created knowledge base for {tool_name} v{version}")
        return collection_name
    
    async def search(
        self,
        queries: List[str],
        n_results: int = 10
    ) -> Dict[str, Any]:
        """Search knowledge base using global vector similarity search.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            Search results: dictionary fcontaining:
                - query
                - content
                - tools: lists of tools with version for which hits were recovered, ordered by relevance
        """
        # Always use global vector search across all collections
        
        # Search all collections and get best matches
        results = await self._search_all_collections(queries, n_results)
        
        # Report tools in order of relevance
        return {
            "query": queries,
            "content": "\n".join([x["content"] for x in results]),
            "tools": np.unique([x["tool"] + "_" + x["version"] for x in results]).tolist(),
        }

    async def _search_all_collections(self, queries: List[str], n_results: int) -> List[Dict[str, Any]]:
        """Search across all tool collections using hierarchical semantic search.
        
        Performs a hierarchical search:
        1. First finds the most semantically relevant collections using collection embeddings
        2. Then searches only those top collections in parallel
        
        Args:
            query: List of natural language search queries
            n_results: Total number of results to return across all collections
            
        Returns:
            List of dictionary for each hit, containing:
                - id
                - content
                - metadata
                - distance:
                - tool:
                - version
                
        Process:
        1. Generates query embedding once for efficiency
        2. Computes collection-level embeddings and finds top 10 most relevant
        3. Searches only those top collections in parallel using asyncio.gather
        4. Aggregates all results with tool and version information
        5. Sorts by semantic distance (relevance) across all collections
        6. Returns top N results regardless of source collection
        
        This provides intelligent coverage by focusing on semantically relevant collections.
        """
        all_results_across_queries = []

        for query in queries:
            all_results = []
            # Step 1: Find the most semantically relevant collections
            top_collections = await self._find_relevant_collections(query, top_k=10)
            # Selected top collections for search
            
            # Fallback: If no collections found via semantic search, use document count approach
            if len(top_collections) == 0:
                # No collections found via semantic search, falling back to document count approach
                tool_kbs = list(self.tool_registries.values())
                tool_kbs.sort(key=lambda x: x.document_count, reverse=True)
                top_collections = tool_kbs[:10]
                # Using top collections by document count
            
            # Step 2: Create tasks for parallel collection queries
            async def search_collection(tool_kb):
                try:
                    collection = self.client.get_collection(tool_kb.collection_name)
                    results = collection.query(
                        query_texts=[query],
                        n_results=max(5, n_results // len(top_collections)),
                        include=["documents", "metadatas", "distances"]
                    )
                    
                    collection_results = []
                    # Check if results exist and are not empty
                    if results["ids"] and len(results["ids"]) > 0 and len(results["ids"][0]) > 0:
                        for i in range(len(results["ids"][0])):
                            collection_results.append({
                                "id": results["ids"][0][i],
                                "content": results["documents"][0][i],
                                "metadata": results["metadatas"][0][i],
                                "distance": results["distances"][0][i],
                                "tool": tool_kb.tool_name,
                                "version": tool_kb.version,
                            })
                    return collection_results
                        
                except Exception as e:
                    logger.error(f"Error searching {tool_kb.collection_name}: {e}")
                    return []
            
            # Step 3: Execute searches only on top collections in parallel
            tasks = [search_collection(tool_kb) for tool_kb in top_collections]
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Step 4: Aggregate results from all collections
            for results in results_list:
                if isinstance(results, list):  # Skip exceptions
                    all_results.extend(results)
            
            # Step 5: Sort by distance and take top results
            all_results.sort(key=lambda x: x["distance"])
            all_results_across_queries.extend(all_results[:n_results])
        
        return all_results_across_queries
    
    async def _find_relevant_collections(self, query: str, top_k: int = 10) -> List[ToolKnowledgeBase]:
        """Find the most semantically relevant collections for a query.

        Args:
            query: Search query
            top_k: Number of top collections to return

        Returns:
            List of ToolKnowledgeBase objects for the most relevant collections
        """
        # Generate query embedding asynchronously
        async def generate_query_embedding():
            return await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.embedding_function([query])[0]
            )

        # Start query embedding generation in parallel with collection loading
        query_embedding_task = asyncio.create_task(generate_query_embedding())

        # Get all tool knowledge bases (fast, from registry)
        tool_kbs = list(self.tool_registries.values())

        # Await query embedding completion
        raw_query_embedding = await query_embedding_task
        # Convert to list for consistency
        if hasattr(raw_query_embedding, 'tolist'):
            query_embedding = raw_query_embedding.tolist()
        else:
            query_embedding = list(raw_query_embedding)

        # Get collection-level embeddings in parallel
        async def get_collection_score(tool_kb: ToolKnowledgeBase):
            try:
                collection_embedding = await self._get_collection_embedding(tool_kb.collection_name)
                if collection_embedding is not None:
                    similarity = self._cosine_similarity(query_embedding, collection_embedding)
                    return (tool_kb, similarity)
                return None
            except Exception as e:
                logger.debug(f"Error computing similarity for {tool_kb.collection_name}: {e}")
                return None

        # Process all collections in parallel
        collection_tasks = [get_collection_score(tool_kb) for tool_kb in tool_kbs]
        results = await asyncio.gather(*collection_tasks, return_exceptions=True)

        # Filter out None results and exceptions
        collection_scores = []
        for result in results:
            if result is not None and not isinstance(result, Exception):
                collection_scores.append(result)

        # Sort by similarity and return top k
        collection_scores.sort(key=lambda x: x[1], reverse=True)
        top_collections = [tool_kb for tool_kb, _ in collection_scores[:top_k]]

        return top_collections
    
    def _get_repository_description(self, collection_name: str) -> Optional[str]:
        """Get repository description from API cache and README files.
        
        Args:
            collection_name: Name of the ChromaDB collection (e.g., "scanpy_vlatest")
            
        Returns:
            Combined repository description or None if not found
        """
        try:
            # Extract tool name from collection name
            # Format: {tool_name}_v{version} or {tool_name}_v{version}_{org}
            # Handle cases where tool name contains underscores
            parts = collection_name.split('_')
            if len(parts) < 2:
                return None
            
            # Try to find the repository by checking different combinations
            possible_names = []
            
            # Try the first part as tool name
            possible_names.append(parts[0])
            
            # Try combining parts until we find a match
            for i in range(1, min(len(parts), 4)):  # Limit to first 4 parts
                possible_names.append('_'.join(parts[:i+1]))
            
            # Also try the full name without version suffix
            if '_v' in collection_name:
                base_name = collection_name.split('_v')[0]
                possible_names.append(base_name)
            
            tool_name = None
            for name in possible_names:
                # Check if this name exists as a repository
                for org_dir in BIOINFORMATICS_CACHE_DIR.iterdir():
                    if not org_dir.is_dir():
                        continue
                    
                    repo_path = org_dir / 'repos' / name
                    if repo_path.exists():
                        tool_name = name
                        break
                
                if tool_name:
                    break
            
            if not tool_name:
                # Fallback to original method
                tool_name = parts[0]
            api_description = None
            readme_description = None
            
            # Check all organizations for the repository
            for org_dir in BIOINFORMATICS_CACHE_DIR.iterdir():
                if not org_dir.is_dir():
                    continue
                
                # Check API cache first
                api_cache_path = get_org_api_cache_file(org_dir.name)
                if api_cache_path.exists():
                    try:
                        with open(api_cache_path) as f:
                            api_data = json.load(f)
                        
                        if tool_name in api_data.get('repositories', {}):
                            api_description = api_data['repositories'][tool_name].get('description', '')
                            if api_description:
                                logger.debug(f"Found API description for {tool_name}: {api_description[:100]}...")
                    except Exception as e:
                        logger.debug(f"Error reading API cache for {org_dir.name}: {e}")
                
                # Check README files in repository
                repo_path = org_dir / 'repos' / tool_name
                if repo_path.exists():
                    # Case-insensitive README search - find any file with "readme" in the name
                    readme_found = False
                    for file_path in repo_path.iterdir():
                        if file_path.is_file() and "readme" in file_path.name.lower():
                            try:
                                content = file_path.read_text()
                                # Extract meaningful description (skip badges and links)
                                readme_description = self._extract_description_from_readme(content)
                                if readme_description:
                                    logger.debug(f"Found README description for {tool_name}: {readme_description[:100]}...")
                                    readme_found = True
                                    break
                            except Exception as e:
                                logger.debug(f"Error reading README for {tool_name}: {e}")
            
            # Combine descriptions if both available
            if api_description and readme_description:
                combined_description = f"{api_description} {readme_description}"
                logger.debug(f"Combined API and README descriptions for {tool_name}")
                return combined_description
            elif api_description:
                logger.debug(f"Using API description for {tool_name}")
                return api_description
            elif readme_description:
                logger.debug(f"Using README description for {tool_name}")
                return readme_description
            
            logger.debug(f"No description found for {tool_name}")
            return None
            
        except Exception as e:
            logger.debug(f"Error getting repository description for {collection_name}: {e}")
            return None
    
    def _extract_description_from_readme(self, content: str) -> Optional[str]:
        """Extract meaningful description from README content.
        
        Args:
            content: README file content
            
        Returns:
            Extracted description or None
        """
        try:
            lines = content.split('\n')
            description_lines = []
            
            for line in lines:
                # Skip badge lines
                if line.startswith('![') or line.startswith('[!['):
                    continue
                # Skip empty lines at start
                if not line.strip() and not description_lines:
                    continue
                # Stop at first heading or section
                if line.startswith('#') and description_lines:
                    break
                # Skip lines that are just links or badges
                if line.strip().startswith('http') or line.strip().startswith('['):
                    continue
                description_lines.append(line)
            
            description = '\n'.join(description_lines).strip()
            
            # Clean up the description
            if description:
                # Remove excessive whitespace
                description = re.sub(r'\s+', ' ', description)
                # Limit length
                if len(description) > 500:
                    description = description[:500] + "..."
                
                return description
            
            return None
            
        except Exception as e:
            logger.debug(f"Error extracting description from README: {e}")
            return None
    
    async def _get_collection_embedding(self, collection_name: str) -> Optional[List[float]]:
        """Get collection embedding from cache or compute on demand.

        Args:
            collection_name: Name of the ChromaDB collection

        Returns:
            Collection embedding vector or None if collection is empty
        """
        # Check cache first
        if collection_name in self._collection_embedding_cache:
            return self._collection_embedding_cache[collection_name]

        try:
            # First, try to get repository description and generate embedding
            description = self._get_repository_description(collection_name)
            if description:
                # Generate embedding from repository description
                raw_embedding = self.embedding_function([description])[0]
                # Convert numpy array to list for JSON serialization
                if hasattr(raw_embedding, 'tolist'):
                    description_embedding = raw_embedding.tolist()
                else:
                    description_embedding = list(raw_embedding)

                # Cache the result
                self._collection_embedding_cache[collection_name] = description_embedding
                # Save cache asynchronously for better performance
                asyncio.create_task(self._async_save_cache())

                logger.debug(f"Generated and cached embedding from repository description for {collection_name}")
                return description_embedding

            # Fallback to average of all document embeddings
            collection = self.client.get_collection(collection_name)

            # Get all embeddings from the collection
            results = collection.get(include=["embeddings"])

            embeddings = results.get("embeddings")
            if embeddings is None or len(embeddings) == 0:
                logger.debug(f"No embeddings found in collection {collection_name}")
                return None

            # Compute average embedding (handle numpy arrays)
            import numpy as np
            embeddings_array = np.array(embeddings)
            avg_embedding: List[float] = np.mean(embeddings_array, axis=0).tolist()

            # Cache the result
            self._collection_embedding_cache[collection_name] = avg_embedding
            # Save cache asynchronously for better performance
            asyncio.create_task(self._async_save_cache())

            logger.debug(f"Computed and cached average embedding for {collection_name} ({len(embeddings)} documents)")
            return avg_embedding

        except Exception as e:
            logger.debug(f"Error getting collection embedding for {collection_name}: {e}")
            return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        import numpy as np
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_repository_info(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Get repository information including stars and organization.
        
        Args:
            collection_name: Name of the ChromaDB collection
            
        Returns:
            Repository info dict or None if not found
        """
        try:
            # Extract tool name using the same logic as _get_repository_description
            parts = collection_name.split('_')
            if len(parts) < 2:
                return None
            
            # Try to find the repository by checking different combinations
            possible_names = []
            possible_names.append(parts[0])
            
            for i in range(1, min(len(parts), 4)):
                possible_names.append('_'.join(parts[:i+1]))
            
            if '_v' in collection_name:
                base_name = collection_name.split('_v')[0]
                possible_names.append(base_name)
            
            tool_name = None
            org_name = None
            
            for name in possible_names:
                for org_dir in BIOINFORMATICS_CACHE_DIR.iterdir():
                    if not org_dir.is_dir():
                        continue
                    
                    repo_path = org_dir / 'repos' / name
                    if repo_path.exists():
                        tool_name = name
                        org_name = org_dir.name
                        break
                
                if tool_name:
                    break
            
            if not tool_name or not org_name:
                return None
            
            # Get API cache info
            api_cache_path = get_org_api_cache_file(org_name)
            if api_cache_path.exists():
                try:
                    with open(api_cache_path) as f:
                        api_data = json.load(f)
                    
                    if tool_name in api_data.get('repositories', {}):
                        repo_info = api_data['repositories'][tool_name]
                        return {
                            'name': tool_name,
                            'organization': org_name,
                            'stars': repo_info.get('stars', 0),
                            'description': repo_info.get('description', ''),
                            'maintenance_level': ORGANIZATION_SCORES.get(org_name, 'unknown')
                        }
                except Exception as e:
                    logger.debug(f"Error reading API cache for {org_name}: {e}")
            
            return None
            
        except Exception as e:
            logger.debug(f"Error getting repository info for {collection_name}: {e}")
            return None
    
    async def get_rag_candidates(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Get RAG candidates with relevance scores and metadata.
        
        Args:
            query: Search query
            top_k: Number of top candidates to return
            
        Returns:
            List of candidate repositories with scores and metadata
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_function([query])[0]
            
            # Get all collections and compute scores
            candidates = []
            
            for tool_kb in self.tool_registries.values():
                try:
                    # Get collection-level embedding
                    collection_embedding = await self._get_collection_embedding(tool_kb.collection_name)
                    if collection_embedding is None:
                        continue
                    
                    # Compute cosine similarity
                    similarity = self._cosine_similarity(query_embedding, collection_embedding)
                    
                    # Get repository info
                    repo_info = self._get_repository_info(tool_kb.collection_name)
                    if repo_info is None:
                        continue
                    
                    candidates.append({
                        'collection_name': tool_kb.collection_name,
                        'similarity_score': similarity,
                        'stars': repo_info['stars'],
                        'organization': repo_info['organization'],
                        'maintenance_level': repo_info['maintenance_level'],
                        'description': repo_info['description'],
                        'document_count': tool_kb.document_count
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing {tool_kb.collection_name}: {e}")
                    continue
            
            # Sort by similarity score (descending)
            candidates.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Error getting RAG candidates: {e}")
            return []
    
    async def search_selected_collections(self, query: str, selected_collections: List[str], n_results: int = 10) -> Dict[str, Any]:
        """Search within selected collections and return top results.
        
        Args:
            query: Search query
            selected_collections: List of collection names to search
            n_results: Number of results per collection
            
        Returns:
            Search results with metadata
        """
        try:
            all_results = []
            
            for collection_name in selected_collections:
                try:
                    collection = self.client.get_collection(collection_name)
                    
                    # Search within this collection
                    results = collection.query(
                        query_texts=[query],
                        n_results=n_results,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    if (results['documents'] and results['documents'][0] and
                        results['metadatas'] and results['metadatas'][0] and
                        results['distances'] and results['distances'][0]):
                        for i, doc in enumerate(results['documents'][0]):
                            all_results.append({
                                'collection': collection_name,
                                'document': doc,
                                'metadata': results['metadatas'][0][i],
                                'distance': results['distances'][0][i]
                            })
                    
                except Exception as e:
                    logger.debug(f"Error searching collection {collection_name}: {e}")
                    continue
            
            # Sort by distance (ascending - lower is better)
            all_results.sort(key=lambda x: x['distance'])
            
            return {
                'results': all_results[:n_results],
                'collections_accessed': selected_collections,
                'total_results': len(all_results)
            }
            
        except Exception as e:
            logger.error(f"Error searching selected collections: {e}")
            return {'results': [], 'collections_accessed': [], 'total_results': 0}
    
    def get_tool_status(self) -> Dict[str, Any]:
        """Get status of all tool knowledge bases.
        
        Returns:
            Tool status information
        """
        status = {
            "total_tools": len(set(kb.tool_name for kb in self.tool_registries.values())),
            "total_versions": len(self.tool_registries),
            "tools": {},
        }
        
        for registry_key, tool_kb in self.tool_registries.items():
            tool_name = tool_kb.tool_name
            if tool_name not in status["tools"]:
                status["tools"][tool_name] = {
                    "versions": [],
                    "current_installed": self.get_installed_version(tool_name),
                }
            
            status["tools"][tool_name]["versions"].append({
                "version": tool_kb.version,
                "created_at": tool_kb.created_at.isoformat(),
                "last_used": tool_kb.last_used.isoformat(),
                "document_count": tool_kb.document_count,
            })
        
        return status
    
    async def add_documents(self, documents: List[Dict[str, Any]], tool_name: str = None, version: str = None):
        """Add documents to the knowledge base.
        
        Args:
            documents: List of documents with 'content' and 'metadata' keys
            tool_name: Tool name (if None, will try to extract from metadata)
            version: Tool version (if None, will use 'latest' or extract from metadata)
        """
        for doc in documents:
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})
            
            # Extract tool info
            doc_tool = tool_name or metadata.get("tool", metadata.get("library", "unknown"))
            doc_version = version or metadata.get("version", "latest")
            
            # Get or create collection for this tool/version
            collection_name = f"{doc_tool}_v{doc_version.replace('.', '_')}"
            collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={
                    "tool": doc_tool,
                    "version": doc_version,
                    "created_at": datetime.now().isoformat(),
                }
            )
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            if not chunks:
                continue
                
            # ChromaDB will generate embeddings automatically via embedding_function
            
            # Create unique IDs
            doc_type = metadata.get("doc_type", "doc")
            # Use more specific ID for uniqueness
            if doc_type == "function":
                doc_id = f"function_{metadata.get('function_name', 'unknown')}"
            elif doc_type == "class":
                doc_id = f"class_{metadata.get('class_name', 'unknown')}"
            elif doc_type == "workflow":
                doc_id = f"workflow_{metadata.get('workflow_name', 'unknown')}"
            elif doc_type == "tutorial":  # Legacy support
                doc_id = f"workflow_{metadata.get('tutorial_name', metadata.get('workflow_name', 'unknown'))}"
            else:
                doc_id = f"{doc_type}_{len(collection.get()['ids'])}"
            
            chunk_ids = [f"{doc_tool}_{doc_version}_{doc_id}_{i}" for i in range(len(chunks))]
            
            # Prepare metadata for each chunk
            chunk_metadata = []
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    **metadata,
                    "tool": doc_tool,
                    "version": doc_version,
                    "chunk_index": i,
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                chunk_metadata.append(chunk_meta)
            
            # Add to collection (ChromaDB will generate embeddings automatically)
            collection.add(
                ids=chunk_ids,
                documents=chunks,
                metadatas=chunk_metadata
            )
            
            # Update registry
            registry_key = f"{doc_tool}_{doc_version}"
            if registry_key not in self.tool_registries:
                self.tool_registries[registry_key] = ToolKnowledgeBase(
                    tool_name=doc_tool,
                    version=doc_version,
                    created_at=datetime.now(),
                    last_used=datetime.now(),
                    collection_name=collection_name,
                    document_count=0
                )
            
            # Update document count
            self.tool_registries[registry_key].document_count = collection.count()
            self.tool_registries[registry_key].last_used = datetime.now()
        
        # Save registry
        self._save_registry()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information.
        
        Returns:
            System status
        """
        total_docs = sum(kb.document_count for kb in self.tool_registries.values())
        
        return {
            "total_collections": len(self.tool_registries),
            "total_documents": total_docs,
            "tools": list(set(kb.tool_name for kb in self.tool_registries.values())),
            "knowledge_path": str(self.knowledge_path)
        }
    
    async def index_library_from_data(self, lib_name: str, lib_docs: Dict[str, Any], organization: str = "unknown", repo_path: Path = None) -> Dict[str, Any]:
        """Index documentation for a library from data structure using new organization scheme.

        Args:
            lib_name: Library/repository name
            lib_docs: Library documentation data
            organization: GitHub organization name
            repo_path: Optional path to git repository for version detection

        Returns:
            Indexing results for the library
        """
        result = {
            "library": lib_name,
            "documents_added": 0,
            "functions_added": 0,
            "workflows_added": 0,
            "examples_added": 0
        }
        
        if "error" in lib_docs:
            raise Exception(lib_docs["error"])
        
        # Extract API version from ReadTheDocs (stable, latest)
        api_version = lib_docs.get("version", "unknown")
        if api_version == "unknown":
            # Try alternative version sources
            if "readthedocs_extraction" in lib_docs:
                rtd_data = lib_docs.get("readthedocs_extraction", {})
                api_version = rtd_data.get("version", "unknown")

        # Extract workflows version from git tags
        workflows_version = "latest"
        if repo_path and repo_path.exists():
            workflows_version = self._get_git_version(repo_path)

        # Current GitHub extractor format
        api_reference = lib_docs.get("api_documentation", {})
        workflows = lib_docs.get("workflows", [])
        
        # Index API reference to {org}_{repo}_api collection (use ReadTheDocs version)
        if api_reference:
            api_result = await self._index_api_reference(lib_name, api_reference, organization, api_version)
            result["functions_added"] = api_result["functions_added"]
            result["documents_added"] += api_result["documents_added"]

        # Index workflows to {org}_{repo}_workflows collection (use git tag version)
        # Current format: workflows is a list of HierarchicalChunk objects
        if workflows:
            workflow_result = await self._index_hierarchical_workflows(lib_name, workflows, organization, workflows_version)
            result["workflows_added"] = workflow_result["workflows_added"]
            result["documents_added"] += workflow_result["documents_added"]
        
        logger.info(f"Indexed {lib_name}: {result['documents_added']} documents, "
                   f"{result['functions_added']} functions, {result['workflows_added']} workflows "
                   f"(API version: {api_version}, Workflows version: {workflows_version})")
        
        return result
    
    async def _index_api_reference(self, lib_name: str, api_reference: Dict[str, Any], organization: str, version: str) -> Dict[str, Any]:
        """Index API reference documentation to {org}_{repo}_api collection.

        Args:
            lib_name: Library/repository name
            api_reference: API reference data
            organization: GitHub organization name

        Returns:
            Indexing results
        """
        result = {"functions_added": 0, "documents_added": 0}
        documents = []
        
        # Create API collection
        collection_name = f"{organization}_{lib_name}_api"
        api_collection = self._get_or_create_collection(
            collection_name=collection_name,
            organization=organization,
            repository=lib_name,
            content_type="api",
            version=version
        )

        # Add overview document first
        overview_doc = self._create_library_overview(lib_name, {"api_documentation": api_reference}, organization, version)
        documents.append(overview_doc)

        # Index functions
        functions = api_reference.get("functions", [])
        for func in functions:
            doc = self._create_function_document(lib_name, func, organization, version)
            documents.append(doc)

        result["functions_added"] = len(functions)

        # Index classes
        classes = api_reference.get("classes", [])
        for cls in classes:
            doc = self._create_class_document(lib_name, cls, organization, version)
            documents.append(doc)

        # Index modules
        modules = api_reference.get("modules", [])
        for module in modules:
            doc = self._create_module_document(lib_name, module, organization, version)
            documents.append(doc)

        if documents:
            await self._add_documents_to_collection(documents, api_collection)
            result["documents_added"] = len(documents)
        
        return result

    def _get_git_version(self, repo_path: Path) -> str:
        """Get the latest git tag from a repository, fallback to 'latest'.

        Args:
            repo_path: Path to the cloned repository

        Returns:
            Latest git tag or 'latest' if no tags found
        """
        try:
            import subprocess

            # Get the latest git tag
            result = subprocess.run(
                ["git", "describe", "--tags", "--abbrev=0"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0 and result.stdout.strip():
                version = result.stdout.strip()
                logger.debug(f"Found git tag for {repo_path.name}: {version}")
                return version
            else:
                logger.debug(f"No git tags found for {repo_path.name}, using 'latest'")
                return "latest"

        except Exception as e:
            logger.debug(f"Error getting git version for {repo_path.name}: {e}")
            return "latest"

    def _get_or_create_collection(self, collection_name: str, organization: str, repository: str, content_type: str, version: str = "unknown"):
        """Get or create a collection with new naming scheme and metadata."""
        return self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "organization": organization,
                "repository": repository,
                "content_type": content_type,
                "version": version,
                "created_at": datetime.now().isoformat(),
            }
        )

    async def _add_documents_to_collection(self, documents: List[Dict[str, Any]], collection):
        """Add documents to a specific collection."""
        if not documents:
            return


        # Split content into chunks
        chunk_ids = []
        chunk_documents = []
        chunk_metadata = []

        for doc_idx, doc in enumerate(documents):
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            # Validate document structure
            if not isinstance(doc, dict):
                raise ValueError(f"Document {doc_idx} must be a dictionary, got {type(doc)}: {doc}")

            # Validate metadata - this should never happen if document creation is correct
            if not isinstance(metadata, dict):
                raise ValueError(f"Document {doc_idx} metadata must be a dict, got {type(metadata)}: {metadata}. Full doc: {doc}")

            # Split content into chunks
            chunks = self.text_splitter.split_text(content)

            # Create unique IDs
            doc_type = metadata.get("doc_type", "doc")
            if doc_type == "function":
                doc_id = f"function_{metadata.get('function_name', 'unknown')}"
            elif doc_type == "class":
                doc_id = f"class_{metadata.get('class_name', 'unknown')}"
            elif doc_type == "workflow":
                # Use chunk_id for hierarchical workflow chunks, otherwise workflow_name
                workflow_id = metadata.get('chunk_id', metadata.get('workflow_name', 'unknown'))
                doc_id = f"workflow_{workflow_id}"
            elif doc_type == "example":
                doc_id = f"example_{metadata.get('example_name', 'unknown')}"
            elif doc_type == "overview":
                doc_id = "overview"
            else:
                doc_id = f"{doc_type}_{len(collection.get()['ids'])}"

            # Validate metadata is a dictionary - this should never fail
            if not isinstance(metadata, dict):
                raise ValueError(f"Metadata must be a dictionary, got {type(metadata)}: {metadata}")

            chunk_doc_ids = [f"{metadata['organization']}_{metadata['library']}_{doc_id}_{i}" for i in range(len(chunks))]

            # Prepare metadata for each chunk
            for i, chunk in enumerate(chunks):
                chunk_meta = {
                    **metadata,
                    "chunk_index": i,
                    "chunk_text": chunk[:100] + "..." if len(chunk) > 100 else chunk
                }
                chunk_metadata.append(chunk_meta)

            chunk_ids.extend(chunk_doc_ids)
            chunk_documents.extend(chunks)

        # Add to collection
        collection.add(
            ids=chunk_ids,
            documents=chunk_documents,
            metadatas=chunk_metadata
        )

    async def _index_hierarchical_workflows(self, lib_name: str, workflows: List, organization: str, version: str) -> Dict[str, Any]:
        """Index HierarchicalChunk workflows to {org}_{repo}_workflows collection."""
        result = {"workflows_added": 0, "documents_added": 0}

        if not workflows:
            return result

        # Create workflows collection
        collection_name = f"{organization}_{lib_name}_workflows"
        workflows_collection = self._get_or_create_collection(
            collection_name=collection_name,
            organization=organization,
            repository=lib_name,
            content_type="workflows",
            version=version
        )

        # Convert HierarchicalChunk objects to documents
        from ..extractors.hierarchical_workflow_parser import HierarchicalWorkflowParser
        hierarchical_parser = HierarchicalWorkflowParser()
        chunk_documents = hierarchical_parser.chunks_to_documents(workflows)

        # Add organization and library metadata to each document
        for doc in chunk_documents:
            # Ensure metadata is a dictionary
            if not isinstance(doc.get("metadata", {}), dict):
                logger.warning(f"Document metadata is not a dict: {type(doc.get('metadata'))}")
                doc["metadata"] = {}

            doc["metadata"]["organization"] = organization
            doc["metadata"]["library"] = lib_name
            doc["metadata"]["version"] = version

        if chunk_documents:
            await self._add_documents_to_collection(chunk_documents, workflows_collection)
            result["documents_added"] = len(chunk_documents)
            result["workflows_added"] = len(workflows)

        return result

    
    
    def _create_library_overview(self, lib_name: str, lib_docs: Dict[str, Any], organization: str, version: str) -> Dict[str, Any]:
        """Create a library overview document.

        Args:
            lib_name: Library name
            lib_docs: Library documentation data
            organization: Organization name
            version: Version string

        Returns:
            Overview document
        """
        # Handle both formats
        if "sections" in lib_docs:
            sections = lib_docs.get("sections", {})
            api_ref = sections.get("api_reference", {})
            func_count = len(api_ref.get("functions", []))
            class_count = len(api_ref.get("classes", []))
            workflow_count = len(sections.get("tutorials", []))
        else:
            # GitHub extractor format
            api_docs = lib_docs.get("api_documentation", {})
            func_count = len(api_docs.get("functions", []))
            class_count = len(api_docs.get("classes", []))
            workflows = lib_docs.get("workflows", [])
            workflow_count = len(workflows) if isinstance(workflows, (list, dict)) else 0
        
        summary = lib_docs.get("summary", f"Documentation for {lib_name}")
        
        content = f"""
{lib_name.upper()} Library Documentation

{summary}

Base URL: {lib_docs.get('base_url', '')}
Documentation Type: {lib_docs.get('type', '')}
Last Crawled: {lib_docs.get('crawled_at', '')}

Available Resources:
- {func_count} API functions
- {class_count} classes
- {workflow_count} workflows

This library is commonly used for bioinformatics analysis and provides comprehensive API documentation.
        """.strip()
        
        return {
            "content": content,
            "metadata": {
                "organization": organization,
                "library": lib_name,
                "version": version,
                "doc_type": "overview",
                "base_url": lib_docs.get("base_url", ""),
                "documentation_type": lib_docs.get("type", ""),
                "function_count": func_count,
                "class_count": class_count,
                "workflow_count": workflow_count,
                "indexed_at": datetime.now().isoformat()
            }
        }
    
    def _create_function_document(self, lib_name: str, func: Dict[str, Any], organization: str, version: str) -> Dict[str, Any]:
        """Create a document for a function.
        
        Args:
            lib_name: Library name
            func: Function data
            
        Returns:
            Function document
        """
        content_parts = [
            f"Function: {func.get('name', 'Unknown')}",
            f"Library: {lib_name}",
        ]
        
        if func.get("signature"):
            content_parts.append(f"Signature: {func['signature']}")
        
        if func.get("description"):
            content_parts.append(f"Description: {func['description']}")
        
        # Add parameters
        parameters = func.get("parameters", [])
        if parameters:
            content_parts.append("Parameters:")
            for param in parameters:
                param_name = param.get("name", "")
                param_desc = param.get("description", "")
                content_parts.append(f"  - {param_name}: {param_desc}")
        
        if func.get("returns"):
            content_parts.append(f"Returns: {func['returns']}")
        
        # Add examples
        examples = func.get("examples", "")
        if examples:
            content_parts.append("Examples:")
            content_parts.append(f"```python\n{examples}\n```")
        
        content = "\n\n".join(content_parts)
        
        return {
            "content": content,
            "metadata": {
                "organization": organization,
                "library": lib_name,
                "version": version,
                "doc_type": "function",
                "function_name": func.get("name", ""),
                "url": func.get("url", ""),
                "has_examples": len(examples) > 0,
                "parameter_count": len(parameters),
                "indexed_at": datetime.now().isoformat()
            }
        }
    
    def _create_class_document(self, lib_name: str, cls: Dict[str, Any], organization: str, version: str) -> Dict[str, Any]:
        """Create a document for a class.
        
        Args:
            lib_name: Library name
            cls: Class data
            
        Returns:
            Class document
        """
        content_parts = [
            f"Class: {cls.get('name', 'Unknown')}",
            f"Library: {lib_name}",
        ]
        
        if cls.get("description"):
            content_parts.append(f"Description: {cls['description']}")
        
        if cls.get("methods"):
            content_parts.append("Methods:")
            for method in cls["methods"]:
                content_parts.append(f"  - {method}")
        
        content = "\n\n".join(content_parts)
        
        return {
            "content": content,
            "metadata": {
                "organization": organization,
                "library": lib_name,
                "version": version,
                "doc_type": "class",
                "class_name": cls.get("name", ""),
                "url": cls.get("url", ""),
                "indexed_at": datetime.now().isoformat()
            }
        }
    
    def _create_module_document(self, lib_name: str, module: Dict[str, Any], organization: str, version: str) -> Dict[str, Any]:
        """Create a document for a module.
        
        Args:
            lib_name: Library name
            module: Module data
            
        Returns:
            Module document
        """
        content_parts = [
            f"Module: {module.get('name', 'Unknown')}",
            f"Library: {lib_name}",
        ]
        
        if module.get("description"):
            content_parts.append(f"Description: {module['description']}")
        
        content = "\n\n".join(content_parts)
        
        return {
            "content": content,
            "metadata": {
                "organization": organization,
                "library": lib_name,
                "version": version,
                "doc_type": "module",
                "module_name": module.get("name", ""),
                "url": module.get("url", ""),
                "indexed_at": datetime.now().isoformat()
            }
        }
    
    def _create_workflow_document(self, lib_name: str, workflow: Dict[str, Any], organization: str, version: str) -> Dict[str, Any]:
        """Create a document for a workflow.
        
        Args:
            lib_name: Library name
            workflow: Workflow data
            
        Returns:
            Workflow document
        """
        content_parts = [
            f"Workflow: {workflow.get('name', 'Unknown')}",
            f"Library: {lib_name}",
            f"Type: {workflow.get('type', 'workflow')}"
        ]
        
        # Add workflow description if available
        if workflow.get('description'):
            content_parts.append(f"Description: {workflow['description']}")
        
        # Extract and include actual step content
        steps = workflow.get('steps', [])
        if steps:
            content_parts.append(f"Workflow Steps ({len(steps)} steps):")
            
            for step in steps:
                step_parts = []
                
                # Step title and number
                step_num = step.get('step_number', '')
                step_title = step.get('title', 'Untitled Step')
                step_parts.append(f"Step {step_num}: {step_title}")
                
                # Step description
                description = step.get('description', '')
                if description:
                    clean_desc = description.strip().replace('\n\n', ' ').replace('\n', ' ')
                    step_parts.append(f"Description: {clean_desc}")
                
                # Include code if available
                code = step.get('code', '')
                if code and code.strip():
                    code_lines = code.strip().split('\n')
                    if len(code_lines) > 10:
                        code_snippet = '\n'.join(code_lines[:10]) + '\n... (truncated)'
                    else:
                        code_snippet = code.strip()
                    step_parts.append(f"Code:\n{code_snippet}")
                
                if step.get('step_type'):
                    step_parts.append(f"Type: {step['step_type']}")
                
                content_parts.append('\n'.join(step_parts))
        
        # Add URL if available
        if workflow.get('url'):
            content_parts.append(f"URL: {workflow['url']}")
        
        if workflow.get('workflow_type'):
            content_parts.append(f"Workflow Type: {workflow['workflow_type']}")
        
        content = "\n\n".join(content_parts)
        
        return {
            "content": content,
            "metadata": {
                "organization": organization,
                "library": lib_name,
                "version": version,
                "doc_type": "workflow",
                "workflow_name": workflow.get("name", ""),
                "url": workflow.get("url", ""),
                "workflow_type": workflow.get("workflow_type", ""),
                "domain": workflow.get("domain", ""),
                "language": workflow.get("language", ""),
                "step_count": len(steps),
                "has_code": any(step.get('code', '').strip() for step in steps),
                "indexed_at": datetime.now().isoformat()
            }
        }
    
    def _create_example_document(self, lib_name: str, example: Dict[str, Any], organization: str, version: str) -> Dict[str, Any]:
        """Create a document for an example.
        
        Args:
            lib_name: Library name
            example: Example data
            
        Returns:
            Example document
        """
        content = f"""
Example: {example.get('name', 'Unknown')}
Library: {lib_name}
Type: {example.get('type', 'example')}

This is a code example demonstrating how to use {lib_name}.
URL: {example.get('url', '')}

This example shows practical implementation patterns and usage of {lib_name} functionality.
        """.strip()
        
        return {
            "content": content,
            "metadata": {
                "organization": organization,
                "library": lib_name,
                "version": version,
                "doc_type": "example",
                "example_name": example.get("name", ""),
                "url": example.get("url", ""),
                "indexed_at": datetime.now().isoformat()
            }
        }
    
    def has_repositories_batch(self, lib_names: List[str], organization_mapping: Dict[str, str] = None) -> Dict[str, bool]:
        """TRUE batch check if repositories already exist in ChromaDB using new collection naming.

        This method checks for the new {org}_{repo}_{content_type} collection naming scheme.

        Args:
            lib_names: List of library names to check
            organization_mapping: Optional mapping from lib_name to organization

        Returns:
            Dictionary mapping library names to existence status
        """
        result = {lib_name: False for lib_name in lib_names}

        try:
            # Get all collections once at the start
            all_collections = self.client.list_collections()
            collection_names = {collection.name for collection in all_collections}

            logger.debug(f"Checking {len(lib_names)} libraries across {len(collection_names)} collections")

            for lib_name in lib_names:
                # Look for collections with patterns: {org}_{repo}_api or {org}_{repo}_workflows
                for collection_name in collection_names:
                    # Check if collection name matches the pattern for this library
                    if collection_name.endswith(f"_{lib_name}_api") or collection_name.endswith(f"_{lib_name}_workflows"):
                        result[lib_name] = True
                        logger.debug(f"Found {lib_name} via collection name pattern: {collection_name}")
                        break

                    # Also check collection metadata for repository field (fallback)
                    try:
                        collection = self.client.get_collection(collection_name)
                        collection_metadata = collection.metadata
                        if collection_metadata and collection_metadata.get("repository") == lib_name:
                            result[lib_name] = True
                            logger.debug(f"Found {lib_name} via collection metadata in: {collection_name}")
                            break
                    except Exception as e:
                        logger.debug(f"Error checking collection metadata for {collection_name}: {e}")
                        continue

            found_count = sum(1 for found in result.values() if found)
            logger.debug(f"New naming scheme check found {found_count}/{len(lib_names)} libraries")

            return result

        except Exception as e:
            logger.error(f"Error in batch repository check: {e}")
            # Return all False if there's an error
            return result

    def has_repository(self, lib_name: str) -> bool:
        """Check if a repository already exists in ChromaDB.

        Args:
            lib_name: Library name to check

        Returns:
            True if repository exists in ChromaDB, False otherwise
        """
        # Use the batch method for single repository checks
        result = self.has_repositories_batch([lib_name])
        return result.get(lib_name, False)