# kai Retrieval Database Scripts

This directory contains scripts for building, managing, and sharing the kai retrieval database. It builds a database in `~/.kai_agent/retrieval/`.

## Prerequisites

**Python Environment**: These scripts should be run in a dedicated Python environment (e.g., mamba/conda) with:
- Python 3.11 or later
- kai package installed (`pip install -e .` from the kai repository root)
- All dependencies installed

Example setup:
```bash
mamba create -n kai_agent python=3.11
mamba activate kai_agent
cd /path/to/kai
pip install -e .
```

## Script Overview

### Knowledge Base Management
- **`knowledge_base_summary.py`** - Display summary statistics about the current retrieval database
- **`build/1_run_github_extraction.py`** - Extract API documentation and workflows from GitHub repositories
- **`build/1_update_repos.py`** - Update locally cloned repositories to their latest versions
- **`build/1b_chromadb_diagnostic.py`** - Diagnose and verify ChromaDB collections and indices
- **`build/1b_extraction_progress.py`** - Monitor progress of ongoing extraction jobs
- **`build/1c_extract_licenses.py`** - Extract and organize LICENSE files from all cached repositories
- **`build/2_generate_notebook_summaries.py`** - Generate LLM-based summaries of workflow notebooks
- **`build/3_build_summary_index.py`** - Build semantic search index for notebook summaries
- **`build/4_rebuild_collection_registry.py`** - Rebuild collection registry from existing ChromaDB collections
- **`build/5_build_summary_collection_cache.py`** - Pre-compute and cache embeddings for all collections
- **`subset/subset_database.py`** - Build filtered database containing only permissively-licensed repositories
- **`download_database.py`** - Downloads a pre-built database from zenodo for immediate use in kai.

## Workflow 1: Loading a Pre-built Retrieval Database

**Use case**: You want to use kai with a pre-built knowledge base without building from scratch.

### Steps:

1. **Download the database** from Zenodo:
   ```bash
   cd /path/to/kai
   python scripts/download_retrieval_data.py 251121
   ```

2. **Verify the installation**:
   ```bash
   python scripts/knowledge_base_summary.py
   ```

   You should see statistics about collections, documents, and organizations.

3. **Start using kai** - the retrieval database is now available at `~/.kai_agent/retrieval/`

## Workflow 2: Building a Retrieval Database from Scratch

**Use case**: You want to build the complete retrieval database from source repositories.

### Prerequisites:
- GitHub API token (optional but recommended for higher rate limits)
- Ollama instance to run LLMs

### Steps:

1. **Extract documentation and workflows from GitHub**:
   ```bash
   python scripts/build/1_run_github_extraction.py --auth-token YOUR_GITHUB_TOKEN
   ```

   This will:
   - Clone repositories from configured organizations
   - Extract API documentation from ReadTheDocs
   - Extract workflow code from Jupyter notebooks
   - Index everything directly into ChromaDB

   **Note**: This is the longest step. Results are cached so you can abort and restart this process. See the following caching key word arguments that are documented in the script

   - `--cache-only`: Use only local caches, no internet access
   - `--no-recache-orgs`: Don't refresh organization repository lists (use existing cache)
   - `--no-recache-html`: Don't re-download existing HTML documentation
   - `--overwrite-chromadb`: Force reprocessing even if collection exists

   For example, you would restart an incomplete run with:

   ```bash
   python scripts/build/run_github_extraction.py --no-recache-orgs --no-recache-html --auth-token YOUR_GITHUB_TOKEN
   ```

2. **Extract license files**:
   ```bash
   python scripts/build/1c_extract_licenses.py
   ```

   Copies LICENSE files from all repositories to `~/.kai_agent/retrieval/licenses/`. Note: license extraction is done as pattern matching and not guaranteed to be exact.

3. **Generate notebook summaries** (optional, for workflow search):
   ```bash
   python scripts/build/2_generate_notebook_summaries.py
   ```

   Creates LLM-generated summaries of workflow notebooks (this requires ollama). These summaries will be indexed in a ChromaDB instance to faciliate retrieval of notebooks.

4. **Build summary search index** (if you generated summaries):
   ```bash
   python scripts/build/3_build_summary_index.py
   ```

5. **Build collection embeddings cache** (optional, improves performance):
   ```bash
   python scripts/build/5_build_summary_collection_cache.py
   ```

6. **Verify the database**:
   ```bash
   python scripts/knowledge_base_summary.py
   ```

### Maintenance:

To update an existing database with new repositories, run the following script before going through the workflow again.

```bash
# Update local repository clones
python scripts/build/1_update_repos.py

python scripts/build/1_run_github_extraction.py --auth-token YOUR_GITHUB_TOKEN
# etc.
```

## Workflow 3: Subsetting a Retrieval Database

### Prerequisites:
- An existing retrieval database at `~/.kai_agent/retrieval/`
- All LICENSE files extracted (run `1c_extract_licenses.py` if needed)
- GitHub API token stored in `~/.github_token` (for accurate license detection)

### Steps:

1. **Build the sharing database**:
   ```bash
   python scripts/subset/subset_database.py
   ```

   This script will:
   - **Phase 1**: Query GitHub API to identify licenses of repositories
   - **Phase 2**: Copy source files (NOTICE, README, summaries, licenses) - filtered to permissive licenses only
   - **Phase 3**: Build semantic search index for notebook summaries
   - **Phase 3.5**: Rebuild ChromaDB from cached repositories (runs in subprocess)
   - **Phase 4**: Rebuild collection registry
   - **Phase 5**: Build embedding cache for all collections
   - **Phase 6**: Deep verification via GitHub API that no restrictive licenses are present

   Real-time output shows every repository being included or excluded with their license type.

2. **Verify the results**:
   ```bash
   python scripts/knowledge_base_summary.py
   ```

   The database is created at `~/.kai_agent/retrieval_sharing/`

3. **Package for distribution**:
   ```bash
   cd ~/.kai_agent
   tar -czf kai_retrieval_sharing_YYMMDD.tar.gz retrieval_sharing/
   ```
