#!/usr/bin/env python3
"""
Knowledge Base Summary Report
==============================
Comprehensive overview of the indexed knowledge base including:
1. API documentation pages in ChromaDB
2. GitHub repositories cloned
3. Jupyter notebook workflows indexed

Usage:
    python scripts/0_knowledge_base_summary.py
"""

from pathlib import Path
from collections import Counter

from kai.config.paths import BIOINFORMATICS_CACHE_DIR, RETRIEVAL_DIR
from kai.retrieval.snippets.storage.chromadb_manager import ChromaDbManager
from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
from kai.config.settings import Settings


def get_chromadb_stats():
    """Get statistics from ChromaDB collections."""
    manager = ChromaDbManager(RETRIEVAL_DIR)
    collections = manager.client.list_collections()

    total_docs = 0
    total_functions = 0
    total_workflows = 0
    repo_stats = {}

    print(f"  Processing {len(collections)} collections...", flush=True)

    for i, collection in enumerate(collections, 1):
        collection_name = collection.name
        try:
            # Use count() for faster stats
            count = collection.count()

            if count == 0:
                continue

            # Sample metadata to estimate doc types (faster than fetching all)
            sample_size = min(100, count)
            results = collection.get(limit=sample_size, include=['metadatas'])

            if not results['metadatas']:
                continue

            functions = 0
            workflows = 0

            for metadata in results['metadatas']:
                doc_type = metadata.get('doc_type', 'unknown')

                if doc_type == 'function':
                    functions += 1
                elif doc_type == 'workflow' or metadata.get('chunk_level'):
                    workflows += 1

            # Extrapolate from sample
            if len(results['metadatas']) > 0:
                scale = count / len(results['metadatas'])
                functions = int(functions * scale)
                workflows = int(workflows * scale)

            repo_stats[collection_name] = {
                'total_docs': count,
                'functions': functions,
                'workflows': workflows
            }

            total_docs += count
            total_functions += functions
            total_workflows += workflows

            if i % 10 == 0:
                print(f"  Processed {i}/{len(collections)} collections...", flush=True)

        except Exception as e:
            continue

    return {
        'total_collections': len(collections),
        'total_docs': total_docs,
        'total_functions': total_functions,
        'total_workflows': total_workflows,
        'repo_stats': repo_stats
    }


def get_github_repo_stats():
    """Get statistics about cloned GitHub repositories."""
    org_stats = {}
    total_repos = 0

    for org_dir in BIOINFORMATICS_CACHE_DIR.iterdir():
        if org_dir.is_dir():
            org_name = org_dir.name
            repos_dir = org_dir / "repos"

            if repos_dir.exists():
                repo_count = len([d for d in repos_dir.iterdir() if d.is_dir()])
                org_stats[org_name] = repo_count
                total_repos += repo_count
            else:
                org_stats[org_name] = 0

    return {
        'total_repos': total_repos,
        'org_stats': org_stats
    }


def get_notebook_summary_stats():
    """Get statistics about indexed Jupyter notebooks."""
    settings = Settings.from_env()
    storage_path = settings.KNOWLEDGE_BASE_PATH / "notebook_summaries"

    if not storage_path.exists():
        return {
            'total_notebooks': 0,
            'notebooks_with_summaries': 0,
            'org_stats': {}
        }

    storage = NotebookStorage(storage_path)
    all_summaries = storage.get_all_summaries()

    # Count notebooks by organization
    org_counter = Counter()

    for notebook_id in all_summaries.keys():
        # Extract org from notebook_id (format: org/repo/notebook_name)
        parts = notebook_id.split('/')
        if len(parts) >= 1:
            org = parts[0]
            org_counter[org] += 1

    return {
        'total_notebooks': len(all_summaries),
        'notebooks_with_summaries': len(all_summaries),
        'org_stats': dict(org_counter)
    }


def print_summary_report():
    """Print comprehensive knowledge base summary report."""
    print("=" * 80)
    print("🧬 KNOWLEDGE BASE SUMMARY REPORT")
    print("=" * 80)
    print()

    # 1. API Documentation (ChromaDB)
    print("📚 API DOCUMENTATION (ChromaDB)")
    print("-" * 80)

    chromadb_stats = get_chromadb_stats()

    print(f"  📦 Total Collections:      {chromadb_stats['total_collections']:,}")
    print(f"  📄 Total Documents:        {chromadb_stats['total_docs']:,}")
    print(f"  🔧 API Functions:          {chromadb_stats['total_functions']:,}")
    print(f"  📓 Workflow Chunks:        {chromadb_stats['total_workflows']:,}")
    print()

    # Show top repositories by document count
    if chromadb_stats['repo_stats']:
        print("  Top 10 Repositories by Document Count:")
        sorted_repos = sorted(
            chromadb_stats['repo_stats'].items(),
            key=lambda x: x[1]['total_docs'],
            reverse=True
        )[:10]

        for repo_name, stats in sorted_repos:
            repo_display = repo_name.replace('_vlatest', '').replace('_stable', '')
            print(f"    • {repo_display:<30} {stats['total_docs']:>6,} docs "
                  f"({stats['functions']:,} funcs, {stats['workflows']:,} workflows)")
        print()

    # 2. GitHub Repositories
    print("🐙 GITHUB REPOSITORIES")
    print("-" * 80)

    github_stats = get_github_repo_stats()

    print(f"  📂 Total Cloned Repos:     {github_stats['total_repos']:,}")
    print()

    if github_stats['org_stats']:
        print("  Repositories by Organization:")
        for org_name, count in sorted(github_stats['org_stats'].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"    • {org_name:<30} {count:>4,} repos")
        print()

    # 3. Jupyter Notebook Workflows
    print("📔 JUPYTER NOTEBOOK WORKFLOWS")
    print("-" * 80)

    notebook_stats = get_notebook_summary_stats()

    print(f"  📓 Total Notebooks Indexed: {notebook_stats['total_notebooks']:,}")
    print(f"  ✅ With Summaries:          {notebook_stats['notebooks_with_summaries']:,}")
    print()

    if notebook_stats['org_stats']:
        print("  Notebooks by Organization:")
        for org_name, count in sorted(notebook_stats['org_stats'].items(),
                                       key=lambda x: x[1], reverse=True):
            print(f"    • {org_name:<30} {count:>4,} notebooks")
        print()

    # 4. Overall Summary
    print("=" * 80)
    print("📊 OVERALL SUMMARY")
    print("=" * 80)
    print(f"  API Documentation Pages:   {chromadb_stats['total_docs']:,}")
    print(f"  GitHub Repositories:       {github_stats['total_repos']:,}")
    print(f"  Jupyter Notebooks:         {notebook_stats['total_notebooks']:,}")
    print()
    print(f"  Total Knowledge Sources:   "
          f"{chromadb_stats['total_docs'] + github_stats['total_repos'] + notebook_stats['total_notebooks']:,}")
    print("=" * 80)


def main():
    """Main entry point."""
    try:
        print_summary_report()
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
