#!/usr/bin/env python3
"""
Phase 3.5 Helper - Rebuild Main ChromaDB from Cache

This runs in a separate process to avoid ChromaDB client caching issues.
"""
import asyncio
import json
import os
import sys
from pathlib import Path


async def rebuild_main_chromadb(dest_dir: Path, whitelist_file: Path):
    """Rebuild main ChromaDB from cached GitHub repos and HTML docs."""
    from kai.retrieval.snippets.extractors.github_extractor import GitHubDocumentationExtractor
    from kai.retrieval.snippets.storage.hierarchical_cache import HierarchicalCache
    from kai.core.llm_interface import LLMInterface

    # Load whitelist
    with open(whitelist_file, 'r') as f:
        whitelist = set(json.load(f))

    print("🤖 Initializing LLM interface...")
    llm_interface = LLMInterface(provider="ollama", model="qwen2.5-coder:32b")

    print("📚 Initializing GitHub extractor...")
    print(f"   ChromaDB will be created at: {dest_dir}/chromadb")
    extractor = GitHubDocumentationExtractor("python", llm_interface=llm_interface)

    # Get all cached repos and filter to whitelist
    cache = HierarchicalCache()
    all_org_repos = cache.get_all_cloned_repos()

    print(f"📊 Found {sum(len(repos) for repos in all_org_repos.values())} cached repos")

    # Filter to only whitelisted repos (maintaining the dict structure)
    filtered_repos = {}
    for org, repos_dict in all_org_repos.items():
        filtered_repos[org] = {}
        for repo_name, repo_info in repos_dict.items():
            repo_key = f"{org}/{repo_name}"
            if repo_key in whitelist:
                filtered_repos[org][repo_name] = repo_info

    total_filtered = sum(len(repos) for repos in filtered_repos.values())
    print(f"✅ Filtered to {total_filtered} whitelisted repos\n")

    print("🔄 Extracting from cached repositories...")
    print("   (This will take a while - extracting API docs and workflow code)")

    # Extract with cache_only=True and custom_repo_filter (no internet access)
    results = await extractor.extract_all_repositories(
        cache_only=True,  # Use cache only, no internet
        no_recache_orgs=True,
        no_recache_html=True,
        overwrite_chromadb=True,  # Force reprocessing
        custom_repo_filter=filtered_repos  # Use our filtered whitelist
    )

    # Count results
    extracted = sum(1 for r in results.values() if "error" not in r and "skipped" not in r)
    skipped = sum(1 for r in results.values() if r.get("skipped"))
    errors = sum(1 for r in results.values() if r.get("error"))

    print(f"\n✅ Extraction complete:")
    print(f"   📚 Extracted: {extracted} repos")
    print(f"   ⏭️  Skipped: {skipped} repos")
    print(f"   ❌ Errors: {errors} repos")

    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: _phase35_helper.py <dest_dir> <whitelist_json>")
        sys.exit(1)

    dest_dir = Path(sys.argv[1])
    whitelist_file = Path(sys.argv[2])

    # Set environment variable BEFORE any kai imports
    # This overrides RETRIEVAL_DIR in kai/config/paths.py
    os.environ['KAI_RETRIEVAL_DIR'] = str(dest_dir)

    exit_code = asyncio.run(rebuild_main_chromadb(dest_dir, whitelist_file))
    sys.exit(exit_code)
