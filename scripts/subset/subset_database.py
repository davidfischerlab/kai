#!/usr/bin/env python3
"""Build version of retrieval database with only permissive licenses.

Process:
1. Scan licenses/ directory and build whitelist of permissive repos
2. Copy source files (NOTICE, README, notebook summaries, licenses) - filtered
3. Build summary search index from notebook summaries
3.5. Rebuild main ChromaDB from cached GitHub repos/docs - only whitelisted repos
4. Rebuild collection registry from ChromaDB
5. Build collection embeddings cache
6. Deep verification - ensure no restrictive licenses present

Note: This script rebuilds everything from cached source data (no old ChromaDB access).
It requires cached GitHub repos and HTML docs from previous extraction runs.
"""

import asyncio
import json
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Set, Optional
import re

# Disable telemetry
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_TELEMETRY_ENABLED'] = 'false'

from kai.config.settings import Settings
from kai.config.paths import RETRIEVAL_DIR, AGENT_BASE_DIR
from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag
from kai.retrieval.snippets.storage.chromadb_manager import ChromaDbManager
from kai.utils import setup_logger

logger = setup_logger(__name__)

# GPL-3.0 compatible SPDX license identifiers (according to GitHub API)
# These licenses are compatible with redistributing under GPL-3.0
PERMISSIVE_LICENSES = {
    'MIT',
    'BSD-3-Clause',
    'BSD-2-Clause',
    'Apache-2.0',
    'ISC',
    'Python-2.0',
    'GPL-3.0',
    'LGPL-3.0',
}


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "━" * 70)
    print(text)
    print("━" * 70 + "\n")


def get_github_license(org: str, repo: str, token: Optional[str] = None) -> dict:
    """
    Fetch license info from GitHub API.

    Returns dict with:
        - license: SPDX license identifier (e.g., 'MIT', 'Apache-2.0')
        - license_name: Full license name
        - error: Error message if failed
    """
    url = f"https://api.github.com/repos/{org}/{repo}"

    headers = {
        'Accept': 'application/vnd.github.v3+json',
        'User-Agent': 'kai-license-checker'
    }

    if token:
        headers['Authorization'] = f'token {token}'

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))

            license_info = data.get('license')
            if license_info and license_info.get('key') != 'other':
                return {
                    'license': license_info.get('spdx_id'),
                    'license_name': license_info.get('name'),
                }
            else:
                return {'error': 'No license or custom license'}

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {'error': 'Repository not found'}
        elif e.code == 403:
            return {'error': 'Rate limit exceeded'}
        else:
            return {'error': f'HTTP {e.code}'}
    except Exception as e:
        return {'error': str(e)}


def build_license_whitelist(licenses_dir: Path, github_token: Optional[str] = None) -> tuple[Dict[str, str], Dict[str, str]]:
    """Build whitelist of permissive repos using GitHub API.

    Args:
        licenses_dir: Directory containing LICENSE files (used to get list of repos)
        github_token: Optional GitHub API token for higher rate limits

    Returns:
        (whitelist, exclusions) where:
        - whitelist: {repo_key: license_type} for permissive licenses
        - exclusions: {repo_key: exclusion_reason} for excluded repos
    """
    print_header("🔍 Phase 1: Building Whitelist from GitHub API")
    print(f"Scanning repos in: {licenses_dir}")

    if github_token:
        print("✓ Using GitHub API token\n")
    else:
        print("⚠️  No GitHub token - rate limits will apply\n")

    whitelist = {}
    exclusions = {}

    if not licenses_dir.exists():
        print(f"❌ Error: Licenses directory not found at {licenses_dir}")
        return whitelist, exclusions

    # Scan all org directories to get list of repos
    all_repos = []
    for org_dir in sorted(licenses_dir.iterdir()):
        if not org_dir.is_dir():
            continue

        org_name = org_dir.name
        for repo_dir in sorted(org_dir.iterdir()):
            if not repo_dir.is_dir():
                continue

            repo_name = repo_dir.name
            all_repos.append((org_name, repo_name))

    print(f"Found {len(all_repos)} repos to check\n")

    # Query GitHub API for each repo
    for i, (org_name, repo_name) in enumerate(all_repos, 1):
        repo_key = f"{org_name}/{repo_name}"

        # Print progress every 50 repos
        if i % 50 == 0:
            print(f"  Progress: {i}/{len(all_repos)} repos checked...")

        result = get_github_license(org_name, repo_name, github_token)

        if 'error' in result:
            error_msg = result['error']
            print(f"❌ {repo_key}: {error_msg}")
            exclusions[repo_key] = error_msg
        elif result.get('license') in PERMISSIVE_LICENSES:
            license_id = result['license']
            print(f"✅ {repo_key}: {license_id}")
            whitelist[repo_key] = license_id
        else:
            license_id = result.get('license', 'Unknown')
            print(f"❌ {repo_key}: {license_id} (non-permissive)")
            exclusions[repo_key] = f"{license_id} (non-permissive)"

        # Rate limiting - be nice to GitHub (even with token)
        time.sleep(0.1)

    # Print summary
    total = len(whitelist) + len(exclusions)
    print("\n" + "─" * 70)
    print(f"Summary:")
    print(f"  Total repos: {total}")
    print(f"  ✅ Whitelisted: {len(whitelist)} repos ({100*len(whitelist)/total:.1f}%)")
    print(f"  ❌ Excluded: {len(exclusions)} repos ({100*len(exclusions)/total:.1f}%)")

    # Count exclusion reasons
    no_license_count = sum(1 for reason in exclusions.values() if 'No license' in reason or 'custom license' in reason)
    non_permissive_count = sum(1 for reason in exclusions.values() if 'non-permissive' in reason)
    error_count = len(exclusions) - no_license_count - non_permissive_count

    print(f"\n  Exclusion breakdown:")
    print(f"    - No/custom license: {no_license_count} repos")
    print(f"    - Non-permissive license: {non_permissive_count} repos")
    print(f"    - Errors/other: {error_count} repos")

    return whitelist, exclusions


def copy_source_files(whitelist: Set[str], source_dir: Path, dest_dir: Path):
    """Copy source files (NOTICE, README, summaries, licenses) filtered by whitelist."""
    print_header("📝 Phase 2: Copying Source Files")

    dest_dir.mkdir(parents=True, exist_ok=True)

    # 1. Copy NOTICE and README
    for filename in ['NOTICE', 'README.md']:
        src_file = source_dir / filename
        if src_file.exists():
            shutil.copy2(src_file, dest_dir / filename)
            print(f"✅ Copied: {filename}")
        else:
            print(f"⚠️  Warning: {filename} not found in source")

    # 2. Copy notebook summaries (filtered)
    summaries_src = source_dir / "notebook_summaries"
    summaries_dest = dest_dir / "notebook_summaries"

    if summaries_src.exists():
        summaries_dest.mkdir(parents=True, exist_ok=True)

        # Copy directory structure first
        for subdir in ['notebooks', 'summaries', 'summary_index']:
            (summaries_dest / subdir).mkdir(parents=True, exist_ok=True)

        copied_count = 0
        skipped_count = 0

        # Copy notebooks subdirectory (filtered by whitelist)
        notebooks_src = summaries_src / "notebooks"
        if notebooks_src.exists():
            for org_dir in notebooks_src.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    repo_key = f"{org_dir.name}/{repo_dir.name}"
                    if repo_key in whitelist:
                        dest_org_dir = summaries_dest / "notebooks" / org_dir.name
                        dest_org_dir.mkdir(parents=True, exist_ok=True)

                        dest_repo_dir = dest_org_dir / repo_dir.name
                        if dest_repo_dir.exists():
                            shutil.rmtree(dest_repo_dir)
                        shutil.copytree(repo_dir, dest_repo_dir)
                        copied_count += 1
                    else:
                        skipped_count += 1

        # Copy summaries subdirectory (filtered)
        summaries_subdir_src = summaries_src / "summaries"
        if summaries_subdir_src.exists():
            for org_dir in summaries_subdir_src.iterdir():
                if not org_dir.is_dir():
                    continue

                for repo_dir in org_dir.iterdir():
                    if not repo_dir.is_dir():
                        continue

                    repo_key = f"{org_dir.name}/{repo_dir.name}"
                    if repo_key in whitelist:
                        dest_org_dir = summaries_dest / "summaries" / org_dir.name
                        dest_org_dir.mkdir(parents=True, exist_ok=True)

                        dest_repo_dir = dest_org_dir / repo_dir.name
                        if dest_repo_dir.exists():
                            shutil.rmtree(dest_repo_dir)
                        shutil.copytree(repo_dir, dest_repo_dir)

        print(f"✅ Copied: {copied_count} notebook summary directories (filtered out {skipped_count})")
    else:
        print(f"⚠️  Warning: notebook_summaries/ not found in source")

    # 3. Copy licenses (filtered)
    licenses_src = source_dir / "licenses"
    licenses_dest = dest_dir / "licenses"

    if licenses_src.exists():
        licenses_dest.mkdir(parents=True, exist_ok=True)
        copied_repos = 0

        for org_dir in licenses_src.iterdir():
            if not org_dir.is_dir():
                continue

            for repo_dir in org_dir.iterdir():
                if not repo_dir.is_dir():
                    continue

                repo_key = f"{org_dir.name}/{repo_dir.name}"
                if repo_key in whitelist:
                    dest_org_dir = licenses_dest / org_dir.name
                    dest_org_dir.mkdir(parents=True, exist_ok=True)

                    dest_repo_dir = dest_org_dir / repo_dir.name
                    if dest_repo_dir.exists():
                        shutil.rmtree(dest_repo_dir)
                    shutil.copytree(repo_dir, dest_repo_dir)
                    copied_repos += 1

        print(f"✅ Copied: {copied_repos} license directories (filtered)")
    else:
        print(f"⚠️  Warning: licenses/ not found in source")


async def rebuild_summary_index(dest_dir: Path, whitelist: Set[str]):
    """Rebuild semantic search index from filtered notebook summaries.

    This is equivalent to script 3_build_summary_index.py.
    Note: Filtering already happened in Phase 2 when copying summaries.
    """
    print_header("🔍 Phase 3: Building Summary Search Index")

    storage_path = dest_dir / "notebook_summaries"

    if not storage_path.exists():
        print("⚠️  No notebook summaries found, skipping index build")
        return

    # Initialize components
    storage = NotebookStorage(storage_path)
    summary_search = WorkflowSummaryRag(storage_path)

    # Get summaries (already filtered in Phase 2)
    existing_summaries = storage.get_all_summaries()
    print(f"📊 Found {len(existing_summaries)} summaries (already filtered)")

    if not existing_summaries:
        print("⚠️  No summaries to index, skipping")
        return

    # Build index
    print("🔍 Building semantic search index...")
    indexed_count = summary_search.index_all_summaries(storage)

    # Print stats
    stats = summary_search.get_collection_stats()
    print(f"✅ Indexed {stats['total_summaries']} summaries")
    print(f"   Embedding model: {stats['embedding_model']}")


async def rebuild_main_chromadb(dest_dir: Path, whitelist: Set[str]):
    """Rebuild main ChromaDB from cached GitHub repos and HTML docs.

    This extracts API documentation and workflow code from cached repositories,
    filtered to only include whitelisted repos.

    Runs in a separate subprocess to avoid ChromaDB client caching issues.
    """
    print_header("🗄️  Phase 3.5: Rebuilding Main ChromaDB from Cache")

    # Temporarily rename summary_index to avoid ChromaDB conflicts
    summary_index_dir = dest_dir / "notebook_summaries" / "summary_index"
    temp_summary_dir = dest_dir / "notebook_summaries" / "_summary_index_temp"

    if summary_index_dir.exists():
        print(f"📦 Temporarily moving summary_index out of the way...")
        summary_index_dir.rename(temp_summary_dir)

    # Save whitelist to temp file for subprocess
    whitelist_file = dest_dir / "_whitelist_temp.json"
    with open(whitelist_file, 'w') as f:
        json.dump(list(whitelist), f)

    try:
        # Run Phase 3.5 in separate subprocess to avoid ChromaDB caching
        helper_script = Path(__file__).parent / "_phase35_helper.py"

        print(f"🚀 Running Phase 3.5 in separate subprocess...")
        print(f"   (This avoids ChromaDB client caching issues)\n")

        result = subprocess.run(
            [sys.executable, str(helper_script), str(dest_dir), str(whitelist_file)],
            env={**os.environ, 'KAI_RETRIEVAL_DIR': str(dest_dir)},
            check=True
        )

        if result.returncode != 0:
            print(f"\n❌ Phase 3.5 failed with exit code {result.returncode}")
            return False

        print(f"\n✅ Phase 3.5 completed successfully!")
        return True

    finally:
        # Restore summary_index
        if temp_summary_dir.exists():
            print(f"📦 Restoring summary_index...")
            temp_summary_dir.rename(summary_index_dir)

        # Clean up temp file
        if whitelist_file.exists():
            whitelist_file.unlink()


async def rebuild_collection_registry(dest_dir: Path):
    """Rebuild collection_registry.json from ChromaDB collections.

    This is equivalent to script 4_rebuild_collection_registry.py.
    """
    print_header("📋 Phase 4: Rebuilding Collection Registry")

    import chromadb
    from datetime import datetime

    chroma_path = dest_dir / "chromadb"
    if not chroma_path.exists():
        print("⚠️  ChromaDB not found, skipping registry rebuild")
        return

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=str(chroma_path))

    # Get all collections
    collections = client.list_collections()
    print(f"Found {len(collections)} collections in ChromaDB")

    # Build registry
    registry = {}
    for collection in collections:
        collection_name = collection.name

        # Parse collection name: format is org_repo_type
        parts = collection_name.rsplit('_', 1)
        if len(parts) == 2:
            tool_part, doc_type = parts

            if '_' in tool_part:
                org_repo_parts = tool_part.split('_', 1)
                if len(org_repo_parts) == 2:
                    org, repo = org_repo_parts
                    tool_name = repo
                else:
                    tool_name = tool_part
            else:
                tool_name = tool_part
        else:
            tool_name = collection_name.replace('_', '-')

        # Get collection metadata
        try:
            collection_obj = client.get_collection(collection_name)
            doc_count = collection_obj.count()

            registry[collection_name] = {
                "tool_name": tool_name,
                "version": "latest",
                "created_at": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "collection_name": collection_name,
                "document_count": doc_count
            }
        except Exception as e:
            print(f"❌ Error processing {collection_name}: {e}")

    # Save registry
    registry_path = dest_dir / "collection_registry.json"
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    print(f"✅ Registry saved: {len(registry)} entries")

    # Clear the ChromaDB singleton to avoid conflicts with Phase 5
    # Phase 5 needs to create a ChromaDbManager with different settings
    import chromadb.api.shared_system_client
    chromadb.api.shared_system_client.SharedSystemClient._identifier_to_system.clear()


async def build_embedding_cache(dest_dir: Path):
    """Build collection embedding cache for performance.

    This is equivalent to script 5_build_summary_collection_cache.py.
    """
    print_header("⚡ Phase 5: Building Embedding Cache")

    chroma_path = dest_dir / "chromadb"
    if not chroma_path.exists():
        print("⚠️  ChromaDB not found, skipping cache build")
        return

    # Initialize ChromaDB manager
    manager = ChromaDbManager(dest_dir)

    print("Starting background initialization...")
    manager.start_background_initialization()

    print("Waiting for completion...")
    await manager.wait_for_background_initialization()

    print(f"✅ Cache created: {len(manager._collection_embedding_cache)} collection embeddings")


async def verify_no_restrictive_licenses(dest_dir: Path, github_token: Optional[str] = None) -> bool:
    """Deep verification that no restrictive licenses are present using GitHub API."""
    print_header("🔒 Phase 6: Deep Verification")

    print("Verifying no restrictive licenses in final database using GitHub API...\n")

    # Check licenses directory
    licenses_dir = dest_dir / "licenses"
    if not licenses_dir.exists():
        print("⚠️  No licenses directory found")
        return True

    all_clean = True
    restrictive_found = []

    # Get all repos
    all_repos = []
    for org_dir in sorted(licenses_dir.iterdir()):
        if not org_dir.is_dir():
            continue
        for repo_dir in sorted(org_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            all_repos.append((org_dir.name, repo_dir.name))

    # Verify each repo via GitHub API
    for org_name, repo_name in all_repos:
        repo_key = f"{org_name}/{repo_name}"

        result = get_github_license(org_name, repo_name, github_token)

        if 'error' in result:
            print(f"❌ VERIFICATION FAILED: {repo_key} - {result['error']}")
            restrictive_found.append((repo_key, result['error']))
            all_clean = False
        elif result.get('license') not in PERMISSIVE_LICENSES:
            license_id = result.get('license', 'Unknown')
            print(f"❌ VERIFICATION FAILED: {repo_key} has non-permissive license: {license_id}")
            restrictive_found.append((repo_key, f"{license_id} (non-permissive)"))
            all_clean = False

        # Rate limiting
        time.sleep(0.1)

    if all_clean:
        print(f"✅ Verification passed: All {len(all_repos)} repos have permissive licenses")
    else:
        print(f"\n❌ Verification failed: Found {len(restrictive_found)} non-permissive repos:")
        for repo, reason in restrictive_found:
            print(f"   - {repo}: {reason}")

    return all_clean


def print_final_summary(whitelist: Set[str], exclusions: Dict[str, str], dest_dir: Path):
    """Print final summary of the build."""
    print_header("✅ Build Complete")

    print(f"Output directory: {dest_dir}\n")
    print(f"📊 Repository Summary:")
    print(f"   ✅ Included: {len(whitelist)} repos with permissive licenses")
    print(f"   ❌ Excluded: {len(exclusions)} repos\n")

    print(f"📁 Created Files:")
    for item in ['NOTICE', 'README.md', 'licenses/', 'notebook_summaries/',
                 'chromadb/', 'collection_registry.json', 'collection_embeddings_cache.json']:
        path = dest_dir / item
        if path.exists():
            if path.is_file():
                size = path.stat().st_size
                print(f"   ✅ {item} ({size:,} bytes)")
            else:
                print(f"   ✅ {item}/")
        else:
            print(f"   ⚠️  {item} (not created)")


async def main():
    """Main execution function."""
    start_time = time.time()

    # Default paths
    source_dir = AGENT_BASE_DIR / "retrieval"
    dest_dir = AGENT_BASE_DIR / "retrieval_sharing"

    # Check for GitHub token
    github_token = None
    token_file = Path.home() / ".github_token"
    if token_file.exists():
        github_token = token_file.read_text().strip()

    print("=" * 70)
    print("Build Sharing Retrieval Database")
    print("=" * 70)
    print(f"\nSource: {source_dir}")
    print(f"Destination: {dest_dir}")

    # Phase 1: Build whitelist using GitHub API
    licenses_dir = source_dir / "licenses"
    whitelist, exclusions = build_license_whitelist(licenses_dir, github_token)

    if not whitelist:
        print("\n❌ Error: No repos with permissive licenses found!")
        return 1

    # Convert to set of keys
    whitelist_keys = set(whitelist.keys())

    # Phase 2: Copy source files
    copy_source_files(whitelist_keys, source_dir, dest_dir)

    # Phase 3: Build summary search index
    await rebuild_summary_index(dest_dir, whitelist_keys)

    # Phase 3.5: Rebuild main ChromaDB from cached GitHub repos/docs
    await rebuild_main_chromadb(dest_dir, whitelist_keys)

    # Phase 4: Rebuild collection registry
    await rebuild_collection_registry(dest_dir)

    # Phase 5: Build embedding cache
    await build_embedding_cache(dest_dir)

    # Phase 6: Verification using GitHub API
    verification_passed = await verify_no_restrictive_licenses(dest_dir, github_token)

    if not verification_passed:
        print("\n⚠️  WARNING: Verification failed. Review errors before sharing.")
        return 1

    # Final summary
    print_final_summary(whitelist_keys, exclusions, dest_dir)

    elapsed = time.time() - start_time
    print(f"\n⏰ Total time: {elapsed/60:.1f} minutes")

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
