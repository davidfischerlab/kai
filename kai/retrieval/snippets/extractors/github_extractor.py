"""GitHub documentation extractor for multiple languages."""
import asyncio
import subprocess
import time
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re

# Removed DocumentationContentProcessor import (unused)
from .llm_prompts import format_repo_screening_prompt
from .readthedocs_crawler import ReadTheDocsCrawler
from .hierarchical_workflow_parser import HierarchicalWorkflowParser
from ..storage.chromadb_manager import ChromaDbManager
from ..storage.hierarchical_cache import HierarchicalCache
from kai.config.paths import BIOINFORMATICS_CACHE_DIR
from kai.config.settings import settings
from kai.utils import setup_logger

logger = setup_logger(__name__)


class GitHubDocumentationExtractor:
    """
    GitHub Documentation and Workflow Extractor.
    
    This class handles repository discovery, cloning, and extraction of documentation and workflows
    from GitHub organizations. It works in partnership with ReadTheDocsCrawler for API documentation
    and focuses on local workflow extraction.
    
    Architecture:
    ============
    
    1. Repository Discovery:
       - Fetches repositories from configured GitHub organizations
       - Filters by language, stars, and LLM-based content analysis
       - Caches repository metadata to avoid repeated API calls
    
    2. Documentation Extraction (via ReadTheDocsCrawler):
       - Extracts hints from repository files (pyproject.toml, README, etc.)
       - Passes hints to ReadTheDocsCrawler for comprehensive API documentation
       - Does NOT process documentation locally (delegated to ReadTheDocsCrawler)
    
    3. Workflow Extraction (local processing):
       - Finds workflow files (notebooks, examples, tutorials) using LLM selection
       - Workflow parsing is handled separately by HierarchicalWorkflowParser
    
    Interface with ReadTheDocsCrawler:
    =================================
    
    GitHubExtractor provides hints → ReadTheDocsCrawler handles documentation
    
    Hints provided:
    - detected_url: ReadTheDocs URL extracted from repository files
    - org_name: Organization name for cache organization
    - lib_name: Library name for URL construction
    
    ReadTheDocsCrawler handles:
    - URL discovery and pattern matching
    - Exception handling for non-standard documentation sites
    - HTML downloading and content extraction
    - Function signature and parameter parsing
    - Hierarchical caching by organization/library/version
    
    Key Methods:
    ===========
    
    Initialization & Configuration:
    - __init__(): Initialize extractor with language config, directories, and LLM interface
    - github_organizations @property: Get configured GitHub organizations for this language
    - _is_relevant_language(): Check if repository language matches extractor language
    
    Repository Discovery (GitHub API):
    - discover_all_organizations(): Unified discovery with intelligent rate limiting
    - _bulk_discovery_with_auth(): Sequential fetching for authenticated users (5000 req/hour)
    - _conservative_discovery_no_auth(): Sequential fetching for unauthenticated users (60 req/hour)
    - _fetch_organization_repos(): Core repository fetching logic with retry and rate limiting
    
    Content Filtering:
    - _llm_content_filter(): Use LLM to filter repositories by relevance to bioinformatics
    
    Repository Management:
    - _clone_repository(): Clone or update repositories locally for processing
    - _run_git_command(): Execute git commands safely with error handling
    - _extract_org_name_from_repo_path(): Extract organization name from repository path
    
    Main Extraction Pipeline:
    - extract_all_repositories(): Extract documentation/workflows from all discovered repos
    - _extract_repository_structured(): Core extraction pipeline for single repository
    
    ReadTheDocsCrawler Integration:
    - _fetch_readthedocs_via_crawler(): Interface with ReadTheDocsCrawler using extracted hints
    - _detect_readthedocs_url_from_repo(): Extract ReadTheDocs URL hints from repository files
    
    Workflow Processing:
    - _find_workflow_files(): Find workflow files using configured patterns
    Storage:
    - Direct ChromaDB indexing: Save extraction results directly to ChromaDB knowledge base
    
    Configuration:
    =============
    
    Language-specific settings in LANGUAGE_CONFIGS including:
    - organizations: GitHub orgs to process with filtering criteria
    - workflows: File patterns and folders for workflow detection
    
    Note: API documentation extraction has been fully delegated to ReadTheDocsCrawler.
    This class no longer processes local documentation files directly.
    """
    
    # Language-specific configurations
    LANGUAGE_CONFIGS = {
        "python": {
            "organizations": {
                "aertslab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Aerts lab
                "BayraktarLab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Bayraktar lab
                "bioFAM": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Stegle lab
                "bunnelab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Bunne lab
                "dpeerlab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Pe'er lab
                "epigen": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Bock lab
                "lueckenlab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Lücken lab
                "Lotfollahi-lab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Lotfollahi lab
                "MarioniLab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Marioni lab
                "mlbio-epfl": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Brbic lab
                "saezlab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Saez-Rodriguez lab
                "scverse": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # scverse organization
                "ShalekLab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Shalek lab
                "teichlab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Teichmann lab
                "theislab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Theis lab
                "YosefLab": {"include_forks": False, "min_stars": 0, "use_llm_filter": False},  # Yosef lab
                #"broadinstitute": {"include_forks": False, "min_stars": 1, "use_llm_filter": True},
                #"chanzuckerberg": {"include_forks": False, "min_stars": 1, "use_llm_filter": True}
            },
            "workflows": {
                "file_patterns": ["*.ipynb"]
            },
            "relevant_languages": ["Python", "Jupyter Notebook", None]
        },
        "r": {
            "organizations": {
                #"satijalab": {"include_forks": False, "min_stars": 1, "use_llm_filter": True},
                #"Bioconductor": {"include_forks": False, "min_stars": 1, "use_llm_filter": True},
                #"hbctraining": {"include_forks": False, "min_stars": 1, "use_llm_filter": True},
                #"cole-trapnell-lab": {"include_forks": False, "min_stars": 1, "use_llm_filter": True}
            },
            "workflows": {
                "file_patterns": ["*.Rmd"]
            },
            "relevant_languages": ["R", "Jupyter Notebook", None]
        }
    }
    
    def __init__(self, language: str, llm_interface=None):
        """Initialize the GitHub extractor.
        
        Args:
            language: Target language ('python' or 'r')
            llm_interface: LLM interface for content filtering (optional)
        """
        if language not in self.LANGUAGE_CONFIGS:
            raise ValueError(f"Unsupported language: {language}. Supported: {list(self.LANGUAGE_CONFIGS.keys())}")
        
        self.language = language
        self.config = self.LANGUAGE_CONFIGS[language]
        
        # Use hierarchical structure for repositories
        self.organizations_dir = BIOINFORMATICS_CACHE_DIR
        self.organizations_dir.mkdir(parents=True, exist_ok=True)
        
        self.llm_interface = llm_interface
        self.cache = HierarchicalCache()
        
        # Initialize ReadTheDocs crawler for ReadTheDocs integration
        self.doc_crawler = ReadTheDocsCrawler()
        
        # Initialize workflow parser for local workflow processing
        self.workflow_parser = HierarchicalWorkflowParser()
        
        logger.info(f"GitHub extractor initialized for {language}")
        
        # Initialize ChromaDB manager for direct indexing
        try:
            self.chromadb_manager = ChromaDbManager(settings.KNOWLEDGE_BASE_PATH)
            logger.info("ChromaDB manager initialized in constructor")
        except Exception as e:
            logger.warning(f"Failed to initialize ChromaDB manager in constructor: {e}")
            self.chromadb_manager = None
    
    @property
    def github_organizations(self) -> Dict[str, Dict[str, Any]]:
        """Return language-specific GitHub organizations configuration."""
        return self.config["organizations"]
    

    
    # Removed unused path methods - these were never actually used in the extraction logic
    
    def _is_relevant_language(self, repo_language: str) -> bool:
        """Check if repository language is relevant for this extractor."""
        return repo_language in self.config["relevant_languages"]
    
    async def discover_all_organizations(self, auth_token: Optional[str] = None,
                                        delay_between_pages: int = 2,
                                        delay_between_orgs: int = 10,
                                        no_recache_orgs: bool = False) -> Dict[str, Dict[str, Any]]:
        """Discover repositories from all configured organizations with intelligent rate limiting.

        Automatically chooses between bulk (with auth) and conservative (no auth) discovery
        based on authentication availability to respect GitHub API rate limits.

        Args:
            auth_token: GitHub Personal Access Token for higher rate limits (5000/hour vs 60/hour)
            delay_between_pages: Seconds to wait between page requests (used in conservative mode)
            delay_between_orgs: Seconds to wait between organizations (used in conservative mode)
            no_recache_orgs: If True, skip API calls for already-cached orgs (use existing api_cache.json)

        Returns:
            Dictionary mapping organization names to their discovered repositories
        """
        logger.info(f"Discovering {self.language} repositories from all organizations")

        all_org_repos = {}

        # Check cache first for each organization
        for org_name in self.github_organizations.keys():
            cached_repos = self.cache.get_cached_repos(org_name, self.language)
            if cached_repos:
                all_org_repos[org_name] = cached_repos
                logger.info(f"Using cached data for {org_name}: {len(cached_repos)} repos")
            else:
                all_org_repos[org_name] = {}

        # Determine which organizations need fresh data
        if no_recache_orgs:
            # Only fetch orgs with no cached data
            orgs_to_fetch = [org for org, repos in all_org_repos.items() if not repos]
        else:
            # Fetch all orgs to refresh api_cache.json
            orgs_to_fetch = list(self.github_organizations.keys())
        
        if not orgs_to_fetch:
            logger.info("All organization data found in cache")
            return all_org_repos
        
        logger.info(f"Fetching fresh data for: {', '.join(orgs_to_fetch)}")
        
        # Choose discovery strategy based on authentication
        if auth_token:
            logger.info("Using bulk discovery (authenticated - 5000 req/hour)")
            fetched_repos = await self._bulk_discovery_with_auth(orgs_to_fetch, auth_token)
        else:
            logger.info("Using conservative discovery (unauthenticated - 60 req/hour)")
            fetched_repos = await self._conservative_discovery_no_auth(orgs_to_fetch, delay_between_pages, delay_between_orgs)
        
        # Merge fetched data with cached data
        all_org_repos.update(fetched_repos)
        
        return all_org_repos
    
    async def _bulk_discovery_with_auth(self, orgs_to_fetch: List[str], auth_token: str) -> Dict[str, Dict[str, Any]]:
        """Sequential discovery with authentication to avoid ChromaDB concurrency issues."""
        try:
            import aiohttp
            headers = {
                "Accept": "application/vnd.github.v3+json",
                "Authorization": f"token {auth_token}"
            }

            async with aiohttp.ClientSession(headers=headers) as session:
                # Fetch organizations sequentially to avoid concurrent ChromaDB writes
                fetched_repos = {}
                for i, org_name in enumerate(orgs_to_fetch):
                    try:
                        logger.info(f"📋 Processing organization {i+1}/{len(orgs_to_fetch)}: {org_name}")
                        result = await self._fetch_organization_repos(session, org_name)
                        fetched_repos[org_name] = result

                        # Cache the results
                        self.cache.cache_organization_metadata(org_name, self.language, result)
                        logger.info(f"Discovered and cached {len(result)} {self.language} repositories from {org_name}")

                    except Exception as e:
                        logger.error(f"Error fetching repositories for {org_name}: {e}")
                        fetched_repos[org_name] = {}

                    # Add delay between organizations to prevent rate limiting
                    if i < len(orgs_to_fetch) - 1:
                        await asyncio.sleep(1)

                return fetched_repos
        
        except Exception as e:
            logger.error(f"Error in bulk discovery: {e}")
            return {org: {} for org in orgs_to_fetch}
    
    async def _conservative_discovery_no_auth(self, orgs_to_fetch: List[str], delay_between_pages: int, delay_between_orgs: int) -> Dict[str, Dict[str, Any]]:
        """Conservative discovery without authentication - sequential with delays for low rate limits."""
        import aiohttp
        
        headers = {"Accept": "application/vnd.github.v3+json"}
        
        async with aiohttp.ClientSession(headers=headers) as session:
            fetched_repos = {}
            
            for i, org_name in enumerate(orgs_to_fetch):
                logger.info(f"📦 Processing organization {i+1}/{len(orgs_to_fetch)}: {org_name}")
                
                try:
                    result = await self._fetch_organization_repos(session, org_name, delay_between_pages)
                    fetched_repos[org_name] = result
                    
                    # Cache the results
                    self.cache.cache_organization_metadata(org_name, self.language, result)
                    logger.info(f"Discovered and cached {len(result)} {self.language} repositories from {org_name}")
                    
                except Exception as e:
                    logger.error(f"Error fetching repositories from {org_name}: {e}")
                    fetched_repos[org_name] = {}
                
                # Wait between organizations (except for the last one)
                if i < len(orgs_to_fetch) - 1:
                    logger.info(f"⏳ Waiting {delay_between_orgs} seconds before next organization...")
                    await asyncio.sleep(delay_between_orgs)
            
            return fetched_repos
    
    async def _fetch_organization_repos(self, session: 'aiohttp.ClientSession', org_name: str, delay_between_pages: int = 0) -> Dict[str, Dict[str, Any]]:
        """Fetch all repositories from a single organization in bulk, with retry and rate limit handling."""
        org_config = self.github_organizations.get(org_name, {})
        discovered = {}
        
        max_retries = 5
        base_delay = 5  # seconds
        
        try:
            all_repos = []
            page = 1
            should_continue = True

            while should_continue:
                url = f"https://api.github.com/orgs/{org_name}/repos?per_page=100&page={page}&sort=stars&direction=desc"
                retries = 0
                page_success = False

                while retries < max_retries:
                    async with session.get(url) as response:
                        status = response.status
                        headers = response.headers
                        if status == 200:
                            repos = await response.json()

                            # Check if we got any repositories
                            if not repos:
                                # Empty page means we've reached the end
                                logger.info(f"  ✅ No more repositories found (page {page})")
                                should_continue = False
                                page_success = True
                                break

                            all_repos.extend(repos)
                            logger.info(f"  ✅ Page {page}: Found {len(repos)} repositories")

                            # Check if this is the last page
                            if len(repos) < 100:
                                # Less than 100 means this is the last page
                                logger.info(f"  ✅ Last page reached (page {page})")
                                should_continue = False
                            else:
                                # Full page, there might be more
                                page += 1

                                # Add delay between pages if specified (for conservative mode)
                                if delay_between_pages > 0:
                                    logger.info(f"  ⏳ Waiting {delay_between_pages} seconds before next page...")
                                    await asyncio.sleep(delay_between_pages)

                            page_success = True
                            break  # Success, exit retry loop

                        elif status in (403, 429) or headers.get('X-RateLimit-Remaining') == '0':
                            # Rate limited
                            remaining = headers.get('X-RateLimit-Remaining', '0')
                            reset_time = headers.get('X-RateLimit-Reset')

                            # Cache rate limit info
                            if reset_time:
                                self.cache.update_rate_limit_info(org_name, int(remaining), int(reset_time))
                                wait_seconds = int(reset_time) - int(time.time()) + 5
                                wait_seconds = max(wait_seconds, 10)
                                print(f"⏳ Rate limited by GitHub API for org {org_name}. Waiting {wait_seconds} seconds...")
                                await asyncio.sleep(wait_seconds)
                            else:
                                print(f"⏳ Rate limited by GitHub API for org {org_name}. Waiting 60 seconds...")
                                await asyncio.sleep(60)
                            retries += 1
                        else:
                            # Other error, retry with backoff
                            print(f"⚠️  Error fetching {url} (status {status}), retry {retries+1}/{max_retries}")
                            await asyncio.sleep(base_delay * (2 ** retries))
                            retries += 1
                else:
                    # Max retries exceeded
                    print(f"❌ Failed to fetch {url} after {max_retries} retries.")
                    should_continue = False

                # If page fetch failed completely, stop pagination
                if not page_success:
                    break
            
            # Process all repositories locally
            for repo in all_repos:
                repo_name = repo["name"]
                full_name = repo["full_name"]
                
                # Apply filters
                if repo["fork"] and not org_config.get("include_forks", False):
                    continue
                
                if repo["stargazers_count"] < org_config.get("min_stars", 1):
                    continue
                    
                if repo_name in org_config.get("exclude_repos", []):
                    continue
                
                # Language-specific filtering
                if not self._is_relevant_language(repo["language"]):
                    continue
                
                # LLM filtering is applied during extraction, not discovery.
                # Ensures all org repositories are cached.
                
                # Create configuration for this repository
                discovered[repo_name] = {
                    "repo": full_name,
                    "language": repo["language"],
                    "stars": repo["stargazers_count"],
                    "description": repo["description"] or "",
                    "homepage": repo["homepage"]
                }
        
        except Exception as e:
            logger.error(f"Error fetching repositories from {org_name}: {e}")
        
        return discovered

    def _extract_org_name_from_repo_path(self, repo_path: Path) -> str:
        """Extract organization name from repository path."""
        # Path structure: bioinformatics_cache/{org_name}/repos/{repo_name}
        path_parts = repo_path.parts
        if 'bioinformatics_cache' in path_parts:
            org_index = path_parts.index('bioinformatics_cache')
            if org_index + 1 < len(path_parts):
                return path_parts[org_index + 1]
        
        # Fallback to "unknown" if can't extract
        return "unknown"
    
    # Removed tutorial and example extraction methods - these are now handled by workflow parsing
    
    async def _clone_repository(self, repo: str, lib_name: str) -> Path:
        """Clone repository to organization-specific directory."""
        # Check file descriptor count before attempting clone
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            open_files = len(process.open_files())

            # If too many files open, force cleanup
            if open_files > 100:  # Conservative limit
                logger.warning(f"Too many open files ({open_files}), forcing cleanup...")
                gc.collect()
                # Give it a moment
                await asyncio.sleep(1)
        except ImportError:
            # psutil not installed, use basic cleanup
            gc.collect()
        except Exception:
            pass  # Ignore other errors in check

        # Extract organization from repository name
        org_name = "unknown"
        if "/" in repo:
            org_name = repo.split("/")[0]
        
        # Use hierarchical structure: organizations/{org_name}/repos/{lib_name}
        org_dir = self.organizations_dir / org_name
        org_dir.mkdir(parents=True, exist_ok=True)
        
        repos_dir = org_dir / "repos"
        repos_dir.mkdir(parents=True, exist_ok=True)
        
        repo_path = repos_dir / lib_name
        
        if repo_path.exists():
            logger.info(f"Repository {lib_name} already exists, updating...")
            try:
                # Update existing repository
                returncode, stdout, stderr = self._run_git_command(
                    ["pull"], repo_path
                )
                if returncode != 0:
                    logger.warning(f"Failed to update {lib_name}: {stderr}")
            except Exception as e:
                logger.warning(f"Error updating {lib_name}: {e}")
        else:
            logger.info(f"Cloning {repo} to {repo_path}")
            try:
                returncode, stdout, stderr = self._run_git_command(
                    ["clone", f"https://github.com/{repo}.git", str(repo_path)]
                )
                if returncode != 0:
                    logger.error(f"Failed to clone {repo}: {stderr}")
                    raise Exception(f"Git clone failed: {stderr}")
            except Exception as e:
                logger.error(f"Error cloning {repo}: {e}")
                raise
        
        return repo_path
    
    def _run_git_command(self, args: List[str], cwd: Optional[Path] = None) -> Tuple[int, str, str]:
        """Run a git command with proper resource cleanup."""
        cmd = ["git"] + args
        cwd = cwd or self.organizations_dir

        try:
            # Use Popen with explicit cleanup to avoid file handle leaks
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                close_fds=True  # Critical: close file descriptors
            )

            try:
                stdout, stderr = process.communicate(timeout=300)
                return process.returncode, stdout, stderr
            finally:
                # Ensure process is terminated and cleaned up
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        except Exception as e:
            logger.error(f"Error running git command: {e}")
            return 1, "", str(e)

    def _read_repository_readme(self, org_name: str, repo_name: str) -> str:
        """Read README content from a cloned repository."""
        try:
            repo_path = get_org_repos_dir(org_name) / repo_name
            if not repo_path.exists():
                return ""
            
            # Look for common README file names
            readme_files = ['README.md', 'README.rst', 'README.txt', 'README', 'readme.md', 'readme.rst']
            
            for readme_file in readme_files:
                readme_path = repo_path / readme_file
                if readme_path.exists():
                    try:
                        with open(readme_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Return first 1000 characters to avoid huge prompts
                            return content[:1000]
                    except (UnicodeDecodeError, IOError):
                        continue
            
            return ""
        except Exception as e:
            logger.debug(f"Error reading README for {repo_name}: {e}")
            return ""

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

    async def _llm_content_filter(self, repo_name: str, description: str, topics: List[str], org_name: str = None) -> bool:
        """Use LLM to filter repository content with retry mechanism."""
        if self.llm_interface is None:
            logger.error("No LLM interface provided for content filtering")
            return False
        
        try:
            # Read README content if repository is cloned
            readme_content = ""
            if org_name:
                readme_content = self._read_repository_readme(org_name, repo_name)
            
            # Use the LLM-based filtering with the proper prompt
            prompt = format_repo_screening_prompt(repo_name, description, topics, readme_content)
            
            # Retry loop for unclear responses
            max_retries = 5
            for attempt in range(max_retries):
                response = await self.llm_interface.generate(
                    prompt=prompt,
                    task_type="filtering",
                    max_tokens=50,
                    temperature=0.1
                )
                
                # Parse the response to determine if repository is relevant
                response_text = response.strip().upper()
                
                # Check if the response starts with "RELEVANT" or "NOT_RELEVANT"
                if response_text.startswith("RELEVANT"):
                    logger.debug(f"LLM approved repository: {repo_name} (attempt {attempt + 1})")
                    return True
                elif response_text.startswith("NOT_RELEVANT"):
                    logger.debug(f"LLM rejected repository: {repo_name} (attempt {attempt + 1})")
                    return False
                else:
                    # Response is unclear, try again with clarification prompt
                    logger.warning(f"Unclear LLM response for {repo_name} (attempt {attempt + 1}): {response_text}")
                    if attempt < max_retries - 1:  # Don't retry on last attempt
                        from .llm_prompts import REPO_SCREENING_CLARIFICATION_PROMPT
                        prompt = REPO_SCREENING_CLARIFICATION_PROMPT
                    else:
                        # Final attempt failed, default to rejecting
                        logger.error(f"LLM failed to provide clear answer for {repo_name} after {max_retries} attempts, defaulting to reject")
                        return False
            
            # Should not reach here, but default to rejecting if it does
            return False
                
        except Exception as e:
            logger.error(f"Error in LLM content filtering for {repo_name}: {e}")
            return False
    
    async def extract_all_repositories(self, cache_only: bool = False, no_recache_orgs: bool = False,
                                       no_recache_html: bool = False, overwrite_chromadb: bool = False,
                                       auth_token: Optional[str] = None,
                                       custom_repo_filter: Optional[Dict[str, List[str]]] = None) -> Dict[str, Dict[str, Any]]:
        """Extract documentation and workflows from all discovered repositories using bulk discovery.

        Args:
            cache_only: If True, only use local caches, no internet access for downloading
            no_recache_orgs: If True, skip GitHub API calls for already-cached orgs (use existing api_cache.json)
            no_recache_html: If True, don't redownload existing HTML caches, but download missing ones
            overwrite_chromadb: If True, reprocess repositories even if already in ChromaDB
            custom_repo_filter: If provided, only process repos in this dict (org -> list of repo names)
            auth_token: GitHub Personal Access Token for higher rate limits (5000/hour vs 60/hour)

        Cache behavior:
            - cache_only=True: No internet access, only use existing local HTML caches
            - no_recache_orgs=True: Use existing api_cache.json for org repo lists, skip GitHub API calls
            - no_recache_html=True: Use existing HTML caches if available, download only missing objects
            - overwrite_chromadb=False: Skip repositories already in ChromaDB
            - All False: Normal mode, refresh api_cache.json and may redownload HTML caches for freshness

        Returns:
            Dictionary mapping library names to extraction results, containing either:
            - Extraction results from _extract_repository_structured()
            - Skip status: {"skipped": "reason"}
            - Error status: {"error": "error_message"}
        """
        all_results = {}

        # Use discovery to get all repositories at once
        if custom_repo_filter:
            # Use custom filter if provided (for subsetting/filtering repos)
            logger.info("Using custom repository filter")
            all_org_repos = custom_repo_filter
            logger.info(f"Custom filter: {sum(len(repos) for repos in all_org_repos.values())} repositories across {len(all_org_repos)} organizations")
        elif cache_only:
            # In cache-only mode, get all locally cloned repositories
            logger.info("Cache-only mode: processing locally cloned repositories")
            all_org_repos = self.cache.get_all_cloned_repos()
            logger.info(f"Found {sum(len(repos) for repos in all_org_repos.values())} cloned repositories across {len(all_org_repos)} organizations")
        else:
            all_org_repos = await self.discover_all_organizations(auth_token=auth_token, no_recache_orgs=no_recache_orgs)

        # Batch check ChromaDB for all repositories at once to avoid connection issues
        if not overwrite_chromadb and self.chromadb_manager is not None:
            logger.info("🔍 Batch checking ChromaDB for existing repositories...")
            # Get all repo names first
            all_repo_names = []
            for org_name, discovered in all_org_repos.items():
                all_repo_names.extend(discovered.keys())

            # Use efficient batch checking method
            existing_repos_dict = self.chromadb_manager.has_repositories_batch(all_repo_names)
            existing_repos = {lib_name for lib_name, exists in existing_repos_dict.items() if exists}

            # Mark existing repos as skipped
            for lib_name in existing_repos:
                all_results[lib_name] = {"skipped": "Already in ChromaDB"}

            logger.info(f"✓ Found {len(existing_repos)} repos already in ChromaDB, will skip those")
            logger.info(f"Total repos to check: {len(all_repo_names)}, Will process: {len(all_repo_names) - len(existing_repos)}")

            # Debug: Check specific problematic repos
            test_repos = ['scproto', 'neural_organoid_atlas', 'single-cell-best-practices']
            for repo in test_repos:
                if repo in existing_repos:
                    logger.debug(f"DEBUG: ✅ {repo} correctly identified as existing, added to all_results")
                elif repo in all_repo_names:
                    logger.debug(f"DEBUG: ❌ {repo} NOT identified as existing but is in repo list - BUG!")
                else:
                    logger.debug(f"DEBUG: ℹ️  {repo} not in current repo discovery")


            # Force cleanup after batch check
            gc.collect()

        # Process only repositories that need processing
        repo_batch_size = 5  # Can increase batch size since we're skipping most repos
        repo_count = 0

        for org_name, discovered in all_org_repos.items():
            for lib_name, config in discovered.items():
                # Skip if already in ChromaDB
                if lib_name in all_results:
                    continue


                # Process repos in batches
                if repo_count > 0 and repo_count % repo_batch_size == 0:
                    logger.info(f"Processed {repo_count} repos, running garbage collection...")
                    gc.collect()  # Free memory and file handles between batches

                try:

                    # LLM filter (if enabled) - only after ChromaDB check
                    org_cfg = self.github_organizations.get(org_name, {})
                    use_llm = org_cfg.get("use_llm_filter", False)
                    passes_llm = True
                    if use_llm and self.llm_interface is not None:
                        passes_llm = await self._llm_content_filter(
                            lib_name,
                            config.get("description", ""),
                            [],  # topics not available in config
                            org_name  # pass org_name for README reading
                        )
                    if not passes_llm:
                        logger.info(f"LLM filter rejected {lib_name}, skipping.")
                        all_results[lib_name] = {"skipped": "LLM filter rejected"}
                        continue

                    # Use structured pipeline for extraction with cache mode parameters
                    result = await self._extract_repository_structured(
                        lib_name, config, cache_only=cache_only, no_recache_html=no_recache_html,
                        overwrite_chromadb=overwrite_chromadb
                    )
                    all_results[lib_name] = result

                    # Increment counter after processing
                    repo_count += 1

                except Exception as e:
                    logger.error(f"Error extracting {lib_name}: {e}")
                    all_results[lib_name] = {"error": str(e)}

        return all_results

    async def _fetch_readthedocs_via_crawler(self, repo_path: Path, lib_name: str,
                                           cache_only: bool = False, no_recache_html: bool = False) -> Dict[str, Any]:
        """Fetch documentation from ReadTheDocs using ReadTheDocsCrawler with cache control.

        Args:
            repo_path: Path to the local repository
            lib_name: Name of the library
            cache_only: If True, only use local caches, no internet access for downloading
            no_recache_html: If True, don't redownload existing HTML caches, but download missing ones

        Cache behavior for API documentation:
            - HTML files are cached in hierarchical structure by org/library/version
            - If HTML cache exists AND (no_recache_html OR cache_only): Use existing HTML cache
            - If HTML cache exists AND neither flag set: Delete cache and recreate for freshness
            - If HTML cache missing AND cache_only: Skip download (no internet access)
            - If HTML cache missing AND no_recache_html: Download new HTML cache
            - If HTML cache missing AND neither flag: Download new HTML cache
            - Python dictionaries are returned directly from HTML processing

        Returns:
            Dictionary with success status and error details if any:
            {
                "success": bool,
                "error": str (if failed),
                "version": str (if successful),
                "api_documentation": dict (if successful)
            }
        """
        try:
            # Check if HTML cache exists
            org_name = self._extract_org_name_from_repo_path(repo_path)
            
            # Check if HTML cache exists
            from kai.config.paths import get_org_html_dir
            html_cache_dir = get_org_html_dir(org_name)
            
            # Check for existing HTML cache directory
            cache_patterns = [
                f"{lib_name}_stable",
                f"{lib_name}_latest"
            ]
            
            existing_cache = None
            existing_version = None
            for pattern in cache_patterns:
                cache_dir = html_cache_dir / pattern
                if cache_dir.exists() and any(cache_dir.iterdir()):
                    existing_cache = cache_dir
                    existing_version = pattern.split('_')[-1]  # Extract version
                    break
            
            # Implement cache control logic
            if existing_cache:
                # Check if this download was marked as incomplete
                if self.cache.is_html_download_incomplete(org_name, lib_name, existing_version):
                    logger.info(f"Found incomplete HTML download for {lib_name} - cleaning up and re-downloading")
                    import shutil
                    shutil.rmtree(existing_cache)
                    self.cache.mark_html_download_complete(org_name, lib_name, existing_version)
                    existing_cache = None
                elif no_recache_html or cache_only:
                    # Check if this is a failure marker cache
                    failure_marker_path = existing_cache / "no_documentation_found.json"
                    if failure_marker_path.exists():
                        logger.info(f"Found failure marker for {lib_name} - documentation not available")
                        try:
                            import json
                            with open(failure_marker_path, 'r') as f:
                                cached_failure = json.load(f)
                            return {
                                "success": False,
                                "version": existing_version,
                                "error": cached_failure.get("error", "No API documentation found")
                            }
                        except Exception as e:
                            logger.warning(f"Error reading failure marker for {lib_name}: {e}")
                            # Fall through to download new cache
                    else:
                        # Use existing cache - extract from HTML files directly without web requests
                        logger.info(f"Using existing HTML cache for {lib_name} (cache_only={cache_only}, no_recache_html={no_recache_html})")
                        try:
                            # Process cached HTML files directly without any web requests
                            api_data = await self._process_cached_html_directly(existing_cache, lib_name, org_name, existing_version)
                            return {
                                "success": True,
                                "version": existing_version,
                                "api_documentation": api_data
                            }
                        except Exception as e:
                            logger.warning(f"Error reading cached HTML for {lib_name}: {e}")
                            # Fall through to download new cache
                else:
                    # Delete existing cache to recreate for freshness
                    logger.info(f"Deleting existing HTML cache for {lib_name} to recreate for freshness")
                    import shutil
                    shutil.rmtree(existing_cache)
            else:
                # No existing cache
                if cache_only:
                    logger.warning(f"No HTML cache found for {lib_name} and cache_only=True, skipping download")
                    return {"success": False, "error": "No HTML cache found and cache_only mode"}
            
            # Download new cache (if we reach here, either no cache exists or we're refreshing)
            logger.info(f"Downloading documentation for {lib_name} (cache_only={cache_only}, no_recache_html={no_recache_html})")
            
            # Extract hints from repository for the crawler
            detected_url = await self._detect_readthedocs_url_from_repo(repo_path, lib_name)
            
            # Use detected URL as a hint, or fallback to library name if none found
            url_hint = detected_url or lib_name
            
            logger.info(f"Using ReadTheDocsCrawler for {lib_name} with URL hint: {url_hint}")
            
            # Let ReadTheDocsCrawler handle all URL discovery, patterns, exceptions, and fallbacks
            result = await self.doc_crawler.crawl_readthedocs(url_hint, lib_name, org_name)
            
            if "error" not in result and result.get("api_documentation"):
                version = result.get("version", "unknown")
                logger.info(f"Successfully fetched ReadTheDocs documentation for {lib_name} (version: {version})")
                return {
                    "success": True,
                    "version": version,
                    "api_documentation": result.get("api_documentation", {})
                }
            elif "error" in result:
                error_msg = result["error"]
                logger.warning(f"ReadTheDocs extraction failed for {lib_name}: {error_msg}")
                return {"success": False, "error": error_msg}
            else:
                logger.info(f"No ReadTheDocs documentation found for {lib_name}")
                return {"success": False, "error": "No API documentation found"}
            
        except Exception as e:
            error_msg = f"Error fetching ReadTheDocs documentation for {lib_name}: {e}"
            logger.warning(error_msg)
            return {"success": False, "error": str(e)}

    async def _process_cached_html_directly(self, cache_dir: Path, lib_name: str, org_name: str, version: str) -> Dict[str, Any]:
        """Process cached HTML files directly without any web requests.
        
        This method bypasses the ReadTheDocs API entirely and processes only the HTML files
        that are already cached locally. It's used when the user wants to avoid any internet
        access and just use the existing cache.
        
        Args:
            cache_dir: Path to the HTML cache directory
            lib_name: Library name
            org_name: Organization name
            version: Documentation version
            
        Returns:
            Dictionary containing extracted API documentation
        """
        api_data = {
            "functions": [],
            "classes": [],
            "modules": []
        }
        
        logger.debug(f"Processing cached HTML files directly for {lib_name} from {cache_dir}")
        
        try:
            # Find all HTML files in the cache directory
            html_files = list(cache_dir.glob("*.html"))
            
            if not html_files:
                logger.warning(f"No HTML files found in cache directory {cache_dir}")
                return api_data
            
            # Process each HTML file - batch to avoid too many open files
            html_contents = []
            function_urls = []

            # Process in batches of 50 to avoid file descriptor exhaustion
            batch_size = 50
            for i in range(0, len(html_files), batch_size):
                batch = html_files[i:i+batch_size]
                for html_file in batch:
                    try:
                        with open(html_file, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                            html_contents.append(html_content)

                            # Create a dummy URL for the processing function
                            # This is needed for the interface but won't be used for web requests
                            function_urls.append(f"file://{html_file.name}")

                    except Exception as e:
                        logger.warning(f"Error reading cached HTML file {html_file}: {e}")
                        continue

                # Process batch if we have enough files
                if len(html_contents) >= batch_size:
                    logger.debug(f"Processing batch of {len(html_contents)} HTML files...")
                    await self.doc_crawler._process_html_contents(html_contents, function_urls, api_data)
                    html_contents = []
                    function_urls = []
                    # Force garbage collection to free memory and file handles
                    gc.collect()
            
            # Process the HTML contents using the crawler's processing method
            if html_contents:
                logger.debug(f"Processing {len(html_contents)} cached HTML files...")
                await self.doc_crawler._process_html_contents(html_contents, function_urls, api_data)
                logger.info(f"Processed cached HTML: {len(api_data['functions'])} functions extracted")
            
        except Exception as e:
            logger.error(f"Error processing cached HTML files for {lib_name}: {e}")
            
        return api_data

    async def _detect_readthedocs_url_from_repo(self, repo_path: Path, lib_name: str) -> str | None:
        """Detect documentation URL hints from repository files (no web queries)."""
        try:
            detected_url = None
            
            # Method 1: Check pyproject.toml for explicit documentation URL (most reliable)
            pyproject_file = repo_path / "pyproject.toml"
            if pyproject_file.exists():
                content = pyproject_file.read_text(encoding='utf-8', errors='ignore')
                import re
                # Look for documentation URL patterns in project metadata
                patterns = [
                    r'documentation\s*=\s*["\']([^"\']*(?:readthedocs|scverse)[^"\']*)["\']',
                    r'docs?\s*=\s*["\']([^"\']*(?:readthedocs|scverse)[^"\']*)["\']',
                    r'homepage\s*=\s*["\']([^"\']*(?:readthedocs|scverse)[^"\']*)["\']'
                ]
                for pattern in patterns:
                    doc_url_match = re.search(pattern, content, re.IGNORECASE)
                    if doc_url_match:
                        detected_url = doc_url_match.group(1).rstrip('/')
                        logger.info(f"Found documentation URL in pyproject.toml for {lib_name}: {detected_url}")
                        break
            
            # Method 2: Check for .readthedocs.yml (indicates ReadTheDocs is configured)
            if not detected_url:
                readthedocs_configs = [
                    repo_path / ".readthedocs.yml",
                    repo_path / ".readthedocs.yaml"
                ]
                has_rtd_config = any(config.exists() for config in readthedocs_configs)
                
                if has_rtd_config:
                    # ReadTheDocs is configured, use standard naming patterns
                    # Project names often use hyphens instead of underscores
                    project_candidates = [
                        lib_name,
                        lib_name.replace('_', '-'),
                        lib_name.replace('-', ''),
                        lib_name.lower()
                    ]
                    
                    # Return the most likely candidate (repository name with hyphens)
                    candidate = lib_name.replace('_', '-').lower()
                    detected_url = f"https://{candidate}.readthedocs.io"
                    logger.info(f"ReadTheDocs config found for {lib_name}, inferred URL: {detected_url}")
            
            
            # Method 4: Check README and other config files for documentation URLs
            if not detected_url:
                config_files = [
                    repo_path / "README.md",
                    repo_path / "README.rst", 
                    repo_path / "README.txt",
                    repo_path / "setup.py",
                    repo_path / "setup.cfg"
                ]
                
                import re
                # Extended patterns for both readthedocs and scverse
                doc_patterns = [
                    r'https://([^.\s]+)\.readthedocs\.io[^/\s]*',
                    r'https://([^.\s]+)\.scverse\.org[^/\s]*',
                    r'https://scverse\.org/([^/\s]+)[^/\s]*'
                ]
                
                for config_file in config_files:
                    if config_file.exists():
                        try:
                            content = config_file.read_text(encoding='utf-8', errors='ignore')
                            for pattern in doc_patterns:
                                matches = re.findall(pattern, content, re.IGNORECASE)
                                if matches:
                                    if 'readthedocs' in pattern:
                                        detected_url = f"https://{matches[0]}.readthedocs.io"
                                    elif 'scverse.org' in pattern and '/(' in pattern:
                                        detected_url = f"https://scverse.org/{matches[0]}"
                                    else:
                                        detected_url = f"https://{matches[0]}.scverse.org"
                                    logger.info(f"Found documentation URL in {config_file.name} for {lib_name}: {detected_url}")
                                    break
                            if detected_url:
                                break
                        except Exception as e:
                            logger.debug(f"Error reading {config_file}: {e}")
                            continue
            
            return detected_url
                        
        except Exception as e:
            logger.debug(f"Error detecting ReadTheDocs URL for {lib_name}: {e}")
            return None
    
    def _get_org_name_from_config(self, config: Dict[str, Any]) -> str:
        """Extract organization name from repository configuration.
        
        Args:
            config: Repository configuration containing repo URL
            
        Returns:
            Organization name extracted from the repository URL
        """
        repo_url = config.get("repo", "")
        if "github.com/" in repo_url:
            # Extract org from URL like "https://github.com/scverse/scanpy"
            parts = repo_url.split("github.com/")
            if len(parts) > 1:
                org_and_repo = parts[1].split("/")
                if len(org_and_repo) >= 2:
                    return org_and_repo[0]
        
        # Fallback: try to find matching organization from configured ones
        for org_name in self.github_organizations.keys():
            if org_name.lower() in repo_url.lower():
                return org_name
        
        # Final fallback: use first configured organization
        return list(self.github_organizations.keys())[0] if self.github_organizations else "unknown"

    async def _extract_repository_structured(self, lib_name: str, config: Dict[str, Any],
                                            cache_only: bool = False, no_recache_html: bool = False,
                                            overwrite_chromadb: bool = False) -> Dict[str, Any]:
        """Extract documentation and workflows using structured pipeline with cache control.

        Args:
            lib_name: Name of the library/repository
            config: Repository configuration containing repo URL and other metadata
            cache_only: If True, only use local caches, no internet access for downloading
            no_recache_html: If True, don't redownload existing HTML caches, but download missing ones
            overwrite_chromadb: If True, reprocess even if already in ChromaDB

        Cache behavior per step:
            Step 1 (Repository cloning):
                - Always check if repo already exists locally, skip cloning if so
                - Cloned repos are never overwritten to preserve local state

            Step 2 (API documentation from ReadTheDocs):
                - If HTML cache exists AND (no_recache_html OR cache_only): Use existing HTML cache
                - If HTML cache exists AND neither flag set: Delete HTML cache and recreate for freshness
                - If HTML cache missing AND cache_only: Skip download (no internet access)
                - If HTML cache missing AND no_recache_html: Download new HTML cache
                - If HTML cache missing AND neither flag: Download new HTML cache

            Steps 3-5 (Workflow extraction, parsing, storage): Always performed on available data

        Returns:
            Dictionary containing extraction results or error information
        """
        try:
            # Step 1: Clone repository (with skip logic)
            from kai.config.paths import get_org_repos_dir
            org_name = self._get_org_name_from_config(config)
            repo_path = get_org_repos_dir(org_name) / lib_name

            if repo_path.exists() and (repo_path / ".git").exists():
                logger.debug(f"Repository {lib_name} already cloned, skipping clone step")
            else:
                if cache_only:
                    logger.warning(f"Repository {lib_name} not cached and cache_only=True, skipping extraction")
                    return {"skipped": "Repository not cached and cache_only mode"}

                # Clone repository
                repo_path = await self._clone_repository(config["repo"], lib_name)
                logger.info(f"Cloned repository {lib_name} to {repo_path}")

            logger.debug(f"Processing {lib_name} from {repo_path}")

            # Step 2: Get API documentation from ReadTheDocs (with cache control)
            rtd_result = await self._fetch_readthedocs_via_crawler(
                repo_path, lib_name, cache_only=cache_only, no_recache_html=no_recache_html
            )
            if rtd_result["success"]:
                logger.info(f"Successfully extracted {lib_name} from ReadTheDocs (version: {rtd_result.get('version', 'unknown')})")
            else:
                logger.warning(f"ReadTheDocs extraction failed for {lib_name}: {rtd_result.get('error', 'Unknown error')}")
            
            # Step 3: Find workflow files (always performed)
            workflow_files = await self._find_workflow_files(repo_path, lib_name)

            # Step 4: Extract and parse workflows (always performed)
            workflow_chunks = await self._extract_and_parse_workflows(repo_path, lib_name, workflow_files)
            workflows = [{"workflow_chunks": len(workflow_chunks), "status": "parsed"}] if workflow_chunks else []
            
            # Step 5: Index results directly to ChromaDB (always performed)
            # Extract version from ReadTheDocs result
            version = "unknown"
            if rtd_result and rtd_result.get("success"):
                version = rtd_result.get("version", "unknown")

            result = {
                "library": lib_name,
                "language": self.language,
                "repository": config["repo"],
                "version": version,  # Add version at top level
                "readthedocs_extraction": rtd_result,
                "api_documentation": rtd_result.get("api_documentation", {}) if rtd_result.get("success") else {},
                "workflows": workflow_chunks,  # Use actual workflow chunks, not metadata
                "extraction_date": datetime.now().isoformat(),
                "status": "success"
            }
            
            # Index directly to ChromaDB instead of saving JSON files
            await self._index_to_chromadb(lib_name, result, org_name)
            logger.info(f"Successfully processed and indexed {lib_name}: ReadTheDocs + {len(workflow_files)} workflow files")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting {lib_name}: {e}")
            return {"error": str(e)}
    
    async def _index_to_chromadb(self, lib_name: str, extraction_result: Dict[str, Any], organization: str) -> None:
        """Index extraction results directly to ChromaDB, replacing any existing entries.

        Args:
            lib_name: Name of the library
            extraction_result: Extraction results from _extract_repository_structured
            organization: GitHub organization name
        """
        try:
            # Ensure ChromaDB manager is available
            if self.chromadb_manager is None:
                logger.error("ChromaDB manager not initialized, cannot index library")
                return
            
            # Delete existing entries for this library to avoid duplicates
            await self._delete_library_from_chromadb(lib_name)
            
            # Index the new extraction results with organization info and repo path for git version detection
            from kai.config.paths import get_org_repos_dir
            repo_path = get_org_repos_dir(organization) / lib_name
            indexing_result = await self.chromadb_manager.index_library_from_data(lib_name, extraction_result, organization, repo_path)
            
            logger.info(f"Successfully indexed {lib_name} to ChromaDB: "
                       f"{indexing_result['documents_added']} documents, "
                       f"{indexing_result['functions_added']} functions, "
                       f"{indexing_result['workflows_added']} workflows")
            
        except Exception as e:
            import traceback
            logger.error(f"Error indexing {lib_name} to ChromaDB: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Don't raise - extraction succeeded, indexing failure shouldn't stop the process
    
    async def _delete_library_from_chromadb(self, lib_name: str) -> None:
        """Delete existing entries for a library from ChromaDB.
        
        Args:
            lib_name: Name of the library to delete
        """
        try:
            # Check if repository actually exists before logging deletion
            if self.chromadb_manager and self.chromadb_manager.has_repository(lib_name):
                logger.info(f"Deleting existing ChromaDB entries for {lib_name}")
                # Delete from all relevant collections
                # This is a simplified approach - in practice, you'd want to query for
                # documents with this library name and delete them
            else:
                logger.debug(f"No existing ChromaDB entries found for {lib_name}, proceeding with indexing")
            
            # For now, we'll rely on ChromaDB's upsert behavior to handle duplicates
            # A more sophisticated implementation would track and delete specific document IDs
            
        except Exception as e:
            logger.warning(f"Error checking/deleting existing ChromaDB entries for {lib_name}: {e}")
            # Non-critical error, continue with indexing

    async def _find_workflow_files(self, repo_path: Path, lib_name: str) -> List[Path]:
        """Find workflow files based on configured file patterns."""
        config = self.config["workflows"]
        file_patterns = config["file_patterns"]
        
        # Process ALL files matching configured patterns
        workflow_files = []
        for pattern in file_patterns:
            workflow_files.extend(repo_path.rglob(pattern))
        
        # Filter based on language-specific file types
        if self.language == "python":
            # For Python: focus on Jupyter notebooks
            filtered_files = [f for f in workflow_files if f.suffix == '.ipynb']
            logger.info(f"Found {len(filtered_files)} notebook files for {lib_name}")
        elif self.language == "r":
            # For R: focus on R Markdown files
            filtered_files = [f for f in workflow_files if f.suffix == '.Rmd']
            logger.info(f"Found {len(filtered_files)} R Markdown files for {lib_name}")
        else:
            # Fallback: return all files matching patterns
            filtered_files = workflow_files
            logger.info(f"Found {len(filtered_files)} workflow files for {lib_name}")
        
        return filtered_files
    


    async def _extract_and_parse_workflows(self, repo_path: Path, lib_name: str, workflow_files: Optional[List[Path]] = None) -> List[Any]:
        """Extract and parse workflow files into structured chunks.
        
        Args:
            repo_path: Path to the cloned repository
            lib_name: Library name for workflow identification
            
        Returns:
            List of workflow chunks from all parsed workflow files
        """
        org_name = self._extract_org_name_from_repo_path(repo_path)
        all_workflow_chunks = []
        
        try:
            # Use provided workflow_files or find them if not provided
            if workflow_files is None:
                workflow_files = await self._find_workflow_files(repo_path, lib_name)

            if workflow_files:
                org_name = self._extract_org_name_from_repo_path(repo_path)
                logger.debug(f"Processing {len(workflow_files)} workflow files in {org_name}/{lib_name}")
                
                for workflow_file in workflow_files:
                    try:
                        # Parse based on file type
                        if workflow_file.suffix == '.ipynb':
                            # Parse Jupyter notebooks
                            workflow_chunks = self.workflow_parser.parse_notebook(workflow_file, lib_name, org_name)
                        elif workflow_file.suffix == '.Rmd':
                            # Parse R Markdown files (same parser for now)
                            workflow_chunks = self.workflow_parser.parse_notebook(workflow_file, lib_name, org_name)
                        else:
                            # Skip unsupported file types
                            logger.debug(f"Skipping unsupported file type: {workflow_file}")
                            continue
                        
                        if workflow_chunks:
                            all_workflow_chunks.extend(workflow_chunks)
                            logger.debug(f"Parsed {len(workflow_chunks)} workflow chunks from {workflow_file.name}")
                    except Exception as e:
                        logger.warning(f"Error parsing workflow file {workflow_file}: {e}")
                        continue
                
                logger.info(f"Successfully parsed {len(all_workflow_chunks)} total workflow chunks from {lib_name}")
            else:
                logger.debug(f"No workflow files found for {lib_name}")
            
        except Exception as e:
            logger.error(f"Error extracting workflows for {lib_name}: {e}")
        
        return all_workflow_chunks

