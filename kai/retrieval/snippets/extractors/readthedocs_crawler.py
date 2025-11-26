"""
ReadTheDocs API Documentation Crawler.

This module provides functionality to crawl and extract API documentation from ReadTheDocs-hosted
sites, including standard ReadTheDocs.io sites, scverse.org hosted documentation, and other
documentation hosting patterns commonly used in the bioinformatics ecosystem.

Main Entry Points:
1. crawl_readthedocs() - Primary method for extracting documentation from any ReadTheDocs URL
2. _find_api_url() - Core URL discovery engine that tries multiple patterns
3. _smart_homepage_discovery() - Intelligent homepage exploration for API links

Workflow:
ReadTheDocs URL → URL Discovery → HTML Download → Content Extraction → HTML Caching

The crawler handles various documentation hosting patterns:
- Standard ReadTheDocs: {package}.readthedocs.io/en/{version}/api.html
- scverse.org: scverse.org/{package}/api/index.html  
- Custom domains: docs.{package}.org (handled via exceptions)
- GitHub Pages: {user}.github.io/{package}/api.html
"""
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from bs4 import BeautifulSoup
import json
import re
from urllib.parse import urljoin, urlparse

from kai.utils import setup_logger
from ..storage.hierarchical_cache import HierarchicalCache
from kai.config.paths import AGENT_BASE_DIR

logger = setup_logger(__name__)


class ReadTheDocsCrawler:
    """
    Intelligent ReadTheDocs API Documentation Crawler.
    
    This class provides comprehensive functionality for discovering, downloading, and extracting
    API documentation from various ReadTheDocs hosting patterns. It uses adaptive URL discovery,
    smart fallback mechanisms, and efficient bulk processing to extract structured API data.
    
    Architecture:
    
    Entry Point Methods:
    ==================
    crawl_readthedocs() - Main entry point for external callers
        ├── Determines if URL is specific API URL or base documentation URL
        ├── Delegates to _find_api_url() for URL discovery if needed  
        ├── Calls _bulk_download_and_extract_api() for content extraction
        └── Handles HTML caching via hierarchical cache system
    
    Core Discovery Engine:
    =====================
    _find_api_url() - Tries exceptional URLs first, then standard patterns
        ├── Uses readthedocs_exceptions.py for non-standard hosting
        ├── Tests 23+ URL patterns across stable/latest versions
        └── Falls back to _smart_homepage_discovery() if patterns fail
        
    _smart_homepage_discovery() - Explores homepage for API links
        ├── Delegates to _extract_api_links() for link extraction
        ├── Tests discovered links for actual API content
        └── Returns first valid API URL found
    
    Content Extraction Pipeline:
    ===========================
    _bulk_download_and_extract_api() - Main extraction coordinator
        ├── Downloads main API page via _fetch_and_cache_page()
        ├── Extracts direct functions via _extract_functions_from_api_tables()
        ├── Discovers individual pages via _adaptive_discover_function_urls()
        ├── Bulk downloads via _bulk_download_function_pages()
        └── Processes HTML locally via _process_html_contents()
    
    URL Discovery Subsystem:
    =======================
    _adaptive_discover_function_urls() - Finds individual function page URLs
        ├── Uses 6 different regex patterns for URL matching
        ├── Handles relative URL resolution with _is_valid_function_url()
        └── Falls back to _explore_site_structure() if no patterns match
    
    Content Processing:
    ==================
    _extract_readthedocs_function_details() - Extracts structured data from HTML
        ├── Parses function signatures, parameters, returns
        ├── Uses _extract_comprehensive_documentation() for detailed docs
        └── Delegates parameter parsing to _extract_parameter_details()
    
    Caching & Performance:
    =====================
    - Uses HierarchicalCache for organized HTML storage by org/library/version
    - Bulk downloads with rate limiting to respect server resources
    - HTML caching prevents redundant network requests and enables offline processing
    - Python dictionaries are returned directly from HTML processing
    - Supports both fresh crawling and cached HTML retrieval
    
    Error Handling:
    ==============
    - Graceful degradation when URLs return 404
    - Multiple fallback strategies for URL discovery
    - Rate limiting and retry logic for network requests
    - Comprehensive logging for debugging and monitoring
    """
    
    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize the ReadTheDocs crawler.
        
        Args:
            output_dir: Directory to store extracted documentation (optional, HTML caching uses hierarchical cache)
        """
        # No longer using temp directory - HTML files go directly to hierarchical cache
        self.output_dir = output_dir
        
        # Initialize hierarchical cache
        self.hierarchical_cache = HierarchicalCache(AGENT_BASE_DIR)
        
        self.session_timeout = aiohttp.ClientTimeout(total=30)
        
        logger.info("ReadTheDocs crawler initialized - using hierarchical HTML cache")
    
    async def crawl_readthedocs(self, readthedocs_url: str, library_name: str, org_name: str = "unknown") -> Dict[str, Any]:
        """
        Main entry point for crawling ReadTheDocs API documentation.
        
        This method orchestrates the entire documentation extraction process:
        1. Determines if the provided URL is a specific API URL or base documentation URL
        2. Performs URL discovery using adaptive patterns if needed
        3. Downloads and extracts structured API documentation
        4. Caches results in hierarchical storage for future use
        
        The method handles various URL patterns and gracefully degrades when specific
        patterns fail, using smart fallback mechanisms to maximize extraction success.
        
        Workflow:
        - Check HTML cache first (hierarchical cache system)
        - Determine URL type (specific API vs base URL)
        - Find correct API URL (_find_api_url) if needed
        - Extract documentation (_bulk_download_and_extract_api)
        - Cache HTML files and return structured data directly
        
        Args:
            readthedocs_url: URL to documentation site. Can be:
                - Base documentation URL (e.g., https://package.readthedocs.io)
                - Specific API URL (e.g., https://package.readthedocs.io/en/stable/api.html)
                - scverse.org URL (e.g., https://scverse.org/package)
                - Custom domain URL (e.g., https://docs.package.org)
            library_name: Name of the library being crawled (for caching and logging)
            org_name: Organization name for hierarchical caching structure
            
        Returns:
            Dict containing structured API documentation with keys:
            - library: Library name
            - crawled_at: ISO timestamp of crawl
            - base_url: URL used for extraction
            - type: Documentation type ("readthedocs" or "scverse")
            - version: Documentation version ("stable", "latest", etc.)
            - api_documentation: Dict with extracted functions, classes, modules
            - summary: Human-readable extraction summary
            
        Raises:
            No exceptions are raised; errors are logged and empty results returned.
            This ensures graceful degradation when documentation is unavailable.
        """
        logger.info(f"Crawling documentation for {library_name} from {readthedocs_url}")
        
        # Check if we have a failure marker (no documentation found previously)
        failure_marker_path = self._get_failure_marker_path(org_name, library_name, "stable")
        if failure_marker_path.exists():
            logger.info(f"Found failure marker for {library_name} - documentation not available")
            try:
                import json
                with open(failure_marker_path, 'r') as f:
                    cached_failure = json.load(f)
                return cached_failure
            except Exception as e:
                logger.warning(f"Failed to read failure marker for {library_name}: {e}")
                # Fall through to fresh attempt
        
        # Check if the provided URL already looks like a specific API URL
        api_keywords = ['api', 'reference', 'generated', '_autosummary', 'modules', 'autoapi']
        is_specific_api_url = any(keyword in readthedocs_url.lower() for keyword in api_keywords)
        
        if is_specific_api_url:
            # Use the provided URL directly since it already seems to be an API URL
            api_url = readthedocs_url
            logger.info(f"Using provided API URL directly: {api_url}")
            
            # Quick check if the URL is accessible
            try:
                import aiohttp
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                    async with session.get(api_url) as response:
                        if response.status == 404:
                            logger.warning(f"Provided API URL returned 404, trying alternative discovery")
                            # Try to find alternative URL using _find_api_url
                            alternative_url = await self._find_api_url(readthedocs_url, library_name)
                            if alternative_url:
                                api_url = alternative_url
                                logger.info(f"Found alternative API URL: {api_url}")
            except Exception as e:
                logger.debug(f"Error checking provided URL {api_url}: {e}")
        else:
            # Try to find the correct API URL from base URL
            api_url = await self._find_api_url(readthedocs_url, library_name)
            if not api_url:
                logger.warning(f"Could not find valid API documentation URL for {library_name}")
                # Create failure marker and return empty result
                failure_result = {
                    "library": library_name,
                    "crawled_at": datetime.now().isoformat(),
                    "base_url": readthedocs_url,
                    "type": "readthedocs",
                    "version": "stable",
                    "api_documentation": {
                        "functions": [],
                        "classes": [],
                        "modules": []
                    },
                    "error": "No API documentation found"
                }
                
                # Save failure marker to prevent re-trying
                await self._save_failure_marker(org_name, library_name, "stable", failure_result)
                return failure_result
        
        readthedocs_url = api_url
        logger.info(f"Found valid API URL: {readthedocs_url}")
        
        # Detect documentation site type and adjust URL accordingly
        is_scverse = 'scverse.org' in readthedocs_url
        if is_scverse:
            # For scverse.org sites, ensure we're targeting the API
            if not readthedocs_url.endswith(('/api/', '/api/index.html', '/api')):
                # Remove any existing /en/stable/ or /api.html parts and add correct API path
                base_url = readthedocs_url.split('/en/')[0] if '/en/' in readthedocs_url else readthedocs_url
                base_url = base_url.rstrip('/')
                readthedocs_url = base_url + '/api/index.html'
                logger.info(f"Adjusted scverse.org URL to target API: {readthedocs_url}")
            version = "stable"  # scverse.org doesn't use version paths
        else:
            # Traditional ReadTheDocs handling
            version = self._extract_version_from_url(readthedocs_url)
            
            # Only use stable version
            if version != "stable":
                # Convert to stable URL
                readthedocs_url = readthedocs_url.replace(f"/en/{version}/", "/en/stable/")
                version = "stable"
                logger.info(f"Converting to stable version: {readthedocs_url}")
        
        doc_type = "scverse" if is_scverse else "readthedocs"
        
        # Check if HTML cache exists - if so, extract from cached HTML
        if self._has_html_files_cached(org_name, library_name, version):
            logger.info(f"Using cached HTML files for {library_name}")
            try:
                # Extract from cached HTML files
                api_data = await self._bulk_download_and_extract_api(readthedocs_url, library_name, org_name, version, use_cache_only=True)
                result = {
                    "library": library_name,
                    "crawled_at": datetime.now().isoformat(),
                    "base_url": readthedocs_url,
                    "type": doc_type,
                    "version": version,
                    "api_documentation": api_data
                }
                
                # Generate summary
                func_count = len(api_data.get("functions", []))
                class_count = len(api_data.get("classes", []))
                module_count = len(api_data.get("modules", []))
                result["summary"] = f"Extracted {func_count} functions, {class_count} classes, {module_count} modules from cached HTML"
                
                return result
            except Exception as e:
                logger.warning(f"Failed to extract from cached HTML for {library_name}: {e}")
                # Fall through to fresh download
        
        doc_type = "scverse" if is_scverse else "readthedocs"
        result = {
            "library": library_name,
            "crawled_at": datetime.now().isoformat(),
            "base_url": readthedocs_url,
            "type": doc_type,
            "version": version,
            "api_documentation": {
                "functions": [],
                "classes": [],
                "modules": []
            }
        }
        
        try:
            # Mark HTML download as in progress
            self.hierarchical_cache.mark_html_download_in_progress(org_name, library_name, version)
            
            # All documentation is now cached in bioinformatics_cache structure
            api_data = await self._bulk_download_and_extract_api(readthedocs_url, library_name, org_name, version, use_cache_only=False)
            
            result["api_documentation"] = api_data
            
            # Generate summary
            func_count = len(api_data.get("functions", []))
            class_count = len(api_data.get("classes", []))
            module_count = len(api_data.get("modules", []))
            result["summary"] = f"Extracted {func_count} functions, {class_count} classes, {module_count} modules from ReadTheDocs"
            
            # HTML files are already cached by _bulk_download_and_extract_api
            logger.info(f"HTML files cached for {library_name}")
            
            # Mark HTML download as complete
            self.hierarchical_cache.mark_html_download_complete(org_name, library_name, version)
                
        except Exception as e:
            logger.error(f"Error crawling ReadTheDocs for {library_name}: {e}")
            result["error"] = str(e)
            
            # Mark HTML download as complete even on failure to clean up the flag
            self.hierarchical_cache.mark_html_download_complete(org_name, library_name, version)
            
            # Save failure marker for general errors too
            await self._save_failure_marker(org_name, library_name, version, result)
        
        return result
    
    def _get_failure_marker_path(self, org_name: str, library_name: str, version: str) -> Path:
        """Get the path for the failure marker file."""
        html_cache_dir = self.hierarchical_cache.get_html_cache_dir(org_name) / f"{library_name}_{version}"
        return html_cache_dir / "no_documentation_found.json"
    
    def _has_html_files_cached(self, org_name: str, library_name: str, version: str) -> bool:
        """Check if HTML files (not failure markers) are cached for a repository."""
        html_cache_dir = self.hierarchical_cache.get_html_cache_dir(org_name) / f"{library_name}_{version}"
        if not html_cache_dir.exists():
            return False
        
        # Check if there are HTML files (not just the failure marker)
        html_files = [f for f in html_cache_dir.iterdir() if f.is_file() and f.suffix == '.html']
        return len(html_files) > 0
    
    async def _save_failure_marker(self, org_name: str, library_name: str, version: str, failure_result: Dict[str, Any]) -> None:
        """Save a failure marker to prevent re-trying failed libraries."""
        try:
            failure_marker_path = self._get_failure_marker_path(org_name, library_name, version)
            
            # Create cache directory if it doesn't exist
            failure_marker_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save failure result
            import json
            with open(failure_marker_path, 'w') as f:
                json.dump(failure_result, f, indent=2)
            
            logger.info(f"Saved failure marker for {library_name} to prevent re-trying")
        except Exception as e:
            logger.warning(f"Failed to save failure marker for {library_name}: {e}")
    
    async def _find_api_url(self, base_url: str, library_name: str) -> Optional[str]:
        """
        Core URL discovery engine that finds valid API documentation URLs.
        
        This method implements a sophisticated URL discovery strategy using multiple approaches:
        1. Exceptional URLs - Tries known non-standard hosting patterns first
        2. Standard Patterns - Tests 23+ common ReadTheDocs URL patterns
        3. Smart Discovery - Falls back to homepage exploration if patterns fail
        
        The method tests both 'stable' and 'latest' versions for each pattern to maximize
        success rate. It validates each discovered URL by checking for actual API content.
        
        Pattern Categories Tested:
        - Direct API paths: /api.html, /api/, /api/index.html
        - Reference docs: /reference/, /reference.html, /reference/index.html  
        - Generated docs: /generated/, /generated/index.html
        - Autosummary: /_autosummary/, /_autosummary/index.html
        - Module docs: /modules/, /modules.html, /modules/index.html
        - Autoapi: /autoapi/, /autoapi/index.html
        - Package-specific: /{library_name}/, /{library_name}.html
        - User guide: /user_guide/api.html, /user_guide/api/
        - Docs subdirs: /docs/api.html, /docs/api/
        
        Args:
            base_url: Base documentation URL. Can be:
                - Full version URL (e.g., https://package.readthedocs.io/en/stable/)
                - Base domain (e.g., https://package.readthedocs.io)
                - Custom domain (e.g., https://docs.package.org)
            library_name: Library name used for:
                - Package-specific URL patterns
                - Content validation
                - Exception lookup
                
        Returns:
            Valid API URL if discovered, None if no valid API documentation found.
            
        Delegates:
            - get_exception_urls() for non-standard hosting patterns
            - _looks_like_api_page() for content validation
            - _smart_homepage_discovery() as final fallback
            
        Note:
            This method is the core of the adaptive discovery system and handles
            the majority of ReadTheDocs URL patterns found in bioinformatics packages.
        """
        import aiohttp
        from .readthedocs_exceptions import get_exception_urls
        
        # First try exceptional URLs for packages with non-standard hosting
        exception_urls = get_exception_urls(library_name)
        if exception_urls:
            logger.info(f"Trying exceptional URLs for {library_name}")
            try:
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    for test_url in exception_urls:
                        try:
                            async with session.get(test_url) as response:
                                if response.status == 200:
                                    content = await response.text()
                                    if self._looks_like_api_page(content, library_name):
                                        logger.info(f"Found API documentation at exceptional URL: {test_url}")
                                        return test_url
                        except Exception as e:
                            logger.debug(f"Error checking exceptional URL {test_url}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Error trying exceptional URLs for {library_name}: {e}")
        
        # Standard ReadTheDocs patterns - try both stable and latest
        versions = ['stable', 'latest']
        
        # Common API URL patterns to try
        api_patterns = [
            'api.html',           # anndata, sfaira
            'api/',               # scanpy, cellrank, diffxpy
            'api/index.html',     # alternative
            'reference/',         # some use reference instead of api
            'reference.html',     # single page reference
            'reference/index.html', # reference with index
            'generated/',         # autogenerated docs
            'generated/index.html', # generated with index
            '_autosummary/',      # sphinx autosummary
            '_autosummary/index.html', # autosummary with index
            'modules/',           # module documentation
            'modules.html',       # modules as single page
            'modules/index.html', # modules with index
            'autoapi/',           # sphinx autoapi
            'autoapi/index.html', # autoapi with index
            'docs/',              # docs subdirectory
            'docs/api.html',      # docs/api
            'docs/api/',          # docs/api directory
            'user_guide/api.html', # user guide api
            'user_guide/api/',    # user guide api directory
            f'{library_name}/',   # library-specific path
            f'{library_name}.html', # library-specific single page
            f'{library_name}/index.html', # library-specific with index
        ]
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for version in versions:
                    # Extract base domain from URL and rebuild with version
                    if '/en/' in base_url:
                        base_domain = base_url.split('/en/')[0]
                    else:
                        base_domain = base_url.rstrip('/')
                    
                    version_base_url = f"{base_domain}/en/{version}/"
                    
                    for pattern in api_patterns:
                        test_url = version_base_url + pattern
                        try:
                            async with session.get(test_url) as response:
                                if response.status == 200:
                                    # Check if it actually contains API documentation
                                    content = await response.text()
                                    if self._looks_like_api_page(content, library_name):
                                        logger.info(f"Found API documentation at: {test_url}")
                                        return test_url
                                    else:
                                        logger.debug(f"URL {test_url} exists but doesn't look like API docs")
                                else:
                                    logger.debug(f"URL {test_url} returned {response.status}")
                        except Exception as e:
                            logger.debug(f"Error checking {test_url}: {e}")
                            continue
        
        except Exception as e:
            logger.warning(f"Error during API URL discovery for {library_name}: {e}")
        
        # Final fallback: try to discover from documentation homepage
        logger.info(f"Trying smart homepage discovery for {library_name}")
        try:
            return await self._smart_homepage_discovery(base_url, library_name)
        except Exception as e:
            logger.debug(f"Homepage discovery failed for {library_name}: {e}")
        
        return None
    
    async def _smart_homepage_discovery(self, base_url: str, library_name: str) -> Optional[str]:
        """
        Intelligent homepage exploration for API documentation links.
        
        When standard URL patterns fail, this method explores documentation homepages
        to find API links. It tests multiple homepage variations and uses semantic
        analysis to identify likely API documentation links.
        
        Strategy:
        1. Generate multiple homepage URL variations (stable, latest, root)
        2. Download and parse each homepage
        3. Extract and prioritize potential API links
        4. Test each link for actual API content
        5. Return first valid API URL found
        
        Homepage Patterns Tested:
        - /en/stable/ (standard ReadTheDocs stable)
        - /en/latest/ (standard ReadTheDocs latest)  
        - / (root homepage)
        - /stable/ (alternative stable pattern)
        - /latest/ (alternative latest pattern)
        
        Link Detection:
        Uses _extract_api_links() to find links containing API-related keywords
        like 'api', 'reference', 'generated', etc. Links are prioritized by
        semantic relevance.
        
        Args:
            base_url: Base documentation URL to explore
            library_name: Library name for content validation
            
        Returns:
            Valid API URL if found through homepage exploration, None otherwise
            
        Delegates:
            - _extract_api_links() for semantic link extraction
            - _looks_like_api_page() for content validation
            
        Note:
            This is the final fallback in the URL discovery pipeline and handles
            cases where documentation uses non-standard organization or naming.
        """
        import aiohttp
        from bs4 import BeautifulSoup
        
        # Try common homepage URLs
        homepage_urls = []
        
        # Extract base domain from URL
        if '/en/' in base_url:
            base_domain = base_url.split('/en/')[0]
        else:
            base_domain = base_url.rstrip('/')
        
        # Add homepage patterns to try
        homepage_urls.extend([
            f"{base_domain}/en/stable/",
            f"{base_domain}/en/latest/",
            f"{base_domain}/",
            base_url if not base_url.endswith(('api.html', 'api/')) else base_domain + "/",
        ])
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                for homepage_url in homepage_urls:
                    try:
                        async with session.get(homepage_url) as response:
                            if response.status == 200:
                                content = await response.text()
                                soup = BeautifulSoup(content, 'html.parser')
                                
                                # Look for links that might be API documentation
                                api_links = self._extract_api_links(soup, homepage_url)
                                
                                for link_url in api_links:
                                    # Test if this link actually contains API documentation
                                    try:
                                        async with session.get(link_url) as api_response:
                                            if api_response.status == 200:
                                                api_content = await api_response.text()
                                                if self._looks_like_api_page(api_content, library_name):
                                                    logger.info(f"Smart discovery found API at: {link_url}")
                                                    return link_url
                                    except:
                                        continue
                    except:
                        continue
        except Exception as e:
            logger.debug(f"Error in smart homepage discovery: {e}")
        
        return None
    
    def _extract_api_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract potential API documentation links from a webpage."""
        potential_links = []
        
        # Look for links with API-related text
        api_keywords = [
            'api', 'reference', 'documentation', 'docs', 'functions', 
            'classes', 'modules', 'methods', 'library reference'
        ]
        
        # Find all links
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text().lower().strip()
            title = link.get('title', '').lower().strip()
            
            # Check if link text or title suggests API documentation
            if any(keyword in text for keyword in api_keywords) or \
               any(keyword in title for keyword in api_keywords) or \
               any(keyword in href.lower() for keyword in ['api', 'reference', 'generated']):
                
                # Convert relative URLs to absolute
                if href.startswith('/'):
                    from urllib.parse import urlparse
                    parsed_base = urlparse(base_url)
                    full_url = f"{parsed_base.scheme}://{parsed_base.netloc}{href}"
                elif href.startswith('http'):
                    full_url = href
                else:
                    full_url = urljoin(base_url, href)
                
                potential_links.append(full_url)
        
        # Remove duplicates and sort by likelihood
        unique_links = list(set(potential_links))
        
        # Prioritize links that are more likely to be API docs
        def link_priority(url):
            url_lower = url.lower()
            score = 0
            if 'api' in url_lower: score += 10
            if 'reference' in url_lower: score += 8
            if 'generated' in url_lower: score += 6
            if 'docs' in url_lower: score += 4
            if url_lower.endswith('.html'): score += 2
            return score
        
        return sorted(unique_links, key=link_priority, reverse=True)[:10]  # Top 10 candidates
    
    def _looks_like_api_page(self, content: str, library_name: str) -> bool:
        """
        Content validator that determines if HTML contains API documentation.
        
        Uses heuristic analysis to identify API documentation by looking for:
        - Sphinx-generated documentation indicators
        - Function/class/method signatures  
        - Library-specific references
        - API-related CSS classes and structure
        
        This validation prevents false positives where URLs exist but don't
        contain actual API documentation (e.g., marketing pages, tutorials).
        
        Detection Criteria:
        - Must find at least 2 API indicators
        - Must have substantial content (>1000 chars)
        - Looks for Sphinx documentation markers
        - Searches for function/class definitions
        - Checks for library-specific API references
        
        Args:
            content: Raw HTML content to analyze
            library_name: Library name to look for in API references
            
        Returns:
            True if content appears to contain API documentation, False otherwise
            
        Note:
            This method is crucial for validating discovered URLs and preventing
            extraction attempts on non-API pages that happen to match URL patterns.
        """
        # Look for common indicators of API documentation
        api_indicators = [
            'class="function"',
            'class="method"',
            'class="class"',
            'class="module"',
            'sphinx-doc',
            'api-reference',
            'function',
            'method',
            'class',
            'module',
            f'{library_name}.',  # Library-specific references
            'def ',              # Function definitions
            'class ',            # Class definitions
        ]
        
        content_lower = content.lower()
        
        # Count how many indicators we find
        indicator_count = sum(1 for indicator in api_indicators if indicator.lower() in content_lower)
        
        # Also check for minimum content length (not just a redirect or error page)
        has_substantial_content = len(content) > 1000
        
        # Consider it API documentation if we find multiple indicators and substantial content
        return indicator_count >= 2 and has_substantial_content

    
    async def _bulk_download_and_extract_api(self, base_url: str, library_name: str, org_name: str = "unknown", version: str = "stable", use_cache_only: bool = False) -> Dict[str, Any]:
        """
        Main content extraction coordinator that orchestrates the entire extraction pipeline.
        
        This method implements the core extraction workflow:
        1. Downloads the main API page and extracts summary functions
        2. Discovers individual function page URLs using adaptive patterns
        3. Bulk downloads all function pages with rate limiting
        4. Processes HTML content locally to extract structured data
        5. Returns comprehensive API documentation data
        
        The method prioritizes efficiency by:
        - Minimizing network requests through bulk downloading
        - Caching all HTML content for future offline processing
        - Processing content locally to avoid repeated network calls
        - Using rate limiting to respect server resources
        - Returning Python dictionaries directly from HTML processing
        
        Extraction Pipeline:
        1. Main Page Processing:
           - Download main API page via _fetch_and_cache_page()
           - Extract summary functions via _extract_functions_from_api_tables()
        
        2. URL Discovery:
           - Find individual function pages via _adaptive_discover_function_urls()
           - Handles relative URL resolution and validation
        
        3. Bulk Download:
           - Download all function pages via _bulk_download_function_pages()
           - Respects rate limits and implements retry logic
        
        4. Content Processing:
           - Process downloaded HTML via _process_html_contents()
           - Extract detailed function documentation and signatures
           - Return structured Python dictionaries directly
        
        Args:
            base_url: Main API page URL to start extraction from
            library_name: Library name for caching and content validation
            org_name: Organization name for hierarchical cache organization
            version: Documentation version (stable, latest, etc.) for cache organization
            
        Returns:
            Dictionary containing extracted API documentation:
            - functions: List of function objects with signatures, docs, parameters
            - classes: List of class objects (currently minimal extraction)
            - modules: List of module objects (currently minimal extraction)
            
        Delegates:
            - _fetch_and_cache_page() for individual page downloads
            - _extract_functions_from_api_tables() for main page function extraction
            - _adaptive_discover_function_urls() for URL discovery
            - _bulk_download_function_pages() for efficient bulk downloads
            - _process_html_contents() for content extraction
            
        Note:
            This method coordinates the entire extraction process and is where
            the bulk of the crawling work occurs. It balances thoroughness with
            efficiency through strategic caching and bulk operations.
        """
        api_data = {
            "functions": [],
            "classes": [],
            "modules": []
        }
        
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                # All HTML caching now handled by hierarchical cache system
                # Step 1: Download main API page
                main_html = await self._fetch_and_cache_page(session, base_url, library_name, "api_index", org_name, version, use_cache_only)
                if not main_html:
                    logger.error(f"Failed to fetch main API page for {library_name}")
                    return api_data
                
                main_soup = BeautifulSoup(main_html, 'html.parser')
                
                # Step 2: Extract functions from main API page tables
                main_functions = self._extract_functions_from_api_tables(main_soup)
                api_data["functions"].extend(main_functions)
                logger.info(f"Extracted {len(main_functions)} functions from main API page")
                
                # Step 3: Adaptively discover function page URLs based on site structure
                function_urls = await self._adaptive_discover_function_urls(session, main_soup, base_url, library_name)
                logger.info(f"Found {len(function_urls)} individual function page URLs using adaptive discovery")
                
                if not function_urls:
                    logger.warning(f"No individual function pages found for {library_name}")
                    return api_data
                
                # Step 4: Bulk download all function pages
                logger.info(f"Bulk downloading {len(function_urls)} function pages...")
                html_contents = await self._bulk_download_function_pages(session, function_urls, library_name, org_name, version, use_cache_only)
                
                # Step 5: Process downloaded HTML content locally
                logger.info(f"Processing {len(html_contents)} downloaded HTML pages locally...")
                await self._process_html_contents(html_contents, function_urls, api_data)
                
                logger.info(f"Bulk extraction complete: {len(api_data['functions'])} total functions")
                
        except Exception as e:
            logger.error(f"Error in bulk ReadTheDocs extraction: {e}")
        
        return api_data
    
    async def _fetch_and_cache_page(self, session: aiohttp.ClientSession, url: str, library_name: str, page_name: str, org_name: str, version: str, use_cache_only: bool = False) -> Optional[str]:
        """Fetch a page and cache it locally.
        
        Args:
            session: HTTP session
            url: URL to fetch
            library_name: Library name for cache file naming
            page_name: Name for the cached file
            org_name: Organization name for hierarchical caching
            version: Documentation version
            use_cache_only: If True, only read from cache, don't fetch from web
            
        Returns:
            HTML content
        """
        # Determine cache file path
        html_cache_dir = self.hierarchical_cache.get_html_cache_dir(org_name) / f"{library_name}_{version}"
        
        # Use page_name for main pages, URL-based filename for others
        if page_name:
            filename = f"{page_name}.html"
        else:
            filename = self._url_to_filename(url)
        
        html_file_path = html_cache_dir / filename
        
        # Try to read from cache first
        if html_file_path.exists():
            try:
                with open(html_file_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                logger.debug(f"Loaded cached HTML: {html_file_path}")
                return html_content
            except Exception as e:
                logger.warning(f"Failed to read cached HTML from {html_file_path}: {e}")
        
        # If cache_only mode and no cache found, return None
        if use_cache_only:
            logger.warning(f"Cache-only mode: no cached HTML found for {url}")
            return None
        
        # Fetch from web and cache to hierarchical cache
        html_content = await self._fetch_page(session, url)
        
        if html_content:
            try:
                # Save to hierarchical cache HTML directory
                html_cache_dir.mkdir(parents=True, exist_ok=True)
                
                with open(html_file_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                logger.debug(f"Cached HTML: {html_file_path}")
            except Exception as e:
                logger.warning(f"Failed to cache HTML for {url}: {e}")
        
        return html_content
    
    async def _adaptive_discover_function_urls(self, session: aiohttp.ClientSession, soup: BeautifulSoup, base_url: str, library_name: str) -> List[str]:
        """Adaptively discover function URLs based on the ReadTheDocs site structure."""
        logger.info(f"Starting adaptive discovery for {library_name}")
        
        # Try multiple URL patterns commonly used in ReadTheDocs sites
        url_patterns = [
            # Pattern 1: Standard scanpy/anndata style (generated/)
            r'generated/[^"]*\.html',
            # Pattern 2: scvi-tools style (reference/)
            r'reference/[^"]*\.html',
            # Pattern 3: Direct api/ links
            r'api/[^"]*\.html',
            # Pattern 4: SnapATAC2 style (_autosummary/)
            r'_autosummary/[^"]*\.html',
            # Pattern 5: scCODA style (module.function.html) - try lowercase library name
            rf'{library_name.lower().replace("-", "")}\.[\w.]+\.html',
            # Pattern 6: General documentation links
            r'(?:api|reference|generated|_autosummary)/[^"]*\.html'
        ]
        
        function_urls = set()
        
        # Try each pattern
        for pattern in url_patterns:
            links = soup.find_all('a', href=re.compile(pattern))
            pattern_urls = []
            
            for link in links:
                href = link.get('href')
                if href:
                    # Build full URL handling relative paths correctly
                    parsed_url = urlparse(base_url)
                    
                    # Handle different URL patterns for relative path resolution
                    
                    if parsed_url.path.endswith('/index.html'):
                        # For URLs like .../api/index.html, relative paths are relative to the parent directory (.../api/)
                        parent_path = '/'.join(parsed_url.path.split('/')[:-1]) + '/'
                        base_dir_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parent_path}"
                    elif parsed_url.path.endswith('.html'):
                        # For URLs like .../api.html, determine if href is relative to document root or parent
                        if href.startswith(('api/', 'reference/', 'generated/', '_autosummary/')):
                            # These paths are typically relative to version root (e.g., /en/stable/)
                            path_parts = parsed_url.path.split('/')
                            try:
                                version_index = path_parts.index('stable') if 'stable' in path_parts else path_parts.index('latest')
                                version_root_parts = path_parts[:version_index + 1]
                                version_root = '/'.join(version_root_parts) + '/'
                                base_dir_url = f"{parsed_url.scheme}://{parsed_url.netloc}{version_root}"
                            except (ValueError, IndexError):
                                # Fallback: use parent directory
                                parent_path = '/'.join(parsed_url.path.split('/')[:-1]) + '/'
                                base_dir_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parent_path}"
                        else:
                            # Other relative paths use parent directory
                            parent_path = '/'.join(parsed_url.path.split('/')[:-1]) + '/'
                            base_dir_url = f"{parsed_url.scheme}://{parsed_url.netloc}{parent_path}"
                    else:
                        # Directory URLs: use as-is
                        base_dir_url = base_url if base_url.endswith('/') else base_url + '/'
                    
                    full_url = urljoin(base_dir_url, href)
                    
                    # Filter out unwanted URLs (navigation, indices, etc.)
                    if self._is_valid_function_url(full_url, library_name):
                        pattern_urls.append(full_url)
            
            if pattern_urls:
                logger.info(f"Pattern '{pattern}' found {len(pattern_urls)} URLs for {library_name}")
                function_urls.update(pattern_urls)
        
        # If we found URLs, return them
        if function_urls:
            return list(function_urls)
        
        # Fallback: Try to discover the site structure by exploring common paths
        logger.info(f"No URLs found with standard patterns, exploring site structure for {library_name}")
        return await self._explore_site_structure(session, base_url, library_name)
    
    def _is_valid_function_url(self, url: str, library_name: str) -> bool:
        """Check if a URL is likely to be a valid function documentation page."""
        url_lower = url.lower()
        library_lower = library_name.lower()
        
        # Skip general documentation and navigation pages
        skip_patterns = [
            'genindex.html',
            'search.html',
            'modules.html', 
            'contents.html',
            'install',
            'tutorial',
            'user_guide',
            'developer',
            'changelog',
            'release',
            'contribute',
            'about',
            'license'
        ]
        
        # Skip main index pages but allow function-specific index pages
        if 'index.html' in url_lower:
            # Allow index pages that are in function-specific directories
            if not any(pattern in url_lower for pattern in ['_autosummary/', 'generated/', 'reference/', 'api/']):
                return False
        
        for pattern in skip_patterns:
            if pattern in url_lower:
                return False
        
        # Must be an HTML file (handle URLs with fragments)
        if not ('.html' in url_lower and (url_lower.endswith('.html') or '#' in url_lower.split('.html')[-1])):
            return False
            
        # Must contain function-like patterns (case-insensitive library name matching)
        function_indicators = [
            f'{library_lower}.',  # Package prefix (lowercase)
            f'{library_lower}_',  # Package prefix with underscore
        ]
        
        # For _autosummary/ URLs, accept if they contain the library name in any case
        if '_autosummary/' in url_lower:
            return any(indicator in url_lower for indicator in function_indicators)
        
        # For other patterns, be more flexible with naming
        return any(indicator in url_lower for indicator in function_indicators) or \
               any(pattern in url_lower for pattern in ['generated/', 'reference/', 'api/'])
    
    async def _explore_site_structure(self, session: aiohttp.ClientSession, base_url: str, library_name: str) -> List[str]:
        """Explore ReadTheDocs site structure to find function documentation."""
        logger.info(f"Exploring site structure for {library_name}")
        
        # Common ReadTheDocs directory structures to explore
        paths_to_try = [
            'api/',
            'reference/',
            'generated/',
            'modules/',
            'autoapi/',
            f'{library_name}/'
        ]
        
        function_urls = []
        parsed_base = urlparse(base_url)
        base_without_file = f"{parsed_base.scheme}://{parsed_base.netloc}{'/'.join(parsed_base.path.split('/')[:-1])}/"
        
        for path in paths_to_try:
            try:
                test_url = urljoin(base_without_file, path)
                logger.info(f"Exploring: {test_url}")
                
                html = await self._fetch_page(session, test_url)
                if html:
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Look for links to function documentation
                    links = soup.find_all('a', href=True)
                    for link in links:
                        href = link.get('href')
                        if href and self._looks_like_function_link(href, library_name):
                            full_url = urljoin(test_url, href)
                            if self._is_valid_function_url(full_url, library_name):
                                function_urls.append(full_url)
                
                if function_urls:
                    logger.info(f"Found {len(function_urls)} function URLs in {path}")
                    break
                    
            except Exception as e:
                logger.debug(f"Error exploring {path}: {e}")
        
        return function_urls
    
    def _looks_like_function_link(self, href: str, library_name: str) -> bool:
        """Check if a link looks like it points to function documentation."""
        href_lower = href.lower()
        
        # Must have these characteristics
        return (
            '.html' in href_lower and
            library_name.lower() in href_lower and
            not any(skip in href_lower for skip in ['index', 'contents', 'modules', 'search'])
        )
    
    def _extract_api_function_urls_from_html(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract API function URLs from main page HTML (legacy method for backward compatibility)."""
        return self._extract_readthedocs_function_urls_sync(soup, base_url)
    
    async def _bulk_download_function_pages(self, session: aiohttp.ClientSession, function_urls: List[str], library_name: str, org_name: str, version: str, use_cache_only: bool = False) -> List[str]:
        """Download function pages and save them to hierarchical cache.
        
        Args:
            session: HTTP session
            function_urls: URLs to download
            library_name: Library name
            org_name: Organization name for hierarchical caching
            version: Documentation version (e.g., 'stable', 'latest')
            use_cache_only: If True, only read from cache, don't fetch from web
            
        Returns:
            List of HTML content strings
        """
        logger.info(f"Downloading {len(function_urls)} function pages for {library_name}...")
        html_contents = []
        
        # Get HTML cache directory for this library
        html_cache_dir = self.hierarchical_cache.get_html_cache_dir(org_name) / f"{library_name}_{version}"
        html_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Download up to first 10,000 pages - this is a safeguard that shouldn't be reached
        # Most libraries have 20-200 functions, but we want to extract all available documentation
        urls_to_fetch = function_urls[:10000]
        
        for i, url in enumerate(urls_to_fetch):
            try:
                # Create filename from URL
                filename = self._url_to_filename(url)
                html_file_path = html_cache_dir / filename
                
                # Try to read from cache first
                html_content = None
                if html_file_path.exists():
                    try:
                        with open(html_file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        logger.debug(f"Loaded cached HTML: {html_file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to read cached HTML from {html_file_path}: {e}")
                
                # If no cache and cache_only mode, skip
                if not html_content and use_cache_only:
                    logger.debug(f"Cache-only mode: skipping {url} (no cache found)")
                    continue
                
                # Fetch from web if no cache or not cache_only
                if not html_content:
                    html_content = await self._fetch_page(session, url)
                    
                    if html_content:
                        # Save HTML file to hierarchical cache
                        try:
                            with open(html_file_path, 'w', encoding='utf-8') as f:
                                f.write(html_content)
                            
                            logger.debug(f"Cached HTML file: {html_file_path}")
                        except Exception as e:
                            logger.warning(f"Failed to cache HTML file for {url}: {e}")
                        
                        # Small delay to be respectful to the server
                        if i < len(urls_to_fetch) - 1:
                            await asyncio.sleep(0.1)
                
                if html_content:
                    html_contents.append(html_content)
                    
            except Exception as e:
                logger.debug(f"Error processing {url}: {e}")
                continue
        
        logger.info(f"Successfully downloaded {len(html_contents)}/{len(urls_to_fetch)} function pages")
        return html_contents
    
    def _url_to_filename(self, url: str) -> str:
        """Convert URL to a safe filename for caching.
        
        Args:
            url: URL to convert
            
        Returns:
            Safe filename string
        """
        import re
        from urllib.parse import urlparse
        
        parsed = urlparse(url)
        # Get the path without leading slash and replace problematic characters
        path = parsed.path.lstrip('/')
        # Replace slashes and other problematic characters with underscores
        filename = re.sub(r'[/\\:*?"<>|]', '_', path)
        # Ensure it ends with .html
        if not filename.endswith('.html'):
            filename += '.html'
        return filename
    
    async def _process_html_contents(self, html_contents: List[str], function_urls: List[str], api_data: Dict[str, Any]) -> None:
        """Process downloaded HTML content to extract function information.
        
        Args:
            html_contents: List of HTML content strings
            function_urls: Corresponding URLs (for reference)
            api_data: Dictionary to update with extracted data
        """
        for i, html_content in enumerate(html_contents):
            try:
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Get corresponding URL for reference
                url = function_urls[i] if i < len(function_urls) else "unknown"
                
                # Extract function details
                func_info = self._extract_readthedocs_function_details(soup, url)
                
                if func_info:
                    # Check for duplicates
                    existing = next((f for f in api_data["functions"] if f["name"] == func_info["name"]), None)
                    if existing:
                        # Update with more detailed information
                        existing.update(func_info)
                    else:
                        # Add new function
                        api_data["functions"].append(func_info)
                
                # Log progress periodically
                if (i + 1) % 10 == 0:
                    logger.debug(f"Processed {i + 1}/{len(html_contents)} downloaded HTML pages")
                    
            except Exception as e:
                logger.debug(f"Error processing HTML content for {function_urls[i] if i < len(function_urls) else 'unknown'}: {e}")
    
    async def _process_cached_html_files(self, html_files: List[Path], function_urls: List[str], api_data: Dict[str, Any]) -> None:
        """Process cached HTML files locally to extract function information.
        
        Args:
            html_files: List of cached HTML file paths
            function_urls: Corresponding URLs (for reference)
            api_data: Dictionary to update with extracted data
        """
        for i, html_file in enumerate(html_files):
            try:
                html_content = html_file.read_text(encoding='utf-8')
                soup = BeautifulSoup(html_content, 'html.parser')
                
                # Get corresponding URL for reference
                url = function_urls[i] if i < len(function_urls) else "unknown"
                
                # Extract function details
                func_info = self._extract_readthedocs_function_details(soup, url)
                
                if func_info:
                    # Check for duplicates
                    existing = next((f for f in api_data["functions"] if f["name"] == func_info["name"]), None)
                    if existing:
                        # Update with more detailed information
                        existing.update(func_info)
                    else:
                        # Add new function
                        api_data["functions"].append(func_info)
                
                # Log progress periodically
                if (i + 1) % 50 == 0:
                    logger.info(f"Processed {i + 1}/{len(html_files)} cached HTML files")
                    
            except Exception as e:
                logger.debug(f"Error processing cached file {html_file}: {e}")
    
    def _extract_readthedocs_function_urls_sync(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract individual function page URLs from ReadTheDocs main API page."""
        try:
            function_urls = []
            
            # Convert base_url to directory path (remove filename if present)
            parsed_url = urlparse(base_url)
            if parsed_url.path.endswith('.html'):
                # Remove the filename and use the directory path
                directory_path = '/'.join(parsed_url.path.split('/')[:-1]) + '/'
                base_dir_url = f"{parsed_url.scheme}://{parsed_url.netloc}{directory_path}"
            else:
                base_dir_url = base_url if base_url.endswith('/') else base_url + '/'
            
            # Look for links to generated/ pages (ReadTheDocs pattern)
            links = soup.find_all('a', href=re.compile(r'generated/[^"]*\.html'))
            
            for link in links:
                href = link.get('href')
                if href:
                    # Build full URL using directory base
                    full_url = urljoin(base_dir_url, href)
                    
                    if full_url not in function_urls:
                        function_urls.append(full_url)
            
            logger.info(f"Extracted {len(function_urls)} function page URLs from ReadTheDocs main API page")
            return function_urls
            
        except Exception as e:
            logger.error(f"Error extracting ReadTheDocs function page URLs: {e}")
            return []
    
    def _extract_functions_from_api_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract functions from API reference tables on the main page."""
        functions = []
        
        # Look for tables containing API documentation
        tables = soup.find_all('table', class_='autosummary')
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    # First cell typically contains the function signature
                    first_cell = cells[0]
                    
                    # Look for code elements containing function names
                    code_elements = first_cell.find_all('code')
                    for code in code_elements:
                        text = code.get_text(strip=True)
                        
                        # Match patterns like "io.read_h5ad", "AnnData.write_h5ad", etc.
                        if '.' in text and self._is_likely_python_function(text):
                            # Extract the function name (last part after the dot)
                            func_name = text.split('.')[-1]
                            
                            # Get the description from the second cell
                            description = ""
                            if len(cells) > 1:
                                desc_cell = cells[1]
                                description = desc_cell.get_text(strip=True)
                            
                            functions.append({
                                'name': func_name,
                                'signature': f"{text}()",
                                'description': description,
                                'type': 'function',
                                'source': 'api_table'
                            })
        
        return functions
    
    def _extract_readthedocs_function_details(self, soup: BeautifulSoup, func_url: str) -> Optional[Dict[str, Any]]:
        """Extract detailed function information from an individual ReadTheDocs function page."""
        try:
            # Look for the function signature in the page
            # Pattern 1: Look for dt elements with function signatures
            signature_elements = soup.find_all('dt', class_='sig')
            
            for sig_elem in signature_elements:
                sig_text = sig_elem.get_text(strip=True)
                
                # Extract function name from the signature
                func_match = re.search(r'([a-zA-Z_][a-zA-Z0-9_.]*)\s*\(', sig_text)
                if func_match:
                    full_name = func_match.group(1)
                    
                    # Extract module and function parts
                    parts = full_name.split('.')
                    if len(parts) >= 3:  # e.g., scanpy.tl.umap
                        # For full qualified names, preserve module.function format
                        # This prevents tl.umap and pl.umap from being treated as duplicates
                        func_name = '.'.join(parts[-2:])  # "tl.umap"
                        module = parts[-2]  # "tl"
                    elif len(parts) == 2:  # e.g., module.function
                        func_name = parts[-1]
                        module = parts[0]
                    else:
                        func_name = full_name
                        module = "unknown"
                    
                    # Extract detailed documentation from dd element
                    doc_elem = sig_elem.find_next_sibling('dd')
                    if doc_elem:
                        # Extract comprehensive documentation
                        doc_info = self._extract_comprehensive_documentation(doc_elem)
                        
                        return {
                            'name': func_name,
                            'module': module,
                            'signature': sig_text,
                            'description': doc_info.get('description', ''),
                            'parameters': doc_info.get('parameters', ''),
                            'returns': doc_info.get('returns', ''),
                            'examples': doc_info.get('examples', ''),
                            'type': 'function',
                            'url': func_url,
                            'source': 'individual_page'
                        }
            
            # Pattern 2: Look for title elements that might contain function names
            title = soup.find('title')
            if title:
                title_text = title.get_text(strip=True)
                # Check if title looks like a function name (e.g., "anndata.io.read_h5ad")
                if '.' in title_text and self._is_likely_python_function(title_text.split(' — ')[0]):
                    func_name = title_text.split('.')[-1].split(' ')[0]
                    
                    # Look for any description in the page
                    paragraphs = soup.find_all('p')
                    documentation = ""
                    for p in paragraphs[:3]:  # First few paragraphs
                        text = p.get_text(strip=True)
                        if text and not text.startswith('©'):  # Skip copyright
                            documentation += text + " "
                    
                    return {
                        'name': func_name,
                        'signature': f"{title_text.split(' — ')[0]}()",
                        'description': documentation.strip(),
                        'type': 'function',
                        'url': func_url,
                        'source': 'individual_page'
                    }
                    
        except Exception as e:
            logger.debug(f"Error extracting individual function info: {e}")
        
        return None
    
    def _extract_comprehensive_documentation(self, doc_elem) -> Dict[str, str]:
        """Extract comprehensive documentation including parameters, returns, and examples."""
        doc_info = {
            'description': '',
            'parameters': '',
            'returns': '',
            'examples': ''
        }
        
        try:
            # Extract main description (first paragraph(s))
            description_parts = []
            for child in doc_elem.children:
                if hasattr(child, 'name'):
                    if child.name == 'p':
                        text = child.get_text(strip=True)
                        if text and not any(keyword in text.lower() for keyword in ['parameters', 'returns', 'examples', 'raises']):
                            description_parts.append(text)
                    elif child.name in ['dl', 'div']:
                        # Stop at parameter/return sections
                        break
            
            doc_info['description'] = ' '.join(description_parts)
            
            # Extract parameters section
            param_sections = doc_elem.find_all('dl', class_='field-list')
            for dl in param_sections:
                dt_elements = dl.find_all('dt', class_='field-odd')
                for dt in dt_elements:
                    if 'parameters' in dt.get_text().lower():
                        dd = dt.find_next_sibling('dd', class_='field-odd')
                        if dd:
                            # Extract parameter details
                            param_text = self._extract_parameter_details(dd)
                            doc_info['parameters'] = param_text
                
                # Also check for field-even class
                dt_elements = dl.find_all('dt', class_='field-even')
                for dt in dt_elements:
                    if 'returns' in dt.get_text().lower():
                        dd = dt.find_next_sibling('dd', class_='field-even')
                        if dd:
                            doc_info['returns'] = dd.get_text(strip=True)
            
            # Extract examples section
            example_sections = doc_elem.find_all('div', class_='highlight')
            if example_sections:
                examples = []
                for section in example_sections:
                    code_text = section.get_text(strip=True)
                    if code_text:
                        examples.append(code_text)
                doc_info['examples'] = '\n'.join(examples)
            
            # Alternative: look for doctest sections
            doctest_sections = doc_elem.find_all('div', class_='doctest')
            if doctest_sections and not doc_info['examples']:
                examples = []
                for section in doctest_sections:
                    code_text = section.get_text(strip=True)
                    if code_text:
                        examples.append(code_text)
                doc_info['examples'] = '\n'.join(examples)
            
        except Exception as e:
            logger.debug(f"Error extracting comprehensive documentation: {e}")
        
        return doc_info
    
    def _extract_parameter_details(self, dd_elem) -> List[Dict[str, str]]:
        """Extract detailed parameter information from a dd element."""
        try:
            param_details = []

            # Look for dl elements containing parameter lists
            param_lists = dd_elem.find_all('dl', class_='simple')
            for param_list in param_lists:
                dt_elements = param_list.find_all('dt')
                for dt in dt_elements:
                    param_name_elem = dt.find('strong')
                    if param_name_elem:
                        param_name = param_name_elem.get_text(strip=True)

                        # Get parameter description
                        dd = dt.find_next_sibling('dd')
                        param_desc = dd.get_text(strip=True) if dd else ""

                        param_details.append({
                            "name": param_name,
                            "description": param_desc
                        })
                    else:
                        # If no strong tag, use the whole dt text as parameter name
                        param_info = dt.get_text(strip=True)
                        if param_info:
                            param_details.append({
                                "name": param_info,
                                "description": ""
                            })

            return param_details

        except Exception as e:
            logger.debug(f"Error extracting parameter details: {e}")
            return []
    
    def _is_likely_python_function(self, text: str) -> bool:
        """Check if a text string represents a likely Python function."""
        # Common Python patterns
        python_patterns = [
            r'^[a-zA-Z_][a-zA-Z0-9_.]*\.[a-zA-Z_][a-zA-Z0-9_]*$',  # module.function
            r'^[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*$',  # module.submodule.function
        ]
        
        # Check if it matches Python patterns
        for pattern in python_patterns:
            if re.match(pattern, text):
                # Additional checks to filter out JavaScript
                if not any(js_keyword in text.lower() for js_keyword in ['localstorage', 'document', 'window', 'console']):
                    return True
        
        return False
    
    def _extract_version_from_url(self, url: str) -> str:
        """Extract version information from ReadTheDocs URL."""
        try:
            # Parse URL to extract version
            # e.g., https://anndata.readthedocs.io/en/latest/api.html -> latest
            # e.g., https://anndata.readthedocs.io/en/stable/api.html -> stable
            # e.g., https://anndata.readthedocs.io/en/v0.8.0/api.html -> v0.8.0
            
            import re
            version_match = re.search(r'/en/([^/]+)/', url)
            if version_match:
                return version_match.group(1)
            
            # Default fallback
            return "latest"
            
        except Exception as e:
            logger.debug(f"Error extracting version from URL {url}: {e}")
            return "latest"
    
    async def _bulk_download_and_process_pages(self, session: aiohttp.ClientSession, function_urls: List[str], api_data: Dict[str, Any]) -> None:
        """Bulk download and process all function pages to avoid rate limiting.
        
        Args:
            session: HTTP session
            function_urls: List of URLs to download
            api_data: Dictionary to update with extracted function data
        """
        import asyncio
        
        # Create semaphore to limit concurrent requests (be very conservative)
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent requests to be respectful
        
        async def download_and_process_page(url: str) -> Optional[Dict[str, Any]]:
            """Download and process a single page with rate limiting."""
            async with semaphore:
                try:
                    # Add delay to be respectful (stagger requests)
                    await asyncio.sleep(0.5)
                    
                    func_html = await self._fetch_page(session, url)
                    if func_html:
                        func_soup = BeautifulSoup(func_html, 'html.parser')
                        return self._extract_readthedocs_function_details(func_soup, url)
                    return None
                except Exception as e:
                    logger.debug(f"Error processing ReadTheDocs function page {url}: {e}")
                    return None
        
        # Create tasks for all URLs
        logger.info(f"Starting bulk download of {len(function_urls)} ReadTheDocs pages with rate limiting...")
        tasks = [download_and_process_page(url) for url in function_urls]
        
        # Process in smaller batches to be respectful
        batch_size = 20
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            # Process results
            for func_info in batch_results:
                if isinstance(func_info, dict) and func_info:
                    # Check if we already have this function from the main page
                    existing = next((f for f in api_data["functions"] if f["name"] == func_info["name"]), None)
                    if existing:
                        # Update with more detailed information
                        existing.update(func_info)
                    else:
                        # Add new function
                        api_data["functions"].append(func_info)
            
            # Log progress
            processed_count = min(i + batch_size, len(function_urls))
            logger.info(f"Processed {processed_count}/{len(function_urls)} ReadTheDocs function pages")
            
            # Add longer delay between batches to be respectful
            if i + batch_size < len(function_urls):
                await asyncio.sleep(10)  # 10 second delay between batches

    async def _fetch_page(self, session: aiohttp.ClientSession, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetch a web page with exponential backoff retry logic.
        
        Args:
            session: HTTP session
            url: URL to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            Page HTML content
        """
        import asyncio
        
        for attempt in range(max_retries + 1):
            try:
                headers = {
                    "User-Agent": "KaiAgent",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
                
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        return await response.text()
                    elif response.status == 429:  # Rate limited
                        if attempt < max_retries:
                            # Exponential backoff: 5s, 15s, 45s
                            delay = 5 * (3 ** attempt)
                            logger.warning(f"Rate limited for {url}, retrying in {delay}s (attempt {attempt + 1}/{max_retries + 1})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(f"Rate limited for {url} after {max_retries} retries")
                            return None
                    else:
                        logger.warning(f"Failed to fetch {url}: {response.status}")
                        if attempt < max_retries and response.status >= 500:
                            # Retry server errors
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return None
                        
            except Exception as e:
                if attempt < max_retries:
                    logger.debug(f"Error fetching {url} (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(2 ** attempt)
                    continue
                else:
                    logger.error(f"Error fetching {url} after {max_retries} retries: {e}")
                    return None
        
        return None