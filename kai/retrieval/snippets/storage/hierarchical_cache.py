"""Hierarchical cache interface for organization-based cache structure."""

import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from kai.config.paths import (
    AGENT_BASE_DIR, BIOINFORMATICS_CACHE_DIR,
    get_organization_dir, get_org_repos_dir, get_org_docs_cache_dir,
    get_org_html_dir,
    get_org_api_cache_file, ensure_base_directories, ensure_org_directories
)

logger = logging.getLogger(__name__)


class HierarchicalCache:
    """Cache interface for the new hierarchical organization-based structure."""
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize hierarchical cache.
        
        Args:
            base_dir: Base directory for cache.
        """
        if base_dir is None:
            base_dir = AGENT_BASE_DIR
        
        self.base_dir = Path(base_dir)
        self.organizations_dir = BIOINFORMATICS_CACHE_DIR
        ensure_base_directories()
        
    def get_organization_dir(self, org_name: str) -> Path:
        """Get directory for an organization."""
        ensure_org_directories(org_name)
        return get_organization_dir(org_name)
    
    def get_organization_metadata_file(self, org_name: str) -> Path:
        """Get metadata file path for an organization."""
        return get_org_api_cache_file(org_name)
    
    def get_repos_dir(self, org_name: str) -> Path:
        """Get repos directory for an organization."""
        ensure_org_directories(org_name)
        return get_org_repos_dir(org_name)
    
    def get_docs_cache_dir(self, org_name: str) -> Path:
        """Get documentation cache directory for an organization."""
        ensure_org_directories(org_name)
        return get_org_docs_cache_dir(org_name)
    
    def get_html_cache_dir(self, org_name: str) -> Path:
        """Get HTML files cache directory for an organization."""
        ensure_org_directories(org_name)
        return get_org_html_dir(org_name)
    
    # Organization metadata methods
    def cache_organization_metadata(self, org_name: str, language: str, repos: Dict[str, Any]) -> None:
        """Cache organization metadata (list of repositories)."""
        # Ensure organization directories exist
        ensure_org_directories(org_name)
        metadata_file = self.get_organization_metadata_file(org_name)

        # Load existing metadata if it exists
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Update with new data using api_cache.json structure
        metadata.update({
            "organization": org_name,
            "cache_key": f"{org_name}_{language}",
            "repositories": metadata.get("repositories", {}),
            "cached_at": datetime.now().isoformat(),
            "languages": metadata.get("languages", [])
        })
        
        # Add language if not already present
        if language not in metadata["languages"]:
            metadata["languages"].append(language)
        
        # Update repositories
        metadata["repositories"].update(repos)
        
        # Save metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Cached metadata for {org_name} ({language}): {len(repos)} repositories")
    
    def get_cached_organization_metadata(self, org_name: str) -> Optional[Dict[str, Any]]:
        """Get cached organization metadata."""
        metadata_file = self.get_organization_metadata_file(org_name)
        
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        
        return None
    
    def get_cached_repos(self, org_name: str, language: str) -> Optional[Dict[str, Any]]:
        """Get cached repositories for an organization and language."""
        metadata = self.get_cached_organization_metadata(org_name)
        
        if metadata and "repositories" in metadata:
            # Filter repositories by language if needed
            # For now, return all repos since they're language-agnostic
            return metadata["repositories"]
        
        return None
    
    # Repository clone methods
    def get_repo_path(self, org_name: str, repo_name: str) -> Path:
        """Get path for a cloned repository."""
        return self.get_repos_dir(org_name) / repo_name
    
    def is_repo_cloned(self, org_name: str, repo_name: str) -> bool:
        """Check if a repository is already cloned."""
        repo_path = self.get_repo_path(org_name, repo_name)
        return repo_path.exists() and (repo_path / ".git").exists()
    
    # HTML cache methods
    
    def is_html_cached(self, org_name: str, repo_name: str, version: str = "stable") -> bool:
        """Check if HTML documentation is cached for a repository.
        
        Returns True if either:
        - HTML files exist in the cache directory
        - A failure marker exists indicating no documentation is available
        """
        html_cache_dir = self.get_html_cache_dir(org_name) / f"{repo_name}_{version}"
        if not html_cache_dir.exists():
            return False
        
        # Check if there are any files in the directory
        return any(html_cache_dir.iterdir())
    
    # Statistics and utilities
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = {
            "total_organizations": 0,
            "total_repos_cloned": 0,
            "total_html_cached": 0,
            "organizations": {}
        }
        
        for org_dir in self.organizations_dir.iterdir():
            if org_dir.is_dir():
                org_name = org_dir.name
                stats["total_organizations"] += 1
                
                # Count cloned repos
                repos_dir = org_dir / "repos"
                cloned_repos = 0
                if repos_dir.exists():
                    cloned_repos = len([r for r in repos_dir.iterdir() if r.is_dir() and (r / ".git").exists()])
                
                # Count HTML caches (JSON extractions)
                html_extraction_dir = org_dir / "docs_cache" / "html_extraction"
                html_caches = 0
                if html_extraction_dir.exists():
                    html_caches = len([f for f in html_extraction_dir.iterdir() if f.is_file() and f.name.endswith('.json')])
                
                # Count HTML files
                html_files_dir = org_dir / "docs_cache" / "html"
                html_files = 0
                if html_files_dir.exists():
                    html_files = len([d for d in html_files_dir.iterdir() if d.is_dir()])
                
                stats["organizations"][org_name] = {
                    "cloned_repos": cloned_repos,
                    "html_caches": html_caches,
                    "html_files": html_files
                }
                
                stats["total_repos_cloned"] += cloned_repos
                stats["total_html_cached"] += html_caches
        
        return stats
    
    def get_all_cloned_repos(self) -> Dict[str, Dict[str, Any]]:
        """Get all cloned repositories organized by organization.
        
        Returns:
            Dictionary mapping org_name -> {repo_name -> repo_info}
        """
        all_repos = {}
        
        for org_dir in self.organizations_dir.iterdir():
            if org_dir.is_dir():
                org_name = org_dir.name
                repos_dir = org_dir / "repos"
                
                if repos_dir.exists():
                    org_repos = {}
                    for repo_dir in repos_dir.iterdir():
                        if repo_dir.is_dir() and (repo_dir / ".git").exists():
                            repo_name = repo_dir.name
                            # Create minimal repo info for compatibility
                            org_repos[repo_name] = {
                                "repo": f"{org_name}/{repo_name}",
                                "library": repo_name,
                                "cloned": True
                            }
                    
                    if org_repos:
                        all_repos[org_name] = org_repos
        
        return all_repos

    # GitHub API rate limiting methods (integrated from GitHubCache)
    def update_rate_limit_info(self, org_name: str, remaining: int, reset_time: int) -> None:
        """Update rate limit information for an organization.
        
        Args:
            org_name: Organization name
            remaining: Remaining API calls
            reset_time: Rate limit reset timestamp
        """
        # Store in shared cache file
        shared_file = self.organizations_dir / "shared.json"
        shared_data = {}
        
        if shared_file.exists():
            with open(shared_file, 'r') as f:
                shared_data = json.load(f)
        
        if "rate_limit_info" not in shared_data:
            shared_data["rate_limit_info"] = {}
        
        shared_data["rate_limit_info"][org_name] = {
            "remaining": remaining,
            "reset_time": reset_time,
            "last_updated": datetime.now().isoformat()
        }
        
        with open(shared_file, 'w') as f:
            json.dump(shared_data, f, indent=2)
        
        logger.info(f"Updated rate limit info for {org_name}: {remaining} remaining")
    
    def get_rate_limit_info(self, org_name: str) -> Optional[Dict[str, Any]]:
        """Get rate limit information for an organization.
        
        Args:
            org_name: Organization name
            
        Returns:
            Rate limit info or None if not found
        """
        shared_file = self.organizations_dir / "shared.json"
        
        if shared_file.exists():
            with open(shared_file, 'r') as f:
                shared_data = json.load(f)
                return shared_data.get("rate_limit_info", {}).get(org_name)
        
        return None
    
    def mark_html_download_in_progress(self, org_name: str, library_name: str, version: str = "stable") -> None:
        """Mark HTML cache download as in progress to track incomplete downloads.
        
        This flag helps detect incomplete downloads if the extraction process is aborted
        while HTML files are being cached.
        
        Args:
            org_name: Organization name
            library_name: Library name
            version: Documentation version
        """
        shared_file = self.organizations_dir / "shared.json"
        shared_data = {}
        
        if shared_file.exists():
            with open(shared_file, 'r') as f:
                shared_data = json.load(f)
        
        if "html_downloads_in_progress" not in shared_data:
            shared_data["html_downloads_in_progress"] = {}
        
        cache_key = f"{org_name}/{library_name}_{version}"
        shared_data["html_downloads_in_progress"][cache_key] = {
            "started_at": datetime.now().isoformat(),
            "org_name": org_name,
            "library_name": library_name,
            "version": version
        }
        
        with open(shared_file, 'w') as f:
            json.dump(shared_data, f, indent=2)
        
        logger.debug(f"Marked HTML download as in progress: {cache_key}")
    
    def mark_html_download_complete(self, org_name: str, library_name: str, version: str = "stable") -> None:
        """Mark HTML cache download as complete and remove the in-progress flag.
        
        Args:
            org_name: Organization name
            library_name: Library name
            version: Documentation version
        """
        shared_file = self.organizations_dir / "shared.json"
        
        if shared_file.exists():
            with open(shared_file, 'r') as f:
                shared_data = json.load(f)
            
            cache_key = f"{org_name}/{library_name}_{version}"
            if "html_downloads_in_progress" in shared_data:
                shared_data["html_downloads_in_progress"].pop(cache_key, None)
                
                with open(shared_file, 'w') as f:
                    json.dump(shared_data, f, indent=2)
                
                logger.debug(f"Marked HTML download as complete: {cache_key}")
    
    def is_html_download_incomplete(self, org_name: str, library_name: str, version: str = "stable") -> bool:
        """Check if HTML cache download was marked as incomplete.
        
        Args:
            org_name: Organization name
            library_name: Library name
            version: Documentation version
            
        Returns:
            True if download was marked as incomplete
        """
        shared_file = self.organizations_dir / "shared.json"
        
        if shared_file.exists():
            with open(shared_file, 'r') as f:
                shared_data = json.load(f)
            
            cache_key = f"{org_name}/{library_name}_{version}"
            return cache_key in shared_data.get("html_downloads_in_progress", {})
        
        return False
    
    def cleanup_incomplete_html_downloads(self) -> None:
        """Remove incomplete HTML cache directories and clear their flags.
        
        This method can be called at startup to clean up any incomplete downloads
        from previous runs that were aborted.
        """
        shared_file = self.organizations_dir / "shared.json"
        
        if not shared_file.exists():
            return
            
        with open(shared_file, 'r') as f:
            shared_data = json.load(f)
        
        incomplete_downloads = shared_data.get("html_downloads_in_progress", {})
        
        for cache_key, download_info in list(incomplete_downloads.items()):
            org_name = download_info["org_name"]
            library_name = download_info["library_name"]
            version = download_info["version"]
            
            # Remove the incomplete cache directory
            html_cache_dir = self.get_html_cache_dir(org_name) / f"{library_name}_{version}"
            if html_cache_dir.exists():
                import shutil
                shutil.rmtree(html_cache_dir)
                logger.info(f"Removed incomplete HTML cache directory: {html_cache_dir}")
        
        # Clear all incomplete download flags
        shared_data["html_downloads_in_progress"] = {}
        
        with open(shared_file, 'w') as f:
            json.dump(shared_data, f, indent=2)
        
        if incomplete_downloads:
            logger.info(f"Cleaned up {len(incomplete_downloads)} incomplete HTML downloads")
    
    def is_cache_valid(self, org_name: str, language: str, max_age_hours: int = 24) -> bool:
        """Check if cached data for an organization is still valid.
        
        Args:
            org_name: GitHub organization name
            language: Programming language filter
            max_age_hours: Maximum age of cache in hours (default: 24)
            
        Returns:
            True if cache is valid and not expired
        """
        metadata = self.get_cached_organization_metadata(org_name)
        
        if not metadata or "cached_at" not in metadata:
            return False
        
        try:
            cached_at = datetime.fromisoformat(metadata["cached_at"])
            max_age = timedelta(hours=max_age_hours)
            return datetime.now() - cached_at < max_age
        except Exception as e:
            logger.warning(f"Error parsing cache timestamp: {e}")
            return False
    
    def cleanup_empty_directories(self):
        """Clean up empty directories in the cache."""
        for org_dir in self.organizations_dir.iterdir():
            if org_dir.is_dir():
                # Clean up main subdirectories
                for subdir in ["repos", "logs"]:
                    subdir_path = org_dir / subdir
                    if subdir_path.exists() and not any(subdir_path.iterdir()):
                        subdir_path.rmdir()
                        logger.info(f"Removed empty directory: {subdir_path}")
                
                # Clean up docs_cache subdirectories
                docs_cache_dir = org_dir / "docs_cache"
                if docs_cache_dir.exists():
                    for subdir in ["html", "html_extraction"]:
                        subdir_path = docs_cache_dir / subdir
                        if subdir_path.exists() and not any(subdir_path.iterdir()):
                            subdir_path.rmdir()
                            logger.info(f"Removed empty directory: {subdir_path}")
                    
                    # Remove docs_cache if empty
                    if not any(docs_cache_dir.iterdir()):
                        docs_cache_dir.rmdir()
                        logger.info(f"Removed empty directory: {docs_cache_dir}")
    