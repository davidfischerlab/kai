"""Centralized path configuration for the agent.

This module defines all directory paths used by the agent to ensure consistency
and prevent creation of outdated directories.
"""
import os
from pathlib import Path

# Base directory for all agent data
AGENT_BASE_DIR = Path.home() / ".kai_agent"

# Main cache directory
BIOINFORMATICS_CACHE_DIR = AGENT_BASE_DIR / "github_cache"

# Knowledge base directory - can be overridden by KAI_RETRIEVAL_DIR environment variable
RETRIEVAL_DIR = Path(os.environ.get('KAI_RETRIEVAL_DIR', AGENT_BASE_DIR / "retrieval"))

# Logs directory (only for actual log files)
LOGS_DIR = AGENT_BASE_DIR / "logs"

# Debug prompts directory
DEBUG_PROMPTS_DIR = AGENT_BASE_DIR / "prompt_debugging"

# Organization-specific subdirectories (relative to organization root)
ORG_REPOS_DIR = "repos"
ORG_DOCS_CACHE_DIR = "docs_cache"
ORG_HTML_DIR = "html"
ORG_API_CACHE_FILE = "api_cache.json"


def get_organization_dir(org_name: str) -> Path:
    """Get the directory path for a specific organization.

    Args:
        org_name: Name of the organization

    Returns:
        Path to the organization directory
    """
    return BIOINFORMATICS_CACHE_DIR / org_name


def get_org_repos_dir(org_name: str) -> Path:
    """Get the repositories directory for an organization.

    Args:
        org_name: Name of the organization

    Returns:
        Path to the organization's repos directory
    """
    return get_organization_dir(org_name) / ORG_REPOS_DIR


def get_org_docs_cache_dir(org_name: str) -> Path:
    """Get the documentation cache directory for an organization.

    Args:
        org_name: Name of the organization

    Returns:
        Path to the organization's docs cache directory
    """
    return get_organization_dir(org_name) / ORG_DOCS_CACHE_DIR


def get_org_html_dir(org_name: str) -> Path:
    """Get the HTML files directory for an organization.

    Args:
        org_name: Name of the organization

    Returns:
        Path to the organization's HTML directory
    """
    return get_org_docs_cache_dir(org_name) / ORG_HTML_DIR




def get_org_api_cache_file(org_name: str) -> Path:
    """Get the API cache file for an organization.

    Args:
        org_name: Name of the organization

    Returns:
        Path to the organization's API cache JSON file
    """
    return get_organization_dir(org_name) / ORG_API_CACHE_FILE


def get_shared_cache_file() -> Path:
    """Get the shared cache file for global GitHub data.
    
    Returns:
        Path to the shared.json file
    """
    return BIOINFORMATICS_CACHE_DIR / "shared.json"


def get_debug_prompts_dir() -> Path:
    """Get the base debug prompts directory.
    
    Returns:
        Path to the base debug prompts directory
    """
    return DEBUG_PROMPTS_DIR


def ensure_base_directories():
    """Create base directories if they don't exist."""
    AGENT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    BIOINFORMATICS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)


def ensure_org_directories(org_name: str):
    """Create organization directories if they don't exist.

    Args:
        org_name: Name of the organization
    """
    org_dir = get_organization_dir(org_name)
    org_dir.mkdir(parents=True, exist_ok=True)

    get_org_repos_dir(org_name).mkdir(parents=True, exist_ok=True)
    get_org_docs_cache_dir(org_name).mkdir(parents=True, exist_ok=True)
    get_org_html_dir(org_name).mkdir(parents=True, exist_ok=True)
