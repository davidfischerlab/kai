"""
Simple LLM prompts for repository filtering.
"""

# Generic repository screening prompt for single-cell and spatial omics
REPO_SCREENING_PROMPT = """I am filtering repositories to find only those related to single-cell data analysis or spatial omics data analysis.

Please determine if this repository is relevant for single-cell or spatial omics analysis based on:
- Single-cell RNA sequencing (scRNA-seq)
- Single-cell ATAC sequencing (scATAC-seq)  
- Single-cell multi-omics
- Spatial transcriptomics
- Spatial proteomics
- Cell atlas projects
- Single-cell analysis tools
- Spatial omics analysis tools

Look for keywords like: single-cell, single-nucleus, scRNA, scATAC, spatial, omics, transcriptomics, proteomics, cell atlas, but also consider context and related terms.

EXCLUDE administrative, meta, or infrastructure repositories such as:
- Website repositories (*.github.io, website, docs-only)
- Package management/ecosystem repositories
- Governance, administrative, or organizational repositories
- Bot, automation, or CI/CD repositories  
- Cookiecutter templates or scaffolding repositories
- General infrastructure or tooling repositories

Repository name: {repo_name}
Repository description: {repo_description}
Repository topics: {repo_topics}
README content (first 1000 chars): {readme_content}

Please respond with only "RELEVANT" or "NOT_RELEVANT" followed by a brief explanation."""

# Clarification prompt for when LLM response doesn't follow the template
REPO_SCREENING_CLARIFICATION_PROMPT = """Your previous answer did not adhere to the required template. 

Please respond with ONLY "RELEVANT" or "NOT_RELEVANT" followed by a brief explanation.

Your answer must start with exactly one of these words:
- "RELEVANT" (if the repository is related to single-cell or spatial omics analysis)
- "NOT_RELEVANT" (if the repository is not related to single-cell or spatial omics analysis)

Please provide your answer now:"""


def format_repo_screening_prompt(repo_name: str, repo_description: str, repo_topics: list, readme_content: str = "") -> str:
    """Format repository screening prompt with repository data.
    
    Args:
        repo_name: Repository name
        repo_description: Repository description
        repo_topics: List of repository topics
        readme_content: README content from the repository
        
    Returns:
        Formatted prompt string
    """
    topics_str = ", ".join(repo_topics) if repo_topics else "None"
    
    return REPO_SCREENING_PROMPT.format(
        repo_name=repo_name,
        repo_description=repo_description or "No description available",
        repo_topics=topics_str,
        readme_content=readme_content if readme_content else "No README available"
    )