"""
ReadTheDocs URL exceptions lookup table.

This module contains hardcoded exceptional ReadTheDocs locations for packages
that don't follow the standard {package}.readthedocs.io pattern.
"""

# Dictionary mapping package names to their custom documentation URLs
READTHEDOCS_EXCEPTIONS = {
    "scvi-tools": {
        "base_url": "https://docs.scvi-tools.org",
        "api_paths": [
            "/en/stable/api/index.html",
            "/en/latest/api/index.html",
            "/en/stable/api/",
            "/en/latest/api/"
        ],
        "notes": "Uses custom domain docs.scvi-tools.org instead of readthedocs.io"
    }
}


def get_exception_urls(package_name: str) -> list[str]:
    """
    Get custom ReadTheDocs URLs for packages with non-standard hosting.

    Args:
        package_name: Name of the package

    Returns:
        List of URLs to try, empty list if no exception exists
    """
    if package_name in READTHEDOCS_EXCEPTIONS:
        exception = READTHEDOCS_EXCEPTIONS[package_name]
        base_url = exception["base_url"]
        return [f"{base_url}{path}" for path in exception["api_paths"]]
    return []