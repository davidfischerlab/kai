"""State utilities for accessing dict/object values safely."""

from typing import Any


def safe_get(obj: Any, key: str, default: Any = None) -> Any:
    """Safely get a value from dict or object with attribute access.

    Args:
        obj: Dict or object to access
        key: Key or attribute name
        default: Default value if not found

    Returns:
        Value at key/attribute or default
    """
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
