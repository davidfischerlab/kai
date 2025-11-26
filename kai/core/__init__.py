"""Core module for bioinformatics agent."""
from .llm_interface import LLMInterface


# Import only when needed to avoid circular imports
def get_kai_agent():
    from .agent import KaiAgent
    return KaiAgent


__all__ = ["LLMInterface", "get_kai_agent"]