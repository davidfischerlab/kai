"""Evaluator-optimizer loop subgraphs.

This module contains self-contained subgraphs for evaluator-optimizer loops,
following LangGraph's recommended pattern:
- Direct edge from optimizer to evaluator (no router between them)
- Conditional routing only after evaluator (Accepted → exit, Rejected → loop back)

Each loop is isolated in its own subgraph for clarity and maintainability.
"""

from .task_list_evaluator import build_task_list_evaluator_loop
from .task_update_evaluator import build_task_update_evaluator_loop
from .reasoning_evaluator import build_reasoning_evaluator_loop

__all__ = [
    "build_task_list_evaluator_loop",
    "build_task_update_evaluator_loop",
    "build_reasoning_evaluator_loop",
]
