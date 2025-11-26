from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

from kai.core.orchestration.base_tool import ToolResult


@dataclass
class BacktrackingContext:
    """Context for backtracking operations."""
    recovery_objective: str
    backtrack_to_task: Dict[str, Any]  # The task to backtrack to
    deleted_cells: List[int] = field(default_factory=list)  # Cells that were deleted
    index_translation: Dict[int, int] = field(default_factory=dict)  # Index mapping after deletions
    
    @property
    def is_active(self) -> bool:
        """Check if backtracking is active."""
        return bool(self.recovery_objective)


@dataclass
class ExecutionInputs:
    """Strict structured inputs for ExecutionContext."""
    backtracking_context: Optional[BacktrackingContext]  # All backtracking info in one place
    context: Dict[str, Any]
    task_list: Dict[str, Any]
    user_query: str
    excluded_workflows: List[str] = field(default_factory=list)  # Workflows that returned empty indices in past iterations


@dataclass
class ExecutionContext:
    """Context for workflow execution."""
    inputs: ExecutionInputs
    session_metadata: Dict[str, Any] = field(default_factory=dict)

