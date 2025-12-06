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

    @classmethod
    def from_dict(cls, state: Dict[str, Any]) -> "ExecutionContext":
        """
        Create ExecutionContext from LangGraph state dict.

        Maps state fields to ExecutionContext structure:
        - Entire state → inputs.context (tools expect this for backward compat)
        - Task management → inputs.task_list
        - Backtracking → inputs.backtracking_context
        - Session → session_metadata

        Note: inputs.context = state is intentional - tools access fields via context dict.
        When tools return output_workflow, those updates flow back to state.
        """
        # Extract backtracking context if present
        backtracking_context = None
        if state.get("backtracking_context"):
            bt_dict = state["backtracking_context"]
            backtracking_context = BacktrackingContext(
                recovery_objective=bt_dict.get("recovery_objective", ""),
                backtrack_to_task=bt_dict.get("backtrack_to_task", {}),
                deleted_cells=bt_dict.get("deleted_cells", []),
                index_translation=bt_dict.get("index_translation", {})
            )

        # Build inputs - context is the FULL state dict
        # This allows tools to access any state field via exec_context.inputs.context["field"]
        inputs = ExecutionInputs(
            backtracking_context=backtracking_context,
            context=state,  # Full state as context - tools expect this
            task_list=state.get("task_list", {}),
            user_query=state.get("user_query", ""),
            excluded_workflows=state.get("excluded_workflows", [])
        )

        # Build session metadata
        session_metadata = {
            "session_id": state.get("session_id", ""),
            "request_id": state.get("request_id", ""),
            "autonomous_mode": state.get("autonomous_mode", False),
        }

        return cls(inputs=inputs, session_metadata=session_metadata)

