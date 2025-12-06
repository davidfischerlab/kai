"""Base tool interface for unified tool architecture."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .execution_context import ExecutionContext


class ToolOutputType(Enum):
    """Types of tool outputs for different handling by VSCode."""
    RESPONSE = "response"                     # Normal chat response
    DISPLAY_ONLY = "display"                  # Show in chat but don't expect user response
    TASK_LIST_DISPLAY = "task_list_display"   # Show in chat but don't expect user response
    EXECUTE_ONLY = "execute_code"             # Execute code but don't show in chat (autonomous mode)
    NO_OUTPUT = "no_output"                   # Is not parsed by VSCode but is added to the job queue
    REFERENCE_WORKFLOWS = "reference_workflows"  # Reference workflow IDs to be stored in VSCode


@dataclass
class ToolResult:
    """Result from tool execution."""
    output_ui: Any
    output_workflow: Optional[Dict[str, Any]] = None  # Context for subsequent workflow tools
    output_type: ToolOutputType = ToolOutputType.RESPONSE
    

class BaseTool(ABC):
    """Base class for all tools in the system."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def execute(self, exec_context: "ExecutionContext") -> ToolResult:
        """Execute the tool with ExecutionContext."""
        pass

    def can_execute(self, exec_context: "ExecutionContext") -> bool:
        """Check if tool can execute with given ExecutionContext."""
        return True

    def get_required_inputs(self) -> List[str]:
        """Get list of required input keys."""
        return []

    def get_output_schema(self) -> Dict[str, Any]:
        """Get schema describing tool output."""
        return {}

    def as_graph_node(self):
        """
        Convert tool to LangGraph node function.

        Returns a function that:
        1. Converts dict state to ExecutionContext
        2. Calls the tool's execute method
        3. Merges ToolResult.output_workflow back into state
        """
        async def node_function(state: dict) -> dict:
            from .execution_context import ExecutionContext
            from kai.utils import setup_logger

            logger = setup_logger(__name__)

            # Debug: log state keys
            logger.debug(f"Tool {self.name} received state with keys: {list(state.keys())}")
            if 'notebook_structure' not in state:
                logger.warning(f"Tool {self.name}: notebook_structure missing from state!")

            # Convert state dict to ExecutionContext
            exec_context = ExecutionContext.from_dict(state)

            # Execute tool
            result = await self.execute(exec_context)

            # Merge workflow output back into state
            if result.output_workflow:
                return result.output_workflow
            return {}

        return node_function