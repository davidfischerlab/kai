"""LLM Pool for managing multiple LLM instances with tool-specific routing."""

import copy
from typing import Dict, Any, Optional, TYPE_CHECKING
from .llm_interface import LLMInterface

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from kai.core.orchestration.prompt_tools import BasePromptTool


class LLMPool:
    """Manages multiple LLM instances and routes tools to appropriate models."""
    
    # LLM Pool Configuration - centralized model choices
    LLM_CONFIG = {
        "models": {
            "local": {
                "large": "gpt-oss:20b",
                "small": "qwen2.5-coder:7b"
            },
            "turbo": {
                "large": "gpt-oss:120b", 
                "small": "gpt-oss:20b"
            }
        },
        "tool_llm_mapping": {
            # Large LLM - Complex reasoning/generation tasks
            "AutoLoopIntentClassificationTool": "large_llm",
            "AutonomousMarkCompletionTool": "large_llm",
            "AutonomousUpdateTasksTool": "large_llm",
            "AutonomousUpdateCritiqueTool": "large_llm",
            "BacktrackRecoveryTool": "large_llm",
            "CellSelectionDeletionTool": "large_llm",
            "CellDeletionTool": "large_llm",
            "CellPositioningTool": "large_llm",
            "CodeGenerationTool": "large_llm",
            "CodeGenerationWithGuidanceTool": "large_llm",
            "CodeUpdateTool": "large_llm",
            "ErrorRecoveryTool": "large_llm",
            "ExecutionMonitorTool": "large_llm",
            "IntentClassificationTool": "large_llm",
            "QuestionAnsweringTool": "large_llm",
            "ReasoningCritiqueTool": "large_llm",
            "ReasoningResponseWithGuidanceTool": "large_llm",
            "ReferenceWorkflowSelectionTool": "large_llm",
            "ReferenceWorkflowSelectionOnlyTool": "large_llm",
            "ReferenceWorkflowCellSelectionTool": "large_llm",
            "RestartAndRerunTool": "large_llm",
            "SectionCodeReviewTool": "large_llm",
            "TaskListGenerationTool": "large_llm",
            "TaskListCritiqueTool": "large_llm",
        },
        "tool_reasoning_mapping": {
            # Reasoning levels for each tool
            "AutoLoopIntentClassificationTool": "medium",
            "AutonomousMarkCompletionTool": "medium",
            "AutonomousUpdateTasksTool": "high",
            "AutonomousUpdateCritiqueTool": "high",
            "BacktrackRecoveryTool": "high",
            "CellSelectionDeletionTool": "medium",
            "CellDeletionTool": "medium",
            "CellPositioningTool": "medium",
            "CodeGenerationTool": "high",
            "CodeGenerationWithGuidanceTool": "high",
            "CodeUpdateTool": "high",
            "ErrorRecoveryTool": "high",
            "ExecutionMonitorTool": "high",
            "IntentClassificationTool": "medium",
            "QuestionAnsweringTool": "high",
            "ReasoningCritiqueTool": "medium",
            "ReasoningResponseWithGuidanceTool": "high",
            "ReferenceWorkflowSelectionTool": "high",
            "ReferenceWorkflowSelectionOnlyTool": "high",
            "ReferenceWorkflowCellSelectionTool": "medium",
            "RestartAndRerunTool": "medium",
            "SectionCodeReviewTool": "high",
            "TaskListGenerationTool": "high",
            "TaskListCritiqueTool": "medium",
        },
    }
    
    def __init__(
            self,
            turbo: bool = False,
            large_llm: Optional[str] = None,
            small_llm: Optional[str] = None
    ):
        """
        Initialize the LLM pool with large and small models.
        
        Parameters:
        - turbo: Whether to use Ollama Turbo models (default: False).
        - large_llm: Override default for large LLM model name
        - small_llm: Override default for small LLM model name
        """
        # Store turbo mode and custom model overrides
        self.turbo = turbo
        self._large_llm = large_llm
        self._small_llm = small_llm
        
        # Store tool routing from config
        self.tool_llm_mapping = self.LLM_CONFIG["tool_llm_mapping"].copy()
        self.tool_reasoning_mapping = self.LLM_CONFIG["tool_reasoning_mapping"].copy()
    
    @property
    def large_llm(self) -> str:
        """Get the large LLM instance based on current turbo mode."""
        model_set = "turbo" if self.turbo else "local"
        model = self._large_llm or self.LLM_CONFIG["models"][model_set]["large"]
        return model
    
    @property
    def small_llm(self) -> str:
        """Get the small LLM instance based on current turbo mode."""
        model_set = "turbo" if self.turbo else "local"
        model = self._small_llm or self.LLM_CONFIG["models"][model_set]["small"]
        return model
    
    def get_llm_for_tool(self, tool: 'BasePromptTool') -> str:
        """Get the appropriate LLM instance for a given tool: returns model name"""
        llm_type = self.tool_llm_mapping.get(tool, "large_llm")
        
        if llm_type == "small_llm":
            return self.small_llm
        else:
            return self.large_llm
        
    def get_llmsize_for_tool(self, tool: 'BasePromptTool') -> str:
        """Get the appropriate LLM size for a given tool: return small_llm or large_llm."""
        tool_name = type(tool).__name__
        llm_type = self.tool_llm_mapping.get(tool_name, "large_llm")
        return llm_type
    
    def get_reasoning_for_tool(self, tool: 'BasePromptTool') -> str:
        """Get the appropriate LLM reasoning level for a given tool: return string describing reasoning."""
        tool_name = type(tool).__name__
        reasoning_level = self.tool_reasoning_mapping.get(tool_name, "medium")
        return reasoning_level
    
    def get_config(self) -> Dict[str, Any]:
        """Get the full LLM pool configuration."""
        return copy.deepcopy(self.LLM_CONFIG)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the models in the pool."""
        return {
            "large_model": f"{self.large_llm.provider}/{self.large_llm.model}",
            "small_model": f"{self.small_llm.provider}/{self.small_llm.model}",
            "turbo_enabled": self.turbo,
            "config": self.get_config(),
            "tool_mapping": self.tool_llm_mapping
        }
