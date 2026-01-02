"""Consolidated tools for Kai agent.

This module provides:
- Individual atomic tools (LLM and deterministic)
- Workflow tools for RAG retrieval
- create_consolidated_tools() factory function
- Schema registry for structured output

NOTE: Orchestration is handled by LangGraph routing in langgraph_orchestrator.py,
not by wrapper/composite tools.
"""

from typing import Dict
from kai.core.llm_interface import LLMInterface
from kai.retrieval import ChromaDbManager
from kai.core.tools.base import BaseTool, ToolResult, ToolOutputType

# Workflow tools (multi-step pipelines for RAG)
from .search_workflows import SearchWorkflowsTool, WorkflowRefinementTool
from .execute_cell import ExecuteCellTool
from .notebook_operations import NotebookOperationsTool

# Task tools
from .task_list_generation import TaskListGenerationTool, TaskListGeneration
from .task_list_critique import TaskListCritiqueTool, TaskListCritique
from .autonomous_mark_completion import AutonomousMarkCompletionTool, AutonomousMarkCompletion
from .autonomous_update_tasks import AutonomousUpdateTasksTool, AutonomousTaskUpdate
from .autonomous_update_critique import AutonomousUpdateCritiqueTool, AutonomousUpdateCritique
from .mark_next_task_active import MarkNextTaskActiveTool

# Code tools
from .code_update import CodeUpdateTool
from .cell_positioning import CellPositioningTool, CellPositioning
from .code_generation import CodeGenerationTool, CodeGenerationWithGuidanceTool

# Error/recovery tools
from .error_recovery import ErrorRecoveryTool, ErrorRecoveryStrategy
from .backtrack_recovery import BacktrackRecoveryTool, BacktrackRecoveryStrategy
from .cell_selection_deletion import CellSelectionDeletionTool, CellDeletionSelection
from .execution_monitor import ExecutionMonitorTool, ExecutionMonitor

# Intent/interaction tools
from .intent_classification import IntentClassificationTool, IntentClassification
from .autoloop_intent_classification import AutoLoopIntentClassificationTool, AutoLoopIntentClassification
from .question_answering import QuestionAnsweringTool
from .section_code_review import SectionCodeReviewTool, SectionCodeReview

# Reasoning tools
from .reasoning_response_with_guidance import ReasoningResponseWithGuidanceTool
from .reasoning_critique import ReasoningCritiqueTool, ReasoningCritique

# Reference workflow tools
from .reference_workflow_selection import (
    ReferenceWorkflowSelectionTool,
    ReferenceWorkflowSelectionOnlyTool,
    ReferenceWorkflowSelection,
    ReferenceWorkflowSelectionOnly,
)
from .reference_workflow_cell_selection import (
    ReferenceWorkflowCellSelectionTool,
    ReferenceWorkflowCellSelection,
)

# Deterministic tools
from .rag_retrieval import CodeRetrievalTool
from .set_positioning_from_last_cell import SetPositioningFromLastCellTool, IncrementPositioningTool
from .reference_workflow_query_preparation import (
    ReferenceWorkflowQueryPreparationTool,
    FilterUnusedReferenceWorkflowsTool,
)
from .cell_deletion import CellDeletionTool

# Common schemas
from .common_schemas import TaskItem, TaskStatusUpdate

# Schema registry
from .schema_registry import SCHEMA_REGISTRY


def create_consolidated_tools(
    llm: LLMInterface,
    knowledge_base: ChromaDbManager
) -> Dict[str, BaseTool]:
    """
    Create all tools for the agent.

    Args:
        llm: LLM interface for prompt-based tools
        knowledge_base: ChromaDB for RAG retrieval

    Returns:
        Dictionary mapping tool names to tool instances
    """
    return {
        # Workflow tools
        "search_workflows": SearchWorkflowsTool(llm, knowledge_base, mode="full"),
        "search_workflows_only": SearchWorkflowsTool(
            llm, knowledge_base, mode="selection_only"
        ),
        "workflow_refinement": WorkflowRefinementTool(llm, knowledge_base),

        # Execution tools
        "execute_cell": ExecuteCellTool(),

        # Intent classification
        "classify_intent": IntentClassificationTool(llm),
        "classify_intent_autonomous": AutoLoopIntentClassificationTool(llm),
        "answer_question": QuestionAnsweringTool(llm),
        "review_code": SectionCodeReviewTool(llm),

        "notebook_operations": NotebookOperationsTool(),

        # Task tools
        "mark_next_task_active": MarkNextTaskActiveTool(),
        "autonomous_mark_completion": AutonomousMarkCompletionTool(llm),
        "autonomous_update_tasks": AutonomousUpdateTasksTool(llm),
        "autonomous_update_critique": AutonomousUpdateCritiqueTool(llm),
        "task_list_generation": TaskListGenerationTool(llm),
        "task_list_critique": TaskListCritiqueTool(llm),

        # Code generation tools
        "cell_positioning": CellPositioningTool(llm),
        "code_generation": CodeGenerationTool(llm),  # Simple code gen for regular mode
        "code_generation_with_guidance": CodeGenerationWithGuidanceTool(llm),
        "code_update": CodeUpdateTool(llm),

        # Reasoning tools
        "reasoning_response_with_guidance": ReasoningResponseWithGuidanceTool(llm),
        "reasoning_critique": ReasoningCritiqueTool(llm),

        # Error recovery tools
        "error_recovery": ErrorRecoveryTool(llm),
        "execution_monitor": ExecutionMonitorTool(llm),

        # Backtracking tools
        "backtrack_recovery": BacktrackRecoveryTool(llm),
        "cell_selection_deletion": CellSelectionDeletionTool(llm),
        "cell_deletion": CellDeletionTool(),

        # Positioning tools
        "set_positioning_from_last_cell": SetPositioningFromLastCellTool(),

        # RAG tools
        "rag_retrieval": CodeRetrievalTool(knowledge_base),

        # Reference workflow tools
        "reference_workflow_query_preparation": ReferenceWorkflowQueryPreparationTool(
            llm
        ),
        "filter_unused_reference_workflows": FilterUnusedReferenceWorkflowsTool(),
    }
