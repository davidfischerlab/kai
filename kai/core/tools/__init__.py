"""Consolidated tools for Kai agent."""

from typing import Dict
from kai.core.llm_interface import LLMInterface
from kai.retrieval import ChromaDbManager
from kai.core.orchestration.base_tool import BaseTool

from .rag import SearchCodeSnippetsTool
from .workflow_search import SearchWorkflowsTool, WorkflowRefinementTool
from .code_generation import GenerateCodeTool, UpdateCodeTool
from .execution import ExecuteCellTool, RestartAndRerunTool
from .task_management import ManageProgressTool, PlanTasksTool
from .error_handling import HandleErrorTool, BacktrackTool, ExecutionMonitorTool
from .interaction import (
    ClassifyIntentTool,
    AnswerQuestionTool,
    ReviewCodeTool,
    RespondWithReasoningTool
)
from .notebook import NotebookOperationsTool

# Dev branch tools for deterministic routing
from kai.core.orchestration.deterministic_tools import (
    MarkNextTaskActiveTool,
    CodeRetrievalTool,
    CellDeletionTool,
    ReferenceWorkflowQueryPreparationTool,
    FilterUnusedReferenceWorkflowsTool,
    SetPositioningFromLastCellTool,
)
from kai.core.orchestration.prompt_tools import (
    AutonomousMarkCompletionTool,
    CellPositioningTool,
    CodeGenerationWithGuidanceTool,
    ReasoningResponseWithGuidanceTool,
    CodeUpdateTool,
    ErrorRecoveryTool,
    BacktrackRecoveryTool,
    AutonomousUpdateTasksTool,
    AutonomousUpdateCritiqueTool,
    TaskListGenerationTool,
    TaskListCritiqueTool,
    ReasoningCritiqueTool,
    CellSelectionDeletionTool,
    RestartAndRerunTool as PromptRestartAndRerunTool
)


def create_consolidated_tools(
    llm: LLMInterface,
    knowledge_base: ChromaDbManager
) -> Dict[str, BaseTool]:
    """
    Create all consolidated tools for the agent.

    Args:
        llm: LLM interface for prompt-based tools
        knowledge_base: ChromaDB for RAG retrieval

    Returns:
        Dictionary mapping tool names to tool instances
    """
    return {
        "search_code_snippets": SearchCodeSnippetsTool(knowledge_base),
        "search_workflows": SearchWorkflowsTool(llm, knowledge_base, mode="full"),
        "search_workflows_only": SearchWorkflowsTool(llm, knowledge_base, mode="selection_only"),
        "workflow_refinement": WorkflowRefinementTool(llm, knowledge_base),

        "generate_code": GenerateCodeTool(llm, with_guidance=True),
        "generate_code_simple": GenerateCodeTool(llm, with_guidance=False),
        "update_code": UpdateCodeTool(llm),

        "execute_cell": ExecuteCellTool(),
        "restart_and_rerun": RestartAndRerunTool(llm),

        "plan_tasks": PlanTasksTool(llm, use_critique=True),
        "manage_progress": ManageProgressTool(llm, use_critique=True),

        "handle_error": HandleErrorTool(llm),
        "backtrack": BacktrackTool(llm),
        "execution_monitor": ExecutionMonitorTool(llm),

        "classify_intent": ClassifyIntentTool(llm, mode="regular"),
        "classify_intent_autonomous": ClassifyIntentTool(llm, mode="autonomous"),
        "answer_question": AnswerQuestionTool(llm),
        "review_code": ReviewCodeTool(llm),
        "respond_with_reasoning": RespondWithReasoningTool(llm, use_critique=True),

        "notebook_operations": NotebookOperationsTool(),

        # Dev branch tools for deterministic routing - Core tools
        "mark_next_task_active": MarkNextTaskActiveTool(),
        "autonomous_mark_completion": AutonomousMarkCompletionTool(llm),
        "cell_positioning": CellPositioningTool(llm),
        "code_generation_with_guidance": CodeGenerationWithGuidanceTool(llm),
        "reasoning_response_with_guidance": ReasoningResponseWithGuidanceTool(llm),

        # Error recovery tools
        "code_update": CodeUpdateTool(llm),
        "error_recovery": ErrorRecoveryTool(llm),
        "restart_and_rerun_prompt": PromptRestartAndRerunTool(),  # From prompt_tools

        # Task update tools
        "autonomous_update_tasks": AutonomousUpdateTasksTool(llm),
        "autonomous_update_critique": AutonomousUpdateCritiqueTool(llm),

        # Backtracking tools
        "backtrack_recovery": BacktrackRecoveryTool(llm),
        "cell_selection_deletion": CellSelectionDeletionTool(llm),
        "cell_deletion": CellDeletionTool(),

        # Positioning tools (for standard continue/retry - matches kai_dev behavior)
        "set_positioning_from_last_cell": SetPositioningFromLastCellTool(),

        # Critique tools
        "task_list_generation": TaskListGenerationTool(llm),
        "task_list_critique": TaskListCritiqueTool(llm),
        "reasoning_critique": ReasoningCritiqueTool(llm),

        # RAG tools
        "rag_retrieval": CodeRetrievalTool(knowledge_base),

        # Reference workflow tools (for planning)
        "reference_workflow_query_preparation": ReferenceWorkflowQueryPreparationTool(llm),
        "filter_unused_reference_workflows": FilterUnusedReferenceWorkflowsTool(),
    }


__all__ = [
    'create_consolidated_tools',
    'SearchCodeSnippetsTool',
    'SearchWorkflowsTool',
    'GenerateCodeTool',
    'UpdateCodeTool',
    'ExecuteCellTool',
    'RestartAndRerunTool',
    'ManageProgressTool',
    'PlanTasksTool',
    'HandleErrorTool',
    'BacktrackTool',
    'ExecutionMonitorTool',
    'ClassifyIntentTool',
    'AnswerQuestionTool',
    'ReviewCodeTool',
    'RespondWithReasoningTool',
    'NotebookOperationsTool',
]
