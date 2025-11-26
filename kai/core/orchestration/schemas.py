"""Pydantic schemas for structured LLM outputs via Ollama.

These schemas define the exact JSON format specifications from prompt_manager.py
to ensure consistent structured output from LLM interactions.
"""

from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field, model_validator, ConfigDict

# Schema descriptions from prompt manager - maintain in sync
TASK_LIST_DESCRIPTION = """
The task list is a an analysis plan. 

Adhere to the following for desiging this plan:
- It should break a an objective down into individual steps with specific instructions.
- When sequentially performed in a jupyter notebook, these steps should address the objective.
- Plan analyses that complements the notebook, do not repeat imports, analyses or code that is already there.
- Further analyses may be added to the notebook later, so the planned analyses should strictly only address the objective and not explore other analysis options.
- Give the plan a decriptive and specific title.

When designing the steps, adhere to these guidelines:
- Each task should be specific and correspond to one cell in a jupyter notebook, 
which typically contains a number of lines that form a step in an analysis, for example, that generate a specific plot.
- Don't make individual steps too small - they should correspond to a meanningful analysis step that makes a reasonable jupyter notebook cell size, not necessarily a single line of code.
- If later steps in the plan depend on the outcomes of earlier steps, you can keep these coarse as you will later be able to refine the list.
"""

CELL_SELECTION_INTENTS_DESCRIPTION = """
## Cell Selection Intents:
You distinguish the following intents based on conversation history, execution history, and full notebook structure:

**MOVE_TO_NEXT_FROM_LAST_EXECUTED**: User wants to add a cell after the last executed cell.
- The user wants to add cells after the last executed cell.
- Consider this a default choice.

**MOVE_TO_NEXT_FROM_CURSOR**: User wants to add a cell after the currently selected cell.
- The currently selected cell deviates from the last executed cell and the user indicates that they want to work on a different analysis.
- You will be able to identify this based on the conversation history and the content of the currently selected and last executed cell.

**MOVE_TO_SPECIFIC**: User wants to add a cell at position that is specified in the prompt.
- The position may be identied based on a cell number, relative position, or content description.
- This will mostly involve a jump to a different analysis in the notebook, rather than a continuation of a current analysis by a single step.
- Note that the position of the currently selected cell may help in identifying this new position as the user might be looking at this section they want to move to.
"""

ERROR_RECOVERY_STRATEGIES_DESCRIPTION = """
**REPLACE_AND_RETRY**: Fix the code in the last executed cell that caused the error and execute it again.
- Choose this option if the error did not cause any non-recovarable change to key objects in the session:
For example, a subsetting of or a modification of gene expression features or medata of an anndata instance may not be recoverable.
A failed modification of metadata may be recoverable.
If uncertain and the notebook is relatively small / can be executed quickly, err on the side of caution and prefer REPLACE_AND_RESTART. 
- Consider this a default choice.

**REPLACE_AND_RESTART**: Fix the code in the last executed cell that caused the error, restart the kernel and run all cells up to including the one that showed the error again.
- Choose this option if you find that REPLACE_AND_RETRY is not a suitable choice.
"""


class AutoLoopIntentClassification(BaseModel):
    """Schema for classifying user feedback during autonomous mode.

    Determines whether the user wants to modify the task list or change code implementation.
    """
    model_config = ConfigDict(extra='forbid')

    intent: Literal["TASK_LIST_MODIFICATION", "CODE_IMPLEMENTATION_FEEDBACK", "APPROVAL"] = Field(
        description="Type of feedback - task list changes or implementation changes"
    )
    modification_description: str = Field(
        description="What the user wants to change"
    )
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "intent": "TASK_LIST_MODIFICATION", "CODE_IMPLEMENTATION_FEEDBACK", or "APPROVAL"
    "modification_description": "User wants to change the differential expression method from t-test to wilcoxon"
}

Ensure all JSON is valid and complete."""


class TaskItem(BaseModel):
    """Individual task structure matching prompt manager specifications."""
    model_config = ConfigDict(extra='forbid')

    id: int = Field(description="Unique identifier for the task item")
    task: str = Field(description="Description of the task to be performed")
    status: Literal["pending", "active", "completed"] = Field(
        description="Current status of the task - pending (not started), active (currently working on), or completed"
    )


class AutonomousTaskUpdate(BaseModel):
    """Schema for autonomous task updates (no decision-making)."""
    model_config = ConfigDict(extra='forbid')

    tasks: List[TaskItem] = Field(description="Updated list of non-completed tasks - this will be appended to the completed tasks.")
    retrieval_queries: List[str] = Field(description="Query to retrieve snippets of API documentation and workflow examples to guide code generation for current task.")
    update_rationale: str = Field(description="Reasoning for performing the update.")
    update_rule: Literal["KEEP", "UPDATE"] = Field(description="Whether to keep the current task list or to apply updates.")
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "tasks": [
        {"id": 2, "task": "Process data", "status": "pending"},
    ],
    "retrieval_queries": ["query 1", "query 2"],
    "update_rationale": "Reasoning for performing the update.",
    "update_rule": "UPDATE",
}

Ensure all JSON is valid and complete."""


class BacktrackRecoveryStrategy(BaseModel):
    """Schema for backtracking recovery decisions."""
    model_config = ConfigDict(extra='forbid')

    restart_required: bool = Field(
        description="Whether notebook restart is required before continuing with backtracking recovery"
    )
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "restart_required": true
}

Valid restart_required values: true, false
Ensure all JSON is valid and complete."""


class CellPositioning(BaseModel):
    """Schema for cell positioning decisions."""
    model_config = ConfigDict(extra='forbid')

    target_cell: int = Field(
        description="Target cell number for insertion/replacement"
    )
    reasoning: str = Field(
        description="Reasoning for the positioning choice"
    )
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "target_cell": 2,
    "reasoning": "Reasoning for the positioning choice"
}

Ensure all JSON is valid and complete."""


class IntentClassification(BaseModel):
    """Schema for user intent classification.

    Matches the intent classification system from prompt_tools.py:
    - question_about_code: User is asking about existing code, methods, or concepts
    - generate_code: User wants to generate new code (will create new cells)
    - generate_code_in_place: User wants to modify/fix existing code (will replace current cell)
    - remove_code: User wants to remove code
    """
    model_config = ConfigDict(extra='forbid')

    intent: Literal[
        "question_about_code",
        "generate_code", 
        "generate_code_in_place",
        "remove_code"
    ] = Field(description="Classified user intent")
    reasoning: Optional[str] = Field(
        default=None,
        description="Reasoning for the classification choice"
    )
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "intent": "generate_code",
    "reasoning": "Reasoning for the classification choice"
}

Valid intent values: "question_about_code", "generate_code", "generate_code_in_place", "remove_code"
Ensure all JSON is valid and complete."""


class CellDeletionSelection(BaseModel):
    """Schema for cell deletion selection responses."""
    model_config = ConfigDict(extra='forbid')
    cells_to_delete: List[int] = Field(
        description="List of cell numbers to delete (0-indexed)"
    )
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "cells_to_delete": [3, 5, 7],
}

Ensure all JSON is valid and complete."""


class ErrorRecoveryStrategy(BaseModel):
    """Schema for error recovery decisions."""
    model_config = ConfigDict(extra='forbid')

    intent: Literal["REPLACE_AND_RETRY", "REPLACE_AND_RESTART"] = Field(
        description="Error recovery strategy to apply"
    )
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "intent": "REPLACE_AND_RETRY"
}

Valid intent values: "REPLACE_AND_RETRY", "REPLACE_AND_RESTART"
Ensure all JSON is valid and complete."""


class ReferenceWorkflowSelection(BaseModel):
    """Schema for reference workflow retrieval output."""
    model_config = ConfigDict(extra='forbid')
    selected_notebooks: List[str] = Field(description="List of selected notebook IDs")
    retrieval_queries: List[str] = Field(description="List of queries for further retrieval", default_factory=list)
        
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "selected_notebooks": ["scverse/scanpy-tutorials/pbmc3k_tutorial.ipynb", "scverse/decoupler-tutorials/rna_sc.ipynb"],
    "retrieval_queries": ["classify cell types"],
}

Ensure all JSON is valid and complete."""


class ReferenceWorkflowCellSelection(BaseModel):
    """Schema for selecting relevant cells from a single reference workflow."""
    model_config = ConfigDict(extra='forbid')
    selected_cells: List[int] = Field(description="List of cell indices to include from the notebook")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "selected_cells": [0, 2, 5, 10, 15]
}

Ensure all JSON is valid and complete."""


class ReferenceWorkflowSelectionOnly(BaseModel):
    """Schema for reference workflow retrieval output."""
    model_config = ConfigDict(extra='forbid')
    selected_notebooks: List[str] = Field(description="List of selected notebook IDs")

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "selected_notebooks": ["scverse/scanpy-tutorials/pbmc3k_tutorial.ipynb", "scverse/decoupler-tutorials/rna_sc.ipynb"],
}

Ensure all JSON is valid and complete."""


class SectionCodeReview(BaseModel):
    """Schema for section code review and recovery decisions."""
    model_config = ConfigDict(extra='forbid')

    operation: Literal["delete", "replace", "insert"] = Field(
        description="Type of operation to perform: delete, replace, or insert cells"
    )
    position: Union[int, List[int]] = Field(
        description="For insert: integer index where new code should be placed. For delete/replace: list of cell indices to modify (0-based relative to section)"
    )
    intent: str = Field(
        description="User query/intent describing what the code should accomplish - used for code generation"
    )
    reasoning: str = Field(
        description="Reasoning for why this fix addresses the error"
    )
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "operation": "replace",
    "position": [3, 4],
    "intent": "User query describing what the code should accomplish",
    "reasoning": "Reasoning for the fix"
}

Valid operation values: "delete", "replace", "insert"
For insert: position should be an integer
For delete/replace: position should be a list of integers
Ensure all JSON is valid and complete."""


class TaskListGeneration(BaseModel):
    """Schema for task list generation responses."""
    model_config = ConfigDict(extra='forbid')

    tasks: List[TaskItem] = Field(description="List of analysis steps as tasks")
    retrieval_queries: List[str] = Field(description="List of queries for further retrieval  of reference workflows.", default_factory=list)

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "tasks": [
        {"id": 1, "task": "Description of analysis step", "status": "pending"},
        {"id": 2, "task": "Another analysis step", "status": "pending"}
    ],
    "retrieval_queries": ["List of string queries for further retrieval of reference workflows."]
}

Ensure all JSON is valid and complete."""


class TaskStatusUpdate(BaseModel):
    """Individual task status update."""
    model_config = ConfigDict(extra='forbid')

    id: int = Field(description="Unique identifier of the task to update")
    status: Literal["pending", "active", "completed"] = Field(
        description="New status for this task - pending, active, or completed"
    )


class AutonomousMarkCompletion(BaseModel):
    """Schema for marking task completion status with backtracking support.

    Updates status of ALL tasks. Can backtrace by setting earlier tasks back to pending.
    Must maintain logical order: no completed tasks after pending tasks.
    Backtracking is detected when recovery_objective is provided.
    """
    model_config = ConfigDict(extra='forbid')

    status_updates: List[TaskStatusUpdate] = Field(
        description="Status updates for ALL tasks by ID. Must maintain logical order - no completed after pending."
    )
    retry_objective: Optional[str] = Field(
        default=None,
        description="Optional: if task was not addressed sufficiently, explanation of what needs to change in the next attempt."
    )
    recovery_objective: Optional[str] = Field(
        default=None,
        description="Optional: if backtracking, explanation of what needs to change in earlier tasks based on later analysis."
    )
    
    @model_validator(mode='after')
    def validate_logical_order(self):
        """Ensure logical task ordering: completed -> active -> pending."""
        if not self.status_updates:
            return self
            
        # Sort by ID to check order
        sorted_updates = sorted(self.status_updates, key=lambda x: x.id)
        
        # Define status priority for ordering validation (0 = earliest valid, 2 = latest valid)
        status_priority = {"completed": 0, "active": 1, "pending": 2}
        
        # Check that status priorities are non-decreasing (completed -> active -> pending)
        for i in range(len(sorted_updates) - 1):
            current_priority = status_priority[sorted_updates[i].status]
            next_priority = status_priority[sorted_updates[i + 1].status]
            
            if current_priority > next_priority:
                raise ValueError(
                    f"Invalid task order: Task {sorted_updates[i+1].id} is {sorted_updates[i+1].status} "
                    f"but comes after {sorted_updates[i].status} task {sorted_updates[i].id}. "
                    f"Tasks must be ordered: completed -> active -> pending."
                )
        
        # Ensure only one task is active at a time
        active_count = sum(1 for update in sorted_updates if update.status == "active")
        if active_count > 1:
            active_tasks = [str(update.id) for update in sorted_updates if update.status == "active"]
            raise ValueError(
                f"Only one task can be active at a time. Found active tasks: {', '.join(active_tasks)}"
            )
        
        return self
    
    @property
    def backtrack_detected(self) -> bool:
        """Detect if backtrack was initiated - derived from recovery_objective being provided."""
        return self.recovery_objective is not None
    
    @property
    def retry_detected(self) -> bool:
        """Detect if retry was initiated - derived from recovery_objective being provided."""
        return self.retry_objective is not None
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "status_updates": [
        {"id": 1, "status": "completed"},
        {"id": 2, "status": "completed"},
        {"id": 3, "status": "pending"},
        {"id": 4, "status": "pending"}
    ],
    "retry_objective": "Optional: if task was not addressed sufficiently, explanation of what needs to change in the next attempt.",
    "recovery_objective": "Optional: if backtracking, explanation of what needs to change in earlier tasks based on later analysis."
}

Ensure all JSON is valid and complete."""


## Critiques


class AutonomousUpdateCritique(BaseModel):
    """Schema for task list update critique responses."""
    model_config = ConfigDict(extra='forbid')

    approval: Literal["APPROVED", "MODIFY"] = Field(
        description="Whether the task list update is sufficient or needs to be modified."
    )
    critique: str = Field(description="If applicable, any suggestions for improving the task list update.", default="")
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "approval": "APPROVED" or "MODIFY",
    "critique": "If applicable, any suggestions for improving the task list update."
}

Ensure all JSON is valid and complete."""


class TaskListCritique(BaseModel):
    """Schema for task list critique responses."""
    model_config = ConfigDict(extra='forbid')

    approval: Literal["APPROVED", "MODIFY"] = Field(
        description="Whether the task list is sufficient or needs to be modified."
    )
    critique: str = Field(description="If applicable, any suggestions for improving the task list.", default="")
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "approval": "APPROVED" or "MODIFY",
    "critique": "If applicable, any suggestions for improving the task list."
}

Ensure all JSON is valid and complete."""


class ExecutionMonitor(BaseModel):
    """Schema for execution progress monitoring decisions."""
    model_config = ConfigDict(extra='forbid')

    action: Literal["continue", "terminate"] = Field(
        description="Whether to continue execution or terminate the stuck cell."
    )
    feedback: str = Field(
        description="If a cell was terminated - instructions for updating the cell."
    )

    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "action": "continue" or "terminate",
    "feedback": "If a cell was terminated - instructions for updating the cell"
}

Ensure all JSON is valid and complete."""


class ReasoningCritique(BaseModel):
    """Schema for reasoning critique responses."""
    model_config = ConfigDict(extra='forbid')

    approval: Literal["APPROVED", "MODIFY"] = Field(
        description="Whether the reasoning is valid or needs to be modified."
    )
    critique: str = Field(description="If applicable, any suggestions for improving the reasoning.", default="")
    
    @classmethod
    def get_json_format_instruction(cls) -> str:
        """Get JSON format instruction for prompts when structured output is disabled."""
        return """
IMPORTANT: Respond ONLY with valid JSON matching this exact format:
{
    "approval": "APPROVED" or "MODIFY",
    "critique": "If applicable, any suggestions for improving the reasoning."
}

Ensure all JSON is valid and complete."""


# Schema registry for easy lookup
SCHEMA_REGISTRY = {
    "autoloop_intent_classification": AutoLoopIntentClassification,
    "autonomous_mark_completion": AutonomousMarkCompletion,
    "autonomous_update_tasks": AutonomousTaskUpdate,
    "autonomous_update_critique": AutonomousUpdateCritique,
    "backtrack_recovery": BacktrackRecoveryStrategy,
    "cell_positioning": CellPositioning,
    "cell_selection_deletion": CellDeletionSelection,
    "error_recovery": ErrorRecoveryStrategy,
    "execution_monitor": ExecutionMonitor,
    "intent_classification": IntentClassification,
    "notebook_rag": ReferenceWorkflowSelection,
    "section_code_review": SectionCodeReview,
    "task_list_generation": TaskListGeneration,
    "task_list_critique": TaskListCritique,
    "reasoning_critique": ReasoningCritique,
    "reference_workflow_selection": ReferenceWorkflowSelection,
    "reference_workflow_cell_selection": ReferenceWorkflowCellSelection,
    "reference_workflow_selection_only": ReferenceWorkflowSelectionOnly,
}
