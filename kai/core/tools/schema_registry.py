"""Central schema registry importing schemas from individual tool files.

This module provides a central SCHEMA_REGISTRY dictionary that maps tool names
to their Pydantic schemas. Schemas are imported from their co-located tool files.

Usage:
    from kai.core.tools.schema_registry import SCHEMA_REGISTRY

    schema = SCHEMA_REGISTRY.get("task_list_generation")
"""

# Import schemas from individual tool files
from kai.core.tools.task_list_generation import TaskListGeneration
from kai.core.tools.task_list_critique import TaskListCritique
from kai.core.tools.autonomous_mark_completion import AutonomousMarkCompletion
from kai.core.tools.autonomous_update_tasks import AutonomousTaskUpdate
from kai.core.tools.autonomous_update_critique import AutonomousUpdateCritique

from kai.core.tools.intent_classification import IntentClassification
from kai.core.tools.autoloop_intent_classification import AutoLoopIntentClassification
from kai.core.tools.section_code_review import SectionCodeReview

from kai.core.tools.cell_positioning import CellPositioning
from kai.core.tools.cell_selection_deletion import CellDeletionSelection
from kai.core.tools.error_recovery import ErrorRecoveryStrategy
from kai.core.tools.backtrack_recovery import BacktrackRecoveryStrategy
from kai.core.tools.execution_monitor import ExecutionMonitor

from kai.core.tools.reasoning_critique import ReasoningCritique

from kai.core.tools.reference_workflow_selection import (
    ReferenceWorkflowSelection,
    ReferenceWorkflowSelectionOnly,
)
from kai.core.tools.reference_workflow_cell_selection import (
    ReferenceWorkflowCellSelection,
)


# Schema registry for easy lookup by tool name
SCHEMA_REGISTRY = {
    # Task tools
    "task_list_generation": TaskListGeneration,
    "task_list_critique": TaskListCritique,
    "autonomous_mark_completion": AutonomousMarkCompletion,
    "autonomous_update_tasks": AutonomousTaskUpdate,
    "autonomous_update_critique": AutonomousUpdateCritique,

    # Intent/classification tools
    "intent_classification": IntentClassification,
    "autoloop_intent_classification": AutoLoopIntentClassification,
    "section_code_review": SectionCodeReview,

    # Positioning/cell tools
    "cell_positioning": CellPositioning,
    "cell_selection_deletion": CellDeletionSelection,

    # Error/recovery tools
    "error_recovery": ErrorRecoveryStrategy,
    "backtrack_recovery": BacktrackRecoveryStrategy,
    "execution_monitor": ExecutionMonitor,

    # Reasoning tools
    "reasoning_critique": ReasoningCritique,

    # Workflow tools
    "reference_workflow_selection": ReferenceWorkflowSelection,
    "reference_workflow_selection_only": ReferenceWorkflowSelectionOnly,
    "reference_workflow_cell_selection": ReferenceWorkflowCellSelection,
}
