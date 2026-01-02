"""Common Pydantic schemas shared across multiple tools.

These schemas are extracted from orchestration/schemas.py for reuse
across different tool implementations.
"""

from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class TaskItem(BaseModel):
    """Individual task structure matching prompt manager specifications."""
    model_config = ConfigDict(extra='forbid')

    id: int = Field(description="Unique identifier for the task item")
    task: str = Field(description="Description of the task to be performed")
    status: Literal["pending", "active", "completed"] = Field(
        description="Current status of the task - pending (not started), active (currently working on), or completed"
    )


class TaskStatusUpdate(BaseModel):
    """Individual task status update."""
    model_config = ConfigDict(extra='forbid')

    id: int = Field(description="Unique identifier of the task to update")
    status: Literal["pending", "active", "completed"] = Field(
        description="New status for this task - pending, active, or completed"
    )
