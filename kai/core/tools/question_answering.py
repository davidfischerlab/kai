"""Question answering tool for answering questions about code/analysis.

This module contains the QuestionAnsweringTool, an UnstructuredPromptTool
that provides answers to user questions about code and analysis.
"""

from typing import TYPE_CHECKING

from kai.core.prompt_manager import PromptScenario
from kai.utils import setup_logger
from .base import ToolResult, ToolOutputType
from .prompt_base import UnstructuredPromptTool

if TYPE_CHECKING:
    from kai.core.llm_interface import LLMInterface
    from kai.core.orchestration.state import KaiState

logger = setup_logger(__name__)


class QuestionAnsweringTool(UnstructuredPromptTool):
    """Tool for answering questions about code/analysis.

    **UI Returns:**
    - `output_ui`: Raw LLM response string containing the answer
    - `output_type`: RESPONSE - displayed in chat for user to read

    **Workflow Returns:**
    - None - this tool doesn't propagate workflow data

    **Used by workflows:** Regular request workflow for question_about_code intent after retrieval
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("question_answering", PromptScenario.QUESTION_ANSWERING, llm_interface)

    async def _process_response(self, response: str, state: 'KaiState') -> ToolResult:
        """Process question answering response."""
        return ToolResult(
            output_ui=response,
            output_type=ToolOutputType.RESPONSE
        )
