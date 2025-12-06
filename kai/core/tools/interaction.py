"""User interaction tools for classification, Q&A, and reasoning."""

from typing import TYPE_CHECKING
from kai.core.orchestration.base_tool import BaseTool, ToolResult
from kai.core.orchestration.prompt_tools import (
    IntentClassificationTool,
    AutoLoopIntentClassificationTool,
    QuestionAnsweringTool,
    SectionCodeReviewTool,
    ReasoningResponseWithGuidanceTool,
    ReasoningCritiqueTool
)

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext
    from kai.core.llm_interface import LLMInterface


class ClassifyIntentTool(BaseTool):
    """
    Classify user intent for request routing.

    Supports both regular mode and autonomous loop intents.
    """

    def __init__(self, llm_interface: 'LLMInterface', mode: str = "regular"):
        super().__init__("classify_intent")
        if mode == "autonomous":
            self.intent_tool = AutoLoopIntentClassificationTool(llm_interface)
        else:
            self.intent_tool = IntentClassificationTool(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return await self.intent_tool.execute(exec_context, **kwargs)


class AnswerQuestionTool(BaseTool):
    """
    Answer user questions about code and analysis.

    Expects RAG context to be provided by agent.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("answer_question")
        self.qa_tool = QuestionAnsweringTool(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return await self.qa_tool.execute(exec_context, **kwargs)


class ReviewCodeTool(BaseTool):
    """
    Review and provide feedback on code sections.

    Used for code quality assessment and debugging suggestions.
    """

    def __init__(self, llm_interface: 'LLMInterface'):
        super().__init__("review_code")
        self.review_tool = SectionCodeReviewTool(llm_interface)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return await self.review_tool.execute(exec_context, **kwargs)


class RespondWithReasoningTool(BaseTool):
    """
    Respond with reasoning for planning and explanation tasks.

    Used when active task is marked as [reasoning] rather than code generation.
    Includes internal critique loop.
    """

    def __init__(self, llm_interface: 'LLMInterface', use_critique: bool = True):
        super().__init__("respond_with_reasoning")
        self.reasoning_tool = ReasoningResponseWithGuidanceTool(llm_interface)
        self.critique_tool = ReasoningCritiqueTool(llm_interface)
        self.use_critique = use_critique

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        max_iterations = 2 if self.use_critique else 1

        for iteration in range(max_iterations):
            reasoning_result = await self.reasoning_tool.execute(exec_context, **kwargs)
            exec_context.inputs.context.update(reasoning_result.output_workflow or {})

            if not self.use_critique:
                break

            critique_result = await self.critique_tool.execute(exec_context, **kwargs)
            exec_context.inputs.context.update(critique_result.output_workflow or {})

            if exec_context.inputs.context.get("reasoning_approval") == "APPROVED":
                break
            elif iteration == max_iterations - 1:
                break

        return reasoning_result
