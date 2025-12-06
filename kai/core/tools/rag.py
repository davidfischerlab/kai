"""RAG tools for code snippet and workflow retrieval."""

from typing import TYPE_CHECKING
from kai.core.orchestration.base_tool import BaseTool, ToolResult, ToolOutputType
from kai.core.orchestration.deterministic_tools import CodeRetrievalTool

if TYPE_CHECKING:
    from kai.core.orchestration.execution_context import ExecutionContext
    from kai.retrieval import ChromaDbManager


class SearchCodeSnippetsTool(BaseTool):
    """
    Search code examples and API documentation.

    Direct wrapper around CodeRetrievalTool for deterministic RAG search.
    """

    def __init__(self, knowledge_base: 'ChromaDbManager'):
        super().__init__("search_code_snippets")
        self.knowledge_base = knowledge_base
        self.retrieval_tool = CodeRetrievalTool(knowledge_base)

    async def execute(self, exec_context: "ExecutionContext", **kwargs) -> ToolResult:
        return await self.retrieval_tool.execute(exec_context, **kwargs)
