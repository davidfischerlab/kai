"""LangGraph orchestrator with TypedDict state."""

import time
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from kai.core.state import KaiState
from kai.core.tools import create_consolidated_tools
from kai.core.llm_interface import LLMInterface
from kai.retrieval import ChromaDbManager
from kai.core.orchestration.vscode_communicator import VSCodeCommunicator
from kai.utils import setup_logger

logger = setup_logger(__name__)


class LangGraphOrchestrator:
    """LangGraph orchestrator using TypedDict state."""

    def __init__(
        self,
        llm_interface: LLMInterface,
        knowledge_base: ChromaDbManager,
        vscode_communicator: VSCodeCommunicator = None
    ):
        self.llm = llm_interface
        self.knowledge_base = knowledge_base
        self.vscode = vscode_communicator or VSCodeCommunicator()

        self.tools = create_consolidated_tools(llm_interface, knowledge_base)

        self.autonomous_graph = self._build_autonomous_graph()
        self.regular_graph = self._build_regular_graph()

        self.is_cancelled = False

    def _send_message(self, message: str):
        """Send console message via VSCode communicator."""
        self.vscode.send_console_message(message)

    def _build_autonomous_graph(self):
        """Build graph for autonomous mode execution."""
        graph = StateGraph(KaiState)

        graph.add_node("agent_router", self._autonomous_router_node)

        for name, tool in self.tools.items():
            if name in ["search_code_snippets", "generate_code", "update_code",
                        "execute_cell", "restart_and_rerun", "manage_progress",
                        "handle_error", "backtrack", "respond_with_reasoning",
                        "plan_tasks", "search_workflows", "search_workflows_only"]:
                graph.add_node(name, tool.as_graph_node())

        graph.set_entry_point("agent_router")

        graph.add_conditional_edges(
            "agent_router",
            self._route_autonomous_action,
            {
                "plan_tasks": "plan_tasks",
                "search_workflows": "search_workflows",
                "search_workflows_only": "search_workflows_only",
                "generate_code": "generate_code",
                "execute_cell": "execute_cell",
                "manage_progress": "manage_progress",
                "handle_error": "handle_error",
                "backtrack": "backtrack",
                "respond_with_reasoning": "respond_with_reasoning",
                "restart_and_rerun": "restart_and_rerun",
                "update_code": "update_code",
                "complete": END
            }
        )

        for tool_name in ["plan_tasks", "search_workflows", "search_workflows_only",
                          "generate_code", "execute_cell", "manage_progress",
                          "handle_error", "backtrack", "respond_with_reasoning",
                          "restart_and_rerun", "update_code"]:
            graph.add_edge(tool_name, "agent_router")

        memory = MemorySaver()
        return graph.compile(checkpointer=memory)

    def _build_regular_graph(self):
        """Build graph for regular mode."""
        graph = StateGraph(KaiState)

        graph.add_node("classify_intent", self.tools["classify_intent"].as_graph_node())
        graph.add_node("search_code_snippets", self.tools["search_code_snippets"].as_graph_node())
        graph.add_node("generate_code_simple", self.tools["generate_code_simple"].as_graph_node())
        graph.add_node("answer_question", self.tools["answer_question"].as_graph_node())

        graph.set_entry_point("classify_intent")

        graph.add_conditional_edges(
            "classify_intent",
            lambda state: state.get("intent", "question_about_code"),
            {
                "generate_code": "search_code_snippets",
                "generate_code_in_place": "search_code_snippets",
                "question_about_code": "search_code_snippets",
                "remove_code": END
            }
        )

        graph.add_conditional_edges(
            "search_code_snippets",
            lambda state: "generate" if state.get("intent", "").startswith("generate_code") else "answer",
            {
                "generate": "generate_code_simple",
                "answer": "answer_question"
            }
        )

        graph.add_edge("generate_code_simple", END)
        graph.add_edge("answer_question", END)

        return graph.compile()

    async def _autonomous_router_node(self, state: dict) -> dict:
        """Central routing node - just returns empty dict."""
        return {}

    def _route_autonomous_action(self, state: dict) -> str:
        """Determine next action based on validated state."""
        # Helper to safely get values from dict or Pydantic model
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        task_list = safe_get(state, "task_list", {})

        if not task_list or not safe_get(task_list, "tasks"):
            if safe_get(state, "retrieval_queries"):
                return "search_workflows"
            return "plan_tasks"

        if not safe_get(state, "autonomous_mode_continue"):
            return "complete"

        tasks = safe_get(task_list, "tasks", [])
        all_complete = all(safe_get(t, "status") == "completed" for t in tasks)
        if all_complete:
            return "complete"

        if safe_get(state, "error_context") or safe_get(state, "last_execution_failed"):
            if self._should_backtrack(state):
                return "backtrack"
            return "handle_error"

        if safe_get(state, "just_executed"):
            return "manage_progress"

        active_task = safe_get(state, "active_task")
        if not active_task:
            return "manage_progress"

        if safe_get(active_task, "task", "").lower().startswith("reason"):
            return "respond_with_reasoning"

        return "generate_code"

    def _should_backtrack(self, state: dict) -> bool:
        """Check if we should backtrack."""
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        error_ctx = safe_get(state, "error_context", {})
        if not error_ctx:
            return False

        strategy = safe_get(error_ctx, "recovery_strategy")
        if strategy == "BACKTRACK":
            return True

        backtrack_ctx = safe_get(state, "backtracking_context")
        return backtrack_ctx is not None and safe_get(backtrack_ctx, "recovery_objective")

    async def process_request(
        self,
        message: str,
        context: dict
    ) -> None:
        """
        Process request with TypedDict state (LangGraph standard).

        Args:
            message: User query
            context: Dict with all context fields including session_metadata, autonomous_mode, etc.
        """
        # Extract session metadata if present
        session_metadata = context.pop("session_metadata", {})
        autonomous_mode = context.get("autonomous_mode", False)

        # Build initial state (TypedDict - no validation needed)
        initial_state: KaiState = {
            "user_query": message,
            "messages": [{"role": "user", "content": message}] if message else [],
            **context,
            **session_metadata
        }

        # Select graph
        if autonomous_mode:
            graph = self.autonomous_graph
        else:
            graph = self.regular_graph

        # Execute graph (streaming like old orchestrator)
        try:
            start_time = time.time()
            config = {"configurable": {"thread_id": session_metadata.get("session_id", "default")}}

            async for output in graph.astream(initial_state, config):
                if self.is_cancelled:
                    self._send_message("Workflow cancelled by user")
                    break

                for node_name, node_output in output.items():
                    if node_output:
                        logger.debug(f"Node {node_name} completed")

            duration = time.time() - start_time
            self._send_message(f"Request processing completed in {duration:.3f}s")

            # Send completion status
            if autonomous_mode:
                final_state = await graph.aget_state(config)
                state_values = final_state.values if hasattr(final_state, 'values') else final_state

                tasks = state_values.get("task_list", {}).get("tasks", [])
                all_complete = all(t.get("status") == "completed" for t in tasks)

                if all_complete:
                    await self.vscode.send_workflow_result(
                        field="auto_loop_update",
                        state="LOOP_COMPLETE"
                    )
                else:
                    await self.vscode.send_workflow_result(
                        field="auto_loop_update",
                        state="LOOP_INCOMPLETE"
                    )
            else:
                await self.vscode.send_workflow_result(
                    field="regular_chat_update",
                    state="STEP_COMPLETE"
                )

        except Exception as e:
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            raise
