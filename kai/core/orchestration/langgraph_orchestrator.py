"""LangGraph orchestrator with TypedDict state."""

import time
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from kai.core.state import KaiState
from kai.core.tools import create_consolidated_tools
from kai.core.llm_interface import LLMInterface
from kai.retrieval import ChromaDbManager
from kai.core.orchestration.ui_communicator import UICommunicator
from kai.utils import setup_logger

logger = setup_logger(__name__)


# State Management Constants
#
# PERSISTENT fields (carry over across iterations - like kai_dev's OrchestratorState):
#   - task_list: Central task list
#   - reference_workflow_ids: Selected reference workflow IDs
#   - reference_workflow_content: Reference workflow content
#   - retrieval_queries: Queries for iterative reference workflow retrieval
#   - excluded_workflows: Workflows to exclude
#   - auto_mode_first_execution_done: Whether first exec completed
#   - active_task: Currently active task dict (set by mark_next_task_active)
#   - active_task_objective: Description of active task for code generation prompts
#   - is_reasoning_task: Whether active task is a reasoning task
#   - next_pending_task_objective: Next pending task for context
#
# These fields are managed by LangGraph's checkpointer. UI should NOT pass them
# on every iteration - they persist automatically via checkpoint.
#
PERSISTENT_STATE_FIELDS = {
    "task_list",
    "reference_workflow_ids",
    "reference_workflow_content",
    "retrieval_queries",
    "excluded_workflows",
    "auto_mode_first_execution_done",
    "planning_phase",  # Explicit phase tracking for planning control flow
    "workflow_retrieval_iteration",  # Iteration counter for workflow retrieval loop
    "task_planning_iteration",  # Iteration counter for task planning + workflow refinement loop
    # Active task tracking (must persist for code generation to know which task to address)
    "active_task",
    "active_task_objective",
    "is_reasoning_task",
    "next_pending_task_objective",
}

# TRANSIENT fields (cleared each iteration unless explicitly provided by UI):
#   - All fields in TRANSIENT_STATE_FIELDS below
#   - These represent execution state that should NOT persist
#
TRANSIENT_STATE_FIELDS = {
    # Code generation state
    "generated_code", "target_cell", "positioning_info",

    # Execution tracking
    "last_execution_failed", "last_output",

    # Reasoning state
    "reasoning_response", "reasoning_approval", "reasoning_critique",

    # Phase tracking flags (reset each iteration)
    "task_completion_analyzed", "next_task_activated",
    "tasks_updated", "update_approved",

    # Critique iterations (reset each iteration)
    "critique_iteration",

    # Error recovery flags
    "retry_objective", "recovery_objective", "backtrack_to_task",
    "error_recovery_strategy", "restart_required",

    # Task update state (includes backup for reversion within same iteration)
    "task_list_backup", "task_list_update_rule", "task_text_old",
    "task_list_approval", "autonomous_update_approval",

    # Backtracking state
    "reset_tasks", "cells_to_delete", "cells_deleted",
    "backtrack_recovery_done",

    # RAG retrieval (used once then cleared)
    "snippet_retrieval_query", "rag_retrieval", "rag_text",

    # Internal tool communication
    "_last_tool_result",
}


class LangGraphOrchestrator:
    """
    LangGraph orchestrator with proper state management.

    State Management:
    - PERSISTENT fields: task_list, reference_workflow_ids, excluded_workflows,
      auto_mode_first_execution_done (carry over across iterations)
    - TRANSIENT fields: generated_code, positioning_info, task_completion_analyzed, etc.
      (cleared each iteration unless explicitly provided by UI)

    This matches kai_dev behavior where only OrchestratorState fields persisted.
    """

    def __init__(
        self,
        llm_interface: LLMInterface,
        knowledge_base: ChromaDbManager,
        ui_communicator: UICommunicator = None,
        use_deterministic_routing: bool = True,
        graph_recursion_limit: Optional[int] = 100,
        max_task_planning_iterations: int = 10
    ):
        self.llm = llm_interface
        self.knowledge_base = knowledge_base
        self.vscode = ui_communicator or UICommunicator()
        self.use_deterministic_routing = use_deterministic_routing
        self.graph_recursion_limit = graph_recursion_limit  # Default: 100 (LangGraph default is 25)
        self.max_task_planning_iterations = max_task_planning_iterations

        self.tools = create_consolidated_tools(llm_interface, knowledge_base)

        self.planning_graph = self._build_planning_graph()
        self.autonomous_graph = self._build_autonomous_graph()
        self.regular_graph = self._build_regular_graph()

        self.is_cancelled = False

    def set_graph_recursion_limit(self, limit: Optional[int]):
        """Set graph recursion limit for LangGraph execution.

        Args:
            limit: Maximum number of graph steps per iteration (default: 100, LangGraph default is 25)
        """
        self.graph_recursion_limit = limit

    def _send_message(self, message: str):
        """Send console message via VSCode communicator."""
        self.vscode.send_console_message(message)

    def _build_planning_graph(self):
        """Build graph for planning phase (workflow retrieval + task generation).

        This graph handles:
        1. Initial workflow retrieval (max 2 iterations)
        2. Task list generation + critique loop (max 10 iterations)
        3. Workflow refinement based on retrieval queries from task generation

        Matches kai_dev/core/orchestration/workflow_orchestrator.py lines 240-319 exactly:
        - Initial workflow retrieval: max 2 iterations
        - Task generation loop: max 10 iterations
          - Generate tasks ONCE
          - If retrieval_queries: refine workflows, continue to next iteration
          - If no retrieval_queries: run critique (if enabled)
          - If critique approves or max iterations: exit

        Entry points:
        - search_workflows (if RAG enabled and retrieval queries exist)
        - task_list_generation (if no RAG or no queries)

        Exit: When task planning is complete (approved or max iterations)
        """
        graph = StateGraph(KaiState)

        # Add router
        graph.add_node("planning_router", self._planning_router_node)

        # Add iteration counter node (matches kai_dev's for loop increment)
        graph.add_node("increment_task_planning_iteration", self._increment_task_planning_iteration_node)

        # Add planning tools - using SEPARATE generation and critique nodes
        # (not bundled PlanTasksTool which has its own internal loop)
        planning_tools = [
            "search_workflows",
            "workflow_refinement",
            "task_list_generation",  # Single generation step
            "task_list_critique",     # Single critique step
        ]

        for name in planning_tools:
            if name in self.tools:
                graph.add_node(name, self.tools[name].as_graph_node())

        # Add filter tool for cleanup after planning (kai_dev lines 325-327)
        if "filter_unused_reference_workflows" in self.tools:
            graph.add_node("filter_unused_reference_workflows", self.tools["filter_unused_reference_workflows"].as_graph_node())

        # Entry point is the router
        graph.set_entry_point("planning_router")

        # Router decides next step
        routing_map = {tool: tool for tool in planning_tools}
        routing_map["increment_task_planning_iteration"] = "increment_task_planning_iteration"
        routing_map["filter_and_complete"] = "filter_unused_reference_workflows"  # Filter before ending
        routing_map["complete"] = END  # Skip filter if no RAG

        graph.add_conditional_edges(
            "planning_router",
            self._route_planning_phase,
            routing_map
        )

        # All tools and nodes return to router
        for tool_name in planning_tools:
            graph.add_edge(tool_name, "planning_router")
        graph.add_edge("increment_task_planning_iteration", "planning_router")

        # Filter tool goes to END (last step before completing planning)
        graph.add_edge("filter_unused_reference_workflows", END)

        # Planning graph needs in-memory state tracking for iteration counters
        # (but not persistent checkpointing across process_request calls)
        return graph.compile()

    async def _planning_router_node(self, _state: dict) -> dict:
        """Planning router node - just returns empty dict."""
        return {}

    async def _increment_task_planning_iteration_node(self, state: dict) -> dict:
        """Set task_planning_iteration counter for current iteration.

        Matches kai_dev's for loop variable assignment (line 262):
        - First call: iteration = 0
        - Second call: iteration = 1
        - ...

        The counter is initialized to -1, so first increment gives 0.
        Sets planning_phase to "ready_to_generate" to signal router.
        """
        current = state.get("task_planning_iteration", -1)  # Start at -1 so first increment gives 0
        new_value = current + 1

        # Log planning iteration start on its own line
        # Workflow retrieval takes iterations 1-2, task generation takes 3-12
        # new_value is 0-indexed internally (0-9), display as 3-12 (offset by +3)
        if new_value < self.max_task_planning_iterations:
            logger.info(f"[PLANNING ITERATION {new_value + 1}/{self.max_task_planning_iterations}]")
            self._send_message(f"[KAI] Planning iteration {new_value + 1}/{self.max_task_planning_iterations}")

        return {
            "task_planning_iteration": new_value,
            "planning_phase": "ready_to_generate",  # Signal that we're ready for task generation
            "task_list_approval": None  # Clear previous critique result for next iteration
        }

    def _build_autonomous_graph(self):
        """Build graph for autonomous mode execution (after planning is complete).

        This graph handles task execution with checkpointing for persistence across iterations.
        Planning is done in a separate graph before this graph is invoked.
        """
        graph = StateGraph(KaiState)

        graph.add_node("agent_router", self._autonomous_router_node)
        graph.add_node("mark_first_execution_done", self._mark_first_execution_done_node)
        graph.add_node("mark_reasoning_completed", self._mark_reasoning_completed_node)
        graph.add_node("backup_task_list", self._backup_task_list_node)
        graph.add_node("revert_task_list", self._revert_task_list_node)

        # Add all tools needed for autonomous mode EXECUTION (no planning tools)
        autonomous_tools = [
            # Core execution tools
            "mark_next_task_active", "autonomous_mark_completion",
            "cell_positioning", "code_generation_with_guidance",
            "reasoning_response_with_guidance", "reasoning_critique",
            # Positioning tools (matches kai_dev - use last_cell instead of LLM for continuation)
            "set_positioning_from_last_cell",
            # Task update tools
            "autonomous_update_tasks", "autonomous_update_critique",
            # Error recovery tools
            "error_recovery", "code_update", "restart_and_rerun", "restart_and_rerun_prompt",
            # Backtracking tools
            "backtrack_recovery", "cell_selection_deletion", "cell_deletion",
            # RAG tools
            "rag_retrieval", "search_code_snippets",
            # Legacy tools (for smart router)
            "generate_code", "update_code", "execute_cell", "manage_progress",
            "handle_error", "backtrack", "respond_with_reasoning"
        ]

        for name in autonomous_tools:
            if name in self.tools:
                graph.add_node(name, self.tools[name].as_graph_node())

        graph.set_entry_point("agent_router")

        # Build routing map for all tools
        routing_map = {tool: tool for tool in autonomous_tools}
        routing_map["mark_first_execution_done"] = "mark_first_execution_done"
        routing_map["mark_reasoning_completed"] = "mark_reasoning_completed"
        routing_map["backup_task_list"] = "backup_task_list"
        routing_map["revert_task_list"] = "revert_task_list"
        routing_map["complete"] = END

        graph.add_conditional_edges(
            "agent_router",
            self._route_autonomous_action,
            routing_map
        )

        # All tools return to router after execution
        for tool_name in autonomous_tools:
            graph.add_edge(tool_name, "agent_router")

        graph.add_edge("mark_first_execution_done", END)
        graph.add_edge("mark_reasoning_completed", END)
        graph.add_edge("backup_task_list", "agent_router")
        graph.add_edge("revert_task_list", "agent_router")

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

    async def _mark_first_execution_done_node(self, state: dict) -> dict:
        """Mark first execution as complete."""
        self._send_message("[KAI] Marking first execution as done")
        return {"auto_mode_first_execution_done": True}

    async def _mark_reasoning_completed_node(self, state: dict) -> dict:
        """Mark the active reasoning task as completed.

        This is called after reasoning is approved to ensure the task is marked
        completed before the next iteration starts. Without this, the next iteration's
        autonomous_mark_completion would see the task still as 'active' and try to
        re-evaluate it, potentially setting a retry_objective.

        Also marks first execution as done if we're still in first execution phase.
        """
        import copy

        task_list = state.get("task_list", {})
        if not task_list or "tasks" not in task_list:
            logger.warning("[MARK_REASONING_COMPLETED] No task list found")
            return {}

        # Deep copy to avoid mutating original
        updated_task_list = copy.deepcopy(task_list)

        # Find and mark the active task as completed
        active_task_id = None
        for task in updated_task_list["tasks"]:
            if task.get("status") == "active":
                task["status"] = "completed"
                active_task_id = task.get("id")
                logger.info(
                    f"[MARK_REASONING_COMPLETED] Marked task {active_task_id} "
                    f"as completed: {task.get('task', '')[:50]}..."
                )
                break

        if active_task_id is None:
            logger.warning("[MARK_REASONING_COMPLETED] No active task found")
            return {}

        # Send simple message to UI
        self._send_message(f"[KAI] Reasoning task {active_task_id} completed")

        # Build result - clear transient reasoning state for next task
        result = {
            "task_list": updated_task_list,
            "reasoning_response": None,
            "reasoning_approval": None,
            "reasoning_critique": None,
            "critique_iteration": 0,
            "active_task": None,
            "active_task_objective": None,
            "is_reasoning_task": False,
        }

        # Also mark first execution as done if we're still in first execution
        if not state.get("auto_mode_first_execution_done"):
            result["auto_mode_first_execution_done"] = True
            logger.debug("[MARK_REASONING_COMPLETED] Also marking first execution done")

        return result

    async def _backup_task_list_node(self, state: dict) -> dict:
        """Backup task list before autonomous_update_tasks (for reversion if critique fails)."""
        import copy
        task_list = state.get("task_list", {})
        logger.debug(f"[BACKUP] Saving task list backup: {len(task_list.get('tasks', []))} tasks")
        return {"task_list_backup": copy.deepcopy(task_list)}

    async def _revert_task_list_node(self, state: dict) -> dict:
        """Revert task list to backup (critique failed after max iterations)."""
        backup = state.get("task_list_backup")
        if backup:
            logger.warning(f"[REVERT] Reverting task list to backup: {len(backup.get('tasks', []))} tasks")
            return {
                "task_list": backup,
                "task_list_backup": None,  # Clear backup after reversion
                "tasks_updated": True,  # Mark as updated (even though reverted) so router skips update flow
                "update_approved": True,  # Mark as approved to skip critique
                "critique_iteration": 0,  # Reset critique counter
                "task_list_update_rule": None,  # Clear update rule
                "task_text_old": None,  # Clear old task text
            }
        else:
            logger.error("[REVERT] No backup found! Cannot revert task list")
            # Even if backup is missing, mark as updated to prevent looping
            return {
                "tasks_updated": True,
                "update_approved": True,
                "critique_iteration": 0,
                "task_list_update_rule": None,
                "task_text_old": None,
            }

    def _route_autonomous_action(self, state: dict) -> str:
        """Route to next action - uses deterministic or smart routing."""
        logger.debug(f"[ROUTE_DISPATCH] use_deterministic_routing={self.use_deterministic_routing}")
        if self.use_deterministic_routing:
            logger.debug("[ROUTE_DISPATCH] → _route_deterministic")
            return self._route_deterministic(state)
        else:
            logger.debug("[ROUTE_DISPATCH] → _route_smart")
            return self._route_smart(state)

    def _route_deterministic(self, state: dict) -> str:
        """
        Deterministic routing for execution phase (after planning is complete).

        Assumes planning graph has already run and produced a task list.

        Phases:
        1. First iteration: Activate first task and exit to show user
        2. First execution: After user approval, generate and execute first code
        3. Standard execution: Execute remaining tasks with error handling and backtracking
        """
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        task_list = safe_get(state, "task_list", {})
        tasks = safe_get(task_list, "tasks", [])
        autonomous_mode_continue = safe_get(state, "autonomous_mode_continue")

        auto_mode_first_execution_done = safe_get(state, "auto_mode_first_execution_done", False)
        logger.debug(f"[DET ROUTER] tasks={len(tasks)}, continue={autonomous_mode_continue}, first_exec_done={auto_mode_first_execution_done}")

        # ===== Sanity check: Should have tasks from planning graph =====
        if not tasks:
            logger.error("[DET ROUTER] No tasks found! Planning graph should have created tasks.")
            return "complete"

        # ===== PHASE 2: FIRST ITERATION (after planning) =====
        # Activate first task, then exit to show user (VSCode) or continue to execution (Jupyter)
        # Only apply this logic if we haven't done first execution yet (to avoid exiting on every iteration)
        auto_mode_first_execution_done = safe_get(state, "auto_mode_first_execution_done", False)

        if not autonomous_mode_continue and not auto_mode_first_execution_done:
            all_pending = all(safe_get(t, "status") == "pending" for t in tasks)
            has_active = any(safe_get(t, "status") == "active" for t in tasks)

            logger.debug(f"[DET ROUTER] FIRST_ITER check: all_pending={all_pending}, has_active={has_active}, auto_continue={autonomous_mode_continue}, first_exec_done={auto_mode_first_execution_done}")

            if all_pending:
                logger.debug("[DET ROUTER] FIRST_ITER: activating first task → mark_next_task_active")
                return "mark_next_task_active"
            elif has_active:
                # Check if we should pause for user confirmation (VSCode) or continue directly (Jupyter)
                confirm_plan = safe_get(state, "confirm_plan", True)  # Default True for backwards compat
                logger.debug(f"[DET ROUTER] FIRST_ITER: task active, confirm_plan={confirm_plan}")
                if confirm_plan:
                    logger.debug("[DET ROUTER] FIRST_ITER: exiting to show user (VSCode mode) → complete")
                    return "complete"
                else:
                    logger.debug("[DET ROUTER] FIRST_ITER: continuing to first execution (Jupyter mode) → falling through to PHASE 3")
                    # Continue to first execution phase (bypass the autonomous_mode_continue check)
                    # by falling through to PHASE 3 below
            else:
                logger.debug("[DET ROUTER] FIRST_ITER: no tasks active or pending → complete")
                return "complete"

        # ===== Check completion first =====
        all_complete = all(safe_get(t, "status") == "completed" for t in tasks)
        if all_complete:
            logger.debug("[DET ROUTER] All tasks complete!")
            return "complete"

        # ===== PHASE 3: FIRST EXECUTION (after user approval, first code generation) =====
        auto_mode_first_execution_done = safe_get(state, "auto_mode_first_execution_done", False)
        if not auto_mode_first_execution_done:
            return self._route_first_execution(state)

        # ===== PHASE 4: STANDARD EXECUTION (all subsequent iterations) =====
        return self._route_standard_execution(state)

    def _log_task_list_summary(self, state: dict):
        """Log final task list summary at end of planning."""
        task_list = state.get("task_list", {})
        tasks = task_list.get("tasks", [])

        if not tasks:
            logger.warning("[PLANNING] No tasks generated!")
            return

        logger.info(f"Task list:")
        for i, task in enumerate(tasks, 1):
            # Tasks use "task" field, not "objective"
            task_obj = task.get("task", task.get("objective", "No objective"))
            logger.info(f"  {i}. {task_obj}")

    def _route_planning_phase(self, state: dict) -> str:
        """
        Route planning phase based on current state.

        Matches kai_dev planning logic (lines 240-319) EXACTLY:

        PHASE 1: Initial workflow retrieval (max 2 iterations)
        - Entry → search_workflows (if RAG enabled)
        - search_workflows → router checks queries → search_workflows OR task_list_generation

        PHASE 2: Task generation + refinement loop (max 10 iterations)
        For each iteration:
        1. task_list_generation (generates tasks ONCE)
        2. Router checks retrieval_queries:
           - If has queries: workflow_refinement → back to task_list_generation
           - If no queries: task_list_critique (if enabled)
        3. Router checks critique approval:
           - If APPROVED: complete
           - If REJECTED: back to task_list_generation
        4. Max 10 iterations total
        """
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        planning_phase = safe_get(state, "planning_phase")
        retrieval_queries = safe_get(state, "retrieval_queries", [])
        workflow_iteration = safe_get(state, "workflow_retrieval_iteration", 0)
        task_planning_iteration = safe_get(state, "task_planning_iteration", 0)
        rag_enabled = safe_get(state, "rag_enabled", False)
        use_critique = safe_get(state, "use_critique", False)
        task_list_approval = safe_get(state, "task_list_approval")

        self._send_message(f"[KAI] PLANNING ROUTER: phase={planning_phase}, rag={rag_enabled}, queries={len(retrieval_queries) if retrieval_queries else 0}")
        logger.debug(f"[PLANNING ROUTER] phase={planning_phase}, rag={rag_enabled}, use_critique={use_critique}, "
                    f"workflow_iter={workflow_iteration}, task_iter={task_planning_iteration}, "
                    f"queries={len(retrieval_queries) if retrieval_queries else 0}, approval={task_list_approval}")

        # ===== PHASE 1: Initial workflow retrieval (max 2 iterations) =====
        # Matches kai_dev lines 243-256
        if planning_phase is None:
            if rag_enabled and retrieval_queries:
                self._send_message(f"[KAI] → search_workflows (initial)")
                return "search_workflows"
            else:
                self._send_message(f"[KAI] → increment_task_planning_iteration (no RAG or no queries)")
                return "increment_task_planning_iteration"

        if planning_phase == "workflow_retrieval":
            # Check termination conditions
            if workflow_iteration >= 2:
                return "increment_task_planning_iteration"

            if not retrieval_queries or len(retrieval_queries) == 0:
                return "increment_task_planning_iteration"

            # Continue retrieval
            return "search_workflows"

        # ===== PHASE 2: Task generation + workflow refinement loop =====
        # Matches kai_dev lines 262-313
        if planning_phase == "task_planning":
            # Check max iterations FIRST (kai_dev line 310)
            if task_planning_iteration >= self.max_task_planning_iterations:
                logger.info(f"[PLANNING ROUTER] Max task planning iterations ({self.max_task_planning_iterations}) reached → {'filter_and_complete' if rag_enabled else 'complete'}")
                self._log_task_list_summary(state)
                return "filter_and_complete" if rag_enabled else "complete"

            # ALWAYS run workflow refinement when rag_enabled (kai_dev lines 271-297)
            # This runs even without new queries to allow selection changes (protecting cited, removing uncited)
            # The continue logic (line 296-297) only applies if there WERE queries
            if rag_enabled:
                return "workflow_refinement"

            # No RAG - run critique if enabled (kai_dev lines 300-306)
            if use_critique:
                # Check if we just came from critique
                if task_list_approval is not None:
                    # We have a critique result
                    if task_list_approval == "APPROVED":
                        logger.info(f"[PLANNING ROUTER] Task list approved after {task_planning_iteration} iterations → {'filter_and_complete' if rag_enabled else 'complete'}")
                        self._log_task_list_summary(state)
                        return "filter_and_complete" if rag_enabled else "complete"
                    else:
                        # Rejected - generate again (increment iteration counter first)
                        logger.info(f"[PLANNING ROUTER] Task list rejected, iteration {task_planning_iteration + 1}/{self.max_task_planning_iterations} → increment_task_planning_iteration")
                        return "increment_task_planning_iteration"
                else:
                    # No critique result yet - run critique
                    logger.info(f"[PLANNING ROUTER] Task planning iteration {task_planning_iteration + 1}/{self.max_task_planning_iterations}: running critique → task_list_critique")
                    return "task_list_critique"

            # No critique enabled and no retrieval queries - done (kai_dev lines 308-309)
            logger.info(f"[PLANNING ROUTER] Task planning complete (no critique, no queries) → {'filter_and_complete' if rag_enabled else 'complete'}")
            self._log_task_list_summary(state)
            return "filter_and_complete" if rag_enabled else "complete"

        # ===== PHASE 3: After task list critique =====
        # Check if we just came from critique (planning_phase == "task_list_critique")
        if planning_phase == "task_list_critique":
            if task_list_approval is not None:
                # We have a valid critique result
                if task_list_approval == "APPROVED":
                    self._send_message(f"[KAI] Task list approved after {task_planning_iteration + 1} iterations")
                    logger.info(f"[PLANNING ROUTER] Task list approved after {task_planning_iteration + 1} iterations → {'filter_and_complete' if rag_enabled else 'complete'}")
                    self._log_task_list_summary(state)
                    return "filter_and_complete" if rag_enabled else "complete"
                else:
                    # Rejected - generate again (increment iteration counter first)
                    # This will loop back to task_list_generation (kai_dev line 262-264)
                    return "increment_task_planning_iteration"
            else:
                # Critique tool returned invalid/no result - proceed anyway to avoid infinite loop
                logger.warning(f"[PLANNING ROUTER] Task list critique returned no approval status (LLM error) - proceeding anyway → {'filter_and_complete' if rag_enabled else 'complete'}")
                self._log_task_list_summary(state)
                return "filter_and_complete" if rag_enabled else "complete"

        # ===== PHASE 4: After workflow refinement =====
        # After workflow refinement, check if there were queries (kai_dev line 296-297)
        if planning_phase == "workflow_refinement":
            had_queries = safe_get(state, "had_retrieval_queries_before_refinement", False)
            if had_queries:
                # Had queries → continue to next iteration (skip critique, kai_dev line 297)
                return "increment_task_planning_iteration"
            else:
                # No queries → proceed to critique (if enabled)
                if use_critique:
                    return "task_list_critique"
                else:
                    # No critique enabled → complete
                    logger.info(f"[PLANNING ROUTER] Workflow refinement complete (no critique) → {'filter_and_complete' if rag_enabled else 'complete'}")
                    self._log_task_list_summary(state)
                    return "filter_and_complete" if rag_enabled else "complete"

        # ===== PHASE 4: After increment =====
        # After incrementing, check if we've exceeded max iterations (matches kai_dev's range check)
        # kai_dev uses range(10) which gives 0-9, so iteration 10 would exit the loop
        if planning_phase == "ready_to_generate":
            if task_planning_iteration >= self.max_task_planning_iterations:
                self._send_message(f"[KAI] Task list generation reached max iterations ({self.max_task_planning_iterations}) without approval - proceeding anyway")
                logger.info(f"[PLANNING ROUTER] Max task planning iterations ({self.max_task_planning_iterations}) reached → {'filter_and_complete' if rag_enabled else 'complete'}")
                self._log_task_list_summary(state)
                return "filter_and_complete" if rag_enabled else "complete"
            return "task_list_generation"

        # Phase complete (shouldn't reach here in normal flow)
        logger.info("[PLANNING ROUTER] Planning phase complete (unexpected state)")
        self._log_task_list_summary(state)
        return "filter_and_complete" if rag_enabled else "complete"

    def _route_first_execution(self, state: dict) -> str:
        """
        First execution phase routing.
        Sequence: mark_next_task_active (if needed) → cell_positioning → code_generation_with_guidance → exit
        """
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        logger.debug("[DET ROUTER] Phase: FIRST_EXECUTION")

        # Check if we have an active task (mark_next_task_active sets active_task and active_task_objective)
        task_list = safe_get(state, "task_list", {})
        tasks = safe_get(task_list, "tasks", [])
        has_active_task = any(safe_get(t, "status") == "active" for t in tasks)
        active_task_objective = safe_get(state, "active_task_objective")

        has_positioning = safe_get(state, "positioning_info") is not None
        first_exec_done = safe_get(state, "auto_mode_first_execution_done", False)

        logger.debug(f"[DET ROUTER] FIRST_EXEC state: has_active={has_active_task}, active_objective={bool(active_task_objective)}, has_positioning={has_positioning}, first_exec_done={first_exec_done}")

        # Defensive: if first execution already done, shouldn't be here - exit
        if first_exec_done:
            logger.debug("[DET ROUTER] FIRST_EXEC: first exec already done (unexpected state) → complete")
            return "complete"

        # First, ensure we have an active task (mark_next_task_active sets active_task_objective)
        if not has_active_task:
            logger.debug("[DET ROUTER] FIRST_EXEC: no active task → mark_next_task_active")
            return "mark_next_task_active"

        if not has_positioning:
            logger.debug("[DET ROUTER] FIRST_EXEC: need positioning → cell_positioning")
            return "cell_positioning"

        # Check if this is a reasoning task (set by mark_next_task_active)
        is_reasoning = safe_get(state, "is_reasoning_task", False)
        generated_code = safe_get(state, "generated_code")
        reasoning_response = safe_get(state, "reasoning_response")

        if is_reasoning:
            # Reasoning task flow
            if not reasoning_response:
                logger.debug("[DET ROUTER] FIRST_EXEC: reasoning task, generating reasoning")
                return "reasoning_response_with_guidance"

            # Reasoning critique loop
            reasoning_approval = safe_get(state, "reasoning_approval")
            critique_iteration = safe_get(state, "critique_iteration", 0)

            if reasoning_approval != "APPROVED" and critique_iteration < 3:
                logger.debug(f"[DET ROUTER] FIRST_EXEC: reasoning critique (iter {critique_iteration + 1})")
                return "reasoning_critique"

            # Reasoning complete - mark task completed and exit
            if reasoning_approval == "APPROVED":
                self._send_message(f"Reasoning approved after {critique_iteration} critique iterations")
            else:
                self._send_message(f"Reasoning critique reached max iterations (3) without approval - proceeding anyway")
            logger.debug("[DET ROUTER] FIRST_EXEC: reasoning complete → mark_reasoning_completed")
            return "mark_reasoning_completed"
        else:
            # Code task flow
            if not generated_code:
                logger.debug("[DET ROUTER] FIRST_EXEC: need code generation → code_generation_with_guidance")
                return "code_generation_with_guidance"
            else:
                # Code has been generated - exit to UI for execution
                # Mark first execution as done so next iteration goes to STANDARD_EXECUTION
                logger.debug("[DET ROUTER] FIRST_EXEC: code generated, marking done and exiting to UI")
                return "mark_first_execution_done"

    def _route_standard_execution(self, state: dict) -> str:
        """
        Standard execution phase routing with 4 branches.

        Phase 1: Analyze completion & update tasks
        Phase 2: Branch based on execution state:
          - Branch 1: All Complete
          - Branch 2: Standard Continue (no errors, normal progression)
          - Branch 3: Standard Retry (error or LLM detected issue)
          - Branch 4: Backtracking
        """
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        logger.debug("[DET ROUTER] Phase: STANDARD_EXECUTION")

        # ==== SUB-PHASE 1: Analyze Completion & Update Tasks ====

        task_completion_analyzed = safe_get(state, "task_completion_analyzed")

        # Step 1: Analyze completion
        # The UI (VSCode/Jupyter) waits for execution to complete before calling Python again,
        # so when we're in STANDARD_EXECUTION with autonomous_mode_continue=True, we know execution just finished
        if not task_completion_analyzed:
            logger.debug("[DET ROUTER] STANDARD_EXEC: analyzing completion")
            return "autonomous_mark_completion"

        # Branch detection - autonomous_mark_completion sets these flags
        has_error = safe_get(state, "last_execution_failed")
        retry_objective = safe_get(state, "retry_objective")
        recovery_objective = safe_get(state, "recovery_objective")

        is_backtracking = recovery_objective is not None
        is_standard_retry = (has_error or retry_objective) and not is_backtracking

        logger.debug(f"[DET ROUTER] Branch detection: error={has_error}, retry={bool(retry_objective)}, backtrack={is_backtracking}")

        # Step 2: Update task list (if NOT standard retry and NOT backtracking and NOT complete)
        tasks_updated = safe_get(state, "tasks_updated")
        update_approved = safe_get(state, "update_approved")

        task_list = safe_get(state, "task_list", {})
        tasks = safe_get(task_list, "tasks", [])
        all_complete = all(safe_get(t, "status") == "completed" for t in tasks)

        # Skip task updates in standard retry AND backtracking (dev branch behavior)
        if task_completion_analyzed and not is_standard_retry and not is_backtracking and not all_complete and not tasks_updated:
            # Check if we need to backup task list first
            has_backup = safe_get(state, "task_list_backup") is not None
            if not has_backup:
                logger.debug("[DET ROUTER] STANDARD_EXEC: backing up task list before update")
                return "backup_task_list"
            else:
                logger.debug("[DET ROUTER] STANDARD_EXEC: updating tasks")
                return "autonomous_update_tasks"

        # Step 3: Task update critique loop
        task_list_update_rule = safe_get(state, "task_list_update_rule")
        critique_iteration = safe_get(state, "critique_iteration", 0)

        if tasks_updated and task_list_update_rule == "UPDATE" and not update_approved:
            if critique_iteration < 3:  # Max 3 iterations
                autonomous_update_approval = safe_get(state, "autonomous_update_approval")
                if autonomous_update_approval == "APPROVED":
                    self._send_message(f"[KAI] Task list update approved after {critique_iteration} critique iterations")
                    logger.debug("[DET ROUTER] Task update approved")
                    # Log the approved task list
                    tasks = safe_get(task_list, "tasks", [])
                    logger.info(f"Updated task list ({len(tasks)} tasks):")
                    for i, task in enumerate(tasks, 1):
                        task_text = task.get("task", task.get("objective", "No objective"))
                        logger.info(f"  {i}. {task_text}")
                    # Continue to next step
                elif critique_iteration == 0:
                    # First critique - just run critique on initial proposal
                    logger.debug("[DET ROUTER] STANDARD_EXEC: task update critique (iter 1)")
                    return "autonomous_update_critique"
                else:
                    # Critique failed, need to regenerate task list before next critique
                    # This matches kai_dev: after critique failure, regenerate with autonomous_update_tasks
                    logger.debug(f"[DET ROUTER] Task update not approved (iter {critique_iteration}), regenerating task list")
                    return "autonomous_update_tasks"
            else:
                # Max iterations reached without approval - revert to backup
                self._send_message(f"[KAI] Task list update critique reached max iterations (3) without approval - reverting to previous task list")
                logger.warning(f"[DET ROUTER] Task update critique reached max iterations ({critique_iteration}) without approval - reverting to backup")
                return "revert_task_list"

        # Step 4: Activate next task
        next_task_activated = safe_get(state, "next_task_activated")

        if task_completion_analyzed and not next_task_activated:
            # Skip if we're still in task update flow
            if not tasks_updated or update_approved or task_list_update_rule != "UPDATE":
                logger.debug("[DET ROUTER] STANDARD_EXEC: activating next task")
                return "mark_next_task_active"

        # Step 5: RAG retrieval (if standard retry and RAG enabled)
        rag_enabled = safe_get(state, "rag_enabled", False)
        # BUG FIX: Check for "rag_retrieval" not "rag_text" (that's what the tool returns)
        rag_retrieved = safe_get(state, "rag_retrieval") is not None

        logger.debug(f"[DET ROUTER] RAG check: enabled={rag_enabled}, retrieved={rag_retrieved}, retry={is_standard_retry}, next_activated={next_task_activated}")
        logger.debug(f"[DET ROUTER] RAG state value: {safe_get(state, 'rag_retrieval')}")

        if next_task_activated and is_standard_retry and rag_enabled and not rag_retrieved:
            logger.debug("[DET ROUTER] STANDARD_EXEC: RAG retrieval for error recovery")
            return "rag_retrieval"

        # Debug: Log if RAG was skipped
        if is_standard_retry and rag_enabled:
            if rag_retrieved:
                logger.debug(f"[DET ROUTER] RAG already retrieved, proceeding to error recovery")
            elif not next_task_activated:
                logger.debug(f"[DET ROUTER] Skipping RAG - next task not activated yet")

        # ==== SUB-PHASE 2: Branch Based on Execution State ====

        # Re-check completion after task updates
        tasks = safe_get(task_list, "tasks", [])
        all_complete = all(safe_get(t, "status") == "completed" for t in tasks)

        # BRANCH 1: All Complete
        if all_complete:
            logger.debug("[DET ROUTER] Branch 1: ALL COMPLETE")
            return "complete"

        # BRANCH 4: Backtracking
        if is_backtracking:
            return self._route_backtracking_branch(state)

        # BRANCH 3: Standard Retry
        if is_standard_retry:
            return self._route_standard_retry_branch(state)

        # BRANCH 2: Standard Continue (normal progression, no errors)
        return self._route_standard_continue_branch(state)

    def _route_standard_continue_branch(self, state: dict) -> str:
        """Branch 2: Standard continue - no errors, normal progression."""
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        logger.debug("[DET ROUTER] Branch 2: STANDARD_CONTINUE")

        # Set positioning from last_cell_modified_in_auto_mode (matches kai_dev behavior)
        # This is deterministic - we always add after the last modified cell
        has_positioning = safe_get(state, "positioning_info") is not None
        if not has_positioning:
            logger.debug("[DET ROUTER] STANDARD_CONTINUE: setting positioning from last cell")
            return "set_positioning_from_last_cell"

        # Check if reasoning task
        is_reasoning = safe_get(state, "is_reasoning_task")
        generated_code = safe_get(state, "generated_code")
        reasoning_response = safe_get(state, "reasoning_response")

        if is_reasoning:
            if not reasoning_response:
                logger.debug("[DET ROUTER] STANDARD_CONTINUE: generating reasoning response")
                return "reasoning_response_with_guidance"

            # Reasoning critique loop
            reasoning_approval = safe_get(state, "reasoning_approval")
            critique_iteration = safe_get(state, "critique_iteration", 0)

            if reasoning_approval != "APPROVED" and critique_iteration < 3:
                logger.debug(f"[DET ROUTER] STANDARD_CONTINUE: reasoning critique (iter {critique_iteration + 1})")
                return "reasoning_critique"

            # Reasoning complete - mark task completed and exit
            if reasoning_approval == "APPROVED":
                self._send_message(f"Reasoning approved after {critique_iteration} critique iterations")
            else:
                self._send_message(f"Reasoning critique reached max iterations (3) without approval - proceeding anyway")
            logger.debug("[DET ROUTER] STANDARD_CONTINUE: reasoning complete → mark_reasoning_completed")
            return "mark_reasoning_completed"
        else:
            if not generated_code:
                logger.debug("[DET ROUTER] STANDARD_CONTINUE: generating code")
                return "code_generation_with_guidance"

            # Code generated, exit to UI for execution
            logger.debug("[DET ROUTER] STANDARD_CONTINUE: code generated, exiting to UI")
            return "complete"

    def _route_standard_retry_branch(self, state: dict) -> str:
        """Branch 3: Error or LLM detected issue - fix and retry."""
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        logger.debug("[DET ROUTER] Branch 3: STANDARD RETRY")

        # Step 1: Determine error recovery strategy
        error_recovery_strategy = safe_get(state, "error_recovery_strategy")

        if not error_recovery_strategy:
            logger.debug("[DET ROUTER] STANDARD RETRY: determining recovery strategy")
            return "error_recovery"

        # Step 2: Set positioning to failed cell (matches kai_dev behavior)
        # Use set_positioning_from_last_cell which uses last_cell_modified_in_auto_mode
        has_positioning = safe_get(state, "positioning_info") is not None
        if not has_positioning:
            logger.debug("[DET ROUTER] STANDARD RETRY: setting positioning from last cell")
            return "set_positioning_from_last_cell"

        # Step 3: Re-check if reasoning task (flag may be stale!)
        active_task_objective = safe_get(state, "active_task_objective", "")
        is_reasoning_task = "[reasoning]" in active_task_objective.lower()

        # Step 4: Execute recovery based on strategy and task type
        restart_done = safe_get(state, "restart_required") == False  # restart_and_rerun sets this to False after execution
        code_updated = safe_get(state, "generated_code") or safe_get(state, "reasoning_response")

        if is_reasoning_task:
            if not safe_get(state, "reasoning_response"):
                logger.debug("[DET ROUTER] STANDARD RETRY: regenerating reasoning")
                return "reasoning_response_with_guidance"

            # Reasoning critique
            reasoning_approval = safe_get(state, "reasoning_approval")
            critique_iteration = safe_get(state, "critique_iteration", 0)

            if reasoning_approval != "APPROVED" and critique_iteration < 3:
                logger.debug(f"[DET ROUTER] STANDARD RETRY: reasoning critique (iter {critique_iteration + 1})")
                return "reasoning_critique"

            # Reasoning complete - mark task completed and exit
            if reasoning_approval == "APPROVED":
                self._send_message(f"[KAI] Reasoning approved after {critique_iteration} critique iterations")
            else:
                self._send_message(f"[KAI] Reasoning critique reached max iterations (3) without approval - proceeding anyway")
            logger.debug("[DET ROUTER] STANDARD RETRY: reasoning complete → mark_reasoning_completed")
            return "mark_reasoning_completed"
        else:
            # Code task - check strategy
            if error_recovery_strategy == "REPLACE_AND_RESTART":
                if not restart_done:
                    logger.debug("[DET ROUTER] STANDARD RETRY: restarting kernel")
                    return "restart_and_rerun_prompt"
                elif not code_updated:
                    logger.debug("[DET ROUTER] STANDARD RETRY: updating code (after restart)")
                    return "code_update"
                else:
                    logger.debug("[DET ROUTER] STANDARD RETRY: code updated, exiting")
                    return "complete"
            elif error_recovery_strategy == "REPLACE_AND_RETRY":
                if not code_updated:
                    logger.debug("[DET ROUTER] STANDARD RETRY: updating code")
                    return "code_update"
                else:
                    logger.debug("[DET ROUTER] STANDARD RETRY: code updated, exiting")
                    return "complete"
            else:
                logger.error(f"[DET ROUTER] Unknown error recovery strategy: {error_recovery_strategy}")
                return "complete"

    def _route_backtracking_branch(self, state: dict) -> str:
        """Branch 4: Backtracking - delete cells and regenerate."""
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        logger.debug("[DET ROUTER] Branch 4: BACKTRACKING")

        # Backtracking sequence:
        # 1. backtrack_recovery (determine restart need)
        # 2. cell_selection_deletion
        # 3. cell_deletion
        # 4. cell_positioning
        # 5. restart_and_rerun (if needed)
        # 6. code_generation_with_guidance

        backtrack_recovery_done = safe_get(state, "backtrack_recovery_done")
        cells_to_delete = safe_get(state, "cells_to_delete")
        cells_deleted = safe_get(state, "cells_deleted")
        has_positioning = safe_get(state, "positioning_info") is not None
        restart_required = safe_get(state, "restart_required")
        restart_done = restart_required == False  # Tool sets this to False after restart
        generated_code = safe_get(state, "generated_code")

        if not backtrack_recovery_done:
            logger.debug("[DET ROUTER] BACKTRACK: determining recovery strategy")
            return "backtrack_recovery"

        if not cells_to_delete:
            logger.debug("[DET ROUTER] BACKTRACK: selecting cells to delete")
            return "cell_selection_deletion"

        if not cells_deleted:
            logger.debug("[DET ROUTER] BACKTRACK: deleting cells")
            return "cell_deletion"

        if not has_positioning:
            logger.debug("[DET ROUTER] BACKTRACK: positioning after deletion")
            return "cell_positioning"

        if restart_required and not restart_done:
            logger.debug("[DET ROUTER] BACKTRACK: restarting kernel")
            return "restart_and_rerun_prompt"

        if not generated_code:
            logger.debug("[DET ROUTER] BACKTRACK: generating new code")
            return "code_generation_with_guidance"

        logger.debug("[DET ROUTER] BACKTRACK: complete, exiting")
        return "complete"

    def _route_smart(self, state: dict) -> str:
        """Determine next action based on validated state."""
        # Helper to safely get values from dict or Pydantic model
        def safe_get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        task_list = safe_get(state, "task_list", {})
        tasks = safe_get(task_list, "tasks", [])

        # SMART ROUTER: Simple routing based on task status
        # Does NOT check retrieval_queries (that's only in deterministic router)
        if not tasks:
            logger.info("[ROUTER] No tasks → plan_tasks")
            return "plan_tasks"

        # SECOND check: If autonomous_mode_continue=False, this is the initial user request
        # We should plan tasks, activate first task, then exit to let UI display task list
        # before autonomous execution begins in subsequent iterations
        if not safe_get(state, "autonomous_mode_continue"):
            all_pending = all(safe_get(t, "status") == "pending" for t in tasks)
            has_active = any(safe_get(t, "status") == "active" for t in tasks)

            if all_pending:
                # Just finished planning, activate first task
                return "manage_progress"
            elif has_active:
                # Just activated first task, exit to let UI display task list
                return "complete"
            else:
                # User manually stopped with tasks in progress
                return "complete"

        # Check if all tasks complete
        all_complete = all(safe_get(t, "status") == "completed" for t in tasks)
        if all_complete:
            return "complete"

        if safe_get(state, "error_context") or safe_get(state, "last_execution_failed"):
            if self._should_backtrack(state):
                return "backtrack"
            return "handle_error"

        # If code was generated but not yet executed, execute it
        if safe_get(state, "generated_code"):
            return "execute_cell"

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

        Clears transient state fields that aren't provided by the UI layer.
        This mimics kai_dev behavior where only OrchestratorState fields persisted.

        Args:
            message: User query
            context: Dict with all context fields including session_metadata, autonomous_mode, etc.
        """
        # Extract session metadata if present
        session_metadata = context.pop("session_metadata", {})
        autonomous_mode = context.get("autonomous_mode", False)

        # Build initial state with smart persistence:
        # - PERSISTENT fields: Only pass if UI explicitly provides (first iteration or override)
        #   Otherwise let LangGraph checkpointer provide from previous iteration
        # - TRANSIENT fields: Always pass and clear stale values
        # - Other fields: Always pass (notebook context, etc.)

        initial_state: KaiState = {
            "user_query": message,
            "messages": [{"role": "user", "content": message}] if message else [],
            **session_metadata
        }

        # Add fields from context with persistence-aware logic
        for key, value in context.items():
            if key in TRANSIENT_STATE_FIELDS:
                # Always include transient fields from context
                initial_state[key] = value
            elif key in PERSISTENT_STATE_FIELDS:
                # Only include persistent fields if UI explicitly provides them
                # (non-empty value = first iteration or intentional override)
                # Otherwise let checkpointer provide from previous iteration
                if value is not None and value != {} and value != [] and value != "":
                    initial_state[key] = value
                    logger.debug(f"[ORCHESTRATOR] Persistent field '{key}' provided by UI: {value if key != 'retrieval_queries' else f'list[{len(value)}]'}")
                # else: omit from initial_state, checkpointer will provide it
            else:
                # Other fields (notebook structure, execution history, etc.) - always pass
                initial_state[key] = value

        # Initialize PERSISTENT fields ONLY if no checkpoint exists
        #
        # Problem: LangGraph merges initial_state with checkpoint, but initial_state takes precedence.
        # If we always include reference_workflow_content: {} in initial_state, it overrides the
        # checkpoint value on every invocation!
        #
        # Solution: Only initialize PERSISTENT fields on the FIRST invocation (no checkpoint exists).
        # On subsequent invocations, omit them from initial_state so checkpoint provides them.
        config = {
            "configurable": {"thread_id": session_metadata.get("session_id", "default")},
        }

        # Check if checkpoint exists for this thread
        checkpoint = None
        if autonomous_mode:
            try:
                checkpoint = self.autonomous_graph.get_state(config)
            except:
                pass  # No checkpoint yet

        # Only initialize if no checkpoint (first invocation for this thread)
        if checkpoint is None or not hasattr(checkpoint, 'values') or not checkpoint.values:
            logger.debug(f"[ORCHESTRATOR] No checkpoint found - initializing PERSISTENT fields")
            rag_enabled = context.get("rag_enabled", False)
            for field in PERSISTENT_STATE_FIELDS:
                if field not in initial_state:  # Don't overwrite agent-provided values
                    if field in {"task_list", "reference_workflow_content"}:
                        initial_state[field] = {}
                    elif field == "retrieval_queries":
                        # Use user message as initial retrieval query when RAG enabled
                        # This matches kai_dev behavior where first search uses user message
                        if rag_enabled and message:
                            initial_state[field] = [message]
                            logger.info(f"[ORCHESTRATOR] Setting initial retrieval query from user message")
                        else:
                            initial_state[field] = []
                    elif field == "excluded_workflows":
                        initial_state[field] = []
                    elif field in {"reference_workflow_ids", "planning_phase"}:
                        initial_state[field] = None
                    elif field in {"auto_mode_first_execution_done", "is_reasoning_task"}:
                        initial_state[field] = False
                    elif field in {"workflow_retrieval_iteration"}:
                        initial_state[field] = 0
                    elif field in {"task_planning_iteration"}:
                        initial_state[field] = -1  # Start at -1, first increment gives 0 (matching kai_dev's for loop range(10))
                    elif field in {"active_task", "active_task_objective", "next_pending_task_objective"}:
                        initial_state[field] = None  # Set by mark_next_task_active
        else:
            # Checkpoint exists - read PERSISTENT fields from checkpoint into initial_state
            # This is necessary because we read values BEFORE astream merges checkpoint with initial_state
            logger.debug(f"[ORCHESTRATOR] Checkpoint exists - restoring PERSISTENT fields from checkpoint")
            checkpoint_values = checkpoint.values
            for field in PERSISTENT_STATE_FIELDS:
                if field not in initial_state and field in checkpoint_values:
                    initial_state[field] = checkpoint_values[field]
                    if field == "auto_mode_first_execution_done":
                        logger.debug(f"[ORCHESTRATOR] Restored '{field}' from checkpoint: {checkpoint_values[field]}")

        # Clear transient state fields that weren't provided by UI
        # This prevents stale state from previous iterations causing router loops
        for field in TRANSIENT_STATE_FIELDS:
            if field not in context and field not in session_metadata:
                # Determine appropriate default value
                if field == "critique_iteration":
                    initial_state[field] = 0
                elif field in {"last_execution_failed", "task_completion_analyzed",
                              "next_task_activated", "tasks_updated", "update_approved",
                              "restart_required", "cells_deleted", "backtrack_recovery_done"}:
                    initial_state[field] = False
                elif field == "snippet_retrieval_query":
                    initial_state[field] = []  # Empty list for snippet queries
                else:
                    initial_state[field] = None

        # Initialize required fields for prompts if not provided
        if "execution_history" not in initial_state:
            initial_state["execution_history"] = []
        if "conversation_history" not in initial_state:
            initial_state["conversation_history"] = []
        if "notebook_cells" not in initial_state:
            initial_state["notebook_cells"] = []
        if "notebook_structure" not in initial_state:
            initial_state["notebook_structure"] = {'totalCells': 0, 'allCells': []}
        if "current_cell" not in initial_state:
            initial_state["current_cell"] = ""
        if "current_cell_index" not in initial_state:
            initial_state["current_cell_index"] = 0
        if "use_critique" not in initial_state:
            initial_state["use_critique"] = True  # Default True, matching kai_dev line 228

        # Determine if planning is needed (autonomous mode without tasks)
        # Check checkpoint first, then initial_state (for first iteration)
        has_tasks = False
        if checkpoint and hasattr(checkpoint, 'values') and checkpoint.values:
            # Subsequent iterations - check checkpoint
            has_tasks = bool(checkpoint.values.get("task_list", {}).get("tasks"))
        else:
            # First iteration - check initial_state
            has_tasks = bool(initial_state.get("task_list", {}).get("tasks"))

        needs_planning = autonomous_mode and not has_tasks

        # Execute graph(s) with UI communication after each node
        try:
            start_time = time.time()

            # config already defined above at line 884

            # Only set recursion_limit if explicitly provided at initialization (otherwise use LangGraph's default of 25)
            if self.graph_recursion_limit is not None:
                config["recursion_limit"] = self.graph_recursion_limit

            # Track graph steps to detect loops
            graph_step_count = 0
            tool_call_sequence = []  # Track which tools are called

            # ===== PHASE 1: Planning (if needed) =====
            if needs_planning:
                self._send_message("[KAI] Starting autonomous iteration - planning workflow")
                async for output in self.planning_graph.astream(initial_state, {"recursion_limit": config.get("recursion_limit", 25)}):
                    if self.is_cancelled:
                        self._send_message("[KAI] Planning cancelled by user")
                        return

                    for node_name, node_output in output.items():
                        if node_output:
                            graph_step_count += 1
                            tool_call_sequence.append(f"PLAN:{node_name}")
                            logger.debug(f"[PLANNING STEP {graph_step_count}] Tool: {node_name}")

                            # Send UI update if tool returned result
                            if '_last_tool_result' in node_output:
                                tool_data = node_output['_last_tool_result']
                                result = tool_data['result']
                                exec_context = tool_data['exec_context']

                                # Send tool result to UI
                                if result.output_ui:
                                    from .ui_communicator import VscodeInputContext
                                    vscode_context = VscodeInputContext(
                                        session_id=session_metadata.get("session_id", ""),
                                        inputs=exec_context.inputs
                                    )
                                    await self.vscode.send_tool_result(result, vscode_context)

                            # Update initial_state with planning output for execution phase
                            if node_output:
                                initial_state.update(node_output)

                planning_duration = time.time() - start_time
                self._send_message(f"[KAI] Planning workflow completed in {planning_duration:.3f}s")
                logger.info("[ORCHESTRATOR] Planning phase complete")

            # ===== PHASE 2: Execution =====
            # Select execution graph
            if autonomous_mode:
                graph = self.autonomous_graph
                # Determine which execution phase we're in for console messages
                auto_first_exec_done = initial_state.get("auto_mode_first_execution_done", False)
                auto_continue = initial_state.get("autonomous_mode_continue", False)

                self._send_message(f"[KAI] DEBUG: auto_first_exec_done={auto_first_exec_done}, auto_continue={auto_continue}")

                if not auto_continue and not auto_first_exec_done:
                    # First iteration after planning (will just activate task and exit)
                    pass  # No message - planning message already shown
                elif not auto_first_exec_done:
                    # First execution iteration
                    self._send_message("[KAI] Starting autonomous iteration - first execution iteration")
                else:
                    # Standard execution iterations
                    self._send_message("[KAI] Starting autonomous iteration - continuation workflow")
            else:
                graph = self.regular_graph

            async for output in graph.astream(initial_state, config):
                if self.is_cancelled:
                    self._send_message("[KAI] Workflow cancelled by user")
                    break

                for node_name, node_output in output.items():
                    if node_output:
                        graph_step_count += 1
                        tool_call_sequence.append(f"EXEC:{node_name}")
                        logger.debug(f"[EXEC STEP {graph_step_count}/{self.graph_recursion_limit if self.graph_recursion_limit else 'unlimited'}] Tool: {node_name}")

                        # Send UI update if tool returned result
                        if '_last_tool_result' in node_output:
                            tool_data = node_output['_last_tool_result']
                            result = tool_data['result']
                            exec_context = tool_data['exec_context']

                            # Send tool result to UI
                            if result.output_ui:
                                from .ui_communicator import VscodeInputContext
                                vscode_context = VscodeInputContext(
                                    session_id=session_metadata.get("session_id", ""),
                                    inputs=exec_context.inputs
                                )
                                self._send_message(f"[KAI] Sending tool result: type={result.output_type.value}, has_code={isinstance(result.output_ui, dict) and 'code' in result.output_ui}")
                                await self.vscode.send_tool_result(result, vscode_context)

            duration = time.time() - start_time

            # Log tool call sequence for debugging loops (only in debug mode)
            if graph_step_count > 0:
                from collections import Counter
                tool_counts = Counter(tool_call_sequence)
                logger.debug(f"[ORCHESTRATOR] Tool call counts: {dict(tool_counts)}")
                logger.debug(f"[ORCHESTRATOR] Total graph steps: {graph_step_count}")

            # Get final state (only autonomous_graph has checkpointer)
            if autonomous_mode:
                final_state = await graph.aget_state(config)
                state_values = final_state.values if hasattr(final_state, 'values') else final_state
            else:
                # Regular mode doesn't need state retrieval - no persistent state
                state_values = {}

            # Send completion status with descriptive messages
            if autonomous_mode:
                tasks = state_values.get("task_list", {}).get("tasks", [])
                all_complete = all(t.get("status") == "completed" for t in tasks)

                # Warn if no tasks exist (may indicate early termination)
                if not tasks:
                    logger.warning(f"[ORCHESTRATOR] No tasks in task_list - this may indicate early termination!")

                if all_complete:
                    self._send_message(f"[KAI] All tasks completed! Autonomous iteration finished (completed in {duration:.3f}s)")
                    await self.vscode.send_workflow_result(
                        field="auto_loop_update",
                        state="LOOP_COMPLETE"
                    )
                else:
                    # Determine phase and send appropriate message
                    auto_first_exec_done = state_values.get("auto_mode_first_execution_done", False)
                    last_execution_failed = state_values.get("last_execution_failed", False)
                    retry_objective = state_values.get("retry_objective")
                    recovery_objective = state_values.get("recovery_objective")

                    is_backtracking = recovery_objective is not None
                    is_standard_retry = (last_execution_failed or retry_objective) and not is_backtracking

                    if not auto_first_exec_done:
                        self._send_message(f"[KAI] END: autonomous iteration (completed in {duration:.3f}s)")
                    elif is_standard_retry:
                        self._send_message(f"[KAI] END: autonomous iteration - standard retry (with error: {last_execution_failed}) (completed in {duration:.3f}s)")
                    else:
                        self._send_message(f"[KAI] END: autonomous iteration - standard step (completed in {duration:.3f}s)")

                    await self.vscode.send_workflow_result(
                        field="auto_loop_update",
                        state="LOOP_INCOMPLETE"
                    )
            else:
                # Regular mode (non-autonomous)
                await self.vscode.send_workflow_result(
                    field="regular_chat_update",
                    state="STEP_COMPLETE"
                )

            # Send generic completion message for all modes (tests rely on this)
            self._send_message(f"[KAI] Request processing completed in {duration:.3f}s")

            # Return final state to caller (agent.py) so UI can get updated task_list
            return state_values

        except Exception as e:
            # Enhanced error logging for recursion limit errors
            if "recursion limit" in str(e).lower():
                from collections import Counter
                tool_counts = Counter(tool_call_sequence)
                logger.error(f"Graph recursion limit hit after {graph_step_count} steps!")
                logger.error(f"Tool call counts: {dict(tool_counts)}")
                logger.error(f"Last 20 tools called: {' → '.join(tool_call_sequence[-20:])}")
                # Find loops - check if last few tools are repeating
                if len(tool_call_sequence) >= 4:
                    last_4 = tool_call_sequence[-4:]
                    if last_4[0] == last_4[2] and last_4[1] == last_4[3]:
                        logger.error(f"DETECTED LOOP: {last_4[0]} → {last_4[1]} repeating!")
            logger.error(f"Graph execution failed: {e}", exc_info=True)
            raise

    async def _handle_execution_progress_check(
        self, context: Dict[str, Any], session_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle execution progress monitoring for long-running cells.

        Args:
            context: Dict containing:
                - current_cell: Code of the currently executing cell
                - elapsed_time: Seconds since execution started
                - partial_outputs: Outputs captured so far
                - active_task: Description of the active task
            session_metadata: Session metadata

        Returns:
            Dict with:
                - action: "continue" or "terminate"
                - feedback: Explanation for the decision
        """
        self._send_message(f"[KAI] Analyzing execution progress ({context.get('elapsed_time', 0)}s elapsed)")

        # Create execution context for monitoring
        from .execution_context import ExecutionContext, ExecutionInputs
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="",  # Will be populated by tool's _modify_user_query
                context=context,
                task_list={},
                backtracking_context=None,
                excluded_workflows=[]
            ),
            session_metadata=session_metadata or {}
        )

        # Execute monitoring tool
        monitor_tool = self.tools.get("execution_monitor")
        if not monitor_tool:
            # If monitoring tool not available, default to continue
            return {
                "action": "continue",
                "feedback": "Monitoring tool not available, allowing execution to continue"
            }

        result = await monitor_tool.execute(exec_context)

        # Extract decision from workflow output
        workflow_output = result.output_workflow or {}
        action = workflow_output.get("action", "continue")
        feedback = workflow_output.get("feedback", "")

        self._send_message(f"[KAI] Execution monitor decision: {action.upper()}")
        self._send_message(f"[KAI] Feedback: {feedback}")

        return {
            "action": action,
            "feedback": feedback
        }
