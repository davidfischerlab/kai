"""LangGraph orchestrator with TypedDict state.

See kai.core.orchestration.graphs.main for the iteration model documentation
explaining how LangGraph checkpoints interact with Jupyter/VSCode execution.
"""

import asyncio
import time
from typing import Any, Dict, Optional

from kai.core.orchestration.state import KaiState, get_transient_defaults, initialize_state
from kai.core.tools import create_consolidated_tools
from kai.core.llm_interface import LLMInterface
from kai.retrieval import ChromaDbManager
from kai.core.orchestration.ui_communicator import UICommunicator
from kai.core.orchestration.routers import route_planning_phase
from kai.core.persistence import (
    get_checkpointer_for_settings,
    get_checkpoint_mode,
    clear_session_checkpoints,
    CheckpointMode,
)
from kai.core.orchestration.nodes import (
    increment_task_planning_iteration,
    section_check_position,
    section_route_from_position_check,
    section_execute_cell,
    section_check_execution,
    section_route_from_execution_check,
    section_advance_cell,
    section_route_fix_operation,
    section_apply_delete,
    section_apply_replace,
    section_apply_insert,
    section_check_fix_result,
    section_route_from_fix_check,
    section_complete_success,
    section_complete_failure,
)
from kai.core.orchestration.graphs import (
    build_main_graph,
    build_main_graph_for_studio,
    build_planning_subgraph,
    build_execution_subgraph,
    build_execution_subgraph_for_studio,
    build_regular_subgraph,
    build_section_execution_subgraph,
    AUTONOMOUS_TOOLS,
)
from kai.utils import setup_logger, safe_get

logger = setup_logger(__name__)


class LangGraphOrchestrator:

    def __init__(
        self,
        llm_interface: LLMInterface,
        knowledge_base: ChromaDbManager,
        ui_communicator: UICommunicator = None,
        graph_recursion_limit: Optional[int] = 100,
        max_task_planning_iterations: int = 10,
        max_workflow_retrieval_iterations: int = 2,
        checkpointer: Optional[Any] = None,
    ):
        self.llm = llm_interface
        self.knowledge_base = knowledge_base
        self.vscode = ui_communicator or UICommunicator()
        self.graph_recursion_limit = graph_recursion_limit
        self.max_task_planning_iterations = max_task_planning_iterations
        self.max_workflow_retrieval_iterations = max_workflow_retrieval_iterations

        # Set up checkpointer for state persistence
        # AsyncSqliteSaver enables restart/resume, MemorySaver is in-memory only
        from kai.config.settings import settings
        if checkpointer is not None:
            self.checkpointer = checkpointer
        else:
            # get_checkpointer_for_settings is async
            # Handle case where event loop may already be running (e.g., Jupyter, pytest-asyncio)
            self.checkpointer = self._init_checkpointer_sync(settings)

        # Checkpoint mode: TRANSIENT clears on completion, PERSISTENT keeps
        self.checkpoint_mode = get_checkpoint_mode(settings)

        self.tools = create_consolidated_tools(llm_interface, knowledge_base)

        # Build subgraphs using stateless graph builders with dependencies
        self._planning_subgraph = build_planning_subgraph(
            tools=self.tools,
            increment_task_planning_iteration_node=self._increment_task_planning_iteration_node,
            route_planning_phase=self._route_planning_phase,
        )
        self._execution_subgraph = build_execution_subgraph(
            tools=self.tools,
            send_message=self._send_message,
            send_task_list=self.vscode.send_task_list_update,
        )
        self._regular_subgraph = build_regular_subgraph(tools=self.tools)
        self._section_subgraph = build_section_execution_subgraph(
            llm=self.llm,
            section_check_position_node=section_check_position,
            section_route_from_position_check=section_route_from_position_check,
            section_execute_cell_node=section_execute_cell,
            section_check_execution_node=section_check_execution,
            section_route_from_execution_check=section_route_from_execution_check,
            section_advance_cell_node=section_advance_cell,
            section_route_fix_operation=section_route_fix_operation,
            section_apply_delete_node=section_apply_delete,
            section_apply_replace_node=section_apply_replace,
            section_apply_insert_node=section_apply_insert,
            section_check_fix_result_node=section_check_fix_result,
            section_route_from_fix_check=section_route_from_fix_check,
            section_complete_success_node=section_complete_success,
            section_complete_failure_node=section_complete_failure,
        )

        # Build main graph that orchestrates subgraphs
        # Pass checkpointer for state persistence (enables restart/resume)
        self.main_graph = build_main_graph(
            planning_subgraph=self._planning_subgraph,
            execution_subgraph=self._execution_subgraph,
            regular_subgraph=self._regular_subgraph,
            route_mode=self._route_mode,
            route_after_planning=self._route_after_planning,
            checkpointer=self.checkpointer,
        )

        self.is_cancelled = False

    def _init_checkpointer_sync(self, settings):
        """Initialize checkpointer from sync context.

        Handles the async get_checkpointer_for_settings call from sync __init__.
        Falls back to MemorySaver if async initialization fails (e.g., in pytest-asyncio).
        """
        from langgraph.checkpoint.memory import MemorySaver

        # Check if we're in a running event loop
        try:
            asyncio.get_running_loop()
            # We're inside an event loop (e.g., pytest-asyncio, Jupyter)
            # Cannot use asyncio.run() or run_until_complete()
            # Fall back to MemorySaver for in-memory checkpointing
            logger.debug("[PERSISTENCE] Running event loop detected, using MemorySaver")
            return MemorySaver()
        except RuntimeError:
            # No running event loop - safe to use asyncio.run()
            pass

        try:
            return asyncio.run(get_checkpointer_for_settings(settings))
        except Exception as e:
            logger.warning(f"[PERSISTENCE] Failed to initialize AsyncSqliteSaver: {e}, using MemorySaver")
            return MemorySaver()

    def set_graph_recursion_limit(self, limit: int):
        """Set graph recursion limit for testing."""
        self.graph_recursion_limit = limit

    async def clear_session_on_completion(self, session_id: str) -> bool:
        """Clear session checkpoints if in TRANSIENT mode.

        Called when a session completes successfully. In TRANSIENT mode,
        checkpoints are cleared to save space. In PERSISTENT mode, they
        are kept for later analysis or resumption.

        Args:
            session_id: Session/thread ID to clear

        Returns:
            True if cleared (or not needed), False on error
        """
        if self.checkpoint_mode == CheckpointMode.PERSISTENT:
            logger.debug(f"[PERSISTENCE] Keeping checkpoints (PERSISTENT mode)")
            return True

        # TRANSIENT mode - clear checkpoints
        return await clear_session_checkpoints(self.checkpointer, session_id)

    def get_graphs_for_studio(self):
        """Get graphs compiled WITHOUT checkpointers for LangGraph Studio.

        Studio provides its own persistence layer, so we must not include
        custom checkpointers when exposing graphs for Studio visualization.

        Returns:
            Dict with 'main', 'planning', 'execution' compiled graphs
        """
        # Rebuild graphs without checkpointers using stateless builders
        execution_for_studio = build_execution_subgraph_for_studio(
            tools=self.tools,
            send_message=self._send_message,
            send_task_list=self.vscode.send_task_list_update,
        )
        return {
            'main': build_main_graph_for_studio(
                planning_subgraph=self._planning_subgraph,
                execution_subgraph=execution_for_studio,
                regular_subgraph=self._regular_subgraph,
                route_mode=self._route_mode,
                route_after_planning=self._route_after_planning,
            ),
            'planning': self._planning_subgraph,  # Already no checkpointer
            'execution': execution_for_studio,
        }

    # =========================================================================
    # Node Wrappers (call stateless node functions with dependencies)
    # =========================================================================

    async def _increment_task_planning_iteration_node(self, state: dict) -> dict:
        """Wrapper for stateless increment_task_planning_iteration function."""
        return await increment_task_planning_iteration(
            state,
            max_task_planning_iterations=self.max_task_planning_iterations,
            send_message=self._send_message,
        )

    # =========================================================================
    # Communication
    # =========================================================================

    def _send_message(self, message: str):
        """Send console message via VSCode communicator."""
        self.vscode.send_console_message(message)

    async def _send_tool_result_to_ui(
        self,
        tool_result_data: dict,
        session_metadata: dict
    ) -> None:
        """Send tool result to UI.

        Args:
            tool_result_data: Dict with 'result' and 'state'
            session_metadata: Session metadata for context
        """
        from .ui_communicator import VscodeInputContext

        result = tool_result_data['result']
        state = tool_result_data['state']

        if result.output_ui:
            vscode_context = VscodeInputContext.from_state(state)
            await self.vscode.send_tool_result(result, vscode_context)

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

    # =========================================================================
    # Routing methods that map between nodes in workflows:
    # =========================================================================

    def _route_mode(self, state: KaiState) -> str:
        """Route to appropriate subgraph based on mode and state.

        Routing Logic:
        - If not autonomous_mode: → regular
        - If autonomous_mode and no tasks: → planning
        - If autonomous_mode and has tasks: → execution
        """
        autonomous = safe_get(state, "autonomous_mode", False)
        task_list = safe_get(state, "task_list", {})
        has_tasks = bool(safe_get(task_list, "tasks"))

        if not autonomous:
            logger.debug("[ROUTE_MODE] → regular (non-autonomous)")
            return "regular"
        elif not has_tasks:
            logger.debug("[ROUTE_MODE] → planning (autonomous, no tasks)")
            return "planning"
        else:
            logger.debug("[ROUTE_MODE] → execution (autonomous, has tasks)")
            return "execution"

    def _route_after_planning(self, state: dict) -> str:
        """Route after planning subgraph completes.

        If tasks were generated, continue to execution.
        Otherwise, complete (edge case - planning failed to generate tasks).
        """
        task_list = safe_get(state, "task_list", {})
        has_tasks = bool(safe_get(task_list, "tasks"))

        if has_tasks:
            logger.debug("[ROUTE_AFTER_PLANNING] → execution (tasks generated)")
            return "execution"
        else:
            logger.warning("[ROUTE_AFTER_PLANNING] → complete (no tasks generated)")
            return "complete"

    def _route_planning_phase(self, state: dict) -> str:
        """Wrapper for stateless route_planning_phase function."""
        return route_planning_phase(
            state,
            max_task_planning_iterations=self.max_task_planning_iterations,
            send_message=self._send_message,
            log_task_list_summary=self._log_task_list_summary,
        )

    async def process_request(
        self,
        message: str,
        context: dict
    ) -> None:
        """
        Process request using single main graph with subgraphs.

        The main graph handles routing between planning/execution/regular modes
        automatically based on state. LangGraph's checkpointer handles persistence.

        Args:
            message: User query
            context: Dict with session_metadata, autonomous_mode, etc.
        """
        # Extract session metadata
        session_metadata = context.pop("session_metadata", {})
        autonomous_mode = context.get("autonomous_mode", False)

        # Build initial state
        initial_state: KaiState = {
            "user_query": message,
            "messages": [{"role": "user", "content": message}] if message else [],
            **session_metadata,
            **context
        }

        # Config for LangGraph with thread-based checkpointing
        session_id = session_metadata.get("session_id", "default")
        config = {"configurable": {"thread_id": session_id}}

        # Check for existing checkpoint
        checkpoint = None
        try:
            checkpoint = self.main_graph.get_state(config)
        except Exception:
            pass  # No checkpoint yet

        is_first_invocation = (
            checkpoint is None or
            not hasattr(checkpoint, 'values') or
            not checkpoint.values
        )

        # Initialize state fields
        initial_state = initialize_state(
            initial_state,
            context,
            checkpoint,
            is_first_invocation,
            max_task_planning_iterations=self.max_task_planning_iterations,
            max_workflow_retrieval_iterations=self.max_workflow_retrieval_iterations,
        )

        # Increment iteration_id for transient state tracking
        current_iteration = initial_state.get("iteration_id", 0)
        initial_state["iteration_id"] = current_iteration + 1

        # Reset transient state fields at the start of each iteration
        # With reducers, this ensures stale values don't persist
        # IMPORTANT: Transient fields should ALWAYS be reset EXCEPT when explicitly
        # passed from the UI (in context). This prevents checkpoint values from
        # persisting across iterations while still allowing UI to pass error info.
        # Fields the UI can pass: last_execution_failed, error_message
        ui_preserved_fields = {"last_execution_failed", "error_message"}
        for key, default_value in get_transient_defaults().items():
            # Only preserve if: (1) field is in UI preserve list AND (2) UI explicitly passed a value
            if key in ui_preserved_fields and key in context:
                continue  # Keep the UI-provided value
            # Reset all other transient fields to defaults
            initial_state[key] = default_value  # type: ignore[literal-required]

        # Execute main graph
        try:
            start_time = time.time()

            if self.graph_recursion_limit is not None:
                config["recursion_limit"] = self.graph_recursion_limit

            graph_step_count = 0
            tool_call_sequence = []

            # Log mode for debugging
            mode = self._route_mode(initial_state)
            self._send_message(f"[KAI] Starting request - mode: {mode}")

            # Single graph handles planning → execution → completion
            # Use subgraphs=True to stream intermediate node outputs from nested subgraphs
            # (planning, execution, regular) - otherwise we only see subgraph completion
            async for namespace, output in self.main_graph.astream(
                initial_state, config, subgraphs=True
            ):
                if self.is_cancelled:
                    self._send_message("[KAI] Workflow cancelled by user")
                    break

                for node_name, node_output in output.items():
                    if node_output:
                        graph_step_count += 1
                        tool_call_sequence.append(node_name)
                        logger.debug(
                            f"[STEP {graph_step_count}] Node: {node_name} (namespace: {namespace})"
                        )

                        # Send UI update if tool returned result
                        # Only process from inside subgraphs (namespace non-empty)
                        # When namespace is empty (), it's the main graph seeing
                        # subgraph completion - we already processed that tool result
                        if '_last_tool_result' in node_output and namespace:
                            await self._send_tool_result_to_ui(
                                node_output['_last_tool_result'],
                                session_metadata
                            )

            duration = time.time() - start_time

            # Log summary
            if graph_step_count > 0:
                from collections import Counter
                tool_counts = Counter(tool_call_sequence)
                logger.debug(f"[ORCHESTRATOR] Tool counts: {dict(tool_counts)}")
                logger.debug(f"[ORCHESTRATOR] Total steps: {graph_step_count}")

            # Get final state
            final_state = await self.main_graph.aget_state(config)
            state_values = (
                final_state.values if hasattr(final_state, 'values')
                else final_state
            )

            # Send completion status with descriptive messages
            if autonomous_mode:
                # Check if a node (mark_reasoning_completed, autonomous_mark_completion)
                # already signaled LOOP_COMPLETE in the state
                all_complete = state_values.get("auto_loop_update") == "LOOP_COMPLETE"

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

        # Create state dict for monitoring tool
        # Merge context fields with session metadata at top level
        state = {
            "user_query": "",  # Will be populated by tool's _modify_user_query
            "task_list": {},
            "backtracking_context": None,
            "excluded_workflows": [],
            **context,  # Include all context fields
            **(session_metadata or {})  # Include session metadata at top level
        }

        # Execute monitoring tool
        monitor_tool = self.tools.get("execution_monitor")
        if not monitor_tool:
            # If monitoring tool not available, default to continue
            return {
                "action": "continue",
                "feedback": "Monitoring tool not available, allowing execution to continue"
            }

        result = await monitor_tool.execute(state)

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
