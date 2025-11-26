"""Workflow orchestrator for managing tool pipelines and execution."""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs, BacktrackingContext
from kai.core.utils import format_task_list

from .base_tool import BaseTool, ToolResult, ToolOutputType
from .prompt_tools import (
    CodeGenerationTool, CodeGenerationWithGuidanceTool,
    ErrorRecoveryTool, CodeUpdateTool, AutonomousMarkCompletionTool,
    AutonomousUpdateTasksTool, AutonomousUpdateCritiqueTool, AutoLoopIntentClassificationTool,
    QuestionAnsweringTool, CellPositioningTool, IntentClassificationTool,
    CellSelectionDeletionTool, BacktrackRecoveryTool, ExecutionMonitorTool,
    ReasoningCritiqueTool, ReasoningResponseWithGuidanceTool, RestartAndRerunTool,
    ReferenceWorkflowSelectionTool, ReferenceWorkflowSelectionOnlyTool, ReferenceWorkflowCellSelectionTool,
    SectionCodeReviewTool, TaskListGenerationTool, TaskListCritiqueTool
)
from kai.retrieval.snippets.storage.chromadb_manager import ChromaDbManager
from kai.retrieval.workflow_summaries.notebook_selector import NotebookSelector
from kai.retrieval.workflow_summaries.summary_search import WorkflowSummaryRag
from kai.retrieval.workflow_summaries.notebook_storage import NotebookStorage
from kai.core.prompt_manager import PromptScenario
from .deterministic_tools import CellDeletionTool, CodeRetrievalTool, MarkNextTaskActiveTool, ReferenceWorkflowQueryPreparationTool, FilterUnusedReferenceWorkflowsTool
from ..llm_interface import LLMInterface
from .vscode_communicator import VSCodeCommunicator
from kai.utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class OrchestratorState:
    """
    State management for workflow orchestrator.
    These are stateful variables that persist across iterations.
    Everything else is managed in execution context and is cleared with the end of an iteration.
    Accordingly, this object only contains global variables that describe the overall analysis plan.

    ## Fields:
    - task_list: Central task list that is updated in each iteration.
    - reference_workflow_content: String representation of reference workflows (notebooks).
    - excluded_workflows: List of workflow IDs that returned empty cell indices in past iterations.
    """
    task_list: Dict[str, Any] = field(default_factory=dict)
    reference_workflow_ids: str = ""
    reference_workflow_content: Dict[str, str] = field(default_factory=dict)  # {internal_id: content_string}
    auto_mode_first_execution: bool = True
    excluded_workflows: List[str] = field(default_factory=list)


class WorkflowOrchestrator:
    """Orchestrates tool execution chains with context sharing between tools.
    
    ## Tool Chaining
    Tools are chained via `execute_workflow(tool_names, context)`. Each tool's `output_workflow` 
    data is propagated to subsequent tools via `context.inputs.context.update()`.
    
    ## State Management
    - `self.state`: Persistent session state (task_list)
    - `ExecutionContext`: Per-workflow state that flows between tools
    - State synchronization: `self.state.task_list ↔ context.inputs.task_list`
    
    ## Development Notes
    - Register new tools in `_initialize_tools()`
    - Use `result.output_workflow` to share data between tools
    - State-accessing methods take ExecutionContext parameter
    - VSCode communication delegated to VSCodeCommunicator
    """
    llm_interface: LLMInterface
    knowledge_base: ChromaDbManager
    tools: Dict[str, BaseTool]
    vscode: VSCodeCommunicator
    
    def __init__(self, llm_interface, knowledge_base: 'ChromaDbManager' = None, 
                 vscode_communicator: 'VSCodeCommunicator' = None):
        self.llm = llm_interface
        self.knowledge_base = knowledge_base
        
        # Use provided VSCode communicator or create new one
        self.vscode = vscode_communicator or VSCodeCommunicator()
        
        # Initialize tool registry with LLM pool
        self.tools = self._initialize_tools()
        
        # Simple state management with minimal state manager
        self.state = OrchestratorState()

        # Cancellation flag for stopping workflows
        self.is_cancelled = False

    def _send_message(self, message: str):
        """Send message to VSCode console through stdout (delegated to VSCodeCommunicator)."""
        self.vscode.send_console_message(message)
    
    async def _are_all_tasks_completed(self, context: ExecutionContext) -> bool:
        """Check if all tasks in the task list are completed."""
        if not context.inputs.task_list:
            return False
        
        tasks = context.inputs.task_list['tasks']
        if not tasks:  # Empty task list means no work to complete
            return False
            
        for task in tasks:
            if task.get('status') != 'completed':
                return False
        return True
    
    def _get_active_task(self, context: ExecutionContext):
        """Get the currently active task description from the task list."""
        if not context.inputs.task_list or 'tasks' not in context.inputs.task_list:
            return ""

        for task in context.inputs.task_list['tasks']:
            if task.get('status') == 'active':
                return task.get('task', '')
        return ""
    
    def reset_states(self):
        """Reset orchestrator states that are persisted across iterations."""
        self.state = OrchestratorState()
    
    async def process_request(
            self,
            message: str,
            context: Dict[str, Any]
    ) -> None:
        """Process a request directly through the orchestrator."""
        # Extract data from context
        session_metadata = context["session_metadata"]
        autonomous_mode = context["autonomous_mode"]
        
        # Store context for tools to create PromptContext objects
        self._context = context
        
        if autonomous_mode:
            await self._handle_autonomous_unified(
                message, session_metadata, context,
            )

        else:
            # Regular intent-based processing
            await self._handle_regular_request(
                message, session_metadata, context
            )
    
    async def _handle_autonomous_unified(
        self, message: str, session_metadata: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """
        Unified autonomous handler: Routes to planning or execution based on intent classification.

        Flow:
        1. If no task list → force planning mode (initial iteration)
        2. Otherwise → use AutoLoopIntentClassificationTool to determine intent
        3. Route to planning if PLANNING or TASK_LIST_MODIFICATION
        4. Route to execution if APPROVAL or CODE_IMPLEMENTATION_FEEDBACK

        Planning always ends with LOOP_INCOMPLETE_REQUIRE_FEEDBACK to get user feedback.

        @see _handle_autonomous_planning() for task generation/update
        @see _handle_autonomous_execution() for task execution
        """
        # Create execution context for intent classification or planning
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query=message,
                context=context,
                task_list=self.state.task_list,
                backtracking_context=None,
                excluded_workflows=self.state.excluded_workflows
            ),
            session_metadata=session_metadata
        )
        # Add persistant objects to context:
        if self.state.reference_workflow_content:
            exec_context.inputs.context['reference_workflow_ids'] = self.state.reference_workflow_ids
            exec_context.inputs.context['reference_workflow_content'] = self.state.reference_workflow_content
        
        # Start with planning if no task list given - this is the first iteration
        force_planning = not self.state.task_list
        force_continue_step = exec_context.inputs.context["auto_mode_continue"]
        first_execution = self.state.auto_mode_first_execution

        if force_planning:
            # Initial iteration: planning mode - skip intent classification
            user_intent = "PLANNING"
            planning = True
        elif force_continue_step:
            # No user input, do not need intent classification:
            user_intent = ""
            planning = False
        else:
            # Use LLM classification for more complex messages
            exec_context = await self.execute_workflow(["autoloop_intent_classification"], exec_context)
            user_intent = exec_context.inputs.context["intent"]
            self._send_message(f"User intent classified as: {user_intent}")

            # Route to planning only for planning/modification intents
            # APPROVAL and CODE_IMPLEMENTATION_FEEDBACK should go to execution
            planning = user_intent in ["TASK_LIST_MODIFICATION"]

        # Step 2: Route based on intent classification
        if planning:
            await self._handle_autonomous_planning(exec_context, user_intent)
        else:
            # Call original execution method with proper parameters
            if first_execution:
                await self._handle_autonomous_first_execution(exec_context)
            else:
                await self._handle_autonomous_execution(exec_context)

    async def _handle_autonomous_planning(self, exec_context: ExecutionContext, intent: str) -> None:
        """
        Internal planning workflow that generates/updates tasks and prompts for feedback.

        Always ends by:
        1. Sending UI control message to show feedback prompt
        2. Sending LOOP_INCOMPLETE_REQUIRE_FEEDBACK to pause iteration

        This creates a synchronous pause point where the TypeScript loop waits for user input.

        @see AutonomousExecution.waitForUserFeedback() in autonomous-execution.ts
        """
        use_critique = True
        query_feedback = False

        start_time = time.time()
        self._send_message("Starting autonomous iteration - planning workflow")

        # Planning workflow: Generate/update task list
        initial_planning = intent == "PLANNING"
        # Assess if we are in a user feedback setting which might require us to adapt the reference workflows
        further_user_input = exec_context.inputs.context["auto_mode_continue"] == False
        rag_enabled = exec_context.inputs.context["rag_enabled"]

        if initial_planning:
            # Initial planning
            if rag_enabled:
                max_iterations_retrieval = 2
                workflow = [
                    "reference_workflow_query_preparation", 
                    "reference_workflow_selection", 
                    "reference_workflow_cell_selection"
                ]
                # Initialize retrieval queries with user query - later these can be set by the tools:
                exec_context.inputs.context["retrieval_queries"] = [exec_context.inputs.user_query]
                for _ in range(max_iterations_retrieval):
                    exec_context = await self.execute_workflow(workflow, exec_context)
                    # Break if no retrieval queries or empty list
                    retrieval_queries = exec_context.inputs.context.get("retrieval_queries", [])
                    if not retrieval_queries or len(retrieval_queries) == 0:
                        break
            # Iterate over planning based on critique and/or reference updates:
            if use_critique or rag_enabled:
                # Note task_text_old will not be available for first iteration but will available for subsequent iterations
                # to help guide changes (guide direction of change in iterative updates).
                max_iterations = 10
                for iteration in range(max_iterations):
                    # (Re-)generate task list if not approved and not last iteration (already handled by break above)
                    workflow = ["task_list_generation"]
                    exec_context = await self.execute_workflow(workflow, exec_context)
                    retrieval_queries = exec_context.inputs.context.get("retrieval_queries", [])
                    has_retrieval_queries = retrieval_queries and len(retrieval_queries) > 0

                    # Update reference workflows based on retrieval queries from task generation
                    # This runs the full workflow selection pipeline to update the knowledge context
                    if rag_enabled:
                        # Show loading indicator (cell selection will replace with final list)
                        # Always show this since selection could change even without new queries
                        # (e.g., protecting cited workflows, removing uncited ones)
                        loading_result = ToolResult(
                            output_ui={"text": "⏳ Retrieving reference workflows..."},
                            output_type=ToolOutputType.REFERENCE_WORKFLOWS
                        )
                        await self._send_tool_result(loading_result, exec_context)

                        # Execute sequential workflow (each tool decides if work is needed):
                        # 1. Query prep: Searches knowledge base, updates putative summaries
                        # 2. Selection: LLM selects from summaries (auto-converts full ↔ internal IDs)
                        # 3. Cell selection: Detects changes, runs LLM on new workflows only, merges results
                        workflow = [
                            "reference_workflow_query_preparation",
                            "reference_workflow_selection_only",
                            "reference_workflow_cell_selection"
                        ]
                        exec_context = await self.execute_workflow(workflow, exec_context)

                        # Reset for next iteration
                        exec_context.inputs.context["retrieval_queries"] = []

                        # Continue iterating if workflows were queried (which signals that list generating agent was not ready yet)
                        if has_retrieval_queries:
                            continue

                    # Run critique (after generation and reference updates):
                    if use_critique:
                        workflow = ["task_list_critique"]
                        exec_context = await self.execute_workflow(workflow, exec_context)
                        if exec_context.inputs.context["task_list_approval"] == "APPROVED":
                            self._send_message(f"Task list approved after {iteration + 1} iterations")
                            break

                    # If no critique enabled and no workflow retrievals are queried, exit
                    if not use_critique and not has_retrieval_queries:
                        break
                    elif iteration == max_iterations - 1:
                        # Max iterations reached without approval
                        self._send_message(f"Task list generation reached max iterations ({max_iterations}) without approval - proceeding anyway")
                        break

                    # Evaluating task_list_old here leaves the previous-current version in tact for the tools.
                    # Note: this also leaves exec_context.inputs.context["task_text_old"] empty in the first round
                    task_list_old = format_task_list(exec_context.inputs.task_list)
                    # Note: in contrast to update settings of the task list, where task_text_old is set once before the
                    # iteration with the critique, we set it here once per iteration, as we iteratively improve the first draft.
                    exec_context.inputs.context["task_text_old"] = task_list_old
            else:
                workflow = ["task_list_generation"]
                exec_context = await self.execute_workflow(workflow, exec_context)
            # Filter unused reference workflows after planning completes
            if rag_enabled:
                workflow = ["filter_unused_reference_workflows"]
                exec_context = await self.execute_workflow(workflow, exec_context)

        else:
            if further_user_input:
                workflow = ["reference_workflow_query_preparation", "reference_workflow_selection", "reference_workflow_cell_selection", "autonomous_update_tasks"]
            else:
                workflow = ["autonomous_update_tasks"]
            exec_context = await self.execute_workflow(workflow, exec_context)
            # Iterate over planning based on critique:
            if use_critique:
                exec_context.inputs.context["task_text_old"] = format_task_list(exec_context.inputs.task_list)
                # Save original task list in case critique rejects all updates
                original_task_list = exec_context.inputs.task_list.copy()
                max_iterations = 3
                for iteration in range(max_iterations):
                    workflow = ["autonomous_update_critique"]
                    exec_context = await self.execute_workflow(workflow, exec_context)
                    if exec_context.inputs.context["autonomous_update_approval"] == "APPROVED":
                        self._send_message(f"Task list update approved after {iteration + 1} critique iterations")
                        break
                    elif iteration == max_iterations - 1:
                        # Max iterations reached without approval - revert to original task list
                        self._send_message(f"Task list update critique reached max iterations ({max_iterations}) without approval - reverting to previous task list")
                        exec_context.inputs.task_list = original_task_list
                        exec_context.inputs.context["task_list"] = original_task_list
                        break
                    # Only regenerate if not approved and not last iteration
                    workflow = ["autonomous_update_tasks"]
                    exec_context = await self.execute_workflow(workflow, exec_context)

        duration = time.time() - start_time
        self._send_message(f"Planning workflow completed in {duration:.3f}s")

        # Send workflow state to pause iteration for feedback
        # This causes the TypeScript loop to enter waitForUserFeedback()
        if initial_planning and query_feedback:
            return_state = "LOOP_INCOMPLETE_REQUIRE_FEEDBACK"
        else:
            return_state = "LOOP_INCOMPLETE"
        await self.vscode.send_workflow_result(
            field="auto_loop_update",
            state=return_state
        )

    async def _handle_autonomous_first_execution(self, exec_context: ExecutionContext) -> None:
        """Handle continuation autonomous workflow: Update task list and continue/recover based on execution state."""
        start_time = time.time()
        self._send_message("Starting autonomous iteration - first execution iteration")

        # == Phase 1) Update task list
        exec_context = await self.execute_workflow(["mark_next_task_active"], exec_context)
        
        # == Phase 2) Progress based on error and completion states:        
        workflow = ["cell_positioning", "code_generation_with_guidance"]
        exec_context = await self.execute_workflow(workflow, exec_context)
        duration = time.time() - start_time
        self._send_message(f"END: autonomous iteration (completed in {duration:.3f}s)")
        # Update state:
        self.state.auto_mode_first_execution = False
        # Send workflow result via communicator
        await self.vscode.send_workflow_result(
            field="auto_loop_update",
            state="LOOP_INCOMPLETE"
        )
        return

    async def _handle_autonomous_execution(self, exec_context: ExecutionContext) -> None:
        """Handle continuation autonomous workflow: Update task list and continue/recover based on execution state."""
        use_critique = True

        start_time = time.time()
        self._send_message("Starting autonomous iteration - continuation workflow")

        has_error = exec_context.inputs.context['last_execution_failed']
        rag_enabled = exec_context.inputs.context["rag_enabled"]

        # == Phase 1) Analyze completion status and update task list if necessary
        
        # 1a. Update completion
        # Note: if an error was encountered in the last cell or the output interpretation suggests issues, this tool decides between:
        #   1) standard recovery (keep active task description unchanged but switch to pending and try again) and
        #   2) backtracking (set active and earlier tasks to pending and update task list)
        # by setting completed tasks to pending and supplying a recovery_objective.
        exec_context = await self.execute_workflow(["autonomous_mark_completion"], exec_context)
        # Check if tool queried workflow to try again because task was not sufficiently addressed:
        interpreted_as_failure = "retry_objective" in exec_context.inputs.context
        # Assess completion:
        all_complete = await self._are_all_tasks_completed(exec_context)
        # 1b. Check if completion analysis suggests backtracking
        # Note: backtracking may be triggered based on analysis results even if has_error==False
        # Check if backtracking detected - should be available in context after task completion analysis
        is_backtracking = "recovery_objective" in exec_context.inputs.context
        is_standard_retry = (has_error or interpreted_as_failure) and not is_backtracking
        if is_backtracking:
            # Create BacktrackingContext when backtracking is detected
            exec_context.inputs.backtracking_context = BacktrackingContext(
                recovery_objective=exec_context.inputs.context.get("recovery_objective", ""),
                backtrack_to_task=exec_context.inputs.context.get("backtrack_to_task", {})
            )
        # Note: task_list should already be set in exec_context
        # 1d. Update tasks (will show backtracking intention if detected)
        # Note: do not need to change task list content in standard error recovery.
        if not is_standard_retry and not all_complete:
            exec_context.inputs.context["task_text_old"] = format_task_list(exec_context.inputs.task_list)
            # Save original task list in case critique rejects all updates
            original_task_list = exec_context.inputs.task_list.copy()
            exec_context = await self.execute_workflow(["autonomous_update_tasks"], exec_context)
            # Only run the critique if the task list was updated:
            if use_critique and exec_context.inputs.context["task_list_update_rule"] == "UPDATE":
                max_iterations = 3
                for iteration in range(max_iterations):
                    workflow = ["autonomous_update_critique"]
                    exec_context = await self.execute_workflow(workflow, exec_context)
                    if exec_context.inputs.context["autonomous_update_approval"] == "APPROVED":
                        self._send_message(f"Task list update approved after {iteration + 1} critique iterations")
                        break
                    elif iteration == max_iterations - 1:
                        # Max iterations reached without approval - revert to original task list
                        self._send_message(f"Task list update critique reached max iterations ({max_iterations}) without approval - reverting to previous task list")
                        exec_context.inputs.task_list = original_task_list
                        exec_context.inputs.context["task_list"] = original_task_list
                        break
                    # Only regenerate if not approved and not last iteration
                    workflow = ["autonomous_update_tasks"]
                    exec_context = await self.execute_workflow(workflow, exec_context)
        # Ensure next pending task is set as active before code generation
        exec_context = await self.execute_workflow(["mark_next_task_active"], exec_context)
        # Assemble retrieval queries:
        if not all_complete:
            if rag_enabled and is_standard_retry:
                # snippet_retrieval_query comes from autonomous_update_tasks:
                if "snippet_retrieval_query" in exec_context.inputs.context.keys():
                    snippet_retrieval_query = exec_context.inputs.context['snippet_retrieval_query']
                else:
                    snippet_retrieval_query = []
                # Add feedback on last attempt if given:
                if "retry_objective" in exec_context.inputs.context.keys():
                    snippet_retrieval_query = snippet_retrieval_query + [exec_context.inputs.context['retry_objective']]
                # Add error in if occurred:
                if has_error:
                    snippet_retrieval_query = snippet_retrieval_query + [exec_context.inputs.context['error_message']]
                if len(snippet_retrieval_query) > 0:
                    exec_context.inputs.context['snippet_retrieval_query'] = snippet_retrieval_query
            if "snippet_retrieval_query" in exec_context.inputs.context.keys():
                exec_context = await self.execute_workflow(["rag_retrieval"], exec_context)
        
        # == Phase 2) Progress based on error and completion states:
        # Automatically determine next action
        if all_complete:
            # Branch 1: all tasks are completed
            duration = time.time() - start_time
            self._send_message(f"All tasks completed! Autonomous iteration finished (completed in {duration:.3f}s)")
            # Send workflow result via communicator
            await self.vscode.send_workflow_result(
                field="auto_loop_update",
                state="LOOP_COMPLETE"
            )
        elif not is_standard_retry and not is_backtracking:
            # Branch 2: no issues detected - progress with analysis.
            # Replace positioning tool by last cell modified in auto mode to add after:
            exec_context.inputs.context['positioning_info'] = {
                "target_cell": exec_context.inputs.context['last_cell_modified_in_auto_mode']
            }
            # Check if active task is code or reasoning:
            if exec_context.inputs.context['is_reasoning_task']:
                workflow = ["reasoning_response_with_guidance"]
                exec_context = await self.execute_workflow(workflow, exec_context)
                if use_critique:
                    # Reasoning workflow adds a new reasoning cell - 
                    # to be able to replace that cell below, we need to update positioning:
                    exec_context.inputs.context["positioning_info"] = {
                        "target_cell": exec_context.inputs.context["positioning_info"]["target_cell"] + 1
                    }
                    max_iterations = 2
                    for iteration in range(max_iterations):
                        workflow = ["reasoning_critique"]
                        exec_context = await self.execute_workflow(workflow, exec_context)
                        if exec_context.inputs.context["reasoning_approval"] == "APPROVED":
                            self._send_message(f"Reasoning approved after {iteration + 1} critique iterations")
                            break
                        elif iteration == max_iterations - 1:
                            # Max iterations reached without approval
                            self._send_message(f"Reasoning critique reached max iterations ({max_iterations}) without approval - proceeding anyway")
                            break
                        # Only regenerate if not approved and not last iteration
                        workflow = ["reasoning_response_with_guidance"]
                        exec_context = await self.execute_workflow(workflow, exec_context)
            else:
                workflow = ["code_generation_with_guidance"]
                exec_context = await self.execute_workflow(workflow, exec_context)
            duration = time.time() - start_time
            self._send_message(f"END: autonomous iteration - standard step (completed in {duration:.3f}s)")
            # Send workflow result via communicator
            await self.vscode.send_workflow_result(
                field="auto_loop_update",
                state="LOOP_INCOMPLETE"
            )
        elif is_standard_retry:
            # Branch 3: error detected but no backtracking - fix error in current position.          
            # Evaluate recovery strategy: REPLACE_AND_RETRY or RESTART_AND_RETRY
            exec_context = await self.execute_workflow(["error_recovery"], exec_context)
            error_recovery_strategy = exec_context.inputs.context.get("error_recovery_strategy")
            
            # Replace positioning tool by supplying the cell in which the issue occured - not necessarily an error!
            exec_context.inputs.context['positioning_info'] = {
                "target_cell": exec_context.inputs.context['last_cell_modified_in_auto_mode']
            }

            # Re-determine if active task is code or reasoning by checking the actual active task
            # This is crucial because the is_reasoning_task flag may be stale from a previous task
            active_task_description = exec_context.inputs.context.get('active_task_objective', '')
            is_reasoning_task = "[reasoning]" in active_task_description
            exec_context.inputs.context['is_reasoning_task'] = is_reasoning_task

            # Check if active task is code or reasoning:
            if is_reasoning_task:
                workflow = ["reasoning_response_with_guidance"]
                exec_context = await self.execute_workflow(workflow, exec_context)
                if use_critique:
                    max_iterations = 2
                    for iteration in range(max_iterations):
                        workflow = ["reasoning_critique"]
                        exec_context = await self.execute_workflow(workflow, exec_context)
                        if exec_context.inputs.context["reasoning_approval"] == "APPROVED":
                            self._send_message(f"Reasoning approved after {iteration + 1} critique iterations")
                            break
                        elif iteration == max_iterations - 1:
                            # Max iterations reached without approval
                            self._send_message(f"Reasoning critique reached max iterations ({max_iterations}) without approval - proceeding anyway")
                            break
                        # Only regenerate if not approved and not last iteration
                        workflow = ["reasoning_response_with_guidance"]
                        exec_context = await self.execute_workflow(workflow, exec_context)
            elif error_recovery_strategy == "REPLACE_AND_RESTART":
                workflow = ["restart_and_rerun", "code_update"]
                exec_context = await self.execute_workflow(workflow, exec_context)
            elif error_recovery_strategy == "REPLACE_AND_RETRY":
                workflow = ["code_update"]
                exec_context = await self.execute_workflow(workflow, exec_context)
            else:
                raise ValueError(error_recovery_strategy)
            
            duration = time.time() - start_time
            self._send_message(f"End: autonomous iteration - standard retry (with error: {has_error}) (completed in {duration:.3f}s")
            # Send workflow result via communicator
            await self.vscode.send_workflow_result(
                field="auto_loop_update",
                state="LOOP_INCOMPLETE"
            )
        elif is_backtracking:
            # Branch 3: backtracking
            # Enhanced backtracking workflow with cell deletion and restart decision
            reset_tasks = await self._parse_reset_tasks(exec_context)
            exec_context.inputs.context['reset_tasks'] = reset_tasks
            
            # First: Determine if restart is needed, then delete cells, then proceed
            initial_workflow = ["backtrack_recovery", "cell_selection_deletion", "cell_deletion", "cell_positioning"]
            exec_context = await self.execute_workflow(initial_workflow, exec_context)
            
            # Check if restart is required before generating new code
            restart_required = False
            if "backtrack_recovery" in exec_context.inputs.context:
                recovery_result = exec_context.inputs.context["backtrack_recovery"]
                restart_required = recovery_result.output_workflow.get("restart_required", False) if recovery_result.output_workflow else False
            
            if restart_required:
                self._send_message("Notebook restart required for clean backtracking recovery")
                # Execute restart and rerun, then generate code
                final_workflow = ["restart_and_rerun", "code_generation_with_guidance"]
            else:
                self._send_message("No restart needed - continuing with backtracking recovery")
                # Just generate code directly
                final_workflow = ["code_generation_with_guidance"]
            
            exec_context = await self.execute_workflow(final_workflow, exec_context)
            
            duration = time.time() - start_time
            self._send_message(f"END: autonomous iteration - backtracking (completed in {duration:.3f}s")
            # Send workflow result via communicator
            await self.vscode.send_workflow_result(
                field="auto_loop_update",
                state="LOOP_INCOMPLETE"
            )
        else:
            raise ValueError((has_error, is_backtracking, all_complete))
        return
        
    async def _handle_regular_request(
        self, message: str, session_metadata: Dict[str, Any], context: Dict[str, Any]
    ) -> None:
        """Handle regular (non-autonomous) requests through intent classification."""
        start_time = time.time()
        self._send_message("Starting regular request processing")
        
        # Step 1: Classify user intent with retry logic for malformed output
        temp_exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query=message,
                context=context,
                task_list={},
                backtracking_context=None,
                excluded_workflows=self.state.excluded_workflows
            ),
            session_metadata=session_metadata or {}
        )
        
        intent_tool = self.tools["intent_classification"]
        intent_start_time = time.time()
        
        # Retry intent classification if first attempt returns malformed output
        for attempt in range(2):  # Try up to 2 times
            intent_result = await intent_tool.execute(temp_exec_context)
            intent_data = intent_result.output_ui
            
            if isinstance(intent_data, dict):
                intent_value = intent_data.get('intent')
                break  # Success - exit retry loop
            elif hasattr(intent_data, 'intent'):
                intent_value = intent_data.intent
                break  # Success - exit retry loop
            elif attempt == 0:
                # First attempt failed with malformed output - retry once
                continue
            else:
                # Second attempt also failed - fallback to question answering
                intent_value = "question_about_code"
        
        intent_duration = time.time() - intent_start_time
        
        self._send_message(f"Intent classification completed in {intent_duration:.3f}s: {intent_value}")
        
        # Step 2: Execute appropriate workflow based on intent
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query=message,
                context=context,
                task_list={},
                backtracking_context=None,
                excluded_workflows=self.state.excluded_workflows
            ),
            session_metadata=session_metadata
        )
        
        # Build workflow based on intent (using categorical classification from schema)
        exec_context.inputs.context['snippet_retrieval_query'] = [exec_context.inputs.user_query]
        if intent_value == "question_about_code":
            workflow = ["rag_retrieval", "question_answering"]
        elif intent_value in ["generate_code", "generate_code_in_place"]:
            if intent_value == "generate_code":
                workflow = ["cell_positioning", "rag_retrieval", "code_generation"]
            else:
                workflow = ["rag_retrieval", "code_generation"]
        elif intent_value == "remove_code":
            duration = time.time() - start_time
            self._send_message(f"Regular request (remove code) completed in {duration:.3f}s")

            # Send workflow result via communicator
            await self.vscode.send_workflow_result(
                field="regular_chat_update",
                state="STEP_COMPLETE"
            )
            return None
        else:
            # Fallback to question answering
            workflow = ["rag_retrieval", "question_answering"]
        
        exec_context = await self.execute_workflow(workflow, exec_context)

        duration = time.time() - start_time
        self._send_message(f"Regular request completed in {duration:.3f}s")

        # Send workflow result via communicator
        await self.vscode.send_workflow_result(
            field="regular_chat_update",
            state="STEP_COMPLETE"
        )
        return

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
                - reasoning: Explanation for the decision
        """
        self._send_message(f"Analyzing execution progress ({context.get('elapsed_time', 0)}s elapsed)")

        # Create execution context for monitoring
        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query="",  # Will be populated by tool's _modify_user_query
                context=context,
                task_list={},
                backtracking_context=None,
                excluded_workflows=self.state.excluded_workflows
            ),
            session_metadata=session_metadata or {}
        )

        # Execute monitoring tool
        monitor_tool = self.tools["execution_monitor"]
        result = await monitor_tool.execute(exec_context)

        # Extract decision from workflow output
        workflow_output = result.output_workflow or {}
        action = workflow_output.get("action", "continue")
        feedback = workflow_output.get("feedback", "")

        self._send_message(f"Execution monitor decision: {action.upper()}")
        self._send_message(f"Feedback: {feedback}")

        return {
            "action": action,
            "feedback": feedback
        }

    def _reference_workflow_selection_tool(self) -> BaseTool:
        """Create ReferenceWorkflowSelectionTool with notebook selector."""
        # Get the knowledge base path from settings
        from kai.config.settings import Settings
        settings = Settings.from_env()

        # Initialize notebook selector with storage
        storage = NotebookStorage(settings.NOTEBOOK_SUMMARIES_PATH)
        selector = NotebookSelector(storage)
        return ReferenceWorkflowSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION,
            llm_interface=self.llm,
            notebook_selector=selector
        )
    
    def _reference_workflow_selection_only_tool(self) -> BaseTool:
        """Create ReferenceWorkflowSelectionOnlyTool with notebook selector."""
        # Get the knowledge base path from settings
        from kai.config.settings import Settings
        settings = Settings.from_env()
        storage = NotebookStorage(settings.NOTEBOOK_SUMMARIES_PATH)
        selector = NotebookSelector(storage)
        return ReferenceWorkflowSelectionOnlyTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_SELECTION_ONLY,
            llm_interface=self.llm,
            notebook_selector=selector
        )

    def _reference_workflow_cell_selection_tool(self) -> BaseTool:
        """Create ReferenceWorkflowCellSelectionTool with notebook selector."""
        from kai.config.settings import Settings
        settings = Settings.from_env()
        storage = NotebookStorage(settings.NOTEBOOK_SUMMARIES_PATH)
        selector = NotebookSelector(storage)
        return ReferenceWorkflowCellSelectionTool(
            scenario=PromptScenario.REFERENCE_WORKFLOW_CELL_SELECTION,
            llm_interface=self.llm,
            notebook_selector=selector
        )

    def _reference_workflow_query_preparation_tool(self) -> BaseTool:
        """Create ReferenceWorkflowQueryPreparationTool with summary search."""
        if not self.knowledge_base:
            # Return a dummy tool that does nothing if knowledge base is not available
            from .base_tool import BaseTool, ToolResult, ToolOutputType
            class DummyReferenceWorkflowQueryPreparationTool(BaseTool):
                def __init__(self):
                    super().__init__("reference_workflow_query_preparation")

                async def execute(self, exec_context, **kwargs) -> ToolResult:
                    return ToolResult(
                        output_ui={},
                        output_workflow={},
                        output_type=ToolOutputType.NO_OUTPUT
                    )
            return DummyReferenceWorkflowQueryPreparationTool()

        # Get the knowledge base path from settings
        from kai.config.settings import Settings
        settings = Settings.from_env()
        summary_search = WorkflowSummaryRag(settings.NOTEBOOK_SUMMARIES_PATH)
        return ReferenceWorkflowQueryPreparationTool(
            summary_search=summary_search
        )

    def _initialize_tools(self) -> Dict[str, BaseTool]:
        """Initialize all available tools with LLM pool routing."""
        return {
            # LLM-based tools
            "autoloop_intent_classification": AutoLoopIntentClassificationTool(self.llm),
            "autonomous_mark_completion": AutonomousMarkCompletionTool(self.llm),
            "autonomous_update_tasks": AutonomousUpdateTasksTool(self.llm),
            "autonomous_update_critique": AutonomousUpdateCritiqueTool(self.llm),
            "backtrack_recovery": BacktrackRecoveryTool(self.llm),
            "cell_deletion": CellDeletionTool(),
            "execution_monitor": ExecutionMonitorTool(self.llm),
            "code_generation": CodeGenerationTool(self.llm),
            "code_generation_with_guidance": CodeGenerationWithGuidanceTool(self.llm),
            "code_update": CodeUpdateTool(self.llm),
            "cell_positioning": CellPositioningTool(self.llm),
            "cell_selection_deletion": CellSelectionDeletionTool(self.llm),
            "error_recovery": ErrorRecoveryTool(self.llm),
            "intent_classification": IntentClassificationTool(self.llm),
            "question_answering": QuestionAnsweringTool(self.llm),
            "reasoning_critique": ReasoningCritiqueTool(self.llm),
            "reasoning_response_with_guidance": ReasoningResponseWithGuidanceTool(self.llm),
            "restart_and_rerun": RestartAndRerunTool(self.llm),
            "section_code_review": SectionCodeReviewTool(self.llm),
            "task_list_critique": TaskListCritiqueTool(self.llm),
            "task_list_generation": TaskListGenerationTool(self.llm),

            # Dedicated provider methods that assemble tool:
            "reference_workflow_selection": self._reference_workflow_selection_tool(),
            "reference_workflow_selection_only": self._reference_workflow_selection_only_tool(),
            "reference_workflow_cell_selection": self._reference_workflow_cell_selection_tool(),

            # Deterministic tools
            "rag_retrieval": CodeRetrievalTool(self.knowledge_base),
            "mark_next_task_active": MarkNextTaskActiveTool(),
            "reference_workflow_query_preparation": self._reference_workflow_query_preparation_tool(),
            "filter_unused_reference_workflows": FilterUnusedReferenceWorkflowsTool()
        }
    
    def set_turbo_mode(self, enabled: bool):
        """Enable or disable ollama turbo backend."""
        # Update LLM interface:
        self.llm.set_turbo_mode(enabled)
        # Update tools:
        self._initialize_tools()

    async def execute_workflow(self, workflow: List[str], context: ExecutionContext) -> ExecutionContext:
        """Execute a workflow of tools in sequence.
        
        Returns:
            Tuple of (final_tool_result, updated_execution_context)
        """
        # Save original user_query at workflow level to prevent pollution between tools
        original_workflow_user_query = context.inputs.user_query

        for tool_name in workflow:
            if self.is_cancelled:
                self._send_message("Workflow cancelled by user")
                break

            if tool_name not in self.tools:
                logger.error(f"Tool not found: {tool_name}")
                continue

            tool = self.tools[tool_name]

            # Restore clean user_query before each tool
            context.inputs.user_query = original_workflow_user_query

            # Execute tool with retry logic - pass exec_context directly
            tool_start_time = time.time()
            result = await self._execute_tool_with_retry(tool, context, tool_name)
            tool_duration = time.time() - tool_start_time
            self._send_message(f"Tool {tool_name} completed in {tool_duration:.3f}s")
            
            # Propagate output_workflow data to execution context for subsequent tools
            if result.output_workflow:
                context.inputs.context.update(result.output_workflow)
                
                # Also populate specific fields that tools expect directly in inputs
                # TODO deprecate this and handle this entirely in inputs.context in the future.
                if "task_list" in result.output_workflow:
                    context.inputs.task_list = result.output_workflow["task_list"]
                if "backtracking_context" in result.output_workflow:
                    context.inputs.backtracking_context = result.output_workflow["backtracking_context"]
                if "excluded_workflows" in result.output_workflow:
                    context.inputs.excluded_workflows.extend(result.output_workflow["excluded_workflows"])

            # Send result to VSCode if needed
            await self._send_tool_result(result, context)
        
        # Update orchestrator state at workflow completion
        if context.inputs.context.get("task_list"):
            self.state.task_list = context.inputs.context["task_list"]
        if context.inputs.context.get("reference_workflow_ids"):
            self.state.reference_workflow_ids = context.inputs.context["reference_workflow_ids"]
        if context.inputs.context.get("reference_workflow_content"):
            self.state.reference_workflow_content = context.inputs.context["reference_workflow_content"]
        return context

    async def _send_tool_result(self, result: ToolResult, context: ExecutionContext):
        """Send tool result to VSCode based on output type (delegated to VSCodeCommunicator)."""
        # Create minimal execution context for VSCodeCommunicator
        from .vscode_communicator import VscodeInputContext as VSCodeExecutionContext
        vscode_context = VSCodeExecutionContext(
            session_id=context.session_metadata["session_id"],
            inputs=context.inputs
        )
        
        await self.vscode.send_tool_result(result, vscode_context)
    
    async def _execute_tool_with_retry(self, tool, exec_context: ExecutionContext, tool_name: str, max_retries: int = 5) -> ToolResult:
        """Execute tool with retry logic for format validation failures.
        
        Args:
            tool: Tool instance to execute
            exec_context: ExecutionContext for the tool
            tool_name: Name of the tool (for logging)
            max_retries: Maximum number of attempts (default 5)
            
        Returns:
            ToolResult from successful execution or final error result
        """
        last_failed_output = None
        last_error = None
        context_length_factor = 1.
        # reduce reasoning level if reaching end of retries, this reduces context used for reasoning
        # which can alleviate cases in which tool fails because context runs out before output is finished
        reasoning_level_reduction = {
            "low": "low",
            "medium": "low",
            "high": "medium"
        }
        reasoning_level = None

        # Save original user_query to restore between attempts (prevent retry message pollution)
        original_user_query = exec_context.inputs.user_query

        for attempt in range(max_retries):
            try:
                # Restore original user_query before adding attempt-specific format reminder
                exec_context.inputs.user_query = original_user_query

                # Add format reminder for retry attempts, including previous error
                if attempt > 0:
                    exec_context = self._add_format_reminder_to_exec_context(exec_context, attempt, last_failed_output, last_error, tool)

                result = await tool.execute(
                    exec_context,
                    context_length_factor=context_length_factor,
                    reasoning_level=reasoning_level
                )

                # Success - restore original user_query before returning to prevent pollution
                exec_context.inputs.user_query = original_user_query
                return result

            except Exception as e:
                error_str = str(e)
                error_type = type(e).__name__
                error_repr = repr(e)

                # Store error info for next retry
                last_error = f"{error_type}: {error_str}"

                # Extract raw output if available (attached by llm_interface)
                if hasattr(e, 'raw_output'):
                    last_failed_output = e.raw_output

                self._send_message(f"Tool {tool_name} failed on attempt {attempt + 1}: {error_type}: {error_str}")
                if not error_str.strip():
                    self._send_message(f"  (Empty error string, repr: {error_repr})")
                if attempt == max_retries - 1:
                    # Final attempt failed, return error result
                    raise ValueError(f"Error in {tool_name} after {max_retries} attempts: {str(e)}")
                # Increase context length to cover cases in which LLM ran out of context:
                context_length_factor = context_length_factor * 2.
                # Reduce reasoning in final iterations:
                if attempt >= max_retries - 3 and hasattr(tool, "reasoning_level"):
                    previous_reasoning_level = reasoning_level if reasoning_level else tool.reasoning_level
                    reasoning_level = reasoning_level_reduction[previous_reasoning_level]
    
    def _add_format_reminder_to_exec_context(self, exec_context: ExecutionContext, attempt: int, last_failed_output: Optional[str] = None, last_error: Optional[str] = None, tool = None) -> ExecutionContext:
        """Add format validation reminder to exec_context for retry attempts."""
        format_reminder = f"\n\nIMPORTANT: This is attempt #{attempt + 1}. You failed to format your output correctly last time."

        if last_error:
            format_reminder += f"\n\nThe error was:\n{last_error}"

        from kai.core.orchestration.prompt_tools import StructuredPromptTool
        if tool and isinstance(tool, StructuredPromptTool):
            format_reminder += "\n\nPlease ensure your response strictly follows the required JSON schema format. Double-check all brackets, quotes, and commas."
        else:
            format_reminder += "\n\nPlease ensure your response strictly follows the required format."

        if last_failed_output:
            if len(last_failed_output) > 500:
                truncated_output = last_failed_output[:500] + "... [truncated]"
            else:
                truncated_output = last_failed_output

            format_reminder += f"\n\nYour previous failed output was:\n```\n{truncated_output}\n```\n\nPlease correct the formatting issues shown in the error above."

        exec_context.inputs.user_query = str(exec_context.inputs.user_query) + format_reminder

        return exec_context
    
    def _extract_failed_output_from_error(self, error_output: str) -> str:
        """Extract the actual failed LLM output from error messages."""
        # The error output typically contains the raw LLM output that failed to parse
        # Try to extract it from common error message patterns
        
        if isinstance(error_output, str):
            # Look for patterns like "Raw output: ..." or similar
            import re
            
            # Pattern 1: Look for "raw_llm_output" or similar in the error
            if "raw_llm_output" in error_output.lower():
                # Try to extract JSON-like content after the raw output indicator
                match = re.search(r'raw_llm_output["\s:]+([^"]+)', error_output, re.IGNORECASE | re.DOTALL)
                if match:
                    return match.group(1).strip()
            
            # Pattern 2: Extract anything that looks like malformed JSON
            json_like_pattern = r'[{[][\s\S]*?[}\]]'
            json_matches = re.findall(json_like_pattern, error_output)
            if json_matches:
                # Return the longest match (most likely to be the actual output)
                return max(json_matches, key=len)
            
            # Pattern 3: If it contains obvious format markers, return first part
            if any(marker in error_output for marker in ["```", "json", "{"]):
                # Take first 300 chars which likely contain the malformed output
                return error_output[:300]
        
        # Fallback: return the error output itself (truncated)
        return str(error_output)[:200] if error_output else "No output captured"
    
    async def _parse_reset_tasks(self, context: ExecutionContext):
        """Parse which tasks were reset from completed/active back to pending for backtracking.
        
        In backtracking scenarios, some tasks that were previously completed/active
        are reset to pending. This method identifies those reset tasks.
        """
        if not context.inputs.task_list:
            return []
        
        reset_tasks = []
        
        # Look for patterns that indicate reset tasks:
        # 1. Pending tasks that come after completed tasks (violates normal flow)
        # 2. Tasks that are pending but have higher IDs than active/completed tasks
        
        # Find all pending tasks
        pending_tasks = [t for t in context.inputs.task_list['tasks'] if t.get('status') == 'pending']
        completed_tasks = [t for t in context.inputs.task_list['tasks'] if t.get('status') == 'completed']
        active_tasks = [t for t in context.inputs.task_list['tasks'] if t.get('status') == 'active']
        
        if not pending_tasks:
            return []
        
        # If there are completed tasks, any pending tasks with IDs lower than
        # the highest completed task ID were likely reset
        if completed_tasks:
            max_completed_id = max(task.get('id', 0) for task in completed_tasks)
            reset_tasks.extend([
                task for task in pending_tasks 
                if task.get('id', 0) <= max_completed_id
            ])
        
        # Also check for pending tasks that come between completed and active tasks
        if active_tasks and completed_tasks:
            active_ids = set(task.get('id', 0) for task in active_tasks)
            completed_ids = set(task.get('id', 0) for task in completed_tasks)
            
            for task in pending_tasks:
                task_id = task.get('id', 0)
                # If this pending task's ID falls between completed and active tasks
                if (any(c_id < task_id for c_id in completed_ids) and 
                    any(a_id > task_id for a_id in active_ids)):
                    if task not in reset_tasks:
                        reset_tasks.append(task)
        
        return reset_tasks
