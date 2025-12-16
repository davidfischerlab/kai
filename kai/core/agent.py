import os

# Disable telemetry/analytics before any imports that might use them
os.environ.setdefault('LANGCHAIN_TRACING_V2', 'false')
os.environ.setdefault('LANGSMITH_TRACING', 'false')
os.environ.setdefault('POSTHOG_DISABLED', '1')
os.environ.setdefault('DO_NOT_TRACK', '1')

from datetime import datetime
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any

from kai.config.settings import Settings
from .llm_interface import LLMInterface
from .orchestration.langgraph_orchestrator import LangGraphOrchestrator
from .orchestration.ui_communicator import UICommunicator
from kai.retrieval import create_knowledge_base, ChromaDbManager
from kai.utils import setup_logger

logger = setup_logger(__name__)


class KaiAgent:
    """
    Main entry point for the bioinformatics agent.
    
    Provides a unified interface for bioinformatics code generation, analysis,
    and conversation. Uses LangGraphOrchestrator for all request processing.
    
    Attributes:
        llm: LLMInterface for language model interactions
        knowledge_base: ChromaDB knowledge base for RAG
        orchestrator: LangGraphOrchestrator for tool execution and chaining
    
    Example:
        agent = BioinformaticsAgent(llm_provider="ollama", model="gpt-oss:20b")
        response, session_id = await agent.chat("Generate scanpy code")
    """
    llm_interface: LLMInterface
    knowledge_base: ChromaDbManager
    orchestrator: LangGraphOrchestrator
    session_metadata: Dict[str, Any]
    settings: Settings
    vscode: UICommunicator
    
    def __init__(
        self,
        llm_provider: str = "ollama",
        model: Optional[str] = None,  # Use defaults
        knowledge_path: Optional[Path] = None,
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        suppress_vscode_messages: bool = False,  # Suppress all VSCode JSON messages (for Jupyter interface)
        max_task_planning_iterations: Optional[int] = None,  # Max planning iterations (None = use orchestrator default)
    ):
        self.settings = settings or Settings.from_env()
        knowledge_path = knowledge_path or self.settings.KNOWLEDGE_BASE_PATH

        # Initialize core components
        self.llm_interface = LLMInterface(provider=llm_provider, model=model, settings=self.settings, api_key=api_key)
        self.knowledge_base = create_knowledge_base(knowledge_path, self.settings)

        # Start background initialization of knowledge base caches
        # This ensures collection embeddings are ready when first RAG query arrives
        self.knowledge_base.start_background_initialization()

        # Create shared VSCode communicator for centralized message control
        from .orchestration.ui_communicator import UICommunicator
        self.vscode = UICommunicator()

        # Remember suppression preference (for Jupyter interface)
        self._suppress_vscode_messages = suppress_vscode_messages

        # Suppress VSCode messages if requested (for Jupyter interface)
        if suppress_vscode_messages:
            self.vscode._disabled = True

        # Build orchestrator kwargs, only passing max_task_planning_iterations if explicitly set
        orchestrator_kwargs = {
            "llm_interface": self.llm_interface,
            "knowledge_base": self.knowledge_base,
            "ui_communicator": self.vscode,
        }
        if max_task_planning_iterations is not None:
            orchestrator_kwargs["max_task_planning_iterations"] = max_task_planning_iterations

        self.orchestrator = LangGraphOrchestrator(**orchestrator_kwargs)
        
        # Autonomous session metadata - agent owns all session state
        self.session_metadata = {
            "active": False,
            "session_id": None,
            "session_timestamp": None,  # Set once when session starts, never changes
            "notebook_uri": None,       # Captured once, persisted
            "iteration_counter": 0      # Increments each continue
        }
        self.orchestrator._send_message(f"[KAI] Initializing turbo mode: {'ENABLED' if self.llm_interface.provider_name == 'ollama-turbo' else 'DISABLED'}")
        
    def is_autonomous_active(self, session_id: Optional[str] = None) -> bool:
        """Check if autonomous session is currently active."""
        if not session_id:
            return self.session_metadata.get("active", False)
        return (self.session_metadata.get("active", False) and 
                self.session_metadata.get("session_id") == session_id)
    
    def terminate_autonomous_session(self, reason: str = "manual_stop") -> None:
        """Centralized autonomous session termination."""
        if self.session_metadata.get("active"):
            session_id = self.session_metadata.get("session_id", "unknown")
            self.session_metadata["active"] = False
            # Reset session metadata
            self.session_metadata.update({
                "session_id": None,
                "session_timestamp": None,
                "notebook_uri": None,
                "iteration_counter": 0
            })
            # LangGraph checkpoint handles state cleanup automatically

            import json
            import sys
            msg = {
                "type": "console_log",
                "message": f"[KAI] Terminated autonomous session: {session_id} (reason: {reason})"
            }
            print(json.dumps(msg))
            sys.stdout.flush()
    
    def stop_autonomous_execution(self) -> str:
        """Public method for VSCode to stop autonomous execution."""
        if self.session_metadata.get("active"):
            self.terminate_autonomous_session("user_requested")
            return "Autonomous execution stopped successfully."
        else:
            return "No active autonomous session to stop."

    def get_status(self) -> Dict[str, Any]:
        """Get status of the agent."""
        return {
            "status": "running",
            "primary_model": f"{self.llm_interface.provider_name}/{self.llm_interface.model}",
            "tools": len(self.orchestrator.tools)
        }
    
    # Unified conversational interface
    async def chat(self, user_input: str, session_id: str = None, user_id: str = "anonymous", context: Dict[str, Any] = None) -> tuple[Dict[str, Any], str]:
        """
        Main conversational interface for the bioinformatics agent.
        
        This provides a unified entry point that processes messages with provided context.
        
        Args:
            user_input: The user's message or request
            session_id: Session identifier
            user_id: User identifier
            context: Context with conversation history, execution history, current cell, etc.
            
        Returns:
            Tuple of (response, session_id)
        """
        # Actions in autonomous mode initiation - in first iteration
        auto_mode_initiation = not session_id
        if auto_mode_initiation:
            # Generate session_id in first iteration
            session_id = f"session_{hashlib.md5(f'{user_id}_{user_input}'.encode()).hexdigest()[:8]}"

            # Enable VSCode communication for new autonomous session (unless suppression was requested)
            if not self._suppress_vscode_messages:
                self.vscode.enable_communication()

            # Get session timestamp consisting of date and time:
            date_str = datetime.now().strftime("%Y-%m-%d")
            time_str = datetime.now().strftime('%H-%M-%S')
            session_timestamp = f"{date_str}_{time_str}"
            
            # Get notebook URI from original contexts
            notebook_uri = context.get('notebookUri')

            self.session_metadata.update({
                "active": True,
                "session_id": session_id,
                "session_timestamp": session_timestamp,
                "notebook_uri": notebook_uri,
                "iteration_counter": 0,
                "iteration_timestamp": datetime.now().strftime('%H-%M-%S'),
            })
        elif self.is_autonomous_active(session_id):
            # Update iteration meta data:
            self.session_metadata["iteration_counter"] += 1
            self.session_metadata["iteration_timestamp"] = datetime.now().strftime('%H-%M-%S')
        
        # Extract all VSCode items explicitly and rename from camelCase to snake_case
        context_data = {
            # Request data
            'request_id': context.get('request_id'),

            # Execution context
            'execution_history': context.get('executionHistory', []),
            'conversation_history': context.get('conversationHistory', []),
            'notebook_structure': context.get('notebookStructure', {'totalCells': 0, 'allCells': []}),

            # Current state
            'current_cell': context.get('currentCell'),  # Content of current cell
            'current_cell_index': context.get('currentCellIndex'),  # Index of current cell

            # Error information - provide defaults so orchestrator can use direct access
            'error_cell_index': context.get('errorCellIndex', None),
            'execution_result': context.get('executionResult', ''),
            'last_execution_failed': context.get('lastExecutionFailed', False),

            # Autonomous mode flags
            'autonomous_mode': context.get('autonomousMode', False),
            'autonomous_mode_continue': context.get('autonomousModeContinue', False),
            'autonomous_mode_termination': context.get('autonomousModeTermination', False),
            'last_cell_modified_in_auto_mode': context.get('lastCellModifiedInAutoMode', None),

            # Task management
            'task_list': context.get('taskList', {}),
            'excluded_workflows': context.get('excludedWorkflows', []),

            # Backend details
            'turbo_enabled': context.get('turboEnabled', False),
            'rag_enabled': context.get('ragEnabled', False),
        }

        # Initialize planning state for reference workflow selection on first iteration
        # (subsequent iterations get it from checkpointer)
        if not context_data['autonomous_mode_continue']:
            context_data['retrieval_queries'] = [user_input]
            context_data['planning_phase'] = None  # Will be set by router on first routing decision
            context_data['workflow_retrieval_iteration'] = 0  # Start at 0, incremented by search_workflows tool
            context_data['task_planning_iteration'] = -1  # Start at -1, first increment gives 0 (matching kai_dev's for loop)
            logger.debug(f"[AGENT] Initialized planning state: retrieval_queries with user_input (len={len(user_input)})")

        # Parse error messages:
        # Note: need to use same output separating strings as in VSCode extension: formatCellOutputToString
        if context_data['last_execution_failed']:
            # Check if this is a termination message from execution monitor
            if context_data['execution_result'].startswith("[EXECUTION TERMINATED BY MONITORING AGENT]"):
                # For termination, use the entire execution_result (includes feedback + partial outputs)
                context_data['error_message'] = context_data['execution_result']
            else:
                # For normal errors, extract only the error output sections
                context_data['error_message'] = "\n\n>>> ".join([
                    x for x in context_data['execution_result'].split(">>> ") if x.startswith("Error output")
                ])
        else:
            context_data['error_message'] = ""

        # Handle Turbo mode switching - respect DISABLE_TURBO setting
        turbo_enabled = context_data['turbo_enabled'] and not self.settings.DISABLE_TURBO
        
        if turbo_enabled != (self.llm_interface.provider_name == "ollama-turbo"):
            # Switch turbo mode on/off
            self.orchestrator.set_turbo_mode(bool(turbo_enabled))
            self.orchestrator._send_message(f"🔄 Updating turbo mode to: {'ENABLED' if turbo_enabled else 'DISABLED'}")

        # Add session metadata:
        context_data['session_metadata'] = self.session_metadata.copy()

        # Handle stop request
        if context_data['autonomous_mode_termination']:
            self.terminate_autonomous_session("user_stop")
            return {"text": "Autonomous session stopped by user.", "intent": "stop"}, session_id
        
        # Process message directly through orchestrator
        await self.orchestrator.process_request(
            message=user_input,
            context=context_data
        )

        # Check if autonomous mode was disabled while this function was processing:
        if self.is_autonomous_active(session_id):
            structured_response = {"processed": True}
        else:
            structured_response = {"processed": False}
        return structured_response, session_id
