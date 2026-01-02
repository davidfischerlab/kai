"""Message management utilities for LangGraph orchestration.

TODO: This module is not currently integrated into the main orchestration flow.
      It should be incorporated in the future for context compression when
      conversation history exceeds token limits.

Provides:
- Message summarization via Ollama for context compression
- Token counting utilities
- Execution log query functions
"""

from typing import Dict, Any, List, Optional
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage

from kai.utils import setup_logger

logger = setup_logger(__name__)


# =============================================================================
# Token Counting Utilities
# =============================================================================

def count_tokens(messages: List[BaseMessage]) -> int:
    """Estimate token count for messages.

    Uses a simple character-based estimation (4 chars ≈ 1 token).
    For production, consider using tiktoken for accurate counting.

    Args:
        messages: List of LangChain messages

    Returns:
        Estimated token count
    """
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4  # Rough estimate


def count_message_chars(messages: List[BaseMessage]) -> int:
    """Count total characters in messages.

    Args:
        messages: List of LangChain messages

    Returns:
        Total character count
    """
    return sum(len(str(m.content)) for m in messages)


# =============================================================================
# Message Summarizer (Ollama-based)
# =============================================================================

class MessageSummarizer:
    """Hidden internal tool for message compression.

    Uses a small local Ollama model to summarize conversation history.
    Not exposed to agent's tool calling - only used internally by graph.

    Key features:
    - Preserves user's original intent
    - Preserves key decisions made
    - Preserves current task state
    - Uses cheap local model (no API costs)
    """

    def __init__(self, model: str = "llama3.2:3b"):
        """Initialize summarizer with specified Ollama model.

        Args:
            model: Ollama model identifier (default: llama3.2:3b for speed)
        """
        self.model = model
        self._ollama_available = None

    def _check_ollama_available(self) -> bool:
        """Check if Ollama is available."""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            import ollama
            ollama.list()  # Simple health check
            self._ollama_available = True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            self._ollama_available = False

        return self._ollama_available

    def summarize_sync(self, messages: List[BaseMessage]) -> str:
        """Synchronously summarize messages into a concise paragraph.

        Must be synchronous as LangGraph reducers don't support async.

        Args:
            messages: List of messages to summarize

        Returns:
            Summary string (max 500 chars)
        """
        if not messages:
            return ""

        if not self._check_ollama_available():
            # Fallback: simple truncation if Ollama not available
            return self._fallback_summarize(messages)

        try:
            import ollama

            # Format messages for summarization
            formatted = "\n".join([
                f"{self._get_message_type(m).upper()}: {str(m.content)[:500]}"
                for m in messages
            ])

            prompt = f"""Summarize this conversation in 2-3 sentences, preserving:
- The user's original goal/request
- Key decisions made
- Current state of the task

Conversation:
{formatted}

Summary:"""

            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response'][:500]

        except Exception as e:
            logger.error(f"Ollama summarization failed: {e}")
            return self._fallback_summarize(messages)

    def _fallback_summarize(self, messages: List[BaseMessage]) -> str:
        """Fallback summarization when Ollama unavailable.

        Extracts key parts of messages without LLM.
        """
        parts = []

        # Get first user message (original intent)
        for m in messages:
            if isinstance(m, HumanMessage):
                content = str(m.content)[:200]
                parts.append(f"User requested: {content}...")
                break

        # Get message count
        parts.append(f"({len(messages)} messages total)")

        return " ".join(parts)

    def _get_message_type(self, message: BaseMessage) -> str:
        """Get human-readable message type."""
        if isinstance(message, HumanMessage):
            return "user"
        elif isinstance(message, AIMessage):
            return "assistant"
        elif isinstance(message, SystemMessage):
            return "system"
        return "message"


# Global summarizer instance
_summarizer = MessageSummarizer()


# =============================================================================
# Message Summarization Node
# =============================================================================

def summarize_messages_node(
    state: Dict[str, Any],
    token_limit: int = 8000,
    keep_recent: int = 10,
) -> Dict[str, Any]:
    """Summarize messages if they exceed token limit.

    IMPORTANT: This must be a synchronous function - LangGraph nodes that
    modify state via reducers cannot be async.

    This node should be added to the graph BEFORE any LLM-calling nodes
    to ensure context is compressed before generation.

    Args:
        state: Current graph state with 'messages' field
        token_limit: Max tokens before summarization triggers
        keep_recent: Number of recent messages to preserve

    Returns:
        State update with compressed messages, or empty dict if no change
    """
    messages = state.get("messages", [])

    if not messages:
        return {}

    # Check if summarization needed
    token_count = count_tokens(messages)
    if token_count <= token_limit:
        return {}  # No change needed

    logger.info(
        f"[MESSAGE_MGMT] Summarizing messages: "
        f"{len(messages)} msgs, ~{token_count} tokens"
    )

    # Separate system messages (always preserve)
    system_msgs = [m for m in messages if isinstance(m, SystemMessage)]

    # Keep recent messages
    recent = messages[-keep_recent:] if len(messages) > keep_recent else []

    # Messages to summarize (not system, not recent)
    to_summarize = [
        m for m in messages
        if m not in system_msgs and m not in recent
    ]

    if not to_summarize:
        return {}  # Nothing to summarize

    # Summarize old messages
    summary_text = _summarizer.summarize_sync(to_summarize)

    # Create summary message
    summary_msg = SystemMessage(
        content=(
            f"[Prior conversation summary ({len(to_summarize)} messages)]: "
            f"{summary_text}"
        )
    )

    # New message list: system + summary + recent
    new_messages = system_msgs + [summary_msg] + recent

    logger.info(
        f"[MESSAGE_MGMT] Compressed {len(messages)} → {len(new_messages)} messages"
    )

    return {"messages": new_messages}


# =============================================================================
# Execution Log Query Functions
# =============================================================================

def was_cell_backtracked(log: List[Dict[str, Any]], cell_idx: int) -> bool:
    """Check if a cell was backtracked.

    Queries the append-only execution log to derive current state.

    Args:
        log: Execution event log
        cell_idx: Cell index to check

    Returns:
        True if cell was backtracked
    """
    for event in log:
        if event.get("event_type") == "cell_backtracked":
            backtracked_cells = event.get("payload", {}).get("cell_indices", [])
            if cell_idx in backtracked_cells:
                return True
    return False


def get_task_errors(
    log: List[Dict[str, Any]],
    task_id: str
) -> List[Dict[str, Any]]:
    """Get all errors for a specific task.

    Queries the append-only execution log.

    Args:
        log: Execution event log
        task_id: Task ID to filter by

    Returns:
        List of error payloads
    """
    errors = []
    for event in log:
        if event.get("event_type") == "cell_executed":
            payload = event.get("payload", {})
            if (
                payload.get("task_id") == task_id
                and not payload.get("success", True)
            ):
                errors.append(payload)
    return errors


def count_valid_executions(log: List[Dict[str, Any]]) -> int:
    """Count cell executions excluding backtracked ones.

    Replays the event log to derive current state.

    Args:
        log: Execution event log

    Returns:
        Count of currently valid executed cells
    """
    executed = set()
    for event in log:
        event_type = event.get("event_type")
        payload = event.get("payload", {})

        if event_type == "cell_executed":
            cell_idx = payload.get("cell_index")
            if cell_idx is not None:
                executed.add(cell_idx)

        elif event_type == "cell_backtracked":
            backtracked = set(payload.get("cell_indices", []))
            executed -= backtracked

    return len(executed)


def get_execution_timeline(log: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Get chronological timeline of execution events.

    Args:
        log: Execution event log

    Returns:
        List of events with parsed timestamps, sorted chronologically
    """
    from datetime import datetime

    timeline = []
    for event in log:
        try:
            timestamp = datetime.fromisoformat(event.get("timestamp", ""))
            timeline.append({
                "timestamp": timestamp,
                "event_type": event.get("event_type"),
                "payload": event.get("payload", {}),
            })
        except (ValueError, TypeError):
            # Skip events with invalid timestamps
            pass

    return sorted(timeline, key=lambda x: x["timestamp"])
