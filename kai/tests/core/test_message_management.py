"""Tests for message management and execution log utilities.

Tests:
- Message token counting
- Message summarization (with fallback when Ollama unavailable)
- Execution log query functions
"""

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from kai.core.orchestration.message_utils import (
    count_tokens,
    count_message_chars,
    MessageSummarizer,
    summarize_messages_node,
    was_cell_backtracked,
    get_task_errors,
    count_valid_executions,
    get_execution_timeline,
)


# =============================================================================
# Token Counting Tests
# =============================================================================

class TestTokenCounting:
    """Test token counting utilities."""

    def test_count_tokens_empty(self):
        """Empty message list returns 0."""
        assert count_tokens([]) == 0

    def test_count_tokens_single_message(self):
        """Single message token count."""
        messages = [HumanMessage(content="Hello world")]  # 11 chars
        tokens = count_tokens(messages)
        assert tokens == 11 // 4  # 2 tokens (rough estimate)

    def test_count_tokens_multiple_messages(self):
        """Multiple messages token count."""
        messages = [
            HumanMessage(content="a" * 100),  # 100 chars
            AIMessage(content="b" * 200),     # 200 chars
        ]
        tokens = count_tokens(messages)
        assert tokens == 300 // 4  # 75 tokens

    def test_count_message_chars(self):
        """Character counting."""
        messages = [
            HumanMessage(content="Hello"),      # 5 chars
            AIMessage(content="World"),         # 5 chars
        ]
        assert count_message_chars(messages) == 10


# =============================================================================
# Message Summarizer Tests
# =============================================================================

class TestMessageSummarizer:
    """Test message summarization."""

    def test_fallback_summarize_empty(self):
        """Fallback with empty messages."""
        summarizer = MessageSummarizer()
        result = summarizer._fallback_summarize([])
        # Empty messages still gets a count message
        assert "(0 messages total)" in result

    def test_fallback_summarize_extracts_user_intent(self):
        """Fallback extracts first user message."""
        summarizer = MessageSummarizer()
        messages = [
            HumanMessage(content="Please analyze my gene expression data"),
            AIMessage(content="I'll help with that"),
        ]
        result = summarizer._fallback_summarize(messages)
        assert "User requested:" in result
        assert "gene expression" in result

    def test_summarize_sync_empty(self):
        """Sync summarize with empty messages."""
        summarizer = MessageSummarizer()
        result = summarizer.summarize_sync([])
        assert result == ""

    def test_summarize_sync_fallback_when_no_ollama(self):
        """Sync summarize falls back when Ollama unavailable."""
        summarizer = MessageSummarizer()
        summarizer._ollama_available = False  # Force fallback

        messages = [
            HumanMessage(content="Analyze the dataset"),
        ]
        result = summarizer.summarize_sync(messages)
        assert "User requested:" in result

    def test_get_message_type(self):
        """Message type detection."""
        summarizer = MessageSummarizer()
        human = HumanMessage(content="")
        ai = AIMessage(content="")
        system = SystemMessage(content="")
        assert summarizer._get_message_type(human) == "user"
        assert summarizer._get_message_type(ai) == "assistant"
        assert summarizer._get_message_type(system) == "system"


# =============================================================================
# Summarize Messages Node Tests
# =============================================================================

class TestSummarizeMessagesNode:
    """Test the summarize_messages_node graph node."""

    def test_no_messages_returns_empty(self):
        """No messages - no change."""
        state = {"messages": []}
        result = summarize_messages_node(state)
        assert result == {}

    def test_under_limit_returns_empty(self):
        """Messages under token limit - no change."""
        messages = [
            HumanMessage(content="Short message"),
        ]
        state = {"messages": messages}
        result = summarize_messages_node(state, token_limit=1000)
        assert result == {}

    def test_over_limit_triggers_summarization(self):
        """Messages over token limit triggers summarization."""
        # Create many messages to exceed limit
        messages = [
            HumanMessage(content="x" * 1000),
            AIMessage(content="y" * 1000),
            HumanMessage(content="z" * 1000),
        ]
        state = {"messages": messages}

        # Set low limit to trigger summarization
        result = summarize_messages_node(state, token_limit=100, keep_recent=1)

        # Should have new messages
        assert "messages" in result
        new_messages = result["messages"]

        # Should have compressed
        assert len(new_messages) < len(messages)

    def test_preserves_system_messages(self):
        """System messages are always preserved."""
        messages = [
            SystemMessage(content="You are a helpful assistant"),
            HumanMessage(content="x" * 1000),
            AIMessage(content="y" * 1000),
        ]
        state = {"messages": messages}

        result = summarize_messages_node(state, token_limit=100, keep_recent=1)

        if "messages" in result:
            # System message should be preserved
            system_msgs = [
                m for m in result["messages"]
                if isinstance(m, SystemMessage)
            ]
            assert len(system_msgs) >= 1

    def test_keeps_recent_messages(self):
        """Recent messages are preserved."""
        messages = [
            HumanMessage(content="old" * 500),
            AIMessage(content="old" * 500),
            HumanMessage(content="recent message"),
        ]
        state = {"messages": messages}

        result = summarize_messages_node(state, token_limit=100, keep_recent=1)

        if "messages" in result:
            # Last message should be in result
            contents = [str(m.content) for m in result["messages"]]
            assert any("recent" in c for c in contents)


# =============================================================================
# Execution Log Query Tests
# =============================================================================

class TestExecutionLogQueries:
    """Test execution log query functions."""

    def test_was_cell_backtracked_empty_log(self):
        """Empty log - no backtracking."""
        assert was_cell_backtracked([], 5) is False

    def test_was_cell_backtracked_true(self):
        """Cell was backtracked."""
        log = [
            {
                "event_type": "cell_backtracked",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {"cell_indices": [3, 4, 5]},
            }
        ]
        assert was_cell_backtracked(log, 5) is True
        assert was_cell_backtracked(log, 4) is True
        assert was_cell_backtracked(log, 6) is False

    def test_was_cell_backtracked_false(self):
        """Cell was not backtracked."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {"cell_index": 5},
            }
        ]
        assert was_cell_backtracked(log, 5) is False

    def test_get_task_errors_empty(self):
        """No errors for task."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {"task_id": "1", "success": True},
            }
        ]
        errors = get_task_errors(log, "1")
        assert len(errors) == 0

    def test_get_task_errors_with_errors(self):
        """Get errors for task."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {
                    "task_id": "1", "success": False, "error": "NameError"
                },
            },
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:01:00",
                "payload": {"task_id": "1", "success": True},
            },
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:02:00",
                "payload": {
                    "task_id": "2", "success": False, "error": "ValueError"
                },
            },
        ]
        errors = get_task_errors(log, "1")
        assert len(errors) == 1
        assert errors[0]["error"] == "NameError"

    def test_count_valid_executions_empty(self):
        """Empty log - no executions."""
        assert count_valid_executions([]) == 0

    def test_count_valid_executions_simple(self):
        """Simple execution count."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {"cell_index": 0},
            },
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:01:00",
                "payload": {"cell_index": 1},
            },
        ]
        assert count_valid_executions(log) == 2

    def test_count_valid_executions_with_backtrack(self):
        """Backtracked cells not counted."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {"cell_index": 0},
            },
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:01:00",
                "payload": {"cell_index": 1},
            },
            {
                "event_type": "cell_backtracked",
                "timestamp": "2025-01-01T00:02:00",
                "payload": {"cell_indices": [1]},
            },
        ]
        assert count_valid_executions(log) == 1

    def test_count_valid_executions_reexecute_after_backtrack(self):
        """Re-execution after backtrack counted."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {"cell_index": 0},
            },
            {
                "event_type": "cell_backtracked",
                "timestamp": "2025-01-01T00:01:00",
                "payload": {"cell_indices": [0]},
            },
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:02:00",
                "payload": {"cell_index": 0},  # Re-executed
            },
        ]
        assert count_valid_executions(log) == 1

    def test_get_execution_timeline_chronological(self):
        """Timeline is chronologically sorted."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:02:00",
                "payload": {},
            },
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {},
            },
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:01:00",
                "payload": {},
            },
        ]
        timeline = get_execution_timeline(log)
        assert len(timeline) == 3
        # Should be sorted by timestamp
        assert timeline[0]["timestamp"].hour == 0
        assert timeline[0]["timestamp"].minute == 0

    def test_get_execution_timeline_skips_invalid(self):
        """Invalid timestamps are skipped."""
        log = [
            {
                "event_type": "cell_executed",
                "timestamp": "2025-01-01T00:00:00",
                "payload": {},
            },
            {
                "event_type": "cell_executed",
                "timestamp": "invalid",
                "payload": {},
            },
        ]
        timeline = get_execution_timeline(log)
        assert len(timeline) == 1
