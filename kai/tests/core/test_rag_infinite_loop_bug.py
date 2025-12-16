"""Test for RAG infinite loop bug.

Bug: Router was checking for 'rag_text' field but RAG tool returns 'rag_retrieval',
causing infinite loop of RAG calls during error recovery.
"""

import pytest
from unittest.mock import MagicMock
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator


class TestRAGInfiniteLoopBug:
    """Test that RAG retrieval doesn't loop infinitely during error recovery."""

    def test_rag_not_called_twice_during_error_recovery(self):
        """RAG should only be called once during error recovery, not repeatedly.

        Bug scenario:
        1. Error occurs, router routes to RAG retrieval
        2. RAG tool returns {"rag_retrieval": "..."}
        3. Router checks for "rag_text" (wrong field!)
        4. Router thinks RAG wasn't retrieved, calls RAG again → INFINITE LOOP

        Fixed behavior:
        1. Error occurs, router routes to RAG retrieval  
        2. RAG tool returns {"rag_retrieval": "..."}
        3. Router checks for "rag_retrieval" (correct field!)
        4. Router sees RAG was retrieved, proceeds to error recovery tool
        """
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)
        router = orch._route_autonomous_action

        # State after RAG retrieval has completed
        state_after_rag = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Test", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "just_executed": True,
            "task_completion_analyzed": True,
            "next_task_activated": True,
            "last_execution_failed": True,  # Error state
            "retry_objective": "Fix the error",
            "recovery_objective": None,
            "rag_enabled": True,
            "rag_retrieval": "Some retrieved content here",  # RAG HAS RUN
            "generated_code": None,
            "target_cell": None,
            "positioning_info": None,
        }

        result = router(state_after_rag)

        # Should NOT route back to rag_retrieval (that would be the infinite loop)
        assert result != "rag_retrieval", \
            "BUG: Router routing to rag_retrieval again despite RAG already retrieved!"

        # Should route to error recovery tool instead
        assert result == "error_recovery", \
            f"Expected error_recovery, got {result}"

    def test_rag_called_first_time_during_error_recovery(self):
        """RAG should be called the FIRST time during error recovery if enabled."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)
        router = orch._route_autonomous_action

        # State BEFORE RAG retrieval
        state_before_rag = {
            "task_list": {
                "tasks": [{"id": 1, "task": "Test", "status": "active"}]
            },
            "autonomous_mode_continue": True,
            "auto_mode_first_execution_done": True,
            "just_executed": True,
            "task_completion_analyzed": True,
            "next_task_activated": True,
            "last_execution_failed": True,
            "retry_objective": "Fix the error",
            "recovery_objective": None,
            "rag_enabled": True,
            # No rag_retrieval field yet - RAG hasn't run
            "generated_code": None,
            "target_cell": None,
            "positioning_info": None,
        }

        result = router(state_before_rag)

        # Should route to rag_retrieval the FIRST time
        assert result == "rag_retrieval", \
            f"Expected rag_retrieval on first error recovery, got {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
