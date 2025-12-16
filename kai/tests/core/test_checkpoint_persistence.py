"""Test that persistent fields use LangGraph checkpointer correctly."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator, PERSISTENT_STATE_FIELDS


class TestCheckpointPersistence:
    """Test persistent field management via checkpointer."""

    def test_persistent_fields_defined(self):
        """Verify all persistent fields are defined.

        Note: task_list_backup is NOT persistent - it's created/used/cleared
        within a single iteration for critique reversion.
        """
        expected_fields = {
            "task_list",
            "reference_workflow_ids",
            "reference_workflow_content",
            "excluded_workflows",
            "retrieval_queries",
            "auto_mode_first_execution_done",
            "planning_phase",
            "workflow_retrieval_iteration",
            "task_planning_iteration",
            # Active task tracking (must persist for code generation to know which task to address)
            "active_task",
            "active_task_objective",
            "is_reasoning_task",
            "next_pending_task_objective",
        }

        assert PERSISTENT_STATE_FIELDS == expected_fields, \
            f"Missing fields: {expected_fields - PERSISTENT_STATE_FIELDS}, Extra: {PERSISTENT_STATE_FIELDS - expected_fields}"

    @pytest.mark.asyncio
    async def test_first_iteration_initializes_persistent_fields(self):
        """First iteration: No checkpoint exists, so PERSISTENT fields are initialized.

        This test verifies that when autonomous_mode=True and no checkpoint exists,
        the planning graph is invoked and PERSISTENT fields are initialized.
        """
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()
        mock_comm.send_workflow_result = AsyncMock()
        mock_comm.send_tool_result = AsyncMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        # Mock BOTH planning and autonomous graphs (since autonomous_mode triggers planning)
        with patch.object(orch.planning_graph, 'astream') as mock_planning_astream:
            # Planning graph returns task list in final output
            async def planning_generator():
                yield {"task_list": {"tasks": []}}
            mock_planning_astream.return_value = planning_generator()

            with patch.object(orch.autonomous_graph, 'astream') as mock_auto_astream:
                mock_auto_astream.return_value = async_generator_empty()

                # Mock get_state to return None (no checkpoint exists yet)
                with patch.object(orch.autonomous_graph, 'get_state') as mock_get_state:
                    mock_get_state.return_value = None  # No checkpoint

                    with patch.object(orch.autonomous_graph, 'aget_state') as mock_aget_state:
                        mock_aget_state.return_value = MagicMock(
                            values={"task_list": {"tasks": []}}
                        )

                        context = {
                            # No task_list provided by agent
                            "session_metadata": {"session_id": "test_session"},
                            "autonomous_mode": True,
                        }

                        await orch.process_request(
                            message="Start analysis",
                            context=context
                        )

                        # Check that planning_graph.astream was called with PERSISTENT fields initialized
                        assert mock_planning_astream.called, "Planning graph should be called when autonomous_mode=True and no tasks exist"

                        planning_call_args = mock_planning_astream.call_args
                        initial_state = planning_call_args[0][0]

                        # Verify PERSISTENT fields are initialized (no checkpoint exists)
                        assert "task_list" in initial_state, \
                            "First iteration SHOULD initialize task_list (no checkpoint)"
                        assert initial_state["task_list"] == {}, \
                            "Initialized to empty dict"

                        assert "reference_workflow_content" in initial_state, \
                            "First iteration SHOULD initialize reference_workflow_content"
                        assert initial_state["reference_workflow_content"] == {}, \
                            "Initialized to empty dict"

                        assert "reference_workflow_ids" in initial_state, \
                            "Should initialize reference_workflow_ids"
                        assert initial_state["reference_workflow_ids"] is None, \
                            "Initialized to None"

    @pytest.mark.asyncio
    async def test_subsequent_iteration_restores_from_checkpoint(self):
        """Subsequent iterations: Checkpoint exists, persistent fields restored from it."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()
        mock_comm.send_workflow_result = AsyncMock()
        mock_comm.send_tool_result = AsyncMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        # Mock the graph execution
        with patch.object(orch.autonomous_graph, 'astream') as mock_astream:
            mock_astream.return_value = async_generator_empty()

            # Mock get_state to return a checkpoint (exists from previous invocation)
            with patch.object(orch.autonomous_graph, 'get_state') as mock_get_state:
                mock_get_state.return_value = MagicMock(
                    values={
                        "task_list": {"tasks": [{"id": 1}]},
                        "reference_workflow_content": {"workflow_1": "content"},
                        "auto_mode_first_execution_done": True,
                    }
                )

                with patch.object(orch.autonomous_graph, 'aget_state') as mock_aget_state:
                    mock_aget_state.return_value = MagicMock(
                        values={"task_list": {"tasks": [{"id": 1}]}}
                    )

                    context = {
                        # No task_list or reference_workflow_content provided
                        "session_metadata": {"session_id": "test_session"},
                        "autonomous_mode": True,
                        "last_execution_failed": False,  # Transient field
                    }

                    await orch.process_request(
                        message="Continue",
                        context=context
                    )

                    call_args = mock_astream.call_args
                    initial_state = call_args[0][0]

                    # Persistent fields should be RESTORED from checkpoint
                    assert "task_list" in initial_state, \
                        "Should restore task_list from checkpoint"
                    assert initial_state["task_list"] == {"tasks": [{"id": 1}]}, \
                        "Should have checkpoint value for task_list"

                    assert "auto_mode_first_execution_done" in initial_state, \
                        "Should restore auto_mode_first_execution_done from checkpoint"
                    assert initial_state["auto_mode_first_execution_done"] is True, \
                        "Should have True from checkpoint"

                    assert "last_execution_failed" in initial_state, \
                        "Transient fields should always be included"

    @pytest.mark.asyncio
    async def test_ui_can_override_persistent_fields(self):
        """UI can explicitly override persistent fields by providing non-empty value."""
        mock_llm = MagicMock()
        mock_kb = MagicMock()
        mock_comm = MagicMock()
        mock_comm.send_workflow_result = AsyncMock()
        mock_comm.send_tool_result = AsyncMock()

        orch = LangGraphOrchestrator(mock_llm, mock_kb, mock_comm)

        # Mock the graph execution
        with patch.object(orch.autonomous_graph, 'astream') as mock_astream:
            mock_astream.return_value = async_generator_empty()
            
            with patch.object(orch.autonomous_graph, 'aget_state') as mock_get_state:
                mock_get_state.return_value = MagicMock(
                    values={"task_list": {"tasks": []}}
                )

                # UI explicitly provides task_list (e.g., user manually edited)
                override_task_list = {"tasks": [{"id": 99, "task": "Override"}]}
                
                context = {
                    "task_list": override_task_list,  # Explicit override
                    "session_metadata": {"session_id": "test_session"},
                    "autonomous_mode": True,
                }

                await orch.process_request(
                    message="Override",
                    context=context
                )

                call_args = mock_astream.call_args
                initial_state = call_args[0][0]
                
                assert "task_list" in initial_state, \
                    "Explicit override should be included in initial_state"
                assert initial_state["task_list"] == override_task_list, \
                    "Override value should match what UI provided"


async def async_generator_empty():
    """Mock async generator that yields nothing."""
    return
    yield  # Make it a generator


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
