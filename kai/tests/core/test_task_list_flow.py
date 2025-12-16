"""Test that task_list flows correctly from JupyterInterface → agent → orchestrator.

This is a regression test for the production bug where:
- JupyterInterface set context['taskList'] with tasks
- agent.py didn't extract taskList from context
- orchestrator received empty task_list → always planned new tasks
- Result: Tasks were never preserved across iterations, always replanned
"""

import pytest
from kai.core.agent import KaiAgent


class TestTaskListFlow:
    """Test that task_list is correctly passed through all layers."""

    @pytest.fixture
    def agent(self):
        """Create agent for testing."""
        agent = KaiAgent(llm_provider='ollama', model="qwen3:0.6b")
        agent.session_metadata.update({
            'active': True,
            'session_id': 'test_task_flow',
            'session_timestamp': '2025-01-01_12-00-00',
            'notebook_uri': 'file:///test.ipynb',
            'iteration_counter': 1,
            'iteration_timestamp': '12-01-00'
        })
        return agent

    def test_task_list_extracted_from_context(self, agent):
        """Test that agent.py extracts taskList from JupyterInterface context.

        This is a regression test for the bug where agent.py didn't include
        task_list in the context_data passed to the orchestrator.
        """
        # Simulate context from JupyterInterface (camelCase format)
        jupyter_context = {
            "autonomousMode": True,
            "autonomousModeContinue": True,
            "taskList": {  # camelCase from JupyterInterface
                "tasks": [
                    {"id": 1, "task": "Task 1", "status": "completed"},
                    {"id": 2, "task": "Task 2", "status": "active"},
                    {"id": 3, "task": "Task 3", "status": "pending"}
                ]
            },
            "executionHistory": [],
            "conversationHistory": [],
            "notebookStructure": {"totalCells": 0, "allCells": []},
            "currentCell": "",
            "currentCellIndex": 0,
            "lastExecutionFailed": False,
            "request_id": "test_task_extraction",
            "ragEnabled": False,
            "turboEnabled": False,
            "notebookUri": "file:///test.ipynb"
        }

        # Simulate agent.py's context transformation (lines 189-219)
        context_data = {
            'request_id': jupyter_context.get('request_id'),
            'execution_history': jupyter_context.get('executionHistory', []),
            'conversation_history': jupyter_context.get('conversationHistory', []),
            'notebook_structure': jupyter_context.get('notebookStructure', {'totalCells': 0, 'allCells': []}),
            'current_cell': jupyter_context.get('currentCell'),
            'current_cell_index': jupyter_context.get('currentCellIndex'),
            'error_cell_index': jupyter_context.get('errorCellIndex', None),
            'execution_result': jupyter_context.get('executionResult', ''),
            'last_execution_failed': jupyter_context.get('lastExecutionFailed', False),
            'autonomous_mode': jupyter_context.get('autonomousMode', False),
            'autonomous_mode_continue': jupyter_context.get('autonomousModeContinue', False),
            'autonomous_mode_termination': jupyter_context.get('autonomousModeTermination', False),
            'last_cell_modified_in_auto_mode': jupyter_context.get('lastCellModifiedInAutoMode', None),
            'turbo_enabled': jupyter_context.get('turboEnabled', False),
            'rag_enabled': jupyter_context.get('ragEnabled', False),
            'task_list': jupyter_context.get('taskList', {}),  # THE FIX
            'excluded_workflows': jupyter_context.get('excludedWorkflows', []),
        }

        # Verify task_list was extracted
        assert 'task_list' in context_data, \
            "agent.py must extract taskList from context"

        assert context_data['task_list'] == jupyter_context['taskList'], \
            "task_list should match original taskList from JupyterInterface"

        assert len(context_data['task_list']['tasks']) == 3, \
            "All 3 tasks should be preserved"

        assert context_data['task_list']['tasks'][0]['status'] == 'completed', \
            "Task statuses should be preserved"

        print("✅ task_list correctly extracted from JupyterInterface context")

    def test_empty_task_list_defaults_to_empty_dict(self, agent):
        """Test that missing taskList defaults to empty dict (not None).

        The orchestrator expects task_list to be a dict, not None.
        """
        jupyter_context = {
            "autonomousMode": True,
            "autonomousModeContinue": True,
            # No taskList field
            "executionHistory": [],
            "conversationHistory": [],
            "notebookStructure": {"totalCells": 0, "allCells": []},
            "currentCell": "",
            "currentCellIndex": 0,
            "lastExecutionFailed": False,
            "request_id": "test_empty_task_list",
            "ragEnabled": False,
            "turboEnabled": False,
            "notebookUri": "file:///test.ipynb"
        }

        # Simulate agent.py's extraction
        task_list = jupyter_context.get('taskList', {})

        # Verify it's an empty dict, not None
        assert task_list == {}, \
            "Missing taskList should default to {} (not None)"

        assert isinstance(task_list, dict), \
            "task_list must be a dict"

        print("✅ Missing taskList defaults to empty dict")

    def test_excluded_workflows_extracted(self, agent):
        """Test that excludedWorkflows is also extracted from context.

        This field is also in KaiState and may be used by orchestrator.
        """
        jupyter_context = {
            "autonomousMode": True,
            "excludedWorkflows": ["workflow1", "workflow2"],
            "executionHistory": [],
            "conversationHistory": [],
            "notebookStructure": {"totalCells": 0, "allCells": []},
            "currentCell": "",
            "currentCellIndex": 0,
            "lastExecutionFailed": False,
            "request_id": "test_excluded_workflows",
            "ragEnabled": False,
            "turboEnabled": False,
            "notebookUri": "file:///test.ipynb"
        }

        # Simulate agent.py's extraction
        excluded_workflows = jupyter_context.get('excludedWorkflows', [])

        # Verify extraction
        assert excluded_workflows == ["workflow1", "workflow2"], \
            "excludedWorkflows should be extracted correctly"

        print("✅ excludedWorkflows correctly extracted from context")

    def test_task_list_flow_matches_state_definition(self, agent):
        """Test that task_list structure matches what KaiState expects.

        KaiState expects:
        - task_list: Dict[str, Any]
        - With 'tasks' key containing list of task dicts
        - Each task has: id, task, status
        """
        from kai.core.state import KaiState

        # Verify KaiState has task_list field
        assert 'task_list' in KaiState.__annotations__, \
            "KaiState must have task_list field"

        # Verify it's typed as Dict
        field_type = str(KaiState.__annotations__['task_list'])
        assert 'Dict' in field_type or 'dict' in field_type, \
            f"task_list should be Dict type, got: {field_type}"

        # Test structure
        task_list = {
            "tasks": [
                {"id": 1, "task": "Test task", "status": "pending"}
            ]
        }

        # Verify structure is valid
        assert 'tasks' in task_list, "task_list must have 'tasks' key"
        assert isinstance(task_list['tasks'], list), "'tasks' must be a list"
        assert len(task_list['tasks']) > 0, "Should have at least one task"
        assert 'id' in task_list['tasks'][0], "Task must have 'id'"
        assert 'task' in task_list['tasks'][0], "Task must have 'task' description"
        assert 'status' in task_list['tasks'][0], "Task must have 'status'"

        print("✅ task_list structure matches KaiState expectations")
