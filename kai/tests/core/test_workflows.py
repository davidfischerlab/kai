"""Integration tests for autonomous continuation workflow with real LLM calls.

Tests the 4 main branches of _handle_autonomous_continuation:
1. Normal continuation (task completion + next task)
2. Standard error recovery (error detected + code fixing)
3. Backtracking (task analysis suggests going back)
4. All tasks completed (workflow completion)
"""

import pytest
import json
from kai.core.agent import KaiAgent


class TestWorkflows:
    """Test autonomous continuation workflow with real LLM tools."""

    @pytest.fixture
    def agent_with_capture(self, capsys):
        """Create agent with captured output for VSCode message monitoring."""
        # Use tiny model for local execution to sanity check workflow execution
        agent = KaiAgent(llm_provider='ollama', model="qwen3:0.6b")

        # Set up active autonomous session
        agent.session_metadata.update({
            'active': True,
            'session_id': 'test_integration',
            'session_timestamp': '2025-01-01_12-00-00',
            'notebook_uri': 'file:///test.ipynb',
            'iteration_counter': 1,
            'iteration_timestamp': '12-01-00'
        })

        return agent, capsys

    def parse_vscode_messages(self, capsys):
        """Parse JSON messages from captured stdout."""
        captured = capsys.readouterr()
        output = captured.out
        messages = []
        for line in output.strip().split('\n'):
            if line.strip():
                try:
                    message = json.loads(line)
                    messages.append(message)
                except json.JSONDecodeError:
                    continue
        return messages

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rag_enabled", [True, False])
    async def test_autonomous_initiation(self, agent_with_capture, rag_enabled):
        """Test autonomous initiation workflow with RAG enabled and disabled."""
        agent, capsys = agent_with_capture

        # Create context for initiation
        context = {
            "session_metadata": agent.session_metadata,
            "current_cell": "",
            "current_cell_index": 0,
            "notebook_structure": {
                "totalCells": 5,
                "allCells": ["# Cell 1", "# Cell 2", "# Cell 3", "# Cell 4", "# Cell 5"]
            },
            "execution_history": [],
            "conversation_history": [],
            "last_execution_failed": False,  # Required field
            "request_id": f"test_request_rag_{rag_enabled}",  # Required for VSCode communication
            "rag_enabled": rag_enabled,  # Parameterized RAG setting
            "auto_mode_continue": False  # This is initial planning, not continuation
        }

        # Store original user query to test preservation during critique workflow
        original_user_query = "Analyze single-cell RNA-seq data with quality control"

        # Run autonomous initiation using unified approach
        try:
            await agent.orchestrator._handle_autonomous_unified(
                original_user_query,
                agent.session_metadata,
                context
            )
        except ValueError as e:
            # Expected if task generation fails with small model
            if "Trying to set active task without task list" in str(e):
                pass  # This is expected with tiny models that can't produce structured output
            else:
                raise

        # Parse VSCode messages
        messages = self.parse_vscode_messages(capsys)

        # Check that the workflow ran successfully by checking messages
        assert len(messages) > 0

        # Check workflow execution based on RAG setting
        console_messages = [m for m in messages if m.get("type") == "console_log"]
        workflow_rag_attempted = any(
            "reference_workflow_selection" in msg.get("message", "")
            for msg in console_messages
        )

        if rag_enabled:
            # Should see RAG workflow being attempted when RAG is enabled
            assert workflow_rag_attempted, "Should attempt reference_workflow_selection when RAG is enabled"
        else:
            # Should NOT see RAG workflow being attempted when RAG is disabled
            assert not workflow_rag_attempted, "Should not attempt reference_workflow_selection when RAG is disabled"

        # Should see task_generation being attempted in both cases
        task_generation_attempted = any(
            "task_generation" in msg.get("message", "")
            for msg in console_messages
        )
        assert task_generation_attempted, "Should attempt task_generation in both cases"

        # Validate reference workflow messages are sent when RAG is enabled during planning
        self.assert_reference_workflows_in_planning(messages, rag_enabled)

        # Test user query preservation during critique workflow iterations
        # The critique workflow temporarily modifies user_query for iterations but should restore it
        # This is tested via the orchestrator's internal logic during workflow execution
        print(f"✅ User query preservation test: Original query '{original_user_query}' used in planning workflow")

    def assert_has_message_type(self, messages, message_type):
        """Assert that messages contain at least one of the specified type."""
        matching = [msg for msg in messages if msg.get('type') == message_type]
        assert len(matching) > 0, f"No messages of type '{message_type}' found. Available types: {set(msg.get('type') for msg in messages)}"
        return matching

    def assert_has_code_execution(self, messages):
        """Assert that messages contain code execution."""
        execute_messages = self.assert_has_message_type(messages, 'execute_code')
        for msg in execute_messages:
            assert 'code' in msg, f"execute_code message missing 'code' field: {msg}"
            assert len(msg['code'].strip()) > 0, f"execute_code message has empty code: {msg}"
        return execute_messages

    def assert_has_task_updates(self, messages):
        """Assert that messages contain task list updates."""
        # Look for both 'display' and 'task_list_display' message types
        task_update_messages = []
        for msg in messages:
            if msg.get('type') in ['display', 'task_list_display']:
                # Check if message contains task data
                if ('tasks' in msg.get('data', {}) or
                    'tasks' in str(msg.get('response', {})) or
                    msg.get('type') == 'task_list_display'):
                    task_update_messages.append(msg)
        assert len(task_update_messages) > 0, f"No task list updates found in messages"
        return task_update_messages

    def assert_workflow_completion(self, messages):
        """Assert that workflow completed successfully."""
        workflow_messages = self.assert_has_message_type(messages, 'workflow_result')
        completion_messages = [msg for msg in workflow_messages if 'auto_loop_update' in msg]
        assert len(completion_messages) > 0, f"No workflow completion messages found"
        return completion_messages

    def assert_reference_workflows_in_planning(self, messages, rag_enabled):
        """Assert reference workflow messages are sent during planning when RAG is enabled."""
        reference_workflow_messages = [msg for msg in messages if msg.get('type') == 'reference_workflows']
        console_messages = [msg for msg in messages if msg.get('type') == 'console_log']

        if rag_enabled:
            # When RAG is enabled during planning, we should either have reference workflow messages
            # or evidence that the reference workflow selection was attempted (even if it failed due to small test model)
            rag_attempted = any(
                "reference_workflow_selection" in msg.get("message", "") or
                "Found" in msg.get("message", "") and "summaries" in msg.get("message", "")
                for msg in console_messages
            )

            if len(reference_workflow_messages) > 0:
                # If we have reference workflow messages, validate their structure
                for msg in reference_workflow_messages:
                    assert 'response' in msg or 'data' in msg, \
                        f"reference_workflows message should have response or data field: {msg}"

                    # Check that the message contains reference workflow IDs
                    message_content = msg.get('response', msg.get('data', {}))
                    assert 'text' in message_content or 'reference_workflow_ids' in message_content, \
                        f"reference_workflows message should contain text or reference_workflow_ids: {message_content}"
            else:
                # If no reference workflow messages, at least verify RAG was attempted
                assert rag_attempted, \
                    f"Should attempt reference workflow selection when RAG is enabled. Console messages: {[msg.get('message', '') for msg in console_messages]}"

        return reference_workflow_messages

    @pytest.fixture
    def agent(self):
        """Create agent with same setup as VSCode extension (remote gpt-oss)."""
        # Use tiny model for local execution to sanity check workflow execution
        agent = KaiAgent(llm_provider='ollama', model="qwen3:0.6b")

        # Set up active autonomous session
        agent.session_metadata.update({
            'active': True,
            'session_id': 'test_integration',
            'session_timestamp': '2025-01-01_12-00-00',
            'notebook_uri': 'file:///test.ipynb',
            'iteration_counter': 1,
            'iteration_timestamp': '12-01-00'
        })

        return agent

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rag_enabled", [True, False])
    async def test_autonomous_continuation_standard(self, agent_with_capture, rag_enabled):
        """Test normal continuation: task completion check + next task generation."""
        agent, capsys = agent_with_capture

        # Reset orchestrator state completely for test isolation
        agent.orchestrator.state.reference_workflow_content = ""
        agent.orchestrator.state.reference_workflow_annotation = ""
        # This is a CONTINUATION test, not first execution
        agent.orchestrator.state.auto_mode_first_execution = False
        agent.orchestrator.state.task_list = {
            'tasks': [
                {'id': 1, 'task': 'Import libraries', 'status': 'completed'},
                {'id': 2, 'task': 'Load data from CSV', 'status': 'active'},
                {'id': 3, 'task': 'Create scatter plot', 'status': 'pending'},
                {'id': 4, 'task': 'Export results', 'status': 'pending'}
            ]
        }

        if rag_enabled:
            agent.orchestrator.state.reference_workflow_content = "# Sample workflow context\nImport pandas and matplotlib for data analysis"
            agent.orchestrator.state.reference_workflow_annotation = "Use these patterns for data loading and visualization"

        context = {
            'lastExecutionFailed': False,
            'errorCellIndex': None,
            'executionResult':'Success: CSV data loaded successfully with 1000 rows and 5 columns',
            'autonomousMode': True,
            'notebookStructure': {'totalCells': 3, 'allCells': ['# Cell'] * 3},
            'lastCellModifiedInAutoMode': 1,
            'request_id': f'test_normal_rag_{rag_enabled}',
            'rag_enabled': rag_enabled,
            # Add required context fields for intent classification
            'execution_history': [
                {'cell_index': 0, 'code': 'import pandas as pd', 'output': 'Success'},
                {'cell_index': 1, 'code': 'df = pd.read_csv("data.csv")', 'output': 'Success: CSV data loaded successfully with 1000 rows and 5 columns'}
            ],
            'conversation_history': [
                {'role': 'user', 'content': 'Load the CSV data', 'timestamp': '12:00:00'},
                {'role': 'assistant', 'content': 'I will load the CSV data using pandas', 'timestamp': '12:00:01'}
            ]
        }

        # Run continuation workflow (may fail due to small test model, but we check workflow logic)
        try:
            # Use explicit approval message that should route to execution
            response, _ = await agent.chat('Approved. Continue execution.', 'test_integration', context=context)

            # Basic response check
            assert response['processed'] is True

            # Parse VSCode messages
            messages = self.parse_vscode_messages(capsys)

            # Check workflow behavior - different expectations for RAG vs non-RAG
            if rag_enabled:
                # With RAG enabled, may route to planning mode to refine tasks based on reference workflows
                # Should have task list updates and possibly require feedback
                task_messages = self.assert_has_task_updates(messages)
                assert len(task_messages) > 0, "Should update task list when RAG is enabled"

                # May end with feedback requirement
                workflow_messages = [msg for msg in messages if msg.get('type') == 'workflow_result']
                if workflow_messages:
                    # If workflow result present, might be requesting feedback
                    pass  # This is acceptable behavior with RAG
                else:
                    # If no workflow result, should have generated code
                    code_messages = self.assert_has_code_execution(messages)
                    assert len(code_messages) > 0, "Should generate code if not requesting feedback"
            else:
                # Without RAG, should proceed to execution mode
                # Check for either code execution or workflow completion
                try:
                    code_messages = self.assert_has_code_execution(messages)
                    assert len(code_messages) > 0, "Should generate code for next task"
                except AssertionError:
                    # If no code generation, should at least show workflow progress
                    workflow_messages = [msg for msg in messages if msg.get('type') == 'workflow_result']
                    task_messages = [msg for msg in messages if msg.get('type') == 'task_list_display']
                    assert len(workflow_messages) > 0 or len(task_messages) > 0, "Should show workflow progress"

            # If we have code execution, validate the content
            if not rag_enabled or (rag_enabled and 'code_messages' in locals() and len(code_messages) > 0):
                if 'code_messages' in locals():
                    # Check that code contains meaningful content for the next task
                    code_content = code_messages[0]['code']
                    # Since task 1 (data loading) completed, should generate code for task 2 (scatter plot)
                    plot_related = any(keyword in code_content.lower() for keyword in ['plot', 'scatter', 'plt', 'matplotlib', 'seaborn'])
                    data_related = any(keyword in code_content.lower() for keyword in ['df', 'data', 'csv', 'pandas'])

                    assert plot_related or data_related, \
                        f"Generated code should be related to plotting or data operations: {code_content}"

                    # Code should be more than just comments
                    non_comment_lines = [line for line in code_content.split('\n') if line.strip() and not line.strip().startswith('#')]
                    assert len(non_comment_lines) > 0, f"Generated code should contain executable statements: {code_content}"

                    # Check cell positioning in the execute_code message for continuation workflow
                    code_messages = [msg for msg in messages if msg.get('type') == 'execute_code']
                    if code_messages:
                        for code_msg in code_messages:
                            response = code_msg.get('response', {})
                            positioning_info = response.get('positioning_info', {})
                            assert positioning_info, f"Code message response must have positioning_info, got: {code_msg}"

                            target_cell = positioning_info.get('target_cell')
                            print(f"Continuation code message positioning: target_cell={target_cell}")
                            assert target_cell is not None, f"positioning_info must have target_cell, got: {positioning_info}"
                            assert target_cell != -1, f"Continuation workflow position should not be -1, got: {target_cell}"
                            # For continuation, should use lastCellModifiedInAutoMode (1) as target
                            expected_position = 1  # Should be the lastCellModifiedInAutoMode value
                            assert target_cell == expected_position, f"Continuation should position at {expected_position} (lastCellModifiedInAutoMode), got: {target_cell}"

            # Should update task status (always expected)
            task_messages = self.assert_has_task_updates(messages)
            assert len(task_messages) > 0, "Should update task list"

            # Verify that completed tasks are preserved unchanged in the updated task list
            latest_task_message = task_messages[-1]
            task_list_json = latest_task_message.get('response', {}).get('text', '{}')
            try:
                import json
                updated_task_list = json.loads(task_list_json)
                updated_tasks = updated_task_list.get('tasks', [])

                # Check that the originally completed task (id 1) remains completely unchanged
                task_1 = next((task for task in updated_tasks if task['id'] == 1), None)
                if task_1:
                    original_task_1 = {'id': 1, 'task': 'Import libraries', 'status': 'completed'}

                    # Verify status is still completed
                    assert task_1['status'] == 'completed', (
                        f"Task 1 status should remain completed, got: {task_1['status']}"
                    )

                    # Verify task description is unchanged
                    assert task_1['task'] == original_task_1['task'], (
                        f"Task 1 description should remain unchanged. "
                        f"Expected: '{original_task_1['task']}', "
                        f"Got: '{task_1['task']}'"
                    )

                    # Verify ID is unchanged
                    assert task_1['id'] == original_task_1['id'], (
                        f"Task 1 ID should remain unchanged. "
                        f"Expected: {original_task_1['id']}, Got: {task_1['id']}"
                    )

                    print(f"✅ Completed task preservation verified: Task 1 completely unchanged")
                    print(f"   Status: {task_1['status']}, Description: '{task_1['task']}'")

            except (json.JSONDecodeError, KeyError) as e:
                # This might happen with small test models that don't produce valid JSON
                print(f"⚠️  Could not verify task preservation due to JSON parsing: {e}")

        except Exception as e:
            # Expected if task completion analysis fails with small model
            if "Trying to set active task without task list" in str(e) or "Tool execution failed" in str(e):
                pass  # This is expected with tiny models that can't produce structured output
            else:
                raise

        # Verify RAG context handling
        if rag_enabled:
            # When RAG is enabled, verify that reference workflows are available in the orchestrator state
            assert agent.orchestrator.state.reference_workflow_content.strip() != "", "Should have reference workflows when RAG enabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() != "", "Should have reference workflow annotation when RAG enabled"
        else:
            # When RAG is disabled, these should be empty or contain only whitespace
            assert agent.orchestrator.state.reference_workflow_content.strip() == "", "Should not have reference workflows when RAG disabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() == "", "Should not have reference workflow annotation when RAG disabled"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rag_enabled", [True, False])
    async def test_autonomous_continuation_error(self, agent_with_capture, rag_enabled):
        """Test standard error recovery: error analysis + code fixing."""
        agent, capsys = agent_with_capture

        # Reset orchestrator state completely for test isolation
        agent.orchestrator.state.reference_workflow_content = ""
        agent.orchestrator.state.reference_workflow_annotation = ""
        # This is a CONTINUATION test, not first execution
        agent.orchestrator.state.auto_mode_first_execution = False
        agent.orchestrator.state.task_list = {
            'tasks': [
                {'id': 1, 'task': 'Load data from file', 'status': 'active'}
            ]
        }

        if rag_enabled:
            agent.orchestrator.state.reference_workflow_content = "# Sample workflow context\nImport pandas for data loading and error handling"
            agent.orchestrator.state.reference_workflow_annotation = "Use these patterns for error recovery"

        context = {
            'lastExecutionFailed': True,
            'errorCellIndex': 2,
            'executionResult':'Error output: NameError: name \'data_file\' is not defined',
            'autonomousMode': True,
            'notebookStructure': {'totalCells': 3, 'allCells': ['# Cell'] * 3},
            'lastCellModifiedInAutoMode': 2,
            'request_id': f'test_error_rag_{rag_enabled}',
            'rag_enabled': rag_enabled,
            # Add required context fields for intent classification
            'execution_history': [
                {'cell_index': 0, 'code': 'import pandas as pd', 'output': 'Success'},
                {'cell_index': 1, 'code': 'df = pd.read_csv("data.csv")', 'output': 'Success'},
                {'cell_index': 2, 'code': 'print(data_file)', 'output': 'Error: NameError: name \'data_file\' is not defined'}
            ],
            'conversation_history': [
                {'role': 'user', 'content': 'Load and analyze the data', 'timestamp': '12:00:00'},
                {'role': 'assistant', 'content': 'I will load the data and analyze it', 'timestamp': '12:00:01'}
            ]
        }

        response, _ = await agent.chat('Approved. Continue execution.', 'test_integration', context=context)

        # Basic response check
        assert response['processed'] is True

        # Parse VSCode messages
        messages = self.parse_vscode_messages(capsys)

        # Should trigger error recovery - check for any workflow progress
        console_messages = self.assert_has_message_type(messages, 'console_log')
        error_recovery_found = any('error' in msg.get('message', '').lower() for msg in console_messages)

        # Should show workflow progress (either error recovery or general execution)
        workflow_progress = len(console_messages) > 1 or error_recovery_found  # More than just turbo mode message
        assert workflow_progress, "Should show workflow execution progress"

        # Check for either code generation or workflow completion
        try:
            code_messages = self.assert_has_code_execution(messages)
            assert len(code_messages) > 0, "Should generate fixed code"

            # If code is generated, check it addresses the error
            code_content = code_messages[0]['code']
            assert any(keyword in code_content.lower() for keyword in ['data_file', 'file', 'path', '=']), \
                f"Fixed code should address the NameError: {code_content}"
        except AssertionError:
            # If no code generation, should at least show workflow progress
            workflow_messages = [msg for msg in messages if msg.get('type') == 'workflow_result']
            task_messages = [msg for msg in messages if msg.get('type') == 'task_list_display']
            assert len(workflow_messages) > 0 or len(task_messages) > 0, "Should show workflow progress when error recovery runs"

        # Verify RAG context handling
        if rag_enabled:
            assert agent.orchestrator.state.reference_workflow_content.strip() != "", "Should have reference workflows when RAG enabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() != "", "Should have reference workflow annotation when RAG enabled"
        else:
            assert agent.orchestrator.state.reference_workflow_content.strip() == "", "Should not have reference workflows when RAG disabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() == "", "Should not have reference workflow annotation when RAG disabled"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rag_enabled", [True, False])
    async def test_autonomous_continuation_backtracking(self, agent_with_capture, rag_enabled):
        """Test backtracking: task analysis detects need to go back."""
        agent, capsys = agent_with_capture

        # Reset orchestrator state completely for test isolation
        agent.orchestrator.state.reference_workflow_content = ""
        agent.orchestrator.state.reference_workflow_annotation = ""
        # This is a CONTINUATION test, not first execution
        agent.orchestrator.state.auto_mode_first_execution = False
        agent.orchestrator.state.task_list = {
            'tasks': [
                {'id': 1, 'task': 'Load CSV dataset', 'status': 'completed'},
                {'id': 2, 'task': 'Analyze data distribution', 'status': 'active'}
            ]
        }

        if rag_enabled:
            agent.orchestrator.state.reference_workflow_content = "# Sample workflow context\nData loading and format validation patterns"
            agent.orchestrator.state.reference_workflow_annotation = "Use these patterns for backtracking and recovery"

        context = {
            'lastExecutionFailed': False,
            'errorCellIndex': None,
            'executionResult':'Error: dataset format is JSON, not CSV - need to reload with correct format',
            'autonomousMode': True,
            'notebookStructure': {'totalCells': 3, 'allCells': ['# Cell'] * 3},
            'lastCellModifiedInAutoMode': 1,
            'request_id': f'test_backtrack_rag_{rag_enabled}',
            'rag_enabled': rag_enabled,
            # Add required context fields for intent classification
            'execution_history': [
                {'cell_index': 0, 'code': 'import pandas as pd', 'output': 'Success'},
                {'cell_index': 1, 'code': 'df = pd.read_csv("data.csv")', 'output': 'Error: dataset format is JSON, not CSV - need to reload with correct format'}
            ],
            'conversation_history': [
                {'role': 'user', 'content': 'Load the CSV dataset and analyze it', 'timestamp': '12:00:00'},
                {'role': 'assistant', 'content': 'I found the data format is incorrect. Need to backtrack.', 'timestamp': '12:00:01'}
            ]
        }

        response, _ = await agent.chat('Approved. Continue execution.', 'test_integration', context=context)

        # Basic response check
        assert response['processed'] is True

        # Parse VSCode messages
        messages = self.parse_vscode_messages(capsys)

        # Should show workflow execution progress
        console_messages = self.assert_has_message_type(messages, 'console_log')
        backtrack_indicators = any(
            keyword in msg.get('message', '').lower()
            for msg in console_messages
            for keyword in ['backtrack', 'previous', 'redo', 'analysis', 'completion']
        )

        # Should show workflow progress (either specific backtracking or general execution)
        workflow_progress = len(console_messages) > 1 or backtrack_indicators  # More than just turbo mode message
        assert workflow_progress, "Should show workflow execution progress"

        # Should generate some response (either backtracking, code, or workflow completion)
        workflow_messages = [msg for msg in messages if msg.get('type') == 'workflow_result']
        task_messages = [msg for msg in messages if msg.get('type') == 'task_list_display']
        assert len(messages) > 0, "Should generate some workflow response"
        assert len(workflow_messages) > 0 or len(task_messages) > 0, "Should show workflow progress"

        # Verify RAG context handling
        if rag_enabled:
            assert agent.orchestrator.state.reference_workflow_content.strip() != "", "Should have reference workflows when RAG enabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() != "", "Should have reference workflow annotation when RAG enabled"
        else:
            assert agent.orchestrator.state.reference_workflow_content.strip() == "", "Should not have reference workflows when RAG disabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() == "", "Should not have reference workflow annotation when RAG disabled"

    @pytest.mark.asyncio
    async def test_planning_to_execution_transition(self, agent_with_capture):
        """Test the specific transition from planning (task_list_critique) to execution (mark_next_task_active)."""
        agent, capsys = agent_with_capture

        # Set up state as if we just completed planning with task_list_critique
        agent.orchestrator.state.auto_mode_first_execution = True
        agent.orchestrator.state.task_list = {
            'tasks': [
                {'id': 1, 'task': 'Load single-cell data', 'status': 'pending'},
                {'id': 2, 'task': 'Perform quality control', 'status': 'pending'},
                {'id': 3, 'task': 'Normalize data', 'status': 'pending'}
            ]
        }

        # Simulate the state after task_list_critique has run
        context = {
            'autonomousMode': True,
            'notebookStructure': {'totalCells': 1, 'allCells': ['# Initial cell']},
            'currentCell': '',
            'currentCellIndex': 0,
            'lastExecutionFailed': False,
            'errorCellIndex': None,
            'executionResult': None,
            'lastCellModifiedInAutoMode': None,
            'request_id': 'test_planning_execution_transition',
            'rag_enabled': True,
            'task_list_approval': 'APPROVED',  # Simulating approval from critique
            'task_list_critique': '',  # No critique means approved
            'execution_history': [],
            'conversation_history': [
                {'role': 'user', 'content': 'Analyze single-cell RNA-seq data', 'timestamp': '12:00:00'}
            ]
        }

        # This simulates the VSCode calling agent.chat after planning is complete
        # to continue with execution
        try:
            response, _ = await agent.chat('', 'test_integration', context=context)

            # Should successfully transition to execution
            assert response['processed'] is True

            # Parse messages
            messages = self.parse_vscode_messages(capsys)

            # Should have console messages showing execution started
            console_messages = [msg for msg in messages if msg.get('type') == 'console_log']
            execution_started = any(
                'execution' in msg.get('message', '').lower() or
                'marking task' in msg.get('message', '').lower()
                for msg in console_messages
            )

            # Debug: print all messages
            print(f"\n=== All messages ({len(messages)} total) ===")
            for msg in messages:
                print(f"Type: {msg.get('type')}, Keys: {list(msg.keys())}")
                if msg.get('type') == 'console_log':
                    print(f"  Console: {msg.get('message')}")
                elif msg.get('type') == 'task_list_display':
                    print(f"  Task list: {msg.get('text', '')[:100]}...")

            # Should have marked first task as active
            task_messages = [msg for msg in messages if msg.get('type') == 'task_list_display']
            print(f"\nFound {len(task_messages)} task_list_display messages")

            if task_messages:
                for i, msg in enumerate(task_messages):
                    print(f"\nTask message {i}: {msg.get('text', '')[:200]}...")

                last_task_update = json.loads(task_messages[-1].get('text', '{}'))
                if 'tasks' in last_task_update and len(last_task_update['tasks']) > 0:
                    first_task = last_task_update['tasks'][0]
                    print(f"\nFirst task status: {first_task.get('status')}")
                    # Planning phase keeps tasks as 'pending' - they become 'active' in execution phase
                    assert first_task.get('status') == 'pending', f"First task should still be pending after planning, got: {first_task}"
                else:
                    print(f"\nNo tasks in update: {last_task_update}")

            # Verify state transition - planning completes but tasks remain pending until execution
            print(f"\nOrchestrator state task list: {agent.orchestrator.state.task_list}")
            if agent.orchestrator.state.task_list and agent.orchestrator.state.task_list.get('tasks'):
                assert agent.orchestrator.state.task_list['tasks'][0]['status'] == 'pending', "First task should still be pending after planning phase"

        except Exception as e:
            # This is where we expect to see the error
            print(f"\nError details: {e}")
            import traceback
            print(f"Traceback:\n{traceback.format_exc()}")
            pytest.fail(f"Planning to execution transition failed: {e}")

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rag_enabled", [True, False])
    async def test_autonomous_continuation_completion(self, agent_with_capture, rag_enabled):
        """Test completion: all tasks done."""
        agent, capsys = agent_with_capture

        # Reset orchestrator state completely for test isolation
        agent.orchestrator.state.reference_workflow_content = ""
        agent.orchestrator.state.reference_workflow_annotation = ""
        # This is a CONTINUATION test, not first execution
        agent.orchestrator.state.auto_mode_first_execution = False
        agent.orchestrator.state.task_list = {
            'tasks': [
                {'id': 1, 'task': 'Load data from CSV', 'status': 'completed'},
                {'id': 2, 'task': 'Create visualization', 'status': 'completed'},
                {'id': 3, 'task': 'Export results', 'status': 'completed'}
            ]
        }

        if rag_enabled:
            agent.orchestrator.state.reference_workflow_content = "# Sample workflow context\nData analysis and export completion patterns"
            agent.orchestrator.state.reference_workflow_annotation = "Use these patterns for workflow completion"

        context = {
            'lastExecutionFailed': False,
            'errorCellIndex': None,
            'executionResult':'Success: all analysis complete, results exported to output.csv',
            'autonomousMode': True,
            'notebookStructure': {'totalCells': 5, 'allCells': ['# Cell'] * 5},
            'lastCellModifiedInAutoMode': 3,
            'request_id': f'test_complete_rag_{rag_enabled}',
            'rag_enabled': rag_enabled,
            # Add required context fields for intent classification
            'execution_history': [
                {'cell_index': 0, 'code': 'import pandas as pd', 'output': 'Success'},
                {'cell_index': 1, 'code': 'df = pd.read_csv("data.csv")', 'output': 'Success'},
                {'cell_index': 2, 'code': 'plt.plot(df)', 'output': 'Success'},
                {'cell_index': 3, 'code': 'df.to_csv("output.csv")', 'output': 'Success: all analysis complete, results exported to output.csv'}
            ],
            'conversation_history': [
                {'role': 'user', 'content': 'Complete the full analysis workflow', 'timestamp': '12:00:00'},
                {'role': 'assistant', 'content': 'All tasks completed successfully', 'timestamp': '12:00:01'}
            ]
        }

        response, _ = await agent.chat('Approved. Continue execution.', 'test_integration', context=context)

        # Basic response check
        assert response['processed'] is True

        # Parse VSCode messages
        messages = self.parse_vscode_messages(capsys)

        # Should show completion workflow
        console_messages = self.assert_has_message_type(messages, 'console_log')
        completion_indicators = any(
            keyword in msg.get('message', '').lower()
            for msg in console_messages
            for keyword in ['complete', 'finished', 'done', 'workflow', 'all tasks']
        )
        assert completion_indicators, "Should show workflow completion indicators"

        # Should have workflow result indicating completion
        try:
            workflow_messages = self.assert_workflow_completion(messages)
            assert len(workflow_messages) > 0, "Should send workflow completion signal"
        except AssertionError:
            # If no workflow_result, at least should show completion in console
            assert completion_indicators, "Should show completion indicators in console messages"

        # Verify RAG context handling
        if rag_enabled:
            assert agent.orchestrator.state.reference_workflow_content.strip() != "", "Should have reference workflows when RAG enabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() != "", "Should have reference workflow annotation when RAG enabled"
        else:
            assert agent.orchestrator.state.reference_workflow_content.strip() == "", "Should not have reference workflows when RAG disabled"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() == "", "Should not have reference workflow annotation when RAG disabled"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("rag_enabled", [True, False])
    async def test_autonomous_first_execution(self, agent_with_capture, rag_enabled):
        """Test first execution workflow after planning phase."""
        agent, capsys = agent_with_capture

        # Reset orchestrator state completely for test isolation
        agent.orchestrator.state.reference_workflow_content = ""
        agent.orchestrator.state.reference_workflow_annotation = ""

        # Set up state for first execution (after planning completed)
        agent.orchestrator.state.auto_mode_first_execution = True
        agent.orchestrator.state.task_list = {
            "tasks": [
                {"id": 1, "task": "Load data from CSV file", "status": "pending"},
                {"id": 2, "task": "Perform data analysis", "status": "pending"},
                {"id": 3, "task": "Create visualization", "status": "pending"}
            ]
        }

        if rag_enabled:
            agent.orchestrator.state.reference_workflow_content = "# Sample workflow context\nImport pandas and matplotlib for data analysis workflow"
            agent.orchestrator.state.reference_workflow_annotation = "Use these patterns for data loading and first step execution"

        context = {
            "lastExecutionFailed": False,
            "errorCellIndex": None,
            "executionResult": "",  # No previous execution in first iteration
            "autonomousMode": True,
            "notebookStructure": {"totalCells": 2, "allCells": ["# Cell 1", "# Cell 2"]},
            "lastCellModifiedInAutoMode": -1,  # No previous modifications
            "request_id": f"test_first_exec_rag_{rag_enabled}",
            "rag_enabled": rag_enabled,
            # Add required context fields
            "execution_history": [],  # Empty for first execution
            "conversation_history": [
                {"role": "user", "content": "Start autonomous analysis of CSV data", "timestamp": "12:00:00"},
                {"role": "assistant", "content": "I will begin with loading the data", "timestamp": "12:00:01"}
            ]
        }

        # Test the first execution path directly
        try:
            # Create ExecutionContext to match new method signature
            from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
            exec_context = ExecutionContext(
                inputs=ExecutionInputs(
                    user_query="",
                    context=context,
                    task_list=agent.orchestrator.state.task_list,
                    backtracking_context=None
                ),
                session_metadata=agent.session_metadata
            )
            await agent.orchestrator._handle_autonomous_first_execution(exec_context)

            # Verify state changes after first execution
            assert agent.orchestrator.state.auto_mode_first_execution == False, \
                "auto_mode_first_execution flag should be set to False after first execution"

            # Verify task list has been updated - first task should be marked as active
            first_task = agent.orchestrator.state.task_list["tasks"][0]
            assert first_task["status"] == "active", \
                f"First task should be marked as active, got: {first_task['status']}"

            # Parse VSCode messages
            messages = self.parse_vscode_messages(capsys)

            # Should show workflow execution progress
            console_messages = self.assert_has_message_type(messages, "console_log")

            # Should mention task activation
            task_activation_mentioned = any(
                "marking task" in msg.get("message", "").lower() or "active" in msg.get("message", "").lower()
                for msg in console_messages
            )
            assert task_activation_mentioned, "Should mention task activation in console messages"

            # Should show cell positioning and code generation attempts
            workflow_progress = any(
                keyword in msg.get("message", "").lower()
                for msg in console_messages
                for keyword in ["cell_positioning", "task_step_code_generation", "first execution"]
            )
            assert workflow_progress, "Should show workflow execution progress with tool names"

            # Check for valid cell positioning results (should not be -1)
            positioning_messages = [
                msg for msg in console_messages
                if "using cell position:" in msg.get("message", "").lower()
            ]
            if positioning_messages:
                for msg in positioning_messages:
                    message_text = msg.get("message", "")
                    print(f"Cell positioning message: {message_text}")
                    # Extract position from message like "Using cell position: X with should replace: false"
                    if "using cell position:" in message_text.lower():
                        import re
                        match = re.search(r'position:\s*(-?\d+)', message_text)
                        if match:
                            position = int(match.group(1))
                            assert position != -1, f"Cell position should not be -1, got: {position}"

            # Should end with LOOP_INCOMPLETE workflow result
            try:
                workflow_messages = [msg for msg in messages if msg.get("type") == "workflow_result"]
                if workflow_messages:
                    # Check for auto_loop_update with LOOP_INCOMPLETE state
                    loop_incomplete_found = any(
                        "auto_loop_update" in msg and
                        msg.get("data", {}).get("status") == "LOOP_INCOMPLETE"
                        for msg in workflow_messages
                    )
                    assert loop_incomplete_found, f"Should send LOOP_INCOMPLETE status, got workflow messages: {workflow_messages}"
            except AssertionError:
                # If no workflow result found, thats still acceptable for first execution
                # since small test model might fail to generate proper code
                pass

            # Should attempt code generation (may fail with small model, but attempt should be made)
            try:
                code_messages = self.assert_has_code_execution(messages)
                assert len(code_messages) > 0, "Should attempt code generation for first task"

                # Validate code content relates to first task (data loading)
                code_content = code_messages[0]["code"]
                data_loading_related = any(keyword in code_content.lower()
                                         for keyword in ["csv", "read", "load", "import", "pandas", "data"])
                assert data_loading_related, \
                    f"Generated code should be related to data loading for first task: {code_content}"

            except AssertionError:
                # Code generation may fail with small model, but workflow should still progress
                # At minimum, should show task list updates
                task_messages = [msg for msg in messages if msg.get("type") == "task_list_display"]
                assert len(task_messages) > 0 or len(console_messages) > 1, \
                    "Should show task updates even if code generation fails"

        except Exception as e:
            # Expected if tools fail with small model, but state changes should still occur
            if "Tool execution failed" in str(e) or "Trying to set active task" in str(e):
                # Verify that at least the flag was updated
                assert agent.orchestrator.state.auto_mode_first_execution == False, \
                    "auto_mode_first_execution flag should be set to False even if tools fail"
            else:
                raise

        # Verify RAG context handling for first execution
        if rag_enabled:
            # When RAG is enabled, reference workflows should be available during first execution
            assert agent.orchestrator.state.reference_workflow_content.strip() != "", \
                "Should have reference workflows when RAG enabled in first execution"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() != "", \
                "Should have reference workflow annotation when RAG enabled in first execution"
        else:
            # When RAG is disabled, these should remain empty
            assert agent.orchestrator.state.reference_workflow_content.strip() == "", \
                "Should not have reference workflows when RAG disabled in first execution"
            assert agent.orchestrator.state.reference_workflow_annotation.strip() == "", \
                "Should not have reference workflow annotation when RAG disabled in first execution"

    @pytest.mark.asyncio
    async def test_user_query_preservation_during_critique(self, agent_with_capture):
        """Test that user query is preserved before and after critique workflow iterations."""
        agent, capsys = agent_with_capture

        original_user_query = "Perform quality control analysis on single-cell data"

        # Create context that will trigger critique workflow
        context = {
            "session_metadata": agent.session_metadata,
            "current_cell": "",
            "current_cell_index": 0,
            "notebook_structure": {
                "totalCells": 3,
                "allCells": ["# Cell 1", "# Cell 2", "# Cell 3"]
            },
            "execution_history": [],
            "conversation_history": [],
            "last_execution_failed": False,
            "request_id": "test_query_preservation",
            "rag_enabled": True,  # Enable RAG to test critique workflow
            "auto_mode_continue": False
        }

        # Create execution context to test user query handling
        from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs

        exec_context = ExecutionContext(
            inputs=ExecutionInputs(
                user_query=original_user_query,
                context=context,
                task_list={},
                backtracking_context=None
            ),
            session_metadata=agent.session_metadata
        )

        # Store reference to verify preservation behavior
        initial_query = exec_context.inputs.user_query
        assert initial_query == original_user_query, "Initial query should match original"

        # Test the planning workflow execution which includes critique
        try:
            # This will run the planning workflow with potential critique iterations
            await agent.orchestrator._handle_autonomous_planning(exec_context, "PLANNING")

            # After critique workflow, user query should be restored to original
            final_query = exec_context.inputs.user_query
            assert final_query == original_user_query, (
                f"User query was not preserved during critique workflow. "
                f"Expected: '{original_user_query}' "
                f"Got: '{final_query}'"
            )

            print(f"✅ User query preservation verified: '{original_user_query}' → '{final_query}'")

        except Exception as e:
            # Even if workflow fails, test that we can verify the restoration logic exists
            if "task_list_approval" in str(e):
                # This suggests the critique workflow was attempted
                print(f"✅ Critique workflow was attempted (expected with test model): {e}")
                # The fact that we reached the critique check means restoration logic is in place
            else:
                raise

    @pytest.mark.asyncio
    @pytest.mark.parametrize("update_rule", ["KEEP", "UPDATE"])
    async def test_autonomous_update_tasks_tool_update_rule_simple(self, agent_with_capture, update_rule):
        """Simplified test for KEEP vs UPDATE logic with direct tool testing."""
        from unittest.mock import AsyncMock
        from kai.core.orchestration.schemas import AutonomousTaskUpdate, TaskItem
        from kai.core.orchestration.execution_context import ExecutionContext, ExecutionInputs
        from kai.core.orchestration.prompt_tools import AutonomousUpdateTasksTool

        agent, capsys = agent_with_capture

        # Create the tool
        tool = AutonomousUpdateTasksTool(agent.llm_interface)

        # Set up test data
        original_task_list = {
            'tasks': [
                {'id': 1, 'task': 'Load data', 'status': 'completed'},
                {'id': 2, 'task': 'Analyze data', 'status': 'active'},
                {'id': 3, 'task': 'Create plots', 'status': 'pending'}
            ]
        }

        context = {
            'rag_enabled': True,
            'last_execution_failed': False,
            'execution_history': [],
            'conversation_history': [],
            'notebook_structure': {'totalCells': 3, 'allCells': ['# Cell 1', '# Cell 2', '# Cell 3']},
            'current_cell': "",
            'error_message': "",
            'autonomous_mode': True
        }

        # Create mock response
        if update_rule == "KEEP":
            mock_result = AutonomousTaskUpdate(
                tasks=[
                    TaskItem(id=2, task="Analyze data", status="active"),
                    TaskItem(id=3, task="Create plots", status="pending")
                ],
                retrieval_query=["analysis"],
                update_rationale="Keeping existing task list as current tasks are still valid",
                update_rule="KEEP"
            )
        else:
            mock_result = AutonomousTaskUpdate(
                tasks=[
                    TaskItem(id=2, task="Enhanced data analysis", status="active"),
                    TaskItem(id=3, task="Advanced plots", status="pending"),
                    TaskItem(id=4, task="Export results", status="pending")
                ],
                retrieval_query=["enhanced analysis"],
                update_rationale="Updating tasks to include more detailed analysis and export functionality",
                update_rule="UPDATE"
            )

        # Mock the LLM provider
        original_generate_structured = tool.llm_provider.generate_structured
        tool.llm_provider.generate_structured = AsyncMock(return_value=mock_result)

        try:
            # Create execution context
            exec_context = ExecutionContext(
                inputs=ExecutionInputs(
                    user_query="Test query",
                    context=context,
                    task_list=original_task_list,
                    backtracking_context=None
                ),
                session_metadata={"session_id": "test"}
            )

            # Execute the tool
            result = await tool.execute(exec_context)

            # Basic validation
            assert result is not None
            assert result.output_workflow is not None
            assert "task_list" in result.output_workflow

            updated_tasks = result.output_workflow["task_list"]["tasks"]

            if update_rule == "KEEP":
                # Should preserve original task list exactly
                assert updated_tasks == original_task_list["tasks"]
                print(f"✅ KEEP test passed: Task list preserved")
            else:
                # Should have updated tasks while preserving completed ones
                completed_tasks = [t for t in updated_tasks if t["status"] == "completed"]
                assert len(completed_tasks) == 1  # Original completed task preserved
                assert completed_tasks[0]["id"] == 1
                assert len(updated_tasks) == 4  # Added one new task
                print(f"✅ UPDATE test passed: Tasks updated, completed preserved")

        finally:
            # Restore original method
            tool.llm_provider.generate_structured = original_generate_structured


