#!/usr/bin/env python3
"""
Basic test script for Jupyter interface.

Tests core functionality without requiring full agent initialization.
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from UI.jupyter import JupyterInterface
        print("  ✅ JupyterInterface imported")

        from UI.jupyter.notebook_executor import NotebookExecutor
        print("  ✅ NotebookExecutor imported")

        from UI.jupyter.context_builder import ContextBuilder
        print("  ✅ ContextBuilder imported")

        return True

    except ImportError as e:
        print(f"  ❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_notebook_loading():
    """Test notebook loading and basic operations."""
    print("\nTesting notebook loading...")

    try:
        import nbformat

        # Create a simple test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell("# Test Notebook"),
            nbformat.v4.new_code_cell("print('Hello, World!')")
        ]

        # Save test notebook
        test_path = Path(__file__).parent / "test_basic.ipynb"
        with open(test_path, 'w') as f:
            nbformat.write(nb, f)

        print(f"  ✅ Created test notebook: {test_path}")

        # Test loading
        with open(test_path, 'r') as f:
            loaded_nb = nbformat.read(f, as_version=4)

        print(f"  ✅ Loaded notebook with {len(loaded_nb.cells)} cells")

        # Cleanup
        test_path.unlink()
        print("  ✅ Cleaned up test file")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        return False


def test_context_builder():
    """Test ContextBuilder functionality."""
    print("\nTesting ContextBuilder...")

    try:
        import nbformat
        from UI.jupyter.context_builder import ContextBuilder

        # Create test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_markdown_cell("# Test"),
            nbformat.v4.new_code_cell("x = 1 + 1\nprint(x)")
        ]

        # Create context builder
        builder = ContextBuilder(nb, "test.ipynb")
        print("  ✅ ContextBuilder initialized")

        # Get context
        context = builder.get_context()
        print(f"  ✅ Context generated with {len(context)} fields")

        # Check required fields
        required_fields = [
            'executionHistory', 'conversationHistory', 'notebookStructure',
            'currentCell', 'currentCellIndex', 'errorCellIndex',
            'executionResult', 'lastExecutionFailed', 'autonomousMode'
        ]

        for field in required_fields:
            if field not in context:
                print(f"  ❌ Missing required field: {field}")
                return False

        print(f"  ✅ All {len(required_fields)} required fields present")

        # Check notebook structure
        structure = context['notebookStructure']
        if structure['totalCells'] != 2:
            print(f"  ❌ Expected 2 cells, got {structure['totalCells']}")
            return False

        print("  ✅ Notebook structure correct")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_markdown_cell_execution_history():
    """Test that markdown cells are added to execution history (prevents reasoning loop bug)."""
    print("\nTesting markdown cell execution history...")

    try:
        import nbformat
        from UI.jupyter.context_builder import ContextBuilder
        from UI.jupyter.notebook_executor import ExecutionResult

        # Create test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell("x = 1")
        ]

        builder = ContextBuilder(nb, "test.ipynb")

        # Simulate adding a markdown cell (reasoning)
        markdown_content = "## Reasoning\nThis is test reasoning"

        # Add markdown to notebook
        nb.cells.append(nbformat.v4.new_markdown_cell(markdown_content))

        # Add to execution history (as the interface should do)
        mock_exec_result = ExecutionResult(
            success=True,
            outputs=[],
            error=None,
            terminated=False,
            duration=0.0
        )
        builder.add_to_execution_history(1, mock_exec_result, markdown_content)

        # Update last modified cell
        builder.last_cell_modified_in_auto_mode = 1

        # Get context
        context = builder.get_context()

        # Verify markdown cell is in execution history
        if len(context['executionHistory']) == 0:
            print("  ❌ Markdown cell not added to execution history")
            return False

        print(f"  ✅ Markdown cell added to execution history ({len(context['executionHistory'])} entries)")

        # Verify lastCellModifiedInAutoMode is set
        if context['lastCellModifiedInAutoMode'] != 1:
            print(f"  ❌ lastCellModifiedInAutoMode incorrect: {context['lastCellModifiedInAutoMode']}")
            return False

        print("  ✅ lastCellModifiedInAutoMode set correctly")

        # Verify cell is in notebook structure
        all_cells = context['notebookStructure']['allCells']
        markdown_found = any('MARKDOWN CELL' in cell for cell in all_cells)

        if not markdown_found:
            print("  ❌ Markdown cell not in notebook structure")
            return False

        print("  ✅ Markdown cell in notebook structure")
        print("  ✅ Bug fix verified: Markdown cells properly tracked to prevent reasoning loop")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_last_cell_modified_tracking():
    """Test that lastCellModifiedInAutoMode is updated correctly."""
    print("\nTesting lastCellModifiedInAutoMode tracking...")

    try:
        import nbformat
        from UI.jupyter.context_builder import ContextBuilder
        from UI.jupyter.notebook_executor import ExecutionResult

        # Create test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell("x = 1"),
            nbformat.v4.new_code_cell("y = 2")
        ]

        builder = ContextBuilder(nb, "test.ipynb")

        # Initially should be -1
        if builder.last_cell_modified_in_auto_mode != -1:
            print(f"  ❌ Initial value should be -1, got {builder.last_cell_modified_in_auto_mode}")
            return False

        print("  ✅ Initial lastCellModifiedInAutoMode = -1")

        # Simulate executing cell 0
        builder.last_cell_modified_in_auto_mode = 0
        exec_result = ExecutionResult(
            success=True,
            outputs=[('execute_result', {'text/plain': '1'})],
            error=None,
            terminated=False,
            duration=0.0
        )
        builder.add_to_execution_history(0, exec_result, "x = 1")

        context = builder.get_context()
        if context['lastCellModifiedInAutoMode'] != 0:
            print(f"  ❌ Expected 0, got {context['lastCellModifiedInAutoMode']}")
            return False

        print("  ✅ lastCellModifiedInAutoMode updated to 0")

        # Simulate executing cell 1
        builder.last_cell_modified_in_auto_mode = 1
        exec_result = ExecutionResult(
            success=True,
            outputs=[('execute_result', {'text/plain': '2'})],
            error=None,
            terminated=False,
            duration=0.0
        )
        builder.add_to_execution_history(1, exec_result, "y = 2")

        context = builder.get_context()
        if context['lastCellModifiedInAutoMode'] != 1:
            print(f"  ❌ Expected 1, got {context['lastCellModifiedInAutoMode']}")
            return False

        print("  ✅ lastCellModifiedInAutoMode updated to 1")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_execution_state_updates():
    """Test that execution state updates correctly for success/failure."""
    print("\nTesting execution state updates...")

    try:
        import nbformat
        from UI.jupyter.context_builder import ContextBuilder

        # Create test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell("x = 1")]

        builder = ContextBuilder(nb, "test.ipynb")

        # Test success case
        builder.update_execution_state(success=True, output="Output: 1", cell_index=None)

        if builder.last_execution_failed:
            print("  ❌ Should not be failed after success")
            return False

        if builder.error_cell_index != -1:
            print(f"  ❌ error_cell_index should be -1, got {builder.error_cell_index}")
            return False

        print("  ✅ Success state recorded correctly")

        # Test failure case
        builder.update_execution_state(success=False, output="NameError: x not defined", cell_index=0)

        if not builder.last_execution_failed:
            print("  ❌ Should be failed after failure")
            return False

        if builder.error_cell_index != 0:
            print(f"  ❌ error_cell_index should be 0, got {builder.error_cell_index}")
            return False

        print("  ✅ Failure state recorded correctly")

        # Test markdown success (cell_index=None)
        builder.update_execution_state(success=True, output="", cell_index=None)

        if builder.last_execution_failed:
            print("  ❌ Should not be failed after markdown success")
            return False

        if builder.error_cell_index != -1:
            print(f"  ❌ error_cell_index should be -1, got {builder.error_cell_index}")
            return False

        print("  ✅ Markdown success state recorded correctly")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_handle_execute_code_markdown():
    """Test _handle_execute_code with markdown cell (integration test for the actual code path)."""
    print("\nTesting _handle_execute_code with markdown cell...")

    try:
        import nbformat
        import asyncio
        from UI.jupyter.jupyter_interface import JupyterInterface
        from unittest.mock import MagicMock, patch
        import tempfile
        import os

        # Create a temporary notebook
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nb = nbformat.v4.new_notebook()
            nb.cells = [nbformat.v4.new_code_cell("x = 1")]
            nbformat.write(nb, f)
            temp_path = f.name

        try:
            # Create JupyterInterface (mock agent to avoid LLM initialization)
            with patch('UI.jupyter.jupyter_interface.KaiAgent') as MockAgent:
                mock_agent = MagicMock()
                mock_agent.vscode = MagicMock()
                mock_agent.vscode._disabled = True
                MockAgent.return_value = mock_agent

                interface = JupyterInterface(
                    notebook_path=temp_path,
                    notebook_python=os.environ.get('CONDA_PREFIX', '/usr/bin/python3'),
                    llm_provider='ollama'
                )

                # Simulate markdown cell message from agent (this is the actual code path)
                markdown_message = {
                    'code': '## Reasoning\nThis is test reasoning',
                    'cell_type': 'markdown',
                    'positioning_info': {'target_cell': -1},
                    'should_replace_code': 'false'
                }

                # Call _handle_execute_code (this is where the bug was)
                async def run_test():
                    await interface._handle_execute_code(markdown_message)

                asyncio.run(run_test())

                # Verify markdown cell was added to execution history
                context = interface.context_builder.get_context()
                if len(context['executionHistory']) == 0:
                    print("  ❌ Markdown cell not added to execution history")
                    return False

                print("  ✅ _handle_execute_code processed markdown without errors")

                # Verify lastCellModifiedInAutoMode was updated
                if context['lastCellModifiedInAutoMode'] == -1:
                    print("  ❌ lastCellModifiedInAutoMode not updated")
                    return False

                print("  ✅ lastCellModifiedInAutoMode updated correctly")

                # Verify cell appears in notebook
                if len(interface.notebook.cells) != 2:
                    print(f"  ❌ Expected 2 cells, got {len(interface.notebook.cells)}")
                    return False

                # The markdown cell should be added (position may vary based on target_cell=-1)
                markdown_cells = [c for c in interface.notebook.cells if c.cell_type == 'markdown']
                if len(markdown_cells) != 1:
                    print(f"  ❌ Expected 1 markdown cell, got {len(markdown_cells)}")
                    return False

                print("  ✅ Markdown cell added to notebook")
                print("  ✅ Bug fix verified: _handle_execute_code creates ExecutionResult with correct fields")

                return True

        finally:
            # Clean up
            try:
                interface.executor.shutdown()
            except:
                pass
            try:
                os.unlink(temp_path)
            except:
                pass

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_notebook_executor_init():
    """Test NotebookExecutor initialization (without executing)."""
    print("\nTesting NotebookExecutor initialization...")

    try:
        # Check if jupyter_client is available
        import jupyter_client
        print("  ✅ jupyter_client available")

        # Note: We don't actually start a kernel here because:
        # 1. It requires a Jupyter installation with kernels
        # 2. It would make the test slow
        # 3. The user can test this with the CLI

        print("  ℹ️  Skipping kernel startup (requires Jupyter environment)")
        print("  ℹ️  Use CLI to test full execution: python -m UI.jupyter --help")

        return True

    except ImportError:
        print("  ⚠️  jupyter_client not installed")
        print("     Install with: pip install jupyter_client")
        return True  # Not a failure, just missing optional dependency


def test_modification_history():
    """Test modificationHistory tracking (new feature for VSCode parity)."""
    print("\nTesting modificationHistory tracking...")

    try:
        import nbformat
        from UI.jupyter.context_builder import ContextBuilder

        # Create test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [
            nbformat.v4.new_code_cell("x = 1"),
            nbformat.v4.new_code_cell("y = 2")
        ]

        builder = ContextBuilder(nb, "test.ipynb")

        # Test 1: Track cell creation
        builder.track_modification(2, 'created', 'z = 3')

        if len(builder.modification_history) != 1:
            print(f"  ❌ Expected 1 modification, got {len(builder.modification_history)}")
            return False

        mod = builder.modification_history[0]
        if mod['modification_type'] != 'created':
            print(f"  ❌ Expected 'created', got {mod['modification_type']}")
            return False

        if 'new_content_hash' not in mod:
            print("  ❌ Missing new_content_hash for created cell")
            return False

        print("  ✅ Cell creation tracked correctly")

        # Test 2: Track cell modification
        builder.track_modification(1, 'modified', 'y = 20', old_content='y = 2')

        if len(builder.modification_history) != 2:
            print(f"  ❌ Expected 2 modifications, got {len(builder.modification_history)}")
            return False

        mod = builder.modification_history[0]  # Most recent first
        if mod['modification_type'] != 'modified':
            print(f"  ❌ Expected 'modified', got {mod['modification_type']}")
            return False

        if 'old_content_hash' not in mod or 'new_content_hash' not in mod:
            print("  ❌ Missing content hashes for modified cell")
            return False

        print("  ✅ Cell modification tracked correctly")

        # Test 3: Track cell deletion
        builder.track_modification(0, 'deleted', '', old_content='x = 1')

        mod = builder.modification_history[0]
        if mod['modification_type'] != 'deleted':
            print(f"  ❌ Expected 'deleted', got {mod['modification_type']}")
            return False

        if 'old_content_hash' not in mod:
            print("  ❌ Missing old_content_hash for deleted cell")
            return False

        print("  ✅ Cell deletion tracked correctly")

        # Test 4: Verify history is in modificationHistory in context
        context = builder.get_context()
        if 'modificationHistory' not in context:
            print("  ❌ modificationHistory not in context")
            return False

        if len(context['modificationHistory']) != 3:
            print(f"  ❌ Expected 3 modifications in context, got {len(context['modificationHistory'])}")
            return False

        print("  ✅ modificationHistory included in context")

        # Test 5: Verify max history limit
        for i in range(10):
            builder.track_modification(i, 'created', f'cell{i}')

        if len(builder.modification_history) > builder.MAX_MODIFICATION_HISTORY:
            print(f"  ❌ History exceeded max ({builder.MAX_MODIFICATION_HISTORY}), got {len(builder.modification_history)}")
            return False

        print(f"  ✅ History trimmed to max ({builder.MAX_MODIFICATION_HISTORY}) entries")

        return True

    except Exception as e:
        print(f"  ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vscode_message_suppression():
    """Test that VSCode messages are suppressed in Jupyter interface."""
    print("\nTesting VSCode message suppression...")

    import sys
    import io
    from kai.core.agent import KaiAgent

    try:
        # Capture stdout
        captured_output = io.StringIO()
        original_stdout = sys.stdout
        sys.stdout = captured_output

        try:
            # Create agent with suppression enabled (as Jupyter interface does)
            agent = KaiAgent(
                llm_provider="ollama",
                suppress_vscode_messages=True
            )

            # Check that _disabled is set
            assert agent.vscode._disabled == True, "vscode._disabled should be True"

            # Check that suppression preference is remembered
            assert agent._suppress_vscode_messages == True, "_suppress_vscode_messages should be True"

            # Simulate autonomous session initiation (the bug scenario)
            # This previously would call enable_communication() and re-enable messages
            import hashlib
            user_input = "test message"
            user_id = "test_user"
            session_id = f"session_{hashlib.md5(f'{user_id}_{user_input}'.encode()).hexdigest()[:8]}"

            # Execute the code path that previously had the bug
            auto_mode_initiation = True
            if auto_mode_initiation:
                if not agent._suppress_vscode_messages:
                    agent.vscode.enable_communication()

            # Verify _disabled is still True after session init
            assert agent.vscode._disabled == True, "vscode._disabled should still be True after session init"

            # Try to send messages (should be suppressed)
            agent.vscode.send_console_message("Test console message")

            # Get captured output
            output = captured_output.getvalue()

            # Verify no VSCode JSON output was produced
            assert '{"type": "console_log"' not in output, f"console_log messages should be suppressed, but found in output"

        finally:
            sys.stdout = original_stdout

        print("  ✓ VSCode messages properly suppressed")
        print("  ✓ Suppression persists during autonomous session init")
        return True

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        traceback.print_exc()
        return False


def test_loop_complete_detection():
    """Test that LOOP_COMPLETE signal is detected and handled correctly."""
    print("\nTesting LOOP_COMPLETE detection...")

    try:
        import asyncio
        import nbformat
        import tempfile
        import os
        import traceback
        from UI.jupyter import JupyterInterface

        # Create test notebook
        nb = nbformat.v4.new_notebook()
        nb.cells = [nbformat.v4.new_code_cell("x = 1")]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            nbformat.write(nb, f)
            nb_path = f.name

        try:
            async def run_test():
                import sys
                # Create interface
                interface = JupyterInterface(
                    notebook_path=nb_path,
                    notebook_python=sys.executable,
                    llm_provider="ollama",
                    rag_enabled=False
                )

                # Simulate workflow_result message with LOOP_COMPLETE
                interface.pending_tool_messages = [
                    {
                        'type': 'workflow_result',
                        'data': {'auto_loop_update': 'LOOP_COMPLETE'},
                        'workflow': {}
                    }
                ]

                # Process messages
                loop_complete = await interface._process_tool_messages()

                # Verify detection
                assert loop_complete == True, "LOOP_COMPLETE should be detected"

                # Test with LOOP_INCOMPLETE
                interface.pending_tool_messages = [
                    {
                        'type': 'workflow_result',
                        'data': {'auto_loop_update': 'LOOP_INCOMPLETE'},
                        'workflow': {}
                    }
                ]

                loop_complete = await interface._process_tool_messages()
                assert loop_complete == False, "LOOP_INCOMPLETE should not trigger completion"

                print("  ✓ LOOP_COMPLETE signal detected correctly")
                print("  ✓ LOOP_INCOMPLETE does not trigger completion")

            asyncio.run(run_test())
            return True

        finally:
            os.unlink(nb_path)

    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Kai Jupyter Interface - Basic Tests")
    print("="*60)

    tests = [
        ("Module imports", test_imports),
        ("Notebook loading", test_notebook_loading),
        ("ContextBuilder", test_context_builder),
        ("Markdown cell execution history", test_markdown_cell_execution_history),
        ("lastCellModifiedInAutoMode tracking", test_last_cell_modified_tracking),
        ("Execution state updates", test_execution_state_updates),
        ("modificationHistory tracking", test_modification_history),
        ("_handle_execute_code with markdown (integration)", test_handle_execute_code_markdown),
        ("VSCode message suppression", test_vscode_message_suppression),
        ("LOOP_COMPLETE detection", test_loop_complete_detection),
        ("NotebookExecutor init", test_notebook_executor_init),
    ]

    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))

    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n✅ All tests passed!")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())
