"""Test that retrieval_queries is properly initialized and routed."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from kai.core.agent import KaiAgent
from kai.core.orchestration.langgraph_orchestrator import LangGraphOrchestrator


@pytest.fixture
def mock_llm():
    """Mock LLM interface."""
    llm = Mock()
    llm.provider_name = "ollama"
    return llm


@pytest.fixture
def mock_kb():
    """Mock knowledge base."""
    return Mock()


@pytest.fixture
def agent(mock_llm, mock_kb):
    """Create agent with mocked dependencies."""
    from pathlib import Path
    from kai.config.settings import Settings

    # Create proper settings mock with all required fields
    settings = Mock(spec=Settings)
    settings.DISABLE_TURBO = True
    settings.KNOWLEDGE_BASE_PATH = Path("/tmp/test_kb")
    settings.NOTEBOOK_SUMMARIES_PATH = Path("/tmp/test_summaries")

    # Patch create_knowledge_base to avoid real ChromaDB initialization
    with patch('kai.core.agent.create_knowledge_base', return_value=mock_kb):
        agent = KaiAgent(settings=settings)

    agent.llm_interface = mock_llm
    agent.knowledge_base = mock_kb
    agent.orchestrator = Mock(spec=LangGraphOrchestrator)
    agent.orchestrator.process_request = AsyncMock()
    agent.session_metadata = {
        "active": False,
        "session_id": None,
        "session_timestamp": None,
        "notebook_uri": None,
        "iteration_counter": 0,
    }
    return agent


@pytest.mark.asyncio
async def test_retrieval_queries_initialized_on_first_iteration(agent):
    """Test that retrieval_queries is set to [user_input] on first iteration."""

    user_input = "Analyze single-cell RNA-seq data and perform clustering"
    context = {
        'request_id': 'test-123',
        'executionHistory': [],
        'conversationHistory': [],
        'notebookStructure': {'totalCells': 0, 'allCells': []},
        'currentCell': None,
        'currentCellIndex': None,
        'autonomousMode': True,
        'autonomousModeContinue': False,  # First iteration
        'ragEnabled': True,
    }

    # Call chat (session_id=None triggers new session)
    await agent.chat(user_input, session_id=None, context=context)

    # Verify orchestrator.process_request was called
    assert agent.orchestrator.process_request.called

    # Get the context passed to orchestrator
    call_args = agent.orchestrator.process_request.call_args
    passed_context = call_args.kwargs['context']

    # Verify retrieval_queries was initialized
    assert 'retrieval_queries' in passed_context
    assert passed_context['retrieval_queries'] == [user_input]
    print(f"✅ retrieval_queries initialized: {passed_context['retrieval_queries']}")


@pytest.mark.asyncio
async def test_retrieval_queries_not_set_on_continue(agent):
    """Test that retrieval_queries is NOT set on subsequent iterations (let checkpointer provide it)."""

    user_input = "Continue task"
    context = {
        'request_id': 'test-456',
        'executionHistory': [],
        'conversationHistory': [],
        'notebookStructure': {'totalCells': 5, 'allCells': []},
        'currentCell': None,
        'currentCellIndex': None,
        'autonomousMode': True,
        'autonomousModeContinue': True,  # Continue mode
        'ragEnabled': True,
    }

    # Set up existing session
    agent.session_metadata.update({
        "active": True,
        "session_id": "test-session",
        "notebook_uri": "/test.ipynb",
    })

    # Call chat with existing session_id
    await agent.chat(user_input, session_id="test-session", context=context)

    # Get the context passed to orchestrator
    call_args = agent.orchestrator.process_request.call_args
    passed_context = call_args.kwargs['context']

    # Verify retrieval_queries was NOT set (checkpointer will provide it)
    assert 'retrieval_queries' not in passed_context
    print(f"✅ retrieval_queries not set on continue (checkpointer provides it)")


def test_orchestrator_routes_to_search_workflows():
    """Test that planning router routes to search_workflows when
    retrieval_queries is present AND rag_enabled=True."""

    # Create mock state with retrieval_queries AND rag_enabled
    state = {
        "task_list": {},  # No tasks
        "retrieval_queries": ["Analyze single-cell data"],
        "autonomous_mode_continue": False,
        "rag_enabled": True,  # REQUIRED for workflow retrieval
        "planning_phase": None,  # First time
        "workflow_retrieval_iteration": 0,
    }

    # Create orchestrator with mocked tools
    llm = Mock()
    kb = Mock()
    orchestrator = LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    # Test the planning phase router directly
    next_node = orchestrator._route_planning_phase(state)

    # Should route to search_workflows
    assert next_node == "search_workflows", \
        f"Expected search_workflows, got {next_node}"
    print(f"✅ Router correctly routes to search_workflows")


def test_orchestrator_routes_to_plan_tasks_without_queries():
    """Test that planning router routes to increment_task_planning_iteration
    when retrieval_queries is empty or RAG disabled."""

    # Create mock state WITHOUT retrieval_queries
    state = {
        "task_list": {},  # No tasks
        "retrieval_queries": [],  # Empty
        "autonomous_mode_continue": False,
        "rag_enabled": False,  # RAG disabled
        "planning_phase": None,
        "workflow_retrieval_iteration": 0,
    }

    # Create orchestrator
    llm = Mock()
    kb = Mock()
    orchestrator = LangGraphOrchestrator(llm, kb, use_deterministic_routing=True)

    # Test the planning phase router directly
    next_node = orchestrator._route_planning_phase(state)

    # Should route to increment_task_planning_iteration (which leads to plan_tasks)
    assert next_node == "increment_task_planning_iteration"
    print(f"✅ Router correctly routes to task planning without queries")


def test_persistent_state_includes_retrieval_queries():
    """Test that retrieval_queries is in PERSISTENT_STATE_FIELDS."""
    from kai.core.orchestration.langgraph_orchestrator import PERSISTENT_STATE_FIELDS

    assert "retrieval_queries" in PERSISTENT_STATE_FIELDS
    print(f"✅ retrieval_queries is in PERSISTENT_STATE_FIELDS")


def test_transient_state_includes_snippet_queries():
    """Test that snippet_retrieval_query is in TRANSIENT_STATE_FIELDS."""
    from kai.core.orchestration.langgraph_orchestrator import TRANSIENT_STATE_FIELDS

    assert "snippet_retrieval_query" in TRANSIENT_STATE_FIELDS
    assert "rag_retrieval" in TRANSIENT_STATE_FIELDS
    assert "rag_text" in TRANSIENT_STATE_FIELDS
    print(f"✅ RAG fields are in TRANSIENT_STATE_FIELDS")


if __name__ == "__main__":
    import asyncio

    print("\n=== Testing Retrieval Queries Flow ===\n")

    # Test synchronous functions
    test_orchestrator_routes_to_search_workflows()
    test_orchestrator_routes_to_plan_tasks_without_queries()
    test_persistent_state_includes_retrieval_queries()
    test_transient_state_includes_snippet_queries()

    # Test async functions
    async def run_async_tests():
        from unittest.mock import Mock, AsyncMock, patch
        from pathlib import Path
        from kai.config.settings import Settings

        # Create proper settings mock
        settings = Mock(spec=Settings)
        settings.DISABLE_TURBO = True
        settings.KNOWLEDGE_BASE_PATH = Path("/tmp/test_kb")
        settings.NOTEBOOK_SUMMARIES_PATH = Path("/tmp/test_summaries")

        # Create agent with mocked knowledge base
        mock_kb = Mock()
        with patch('kai.core.agent.create_knowledge_base', return_value=mock_kb):
            agent = KaiAgent(settings=settings)

        agent.llm_interface = Mock()
        agent.llm_interface.provider_name = "ollama"
        agent.knowledge_base = mock_kb
        agent.orchestrator = Mock(spec=LangGraphOrchestrator)
        agent.orchestrator.process_request = AsyncMock()
        agent.session_metadata = {
            "active": False,
            "session_id": None,
            "session_timestamp": None,
            "notebook_uri": None,
            "iteration_counter": 0,
        }

        await test_retrieval_queries_initialized_on_first_iteration(agent)

        # Reset for next test
        agent.session_metadata.update({
            "active": True,
            "session_id": "test-session",
            "notebook_uri": "/test.ipynb",
        })
        agent.orchestrator.process_request.reset_mock()

        await test_retrieval_queries_not_set_on_continue(agent)

    asyncio.run(run_async_tests())

    print("\n=== All Tests Passed! ===\n")
