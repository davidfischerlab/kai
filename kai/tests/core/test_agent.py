import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from kai.core.agent import KaiAgent


# Simple tests for the current API
@pytest.mark.asyncio
async def test_agent_instantiation():
    """Test basic agent instantiation."""
    with patch('kai.core.agent.Settings') as mock_settings_class, \
         patch('kai.core.agent.LLMInterface'), \
         patch('kai.core.agent.create_knowledge_base'):
        
        mock_settings = MagicMock()
        mock_settings.KNOWLEDGE_BASE_PATH = Path("/tmp/knowledge")
        mock_settings.WORKSPACE_PATH = Path("/tmp/workspace")
        mock_settings_class.from_env.return_value = mock_settings
        
        agent = KaiAgent()
        assert agent.llm_interface is not None
        assert agent.knowledge_base is not None
