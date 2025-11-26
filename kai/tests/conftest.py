import pytest
import os
from unittest.mock import MagicMock, patch


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    
    # Disable debug prompts during tests
    os.environ['KAI_DEBUG_PROMPTS'] = 'false'
    
    # Disable Turbo mode during tests
    os.environ['KAI_DISABLE_TURBO'] = 'true'


@pytest.fixture(autouse=True)
def patch_llm_interface():
    with patch('kai.core.llm_interface.LLMInterface') as mock_llm:
        mock_llm.return_value.generate = MagicMock(
            return_value='mocked response'
        )
        mock_llm.return_value.generate_code = MagicMock(
            return_value='print("mocked code")'
        )
        yield mock_llm


@pytest.fixture(autouse=True)
def patch_network_calls():
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_get.return_value.__aenter__.return_value.status = 200
        mock_get.return_value.__aenter__.return_value.text = MagicMock(
            return_value='mocked page'
        )
        yield mock_get 