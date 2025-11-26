"""Tests for the LLMInterface class."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, ANY

from kai.core.llm_interface import LLMInterface, OllamaProvider, BaseLLMProvider
from kai.config.settings import Settings


class TestBaseLLMProvider:
    """Test the BaseLLMProvider abstract class."""
    
    def test_abstract_methods(self):
        """Test that BaseLLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseLLMProvider()
    
    def test_abstract_methods_must_be_implemented(self):
        """Test that abstract methods must be implemented in subclasses."""
        
        class IncompleteProvider(BaseLLMProvider):
            pass
        
        with pytest.raises(TypeError):
            IncompleteProvider()


class TestOllamaProvider:
    """Test the OllamaProvider class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = MagicMock(spec=Settings)
        self.settings.MAX_TOKENS = 2048
        self.settings.VERBOSE = False
        
        # Mock ollama client
        self.mock_client = AsyncMock()
        self.mock_sync_client = MagicMock()
        
        with patch('kai.core.llm_interface.ollama') as mock_ollama:
            mock_ollama.AsyncClient.return_value = self.mock_client
            mock_ollama.list.return_value = {'models': [{'name': 'test-model'}]}
            mock_ollama.pull = MagicMock()
            
            self.provider = OllamaProvider("test-model", self.settings)
    
    def test_initialization(self):
        """Test OllamaProvider initialization."""
        assert self.provider.model == "test-model"
        assert self.provider.settings == self.settings
        assert self.provider.client == self.mock_client
        # model_config is None for unknown models, which is expected behavior
        assert self.provider.model_config is None
    
    def test_initialization_with_unknown_model(self):
        """Test initialization with unknown model."""
        with patch('kai.core.llm_interface.ollama') as mock_ollama:
            mock_ollama.AsyncClient.return_value = self.mock_client
            mock_ollama.list.return_value = {'models': []}
            mock_ollama.pull = MagicMock()

            provider = OllamaProvider("unknown-model", self.settings)

            # Unknown models have no config, which is expected
            assert provider.model_config is None
            assert provider.model == "unknown-model"
    
    def test_ensure_model_available_existing(self):
        """Test _ensure_model_available with existing model."""
        with patch('kai.core.llm_interface.ollama') as mock_ollama:
            mock_ollama.list.return_value = {'models': [{'name': 'test-model'}]}
            mock_ollama.pull = MagicMock()
            
            # Should not call pull
            provider = OllamaProvider("test-model", self.settings)
            mock_ollama.pull.assert_not_called()
    
    def test_ensure_model_available_missing(self):
        """Test _ensure_model_available with missing model."""
        with patch('kai.core.llm_interface.ollama') as mock_ollama:
            mock_ollama.list.return_value = {'models': []}
            mock_ollama.pull = MagicMock()
            
            # Should call pull
            provider = OllamaProvider("missing-model", self.settings)
            mock_ollama.pull.assert_called_once_with("missing-model")
    
    def test_ensure_model_available_error(self):
        """Test _ensure_model_available with error."""
        with patch('kai.core.llm_interface.ollama') as mock_ollama:
            mock_ollama.list.side_effect = Exception("Connection error")
            mock_ollama.pull = MagicMock()

            # Should crash with connection error for traceability
            with pytest.raises(Exception, match="Connection error"):
                OllamaProvider("test-model", self.settings)
    
    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful code generation."""
        # Mock ollama response
        self.mock_client.chat.return_value = {
            "message": {"content": "Generated code here"}
        }
        
        result = await self.provider.generate(
            "Generate hello world",
            system_prompt="You are a helpful assistant",
            temperature=0.8,
            max_tokens=1000
        )
        
        # Check client was called correctly
        self.mock_client.chat.assert_called_once()
        call_args = self.mock_client.chat.call_args
        
        assert call_args[1]["model"] == "test-model"
        assert len(call_args[1]["messages"]) == 2
        assert call_args[1]["messages"][0]["role"] == "system"
        assert call_args[1]["messages"][0]["content"] == "You are a helpful assistant"
        assert call_args[1]["messages"][1]["role"] == "user"
        assert call_args[1]["messages"][1]["content"] == "Generate hello world"
        assert call_args[1]["options"]["temperature"] == 0.8
        assert call_args[1]["options"]["num_predict"] == 1000
        
        assert result == "Generated code here"
    
    @pytest.mark.asyncio
    async def test_generate_no_system_prompt(self):
        """Test generation without system prompt should raise error."""
        self.mock_client.chat.return_value = {
            "message": {"content": "Response"}
        }

        # Should raise ValueError when no system_prompt provided
        with pytest.raises(ValueError, match="No system_prompt provided"):
            await self.provider.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_with_defaults(self):
        """Test generation with default parameters."""
        self.mock_client.chat.return_value = {
            "message": {"content": "Response"}
        }

        # Should raise ValueError when no system_prompt provided
        with pytest.raises(ValueError, match="No system_prompt provided"):
            await self.provider.generate("Test prompt")
    
    @pytest.mark.asyncio
    async def test_generate_error(self):
        """Test generation with error."""
        self.mock_client.chat.side_effect = Exception("Connection error")

        # Should raise ValueError for missing system_prompt before connection error
        with pytest.raises(ValueError, match="No system_prompt provided"):
            await self.provider.generate("Test prompt")


class TestLLMInterface:
    """Test the LLMInterface class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.settings = MagicMock(spec=Settings)

        with patch('kai.core.llm_interface.OllamaProvider') as mock_provider_class, \
             patch('kai.core.llm_pool.LLMPool') as mock_pool_class:

            self.mock_provider_small = MagicMock()
            self.mock_provider_large = MagicMock()
            mock_provider_class.side_effect = [self.mock_provider_small, self.mock_provider_large]

            self.mock_pool = MagicMock()
            self.mock_pool.small_llm = "test-model-small"
            self.mock_pool.large_llm = "test-model-large"
            mock_pool_class.return_value = self.mock_pool

            self.llm_interface = LLMInterface(
                provider="ollama",
                model="test-model",
                settings=self.settings
            )
    
    def test_initialization_with_settings(self):
        """Test LLMInterface initialization with settings."""
        assert self.llm_interface.provider_name == "ollama"
        assert self.llm_interface.model == "test-model"
        assert self.llm_interface.settings == self.settings
        assert self.llm_interface.provider_small == self.mock_provider_small
        assert self.llm_interface.provider_large == self.mock_provider_large
    
    def test_initialization_without_settings(self):
        """Test LLMInterface initialization without settings."""
        with patch('kai.core.llm_interface.OllamaProvider') as mock_provider_class, \
             patch('kai.core.llm_interface.Settings') as mock_settings_class:
            
            mock_settings = MagicMock()
            mock_settings_class.from_env.return_value = mock_settings
            
            llm_interface = LLMInterface(provider="ollama", model="test-model")
            
            # Should use default settings
            mock_settings_class.from_env.assert_called_once()
            assert llm_interface.settings == mock_settings
    
    def test_initialization_with_defaults(self):
        """Test LLMInterface initialization with default parameters."""
        with patch('kai.core.llm_interface.OllamaProvider') as mock_provider_class, \
             patch('kai.core.llm_interface.Settings') as mock_settings_class, \
             patch('kai.core.llm_pool.LLMPool') as mock_pool_class:

            mock_settings = MagicMock()
            mock_settings_class.from_env.return_value = mock_settings

            mock_pool = MagicMock()
            mock_pool.small_llm = "default-small"
            mock_pool.large_llm = "default-large"
            mock_pool_class.return_value = mock_pool

            llm_interface = LLMInterface()

            # Should use defaults
            assert llm_interface.provider_name == "ollama"
            # Model comes from initialization parameter
            assert llm_interface.model is None  # Default is None
    
    def test_initialization_unsupported_provider(self):
        """Test initialization with unsupported provider."""
        with pytest.raises(ValueError, match="Unsupported provider: unsupported"):
            LLMInterface(provider="unsupported", model="test-model", settings=self.settings)
    
    def test_get_llm_for_tool_small(self):
        """Test getting small LLM for tool."""
        with patch.object(self.llm_interface.llm_pool, 'get_llmsize_for_tool', return_value="small_llm"):
            provider = self.llm_interface.get_llm_for_tool("test_tool")
            assert provider == self.llm_interface.provider_small
    
    def test_get_llm_for_tool_large(self):
        """Test getting large LLM for tool."""
        with patch.object(self.llm_interface.llm_pool, 'get_llmsize_for_tool', return_value="large_llm"):
            provider = self.llm_interface.get_llm_for_tool("test_tool")
            assert provider == self.llm_interface.provider_large
    
    def test_provider_instances_are_different(self):
        """Test that small and large provider instances are different."""
        # The providers should be separate instances for small vs large models
        assert self.llm_interface.provider_small is self.mock_provider_small
        assert self.llm_interface.provider_large is self.mock_provider_large
        assert self.mock_provider_small != self.mock_provider_large
