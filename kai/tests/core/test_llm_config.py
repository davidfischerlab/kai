"""Tests for the LLMConfig class."""

import pytest
from unittest.mock import MagicMock, patch

from kai.core.llm_config import LLMConfig, ModelConfig


class TestModelConfig:
    """Test the ModelConfig dataclass."""
    
    def test_initialization(self):
        """Test ModelConfig initialization."""
        config = ModelConfig(
            name="test-model",
            provider="ollama",
            default_temperature=0.7,
            supports_functions=True,
            supports_system_prompt=True,
        )

        assert config.name == "test-model"
        assert config.provider == "ollama"
        assert config.default_temperature == 0.7
        assert config.supports_functions is True
        assert config.supports_system_prompt is True
    
    def test_initialization_with_defaults(self):
        """Test ModelConfig initialization with defaults."""
        config = ModelConfig(name="test-model", provider="ollama")

        assert config.name == "test-model"
        assert config.provider == "ollama"
        assert config.default_temperature == 0.7
        assert config.supports_functions is True
        assert config.supports_system_prompt is True


class TestLLMConfig:
    """Test the LLMConfig class."""
    
    def test_ollama_models_exist(self):
        """Test that Ollama model configurations exist."""
        assert hasattr(LLMConfig, 'OLLAMA_MODELS')
        assert isinstance(LLMConfig.OLLAMA_MODELS, dict)
        assert len(LLMConfig.OLLAMA_MODELS) > 0
    
    def test_get_model_config_existing(self):
        """Test getting existing model configuration."""
        # Get first model from configs
        model_name = list(LLMConfig.OLLAMA_MODELS.keys())[0]
        config = LLMConfig.get_model_config(model_name)
        
        assert config is not None
        assert isinstance(config, ModelConfig)
        assert config.name == model_name
        assert config.provider == "ollama"
    
    def test_get_model_config_non_existent(self):
        """Test getting non-existent model configuration."""
        config = LLMConfig.get_model_config("non-existent-model")
        
        assert config is None
    
    def test_tiered_context_sizing_available(self):
        """Test that context sizing is handled by LLM providers with tiered approach."""
        # Context length is now calculated dynamically by LLM providers
        # This test verifies the old static approach was removed
        assert not hasattr(LLMConfig, 'get_model_context_length')

        # Verify model configs no longer have static context_length
        model_name = list(LLMConfig.OLLAMA_MODELS.keys())[0]
        config = LLMConfig.get_model_config(model_name)
        assert not hasattr(config, 'context_length')
    
    def test_model_configs_structure(self):
        """Test structure of model configurations."""
        for model_name, config in LLMConfig.OLLAMA_MODELS.items():
            assert isinstance(model_name, str)
            assert isinstance(config, ModelConfig)
            # Config name may be different from key for some models (e.g., deepseek-coder)
            assert isinstance(config.name, str)
            assert len(config.name) > 0
            assert config.provider == "ollama"
            assert isinstance(config.default_temperature, float)
            assert 0.0 <= config.default_temperature <= 2.0
            assert isinstance(config.supports_functions, bool)
            assert isinstance(config.supports_system_prompt, bool)
    
    def test_dynamic_context_length_approach(self):
        """Test that dynamic context length approach is implemented."""
        # Context lengths are now calculated dynamically by LLM providers
        # based on input size rather than being static configuration values
        from kai.core.llm_interface import OllamaProvider
        from kai.config.settings import Settings

        # Test that providers have the tiered context sizing method
        settings = Settings()
        provider = OllamaProvider("qwen3:0.6b", settings=settings)
        assert hasattr(provider, 'calculate_tiered_context_size')

        # Test with different input sizes
        short_prompt = "Hello"
        medium_prompt = "This is a medium length prompt " * 100
        long_prompt = "This is a very long prompt " * 1000

        short_context = provider.calculate_tiered_context_size(short_prompt)
        medium_context = provider.calculate_tiered_context_size(medium_prompt)
        long_context = provider.calculate_tiered_context_size(long_prompt)

        # Verify tiered approach works
        assert short_context <= medium_context <= long_context
        assert short_context >= 8192  # Minimum tier
        assert long_context <= 131072  # Maximum tier
    
    def test_specific_models_exist(self):
        """Test that specific expected models exist."""
        expected_models = [
            "codellama",
            "deepseek-coder",
            "llama3",
            "llama3.1",
            "mistral",
            "qwen2.5-coder:7b",
            "qwen2.5-coder:14b",
            "qwen2.5-coder:32b",
            "gpt-oss:20b"
        ]
        
        available_models = LLMConfig.list_available_models()
        
        for model in expected_models:
            assert model in available_models
    
    def test_all_models_have_required_fields(self):
        """Test that all models have required fields."""
        for model_name, config in LLMConfig.OLLAMA_MODELS.items():
            # Required fields
            assert hasattr(config, 'name')
            assert hasattr(config, 'provider')
            assert hasattr(config, 'default_temperature')
            assert hasattr(config, 'supports_functions')
            assert hasattr(config, 'supports_system_prompt')

            # Check values are reasonable
            assert config.name is not None
            assert config.provider == "ollama"
            assert config.default_temperature >= 0.0
            assert isinstance(config.supports_functions, bool)
            assert isinstance(config.supports_system_prompt, bool)
