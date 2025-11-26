"""LLM provider configurations."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model."""
    name: str
    provider: str
    default_temperature: float = 0.7
    supports_functions: bool = True
    supports_reasoning_level: bool = False  # whether to add reasoning level to system prompt
    supports_system_prompt: bool = True
    supports_tools: bool = False  # Native tool support
    native_tools: list = field(default_factory=list)  # List of native tools


@dataclass
class LLMConfig:
    """LLM configuration management."""

    # Ollama models
    OLLAMA_MODELS = {
        "codellama": ModelConfig(
            name="codellama",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "deepseek-coder": ModelConfig(
            name="deepseek-coder:33b",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "llama3": ModelConfig(
            name="llama3",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "llama3.1": ModelConfig(
            name="llama3.1",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "mistral": ModelConfig(
            name="mistral",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "qwen2.5-coder:7b": ModelConfig(
            name="qwen2.5-coder:7b",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "qwen2.5-coder:14b": ModelConfig(
            name="qwen2.5-coder:14b",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "qwen2.5-coder:32b": ModelConfig(
            name="qwen2.5-coder:32b",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "qwen3:0.6b": ModelConfig(  # Tiny model for unit testing
            name="qwen3:0.6b",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=False,
            supports_system_prompt=True,
        ),
        "gpt-oss:20b": ModelConfig(
            name="gpt-oss:20b",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=True,
            supports_system_prompt=True,
            supports_tools=True,  # Native tool support
            native_tools=["browser"],  # Built-in browser tool
        ),
        "gpt-oss:120b": ModelConfig(
            name="gpt-oss:120b",
            provider="ollama",
            default_temperature=0.,
            supports_functions=True,
            supports_reasoning_level=True,
            supports_system_prompt=True,
            supports_tools=True,  # Native tool support
            native_tools=["browser"],  # Built-in browser tool
        ),
    }

    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model."""
        return cls.OLLAMA_MODELS.get(model_name)

    @classmethod
    def list_available_models(cls) -> list[str]:
        """List all available model names."""
        return list(cls.OLLAMA_MODELS.keys())

