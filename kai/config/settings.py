"""Global configuration settings for bioinformatics agent."""
import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from .paths import AGENT_BASE_DIR, RETRIEVAL_DIR


@dataclass
class Settings:
    """Global settings for the bioinformatics agent."""

    # Base paths
    BASE_DIR: Path = field(default_factory=lambda: AGENT_BASE_DIR)
    KNOWLEDGE_BASE_PATH: Path = field(default_factory=lambda: RETRIEVAL_DIR)
    WORKSPACE_PATH: Path = field(default_factory=lambda: Path.cwd() / "workspace")

    # Derived paths
    @property
    def NOTEBOOK_SUMMARIES_PATH(self) -> Path:
        """Path to notebook summaries storage."""
        return self.KNOWLEDGE_BASE_PATH / "notebook_summaries"

    @property
    def DEBUG_FAULTY_LLM_RESPONSES_PATH(self) -> Path:
        """Path to debug directory for faulty LLM responses."""
        return self.BASE_DIR / "debug_faulty_llm_responses"

    # LLM settings
    DEFAULT_LLM_PROVIDER: str = "ollama"
    MAX_TOKENS: int = 4096
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.95

    # Agent behavior
    MAX_RETRIES: int = 3
    TIMEOUT_SECONDS: int = 300
    DEBUG_PROMPTS: bool = True  # Log all prompts to debug folder
    VERBOSE: bool = True
    DISABLE_TURBO: bool = False  # Disable Turbo mode (useful for tests)

    # Notebook settings
    NOTEBOOK_TIMEOUT: int = 600  # 10 minutes per cell
    CLEAR_OUTPUT_ON_ERROR: bool = False
    SAVE_CHECKPOINTS: bool = True

    # Git settings
    AUTO_COMMIT: bool = True
    COMMIT_MESSAGE_PREFIX: str = "[kai_agent]"

    # Knowledge base settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_SEARCH_RESULTS: int = 10

    # Web search settings
    USER_AGENT: str = "KaiAgent/0.1.0"
    MAX_WEB_SEARCH_RESULTS: int = 5
    REQUEST_TIMEOUT: int = 30

    def __post_init__(self):
        """Create necessary directories after initialization."""
        # Validate settings
        self._validate_settings()
        
        # Create necessary directories
        for path_attr in ["BASE_DIR", "KNOWLEDGE_BASE_PATH"]:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)
    
    def _validate_settings(self):
        """Validate configuration settings."""
        # Validate temperature
        if not (0.0 <= self.TEMPERATURE <= 2.0):
            raise ValueError(f"Temperature must be between 0.0 and 2.0, got {self.TEMPERATURE}")
        
        # Validate top_p
        if not (0.0 <= self.TOP_P <= 1.0):
            raise ValueError(f"TOP_P must be between 0.0 and 1.0, got {self.TOP_P}")
        
        # Validate positive integers
        positive_int_fields = ["MAX_TOKENS", "MAX_RETRIES", "TIMEOUT_SECONDS", "NOTEBOOK_TIMEOUT", 
                              "CHUNK_SIZE", "CHUNK_OVERLAP", "MAX_SEARCH_RESULTS", 
                              "MAX_WEB_SEARCH_RESULTS", "REQUEST_TIMEOUT"]
        
        for field_name in positive_int_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}")
        
        # Validate chunk overlap is less than chunk size
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({self.CHUNK_SIZE})")

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables."""
        kwargs = {}

        # Map environment variables to settings
        env_mapping = {
            "KAI_BASE_DIR": ("BASE_DIR", Path),
            "KAI_KNOWLEDGE_PATH": ("KNOWLEDGE_BASE_PATH", Path),
            "KAI_WORKSPACE_PATH": ("WORKSPACE_PATH", Path),
            "KAI_DEBUG_PROMPTS": ("DEBUG_PROMPTS", lambda x: x.lower() == "true"),
            "KAI_VERBOSE": ("VERBOSE", lambda x: x.lower() == "true"),
            "KAI_DISABLE_TURBO": ("DISABLE_TURBO", lambda x: x.lower() == "true"),
        }

        for env_var, (setting_name, converter) in env_mapping.items():
            if value := os.getenv(env_var):
                try:
                    kwargs[setting_name] = converter(value)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid value for {env_var}: {value}. Error: {e}")

        return cls(**kwargs)

    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }


# Global settings instance
settings = Settings.from_env()


def get_config_dir() -> Path:
    """Get the configuration directory for user-specific settings."""
    return AGENT_BASE_DIR / "config"
