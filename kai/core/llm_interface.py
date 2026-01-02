"""LLM interface wrapper for multiple providers."""
import asyncio
import logging
from typing import Optional, List, AsyncGenerator, Dict, Any
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
from datetime import datetime

import ollama
from pydantic import ValidationError as PydanticValidationError
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from kai.core.llm_config import LLMConfig
from kai.config.settings import Settings

logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS FOR RETRY LOGIC
# =============================================================================

class RetriableError(Exception):
    """Network/timeout errors that should be retried with exponential backoff.

    These are transient errors that may succeed on retry:
    - Network timeouts
    - Connection errors
    - Rate limiting
    - Service unavailable
    """
    pass


class NonRetriableError(Exception):
    """Validation/parsing errors that need different handling.

    These errors indicate the LLM response was invalid and require
    context reduction or prompt modification rather than simple retry:
    - JSON parsing errors
    - Schema validation errors
    - Empty responses
    """
    def __init__(self, message: str, raw_output: str = None):
        super().__init__(message)
        self.raw_output = raw_output

# Use TYPE_CHECKING to avoid circular import
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from kai.core.tools.prompt_base import BasePromptTool


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""
    provider_name: str
    model: str

    def __init__(
        self,
        use_structured_output: bool = False
    ):
        """
        Args:
            use_structured_output: Whether to use native structured output or JSON prompting
        """
        self.use_structured_output = use_structured_output

    def _log_faulty_response(self, response: str, error: Exception, tool_name: str = "unknown"):
        """Log faulty LLM responses to debug directory."""
        try:
            debug_dir = self.settings.DEBUG_FAULTY_LLM_RESPONSES_PATH
            debug_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{timestamp}_{tool_name}_{type(error).__name__}.txt"
            filepath = debug_dir / filename

            with open(filepath, 'w') as f:
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Tool: {tool_name}\n")
                f.write(f"Error Type: {type(error).__name__}\n")
                f.write(f"Error Message: {str(error)}\n")
                f.write(f"\n{'='*80}\n")
                f.write(f"LLM Response:\n")
                f.write(f"{'='*80}\n\n")
                f.write(response)
        except Exception as e:
            logging.warning(f"Failed to log faulty response: {e}")

    def calculate_tiered_context_size(self, prompt: str, system_prompt: str = None, 
                                      context_length_factor: float = 1.) -> int:
        """Calculate appropriate context size based on input length.

        Args:
            prompt: User prompt text
            system_prompt: System prompt text (optional)
            context_length_factor: 

        Returns:
            Appropriate context size (8k, 16k, 32k, 64k, or 128k)
        """
        # Combine all input text
        total_input = prompt
        if system_prompt:
            total_input += system_prompt

        input_chars = len(total_input)

        # Character-to-token ratio is roughly 4:1 for English text
        # Note: OSS seem to have lower ratios, some reports up to 1.2:1.
        # This seems to cause LLM output errors some times with apparent cut off responses -
        # so these thresholds here are conservative for now (2:1) but are escalated via context_length_factor if LLM output fails.
        # Add 16384 token buffer for output generation and 100% reasoning margin
        # Note: reasoning also takes up tokens, consider this when asking models to reason!
        # Developer note: make reasoning and output margin adaptive to exact tool usage in the future.
        char_to_token = 1.
        estimated_tokens = int((input_chars / char_to_token * 2. + 16384.) * context_length_factor)

        # Tiered context sizes with character thresholds
        if estimated_tokens <= 8192:
            context_size = 8192
        elif estimated_tokens <= 16384:
            context_size = 16384
        elif estimated_tokens <= 32768:
            context_size = 32768
        elif estimated_tokens <= 65536:
            context_size = 65536
        else:  # Maximum context size for OSS models is 128k token
            context_size = 131072

        return context_size
        
    @abstractmethod
    async def _generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the LLM."""
        pass

    async def _generate_structured(self, prompt: str, schema, system_prompt: Optional[str] = None,
                                   tool_name: str = "unknown", context_length_factor: float = 1., **kwargs):
        """Generate structured output using the provided schema."""
        return await self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            context_length_factor=context_length_factor,
            **kwargs)
    
    async def generate(
        self,
        prompt: str,
        task_type: str = "general",
        context_length_factor: float = 1.,
        **kwargs
    ) -> str:
        """Generate a response.
        
        Args:
            prompt: User prompt
            task_type: Type of task for system prompt selection
            **kwargs: Additional parameters
            
        Returns:
            Generated response
        """
        system_prompt = kwargs.pop("system_prompt", None)
        if not system_prompt:
            raise ValueError(f"No system_prompt provided for task_type: {task_type}")
        
        return await self._generate(
            prompt=prompt,
            system_prompt=system_prompt,
            context_length_factor=context_length_factor,
            **kwargs
        )

    async def generate_structured(self, prompt: str, schema, task_type: str = "general", 
                                  tool_name: str = "unknown", context_length_factor: float = 1., **kwargs):
        """Generate structured output using the configured method (native or JSON prompting)."""
        system_prompt = kwargs.pop("system_prompt", None)
        if not system_prompt:
            raise ValueError(f"No system_prompt provided for task_type: {task_type}")

        if self.use_structured_output:
            # Use native structured output
            return await self._generate_structured(
                prompt=prompt,
                schema=schema,
                system_prompt=system_prompt,
                tool_name=tool_name,
                context_length_factor=context_length_factor,
                **kwargs
            )
        else:
            # Use JSON prompting approach
            import json
            import re

            # Add JSON schema to system prompt
            json_schema = schema.model_json_schema()
            enhanced_system_prompt = f"{system_prompt}\n{json.dumps(json_schema, indent=2)}"

            # Generate response with regular generation
            response = await self.generate(
                prompt=prompt,
                system_prompt=enhanced_system_prompt,
                task_type=task_type,
                context_length_factor=context_length_factor,
                **kwargs
            )

            # Parse JSON from response
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                json_text = json_match.group(0)
                try:
                    structured_data = json.loads(json_text)
                    return schema.model_validate(structured_data)
                except (json.JSONDecodeError, ValueError, PydanticValidationError) as e:
                    # Log faulty response for debugging
                    self._log_faulty_response(response, e, tool_name)
                    # Attach raw output to exception for debugging
                    e.raw_output = response
                    raise e
            else:
                error = ValueError(f"No JSON found in response: {response}")
                self._log_faulty_response(response, error, tool_name)
                raise error
    
    def should_use_json_prompting(self) -> bool:
        """Check if JSON prompting should be used instead of native structured output."""
        return not self.use_structured_output


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider."""
    provider_name = "ollama"

    def __init__(
        self, model: str, settings: Settings, use_structured_output: bool = False
    ):
        """Initialize Ollama provider.

        Args:
            model: Model name
            settings: Application settings
        """
        super().__init__(use_structured_output=use_structured_output)
        self.model = model
        self.settings = settings
        self.client = ollama.AsyncClient()

        # Get model config
        self.model_config = LLMConfig.get_model_config(model)

        # Check if model is available
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Ensure the model is available locally."""
        # Check if model exists
        models = ollama.list()
        # Handle both old dict format and new Model object format
        if hasattr(models, 'models'):
            model_list = models.models
        else:
            model_list = models.get('models', [])
        
        # Extract model names - handle both dict and Model object formats
        model_names = []
        for m in model_list:
            if hasattr(m, 'model'):
                # Model object format
                model_names.append(m.model)
            elif isinstance(m, dict) and 'name' in m:
                # Dict format with 'name' key
                model_names.append(m['name'])
            elif isinstance(m, dict) and 'model' in m:
                # Dict format with 'model' key
                model_names.append(m['model'])
            else:
                raise ValueError(f"Unknown model format: {type(m)}, content: {m}")
        
        if self.model not in model_names:
            ollama.pull(self.model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(RetriableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        context_size: int,
        **kwargs
    ) -> str:
        """Internal method to call LLM with tenacity retry for transient errors.

        This method is wrapped with tenacity's @retry decorator to handle
        network timeouts and connection errors with exponential backoff.

        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            context_size: Context window size
            **kwargs: Additional options

        Returns:
            Generated response content

        Raises:
            RetriableError: For network/timeout errors (will be retried)
            NonRetriableError: For validation/parsing errors (not retried)
        """
        timeout_seconds = 300.0

        try:
            response = await asyncio.wait_for(
                self.client.chat(
                    model=self.model,
                    messages=messages,
                    think=False,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": context_size,
                        **kwargs,
                    },
                ),
                timeout=timeout_seconds
            )
            content = response["message"]["content"]
            if not content or not content.strip():
                raise NonRetriableError(
                    f"Empty response from model {self.model}",
                    raw_output=""
                )
            return content

        except asyncio.TimeoutError:
            raise RetriableError(
                f"LLM request timed out after {timeout_seconds}s"
            )
        except ConnectionError as e:
            raise RetriableError(f"Connection error: {e}")
        except ollama.ResponseError as e:
            # Check if it's a rate limit or server error (retriable)
            error_str = str(e).lower()
            if "rate" in error_str or "503" in error_str or "502" in error_str:
                raise RetriableError(f"Server error (retriable): {e}")
            # Otherwise it's likely a model/config error (not retriable)
            raise NonRetriableError(f"Ollama error: {e}")

    async def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        context_length_factor: float = 1.,
        **kwargs
    ) -> str:
        """Generate a response using Ollama.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Calculate appropriate context size based on input
        context_size = self.calculate_tiered_context_size(
            prompt=prompt,
            system_prompt=system_prompt,
            context_length_factor=context_length_factor
        )

        temp = temperature or (
            self.model_config.default_temperature if self.model_config else 0.7
        )
        tokens = max_tokens or self.settings.MAX_TOKENS

        try:
            return await self._call_llm_with_retry(
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                context_size=context_size,
                **kwargs
            )
        except RetriableError as e:
            # All retries exhausted
            raise Exception(
                f"LLM request failed after retries. "
                f"Please check if Ollama is running and '{self.model}' is available. "
                f"Error: {e}"
            )
        except NonRetriableError:
            # Let NonRetriableError propagate to base.py for context-reduction retry
            raise
    
    async def _generate_structured(self, prompt: str, schema, system_prompt: Optional[str] = None, 
                                   tool_name: str = "unknown", context_length_factor: float = 1., **kwargs):
        """Generate structured output using Ollama's format parameter.

        Note: This uses Ollama's native format parameter for structured output.
        The LLMInterface will fall back to JSON prompting if use_structured_output is False.
        """
        import json
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Calculate appropriate context size based on input
        context_size = self.calculate_tiered_context_size(
            prompt=prompt, system_prompt=system_prompt, context_length_factor=context_length_factor)

        response = await asyncio.to_thread(
            ollama.chat,
            model=self.model,
            messages=messages,
            format=schema.model_json_schema(),
            options={
                "temperature": kwargs.get("temperature", self.model_config.default_temperature),
                "num_predict": kwargs.get("max_tokens", self.settings.MAX_TOKENS),
                "num_ctx": context_size
            }
        )

        content = response["message"]["content"]
        if not content or not content.strip():
            raise ValueError(f"Empty response from model {self.model}")

        try:
            structured_data = json.loads(content)
            return schema.model_validate(structured_data)
        except (json.JSONDecodeError, ValueError, PydanticValidationError) as e:
            # Log faulty response for debugging
            self._log_faulty_response(content, e, tool_name)
            # Attach raw output to exception for debugging
            e.raw_output = content
            raise e


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider (works with OpenAI API or Ollama-hosted OpenAI models)."""
    provider_name = "openai"
    
    def __init__(self, model: str, settings: Settings, api_key: str = None, base_url: str = None, use_structured_output: bool = False):
        """Initialize OpenAI provider.
        
        Args:
            model: Model name (e.g., "gpt-oss:20b", "gpt-4", etc.)
            settings: Application settings
            api_key: OpenAI API key (or "ollama" for Ollama)
            base_url: Base URL (e.g., "http://localhost:11434/v1" for Ollama)
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI package not available. Install with: pip install openai")
        
        super().__init__(use_structured_output=use_structured_output)

        self.model = model
        self.settings = settings
        self.last_tool_calls = None
        
        # Configure client for either OpenAI API or Ollama
        if base_url:
            # Using Ollama or custom endpoint
            self.client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key or "ollama"  # Dummy key for Ollama
            )
        else:
            # Using real OpenAI API
            self.client = openai.AsyncOpenAI(api_key=api_key)
            
    async def _generate(self, prompt: str, system_prompt: str = None, tools: List[Dict[str, Any]] = None, **kwargs) -> str:
        """Generate a response using OpenAI Chat Completions API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.settings.MAX_TOKENS),
                "temperature": kwargs.get("temperature", self.settings.TEMPERATURE),
                "top_p": kwargs.get("top_p", self.settings.TOP_P)
            }
            
            # Add tools if provided
            if tools:
                api_params["tools"] = tools
            
            response = await self.client.chat.completions.create(**api_params)
            
            # Store tool calls for later retrieval
            choice = response.choices[0]
            if hasattr(choice.message, 'tool_calls') and choice.message.tool_calls:
                self.last_tool_calls = [
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments
                        }
                    }
                    for tool_call in choice.message.tool_calls
                ]
            else:
                self.last_tool_calls = None
            
            return choice.message.content or ""
            
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    async def stream(self, prompt: str, system_prompt: str = None, tools: List[Dict[str, Any]] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream a response using OpenAI Chat Completions API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Prepare API call parameters
            api_params = {
                "model": self.model,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", self.settings.MAX_TOKENS),
                "temperature": kwargs.get("temperature", self.settings.TEMPERATURE),
                "top_p": kwargs.get("top_p", self.settings.TOP_P),
                "stream": True
            }
            
            # Add tools if provided (note: streaming with tools may not capture tool calls properly)
            if tools:
                api_params["tools"] = tools
            
            stream = await self.client.chat.completions.create(**api_params)
            
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            raise Exception(f"OpenAI streaming error: {e}")


class OllamaTurboProvider(BaseLLMProvider):
    """Ollama Turbo provider for remote model execution."""
    provider_name = "ollama-turbo"

    def __init__(
        self,
        model: str,
        settings: Settings,
        api_key: Optional[str] = None,
        use_structured_output: bool = False
    ):
        """Initialize Ollama Turbo provider.

        Args:
            model: Model name (e.g., "gpt-oss:120b")
            settings: Application settings
            api_key: Ollama API key (if None, tries to use local Ollama auth)
        """
        super().__init__(use_structured_output=use_structured_output)
        self.model = model
        self.settings = settings

        # Initialize Turbo client - don't store the API key
        headers = {}
        if api_key is not None:
            headers['Authorization'] = api_key

        self.client = ollama.AsyncClient(
            host="https://ollama.com",
            headers=headers
        )

        # Get model config (use local model config as fallback)
        self.model_config = LLMConfig.get_model_config(model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type(RetriableError),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    async def _call_llm_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        context_size: int,
        **kwargs
    ) -> str:
        """Internal method to call Turbo LLM with tenacity retry.

        Turbo has longer timeouts (10 min) for large remote models.
        """
        timeout_seconds = 600.0  # 10 minutes for Turbo

        try:
            response = await asyncio.wait_for(
                self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        "temperature": temperature,
                        "num_predict": max_tokens,
                        "num_ctx": context_size,
                        **kwargs,
                    },
                ),
                timeout=timeout_seconds
            )
            content = response["message"]["content"]
            if not content or not content.strip():
                raise NonRetriableError(
                    f"Empty response from Turbo model {self.model}",
                    raw_output=""
                )
            return content

        except asyncio.TimeoutError:
            raise RetriableError(
                f"Turbo request timed out after {timeout_seconds}s"
            )
        except ConnectionError as e:
            raise RetriableError(f"Connection error: {e}")
        except ollama.ResponseError as e:
            error_str = str(e).lower()
            if "rate" in error_str or "503" in error_str or "502" in error_str:
                raise RetriableError(f"Server error (retriable): {e}")
            raise NonRetriableError(f"Ollama Turbo error: {e}")

    async def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        reasoning_level: Optional[str] = "medium",
        context_length_factor: float = 1.,
        **kwargs
    ) -> str:
        """Generate a response using Ollama Turbo.

        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            reasoning_level: Reasoning level for OSS models
            **kwargs: Additional parameters

        Returns:
            Generated response
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        context_size = self.calculate_tiered_context_size(
            prompt=prompt,
            system_prompt=system_prompt,
            context_length_factor=context_length_factor
        )

        temp = temperature or (
            self.model_config.default_temperature if self.model_config else 0.
        )
        tokens = max_tokens or self.settings.MAX_TOKENS

        try:
            return await self._call_llm_with_retry(
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                context_size=context_size,
                **kwargs
            )
        except RetriableError as e:
            raise Exception(
                f"Ollama Turbo request failed after retries for '{self.model}'. "
                f"Error: {e}"
            )
        except NonRetriableError:
            # Let NonRetriableError propagate to base.py for context-reduction retry
            raise


class LLMInterface:
    """
    Unified interface for LLM interactions across different providers.
    
    Provides a consistent API for interacting with various LLM providers
    (Ollama, OpenAI, etc.) and handles provider-specific configurations,
    model management, and response generation.
    
    Attributes:
        provider_name: Name of the LLM provider
        model: Currently active model
        settings: Application settings
        provider: Provider-specific implementation instance
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: Optional[str] = None,  # Use default if None
        settings: Optional[Settings] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        use_structured_output: bool = False
    ):
        """Initialize LLM interface.
        
        Args:
            provider: LLM provider name ("ollama", "ollama-turbo", or "openai")
            model: Model name to use for small and large LLM, otherwise defaulting to optimal choice
            settings: Application settings
            api_key: API key for OpenAI or Ollama Turbo provider
            base_url: Base URL for OpenAI provider (e.g., for Ollama)
            use_structured_output: Whether to use native structured output or JSON prompting
        """
        self.api_key = api_key  # Store API key for turbo switching
        self.base_url = base_url
        self.model = model
        self.settings = settings or Settings.from_env()
        self.use_structured_output = use_structured_output
        
        # Initialize LLM pool
        from .llm_pool import LLMPool
        self.provider_name = provider
        self.llm_pool = LLMPool(
            turbo=provider == "ollama-turbo",
            small_llm=self.model,
            large_llm=self.model
        )
        
        # Initialize provider
        self._set_providers(self.provider_name)
        
    def _set_providers(self, provider_name: str):
        """Set providers based on current provider_name."""
        self.provider_name = provider_name
        if self.provider_name == "ollama":
            self._set_providers_ollama()
        elif self.provider_name == "ollama-turbo":
            self._set_providers_ollama_turbo()
        elif self.provider_name == "openai":
            self._set_providers_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider_name}")
        
    def _set_providers_ollama(self):
        """Set providers to Ollama instances."""
        kwargs = {
            "settings": self.settings,
            "use_structured_output": self.use_structured_output
        }
        self.provider_small = OllamaProvider(model=self.llm_pool.small_llm, **kwargs)
        self.provider_large = OllamaProvider(model=self.llm_pool.large_llm, **kwargs)

    def _set_providers_ollama_turbo(self):
        """Set providers to Ollama Turbo instances."""
        kwargs = {
            "api_key": self.api_key,
            "settings": self.settings,
            "use_structured_output": self.use_structured_output
        }
        self.provider_small = OllamaTurboProvider(model=self.llm_pool.small_llm, **kwargs)
        self.provider_large = OllamaTurboProvider(model=self.llm_pool.large_llm, **kwargs)

    def _set_providers_openai(self):
        """Set providers to OpenAI instances."""
        kwargs = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "settings": self.settings,
            "use_structured_output": self.use_structured_output
        }
        self.provider_small = OpenAIProvider(model=self.llm_pool.small_llm, **kwargs)
        self.provider_large = OpenAIProvider(model=self.llm_pool.large_llm, **kwargs)

    def set_turbo_mode(self, enabled: bool) -> None:
        """Switch turbo mode on/off."""
        from .llm_pool import LLMPool
        switch_on = enabled and self.provider_name != "ollama-turbo"
        switch_off = not enabled and self.provider_name == "ollama-turbo"
        # Update providers
        if switch_on:
            # Update LLM pool
            self.llm_pool = LLMPool(
                turbo=True,
                small_llm=self.model,
                large_llm=self.model
            )
            self._set_providers("ollama-turbo")
        elif switch_off:
            # Update LLM pool
            self.llm_pool = LLMPool(
                turbo=False,
                small_llm=self.model,
                large_llm=self.model
            )
            self._set_providers("ollama")
    
    def get_llm_for_tool(self, tool: 'BasePromptTool') -> BaseLLMProvider:
        """Get the appropriate LLM provider instance for a given tool."""
        llm_size = self.llm_pool.get_llmsize_for_tool(tool)
        if llm_size == "small_llm":
            return self.provider_small
        else:
            return self.provider_large    

    def get_reasoning_for_tool(self, tool: 'BasePromptTool') -> str:
        """Get the appropriate LLM reasoning level for a given tool: return string describing reasoning."""
        return self.llm_pool.get_reasoning_for_tool(tool)
