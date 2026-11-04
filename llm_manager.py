"""AsyncLLMManager: interfaz única para consultar OpenAI o Anthropic con streaming."""

from collections.abc import AsyncIterator

from exceptions import LLMConfigError, LLMConnectionError, LLMError, LLMRateLimitError
from providers import AnthropicClient, BaseLLMClient, OpenAIClient
from schemas import GenerationParams, Provider, Settings

__all__ = [
    "AsyncLLMManager",
    "LLMConfigError",
    "LLMConnectionError",
    "LLMError",
    "LLMRateLimitError",
]


class AsyncLLMManager:
    """Selecciona el cliente según el proveedor indicado y delega en él."""

    def __init__(self, provider: str | Provider | None = None, settings: Settings | None = None):
        self.settings = settings or Settings.from_env(str(provider) if provider else None)
        self.provider = Provider(provider) if provider else self.settings.provider
        self.client = self._select_client()

    def _select_client(self) -> BaseLLMClient:
        settings = self.settings
        # Para sumar un proveedor: agregarlo al enum Provider, a Settings y acá.
        # Cada entrada: (clase del cliente, API key, modelo, nombre de la variable de entorno).
        provider_setup: dict[Provider, tuple[type[BaseLLMClient], str | None, str, str]] = {
            Provider.OPENAI: (
                OpenAIClient,
                settings.openai_api_key,
                settings.openai_model,
                "OPENAI_API_KEY",
            ),
            Provider.ANTHROPIC: (
                AnthropicClient,
                settings.anthropic_api_key,
                settings.anthropic_model,
                "ANTHROPIC_API_KEY",
            ),
        }
        client_class, api_key, model_name, api_key_env_var = provider_setup[self.provider]
        if not api_key:
            raise LLMConfigError(f"Falta {api_key_env_var} en el entorno (.env).")
        return client_class(api_key=api_key, model=model_name)

    def generate(self, params: GenerationParams) -> AsyncIterator[str]:
        """Async generator de fragmentos de texto; usar con `async for`."""
        return self.client.generate(params)
