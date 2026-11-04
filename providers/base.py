"""Clase abstracta común a todos los proveedores."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from exceptions import LLMConnectionError, LLMError, LLMRateLimitError
from schemas import GenerationParams


class BaseLLMClient(ABC):
    """Contrato común de los clientes: todos se usan con `generate(params)`.

    Para sumar un proveedor: crear un archivo en `providers/`, heredar de esta
    clase, implementar `build_client` y `chat`, declarar los errores de su SDK
    y registrarlo en `AsyncLLMManager`.
    """

    name: str
    rate_limit_errors: tuple[type[Exception], ...]
    connection_errors: tuple[type[Exception], ...]
    status_errors: tuple[type[Exception], ...]

    def __init__(self, api_key: str, model: str):
        self.model = model
        self._client = self.build_client(api_key)

    @abstractmethod
    def build_client(self, api_key: str):
        """Crea el cliente async del SDK del proveedor."""

    @abstractmethod
    def chat(self, params: GenerationParams) -> AsyncIterator[str]:
        """Devuelve (async generator) los fragmentos de texto crudos del proveedor."""

    async def generate(self, params: GenerationParams) -> AsyncIterator[str]:
        """Genera la respuesta en streaming, devolviendo fragmentos con yield."""
        try:
            async for chunk in self.chat(params):
                yield chunk
        except self.rate_limit_errors as error:
            raise LLMRateLimitError(
                f"Límite de tasa excedido en {self.name}. Reintentá más tarde. ({error})"
            ) from error
        except self.connection_errors as error:  # incluye timeouts
            raise LLMConnectionError(
                f"Error de red al contactar a {self.name}: {error}"
            ) from error
        except self.status_errors as error:
            raise LLMError(
                f"Error de {self.name} (HTTP {error.status_code}): {error.message}"
            ) from error
