"""Excepciones controladas de la capa LLM."""


class LLMError(Exception):
    """Error controlado de la capa LLM."""


class LLMConfigError(LLMError):
    """Configuración inválida o incompleta (p. ej. falta la API key)."""


class LLMValidationError(LLMError):
    """Parámetros de entrada inválidos; `messages` tiene un mensaje por cada problema."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("Parámetros inválidos: " + " ".join(messages))


class LLMRateLimitError(LLMError):
    """El proveedor rechazó la solicitud por límite de tasa."""


class LLMConnectionError(LLMError):
    """Fallo de red o timeout al contactar al proveedor."""
