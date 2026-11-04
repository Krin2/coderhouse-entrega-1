"""Excepciones controladas de la capa LLM."""


class LLMError(Exception):
    """Error controlado de la capa LLM."""


class LLMConfigError(LLMError):
    """Configuración inválida o incompleta (p. ej. falta la API key)."""


class LLMRateLimitError(LLMError):
    """El proveedor rechazó la solicitud por límite de tasa."""


class LLMConnectionError(LLMError):
    """Fallo de red o timeout al contactar al proveedor."""
