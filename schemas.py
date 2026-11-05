"""Esquemas pydantic: parámetros de generación y configuración desde el entorno."""

import os
from enum import StrEnum

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from exceptions import LLMConfigError, LLMValidationError


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Mensajes específicos por campo. "range" cubre los errores de límites (ge/gt/le/lt),
# "type" los de tipo incorrecto o dato faltante. {value} es lo que se recibió.
FIELD_MESSAGES: dict[str, dict[str, str]] = {
    "prompt": {
        "empty": "El prompt no puede estar vacío.",
        "type": "El prompt es obligatorio y debe ser texto (recibido: {value!r}).",
    },
    "system_prompt": {
        "type": "El system prompt debe ser texto (recibido: {value!r}).",
    },
    "temperature": {
        "range": "temperature debe estar entre 0 y 2 (recibido: {value!r}).",
        "type": "temperature debe ser un número, por ejemplo 0.7 (recibido: {value!r}).",
    },
    "max_tokens": {
        "range": "max_tokens debe estar entre 1 y 1024 (recibido: {value!r}).",
        "type": "max_tokens debe ser un número entero, por ejemplo 256 (recibido: {value!r}).",
    },
    "top_p": {
        "range": "top_p debe ser mayor que 0 y como máximo 1 (recibido: {value!r}).",
        "type": "top_p debe ser un número, por ejemplo 0.9 (recibido: {value!r}).",
    },
    "provider": {
        "type": "Proveedor {value!r} no soportado. Opciones: "
        + ", ".join(p.value for p in Provider)
        + ".",
    },
}
RANGE_ERROR_TYPES = {"greater_than", "greater_than_equal", "less_than", "less_than_equal"}


def describe_validation_error(error: ValidationError) -> list[str]:
    """Traduce cada problema de pydantic a un mensaje específico en español."""
    messages = []
    for issue in error.errors():
        field = str(issue["loc"][0]) if issue["loc"] else "valor"
        error_type = issue["type"]
        if error_type == "string_too_short":
            kind = "empty"
        elif error_type in RANGE_ERROR_TYPES:
            kind = "range"
        else:
            kind = "type"
        template = FIELD_MESSAGES.get(field, {}).get(kind)
        if template is None:  # campo o caso no previsto: mensaje genérico con el detalle de pydantic
            messages.append(f"{field}: {issue['msg']}.")
        else:
            messages.append(template.format(value=issue.get("input")))
    return messages


class GenerationParams(BaseModel):
    """Parámetros de entrada validados para una consulta al modelo."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, description="Texto de la consulta")
    system_prompt: str | None = Field(default=None, description="Instrucción de sistema")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=1024)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)

    @classmethod
    def validated(cls, **data) -> "GenerationParams":
        """Como el constructor, pero lanza LLMValidationError con mensajes específicos."""
        try:
            return cls(**data)
        except ValidationError as error:
            raise LLMValidationError(describe_validation_error(error)) from error


class Settings(BaseModel):
    """Configuración leída de variables de entorno (.env)."""

    provider: Provider = Provider.OPENAI
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    @classmethod
    def from_env(cls, provider: str | None = None) -> "Settings":
        # Lee el .env y arma los valores; solo pisa los modelos si están definidos,
        # así se conservan los defaults del esquema.
        load_dotenv()
        env_values = {
            "provider": provider or os.getenv("LLM_PROVIDER") or Provider.OPENAI,
            "openai_api_key": os.getenv("OPENAI_API_KEY") or None,
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY") or None,
        }
        if openai_model := os.getenv("OPENAI_MODEL"):
            env_values["openai_model"] = openai_model
        if anthropic_model := os.getenv("ANTHROPIC_MODEL"):
            env_values["anthropic_model"] = anthropic_model
        try:
            return cls(**env_values)
        except ValidationError as error:
            raise LLMConfigError(
                "Configuración inválida en el entorno (.env): "
                + " ".join(describe_validation_error(error))
            ) from error
