"""Esquemas pydantic: parámetros de generación y configuración desde el entorno."""

import os
from enum import StrEnum

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class GenerationParams(BaseModel):
    """Parámetros de entrada validados para una consulta al modelo."""

    model_config = ConfigDict(str_strip_whitespace=True)

    prompt: str = Field(min_length=1, description="Texto de la consulta")
    system_prompt: str | None = Field(default=None, description="Instrucción de sistema")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=1024)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


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
        return cls(**env_values)
