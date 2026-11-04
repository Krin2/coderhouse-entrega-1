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
    max_tokens: int = Field(default=1024, gt=0, le=32768)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)

    def temperature_for(self, provider: Provider) -> float:
        """Anthropic solo admite temperatura 0-1; se limita para ese proveedor."""
        if provider is Provider.ANTHROPIC:
            return min(self.temperature, 1.0)
        return self.temperature


class Settings(BaseModel):
    """Configuración leída de variables de entorno (.env)."""

    provider: Provider = Provider.OPENAI
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    anthropic_model: str = "claude-haiku-4-5"

    @classmethod
    def from_env(cls, provider: str | None = None) -> "Settings":
        load_dotenv()
        data = {
            "provider": provider or os.getenv("LLM_PROVIDER") or Provider.OPENAI,
            "openai_api_key": os.getenv("OPENAI_API_KEY") or None,
            "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY") or None,
        }
        if m := os.getenv("OPENAI_MODEL"):
            data["openai_model"] = m
        if m := os.getenv("ANTHROPIC_MODEL"):
            data["anthropic_model"] = m
        return cls(**data)
