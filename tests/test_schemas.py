"""Validaciones de los esquemas pydantic."""

import pytest
from pydantic import ValidationError

from exceptions import LLMConfigError, LLMValidationError
from schemas import GenerationParams, Provider, Settings


def test_defaults_validos():
    params = GenerationParams(prompt="Hola")
    assert params.temperature == 1.0
    assert params.max_tokens == 512
    assert params.system_prompt is None
    assert params.top_p is None


def test_prompt_se_recorta():
    assert GenerationParams(prompt="  Hola  ").prompt == "Hola"


@pytest.mark.parametrize("prompt", ["", "   "])
def test_prompt_vacio_es_invalido(prompt):
    with pytest.raises(ValidationError):
        GenerationParams(prompt=prompt)


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_temperature_fuera_de_rango(temperature):
    with pytest.raises(ValidationError):
        GenerationParams(prompt="Hola", temperature=temperature)


@pytest.mark.parametrize("temperature", [0.0, 2.0])
def test_temperature_en_los_limites(temperature):
    assert GenerationParams(prompt="Hola", temperature=temperature).temperature == temperature


@pytest.mark.parametrize("max_tokens", [0, -5, 1025])
def test_max_tokens_fuera_de_rango(max_tokens):
    with pytest.raises(ValidationError):
        GenerationParams(prompt="Hola", max_tokens=max_tokens)


@pytest.mark.parametrize("top_p", [0.0, 1.1, -1])
def test_top_p_fuera_de_rango(top_p):
    with pytest.raises(ValidationError):
        GenerationParams(prompt="Hola", top_p=top_p)


def test_tipos_incorrectos():
    with pytest.raises(ValidationError):
        GenerationParams(prompt="Hola", temperature="mucha")


def test_provider_invalido_en_settings():
    with pytest.raises(ValidationError):
        Settings(provider="gemini")


def test_settings_from_env(monkeypatch):
    monkeypatch.setattr("schemas.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_MODEL", "mi-modelo")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings.from_env()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.anthropic_model == "mi-modelo"
    assert settings.openai_api_key is None


@pytest.mark.parametrize(
    ("data", "fragmento"),
    [
        ({"prompt": "  "}, "no puede estar vacío"),
        ({"prompt": "Hola", "temperature": 3}, "temperature debe estar entre 0 y 2 (recibido: 3)"),
        ({"prompt": "Hola", "temperature": "mucha"}, "temperature debe ser un número"),
        ({"prompt": "Hola", "max_tokens": 5000}, "max_tokens debe estar entre 1 y 1024"),
        ({"prompt": "Hola", "max_tokens": "x"}, "max_tokens debe ser un número entero"),
        ({"prompt": "Hola", "top_p": 0}, "top_p debe ser mayor que 0"),
    ],
)
def test_validated_da_mensajes_especificos(data, fragmento):
    with pytest.raises(LLMValidationError) as caught:
        GenerationParams.validated(**data)
    assert any(fragmento in message for message in caught.value.messages)
    assert isinstance(caught.value.__cause__, ValidationError)


def test_validated_junta_varios_errores():
    with pytest.raises(LLMValidationError) as caught:
        GenerationParams.validated(prompt="", temperature=9, max_tokens=0)
    assert len(caught.value.messages) == 3


def test_from_env_con_provider_invalido(monkeypatch):
    monkeypatch.setattr("schemas.load_dotenv", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(LLMConfigError, match="'gemini' no soportado"):
        Settings.from_env()
