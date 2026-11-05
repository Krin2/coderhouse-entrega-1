"""AsyncLLMManager: selección de proveedor y camino feliz con streaming."""

import inspect

import pytest

from exceptions import LLMConfigError
from llm_manager import AsyncLLMManager
from providers import AnthropicClient, OpenAIClient
from schemas import GenerationParams, Provider, Settings
from tests.helpers import SETTINGS, collect, fake_anthropic_sdk, fake_openai_sdk


def test_selecciona_openai():
    manager = AsyncLLMManager(provider="openai", settings=SETTINGS)
    assert isinstance(manager.client, OpenAIClient)
    assert manager.provider == Provider.OPENAI


def test_selecciona_anthropic():
    manager = AsyncLLMManager(provider="anthropic", settings=SETTINGS)
    assert isinstance(manager.client, AnthropicClient)


def test_usa_el_proveedor_de_settings_si_no_se_indica():
    settings = Settings(provider="anthropic", anthropic_api_key="k")
    assert isinstance(AsyncLLMManager(settings=settings).client, AnthropicClient)


def test_proveedor_desconocido():
    with pytest.raises(LLMConfigError, match="'gemini' no soportado"):
        AsyncLLMManager(provider="gemini", settings=SETTINGS)


@pytest.mark.parametrize(
    ("provider", "variable"),
    [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")],
)
def test_falta_api_key(provider, variable):
    with pytest.raises(LLMConfigError, match=variable):
        AsyncLLMManager(provider=provider, settings=Settings())


def test_generate_es_un_generador_asincrono():
    manager = AsyncLLMManager(provider="openai", settings=SETTINGS)
    manager.client._client = fake_openai_sdk(["a"])
    stream = manager.generate(GenerationParams(prompt="Hola"))
    assert inspect.isasyncgen(stream)


def test_happy_path_openai_streaming():
    manager = AsyncLLMManager(provider="openai", settings=SETTINGS)
    fake = fake_openai_sdk(["Ho", None, "la", " mundo"])
    manager.client._client = fake

    params = GenerationParams(
        prompt="Hola", system_prompt="Sé breve", temperature=0.5, max_tokens=50, top_p=0.9
    )
    assert collect(manager, params) == ["Ho", "la", " mundo"]  # ignora deltas vacíos

    sent = fake.chat.completions.create.kwargs
    assert sent["stream"] is True
    assert sent["temperature"] == 0.5
    assert sent["max_completion_tokens"] == 50
    assert sent["top_p"] == 0.9
    assert sent["messages"][0] == {"role": "system", "content": "Sé breve"}
    assert sent["messages"][-1] == {"role": "user", "content": "Hola"}


def test_happy_path_anthropic_streaming():
    manager = AsyncLLMManager(provider="anthropic", settings=SETTINGS)
    fake = fake_anthropic_sdk(["Ho", "la"])
    manager.client._client = fake

    params = GenerationParams(prompt="Hola", system_prompt="Sé breve", max_tokens=50)
    assert collect(manager, params) == ["Ho", "la"]

    sent = fake.messages.stream.kwargs
    assert sent["system"] == "Sé breve"
    assert sent["max_tokens"] == 50
    assert sent["messages"] == [{"role": "user", "content": "Hola"}]


def test_anthropic_limita_la_temperatura_a_1():
    manager = AsyncLLMManager(provider="anthropic", settings=SETTINGS)
    fake = fake_anthropic_sdk(["ok"])
    manager.client._client = fake

    collect(manager, GenerationParams(prompt="Hola", temperature=1.8))
    assert fake.messages.stream.kwargs["extra_body"]["temperature"] == 1.0
