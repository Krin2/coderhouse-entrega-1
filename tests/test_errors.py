"""Los errores de los SDK se traducen a excepciones controladas."""

import pytest

from exceptions import LLMConnectionError, LLMError, LLMRateLimitError
from llm_manager import AsyncLLMManager
from schemas import GenerationParams
from tests import helpers
from tests.helpers import SETTINGS, collect

PARAMS = GenerationParams(prompt="Hola")

CASES = [
    ("openai", helpers.fake_openai_sdk, helpers.openai_rate_limit, LLMRateLimitError),
    ("openai", helpers.fake_openai_sdk, helpers.openai_connection_error, LLMConnectionError),
    ("openai", helpers.fake_openai_sdk, helpers.openai_status_error, LLMError),
    ("anthropic", helpers.fake_anthropic_sdk, helpers.anthropic_rate_limit, LLMRateLimitError),
    ("anthropic", helpers.fake_anthropic_sdk, helpers.anthropic_connection_error, LLMConnectionError),
    ("anthropic", helpers.fake_anthropic_sdk, helpers.anthropic_status_error, LLMError),
]
IDS = [f"{provider}-{expected.__name__}" for provider, _, _, expected in CASES]


@pytest.mark.parametrize(("provider", "fake", "make_error", "expected"), CASES, ids=IDS)
def test_error_al_iniciar_la_llamada(provider, fake, make_error, expected):
    manager = AsyncLLMManager(provider=provider, settings=SETTINGS)
    manager.client._client = fake(error=make_error())

    with pytest.raises(expected) as caught:
        collect(manager, PARAMS)

    assert type(caught.value) is expected  # no se confunde con otra subclase
    assert caught.value.__cause__ is not None  # conserva el error original


@pytest.mark.parametrize(("provider", "fake", "make_error", "expected"), CASES, ids=IDS)
def test_error_en_medio_del_streaming(provider, fake, make_error, expected):
    manager = AsyncLLMManager(provider=provider, settings=SETTINGS)
    manager.client._client = fake(chunks=["parcial"], error=make_error())

    received: list[str] = []
    with pytest.raises(expected):
        collect(manager, PARAMS, received)
    assert received == ["parcial"]  # lo ya emitido no se pierde


def test_los_errores_controlados_heredan_de_llmerror():
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(LLMConnectionError, LLMError)
