"""Fakes y utilidades compartidas por los tests (sin llamadas reales a la red)."""

import asyncio
from types import SimpleNamespace

try:  # las versiones nuevas de los SDK usan httpx2
    import httpx2 as httpx
except ImportError:  # pragma: no cover
    import httpx

import anthropic
import openai

from llm_manager import AsyncLLMManager
from schemas import Settings

SETTINGS = Settings(openai_api_key="test-openai", anthropic_api_key="test-anthropic")


def collect(manager: AsyncLLMManager, params, received: list[str] | None = None) -> list[str]:
    """Consume el generador asíncrono y devuelve todos los fragmentos.

    Si se pasa `received`, se va llenando aunque el stream termine con una excepción.
    """
    received = [] if received is None else received

    async def run() -> None:
        async for chunk in manager.generate(params):
            received.append(chunk)

    asyncio.run(run())
    return received


def _request() -> "httpx.Request":
    return httpx.Request("POST", "https://api.example.com/v1/test")


def _response(status: int) -> "httpx.Response":
    return httpx.Response(status, request=_request())


def openai_rate_limit() -> Exception:
    return openai.RateLimitError("rate limited", response=_response(429), body=None)


def openai_connection_error() -> Exception:
    return openai.APIConnectionError(request=_request())


def openai_status_error() -> Exception:
    return openai.InternalServerError("boom", response=_response(500), body=None)


def anthropic_rate_limit() -> Exception:
    return anthropic.RateLimitError("rate limited", response=_response(429), body=None)


def anthropic_connection_error() -> Exception:
    return anthropic.APIConnectionError(request=_request())


def anthropic_status_error() -> Exception:
    return anthropic.InternalServerError("boom", response=_response(500), body=None)


# ---- Fake del SDK de OpenAI --------------------------------------------------

def fake_openai_sdk(chunks: list[str | None] = (), error: Exception | None = None):
    """Imita `AsyncOpenAI`: `chat.completions.create(...)` devuelve un stream async.

    Sin `chunks`, `error` se lanza al crear el stream; con `chunks`, se lanza
    después de emitirlos (fallo en medio del streaming).
    """

    def event(text):
        return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])

    async def stream():
        yield SimpleNamespace(choices=[])  # evento sin choices (p. ej. usage)
        for text in chunks:
            yield event(text)
        if error is not None:
            raise error

    async def create(**kwargs):
        create.kwargs = kwargs
        if error is not None and not chunks:
            raise error
        return stream()

    create.kwargs = {}
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


# ---- Fake del SDK de Anthropic -----------------------------------------------

def fake_anthropic_sdk(chunks: list[str] = (), error: Exception | None = None):
    """Imita `AsyncAnthropic`: `messages.stream(...)` es un context manager async."""

    class FakeStream:
        async def __aenter__(self):
            if error is not None and not chunks:
                raise error
            return self

        async def __aexit__(self, *exc):
            return False

        @property
        def text_stream(self):
            async def gen():
                for text in chunks:
                    yield text
                if error is not None:
                    raise error

            return gen()

    def stream(**kwargs):
        stream.kwargs = kwargs
        return FakeStream()

    stream.kwargs = {}
    return SimpleNamespace(messages=SimpleNamespace(stream=stream))
