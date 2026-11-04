"""Proveedor OpenAI."""

from collections.abc import AsyncIterator

import openai

from providers.base import BaseLLMClient
from schemas import GenerationParams


class OpenAIClient(BaseLLMClient):
    name = "openai"
    rate_limit_errors = (openai.RateLimitError,)
    connection_errors = (openai.APIConnectionError,)
    status_errors = (openai.APIStatusError,)

    def build_client(self, api_key: str):
        return openai.AsyncOpenAI(api_key=api_key)

    async def chat(self, params: GenerationParams) -> AsyncIterator[str]:
        # En OpenAI la instrucción de sistema es un mensaje más de la lista.
        messages = []
        if params.system_prompt:
            messages.append({"role": "system", "content": params.system_prompt})
        messages.append({"role": "user", "content": params.prompt})

        optional_args = {}
        if params.top_p is not None:
            optional_args["top_p"] = params.top_p

        response_stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=params.temperature,
            max_completion_tokens=params.max_tokens,
            stream=True,
            **optional_args,
        )
        # Cada evento trae un delta de texto; se ignoran los que vienen vacíos.
        async for stream_event in response_stream:
            if stream_event.choices and stream_event.choices[0].delta.content:
                yield stream_event.choices[0].delta.content
