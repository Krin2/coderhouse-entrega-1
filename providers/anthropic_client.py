"""Proveedor Anthropic."""

from collections.abc import AsyncIterator

import anthropic

from providers.base import BaseLLMClient
from schemas import GenerationParams


class AnthropicClient(BaseLLMClient):
    name = "anthropic"
    rate_limit_errors = (anthropic.RateLimitError,)
    connection_errors = (anthropic.APIConnectionError,)
    status_errors = (anthropic.APIStatusError,)

    def build_client(self, api_key: str):
        return anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(self, params: GenerationParams) -> AsyncIterator[str]:
        optional_args = {}
        if params.system_prompt:
            optional_args["system"] = params.system_prompt

        # Anthropic solo admite temperatura 0-1. El SDK 2.x ya no expone
        # temperature/top_p como argumentos tipados de stream(): van en el body.
        sampling_params = {"temperature": min(params.temperature, 1.0)}
        if params.top_p is not None:
            sampling_params["top_p"] = params.top_p

        async with self._client.messages.stream(
            model=self.model,
            messages=[{"role": "user", "content": params.prompt}],
            max_tokens=params.max_tokens,
            extra_body=sampling_params,
            **optional_args,
        ) as response_stream:
            async for text_chunk in response_stream.text_stream:
                yield text_chunk
