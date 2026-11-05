"""Prueba manual de la conexión REAL con OpenAI (consume una pequeña cantidad de créditos).

Uso:
    python test_openai_connection.py
    python test_openai_connection.py --prompt "Contá hasta 5"

Requiere OPENAI_API_KEY en el .env. Los tests automáticos (sin red) están en tests/.
"""

import argparse
import asyncio
import sys
import time

from exceptions import LLMError
from llm_manager import AsyncLLMManager
from schemas import GenerationParams


async def main(prompt: str) -> int:
    try:
        manager = AsyncLLMManager(provider="openai")
        print(f"Conectando con OpenAI (modelo {manager.client.model})...")
        params = GenerationParams.validated(prompt=prompt, temperature=0.2, max_tokens=100)

        start = time.perf_counter()
        first_chunk_at = None
        chunks = 0
        async for text_chunk in manager.generate(params):
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter() - start
            chunks += 1
            print(text_chunk, end="", flush=True)
        print()
    except LLMError as error:
        print(f"\nFALLÓ la conexión: {error}", file=sys.stderr)
        return 1

    if chunks == 0:
        print("FALLÓ: la conexión respondió pero no llegó ningún fragmento.", file=sys.stderr)
        return 1
    total = time.perf_counter() - start
    print(f"OK: {chunks} fragmentos, primer fragmento a los {first_chunk_at:.2f}s, total {total:.2f}s.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prueba la conexión con OpenAI.")
    parser.add_argument("--prompt", default="Respondé solo con la palabra: pong")
    sys.exit(asyncio.run(main(parser.parse_args().prompt)))
