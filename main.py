"""Script de prueba: consulta a OpenAI o Anthropic con streaming desde la consola."""

import argparse
import asyncio
import sys

from pydantic import ValidationError

from llm_manager import AsyncLLMManager, LLMError
from schemas import GenerationParams, Provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consulta a un LLM con streaming.")
    parser.add_argument("--prompt", required=True, help="Texto de la consulta")
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in Provider],
        help="Proveedor a usar (por defecto, LLM_PROVIDER del .env)",
    )
    parser.add_argument("--system", dest="system_prompt", help="Instrucción de sistema")
    parser.add_argument("--temperature", type=float, default=1.0, help="0 a 2 (default 1.0)")
    parser.add_argument("--max-tokens", type=int, default=512, help="1 a 1024 (default 512)")
    parser.add_argument("--top-p", type=float, help="Mayor que 0 y hasta 1")
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    """Valida los argumentos, consulta al modelo y devuelve el código de salida."""
    # 1) Pydantic valida los rangos antes de tocar la red (código 2 si fallan).
    try:
        params = GenerationParams(
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            top_p=args.top_p,
        )
    except ValidationError as error:
        print("Parámetros inválidos:", file=sys.stderr)
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"  - {field}: {issue['msg']}", file=sys.stderr)
        return 2

    # 2) El manager elige el proveedor; los errores controlados no muestran traceback.
    try:
        manager = AsyncLLMManager(provider=args.provider)
        print(f"[{manager.provider.value} / {manager.client.model}]")
        # Cada fragmento se imprime apenas llega (flush=True evita el buffering).
        async for text_chunk in manager.generate(params):
            print(text_chunk, end="", flush=True)
        print()
    except LLMError as error:
        print(f"\nError: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run(parse_args())))
