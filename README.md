# Consulta a LLMs por consola (OpenAI / Anthropic)

Aplicación de consola en Python que permite usar indistintamente OpenAI o Anthropic para hacer una consulta, con respuesta en streaming (los fragmentos se imprimen a medida que llegan).

## Requisitos

- Python 3.12 o superior
- Una API key de OpenAI y/o de Anthropic

## Instalación

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> **Importante:** el entorno virtual tiene que estar activado cada vez que abras una terminal nueva para ejecutar `main.py`. Si ves `ModuleNotFoundError: No module named 'anthropic'` (u `openai`), estás usando el Python global en lugar del del `.venv`. Activalo, o ejecutá directamente con el Python del venv:
>
> ```bash
> # Windows
> .venv\Scripts\python main.py --prompt "Hola"
> # macOS / Linux
> .venv/bin/python main.py --prompt "Hola"
> ```
>
> En PowerShell, si la activación se bloquea por la política de ejecución, usá la forma anterior o corré `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Variables de entorno

Copiá `.env.example` a `.env` y completá tus claves. El archivo `.env` está en `.gitignore` y **no se comparte**.

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

| Variable | Obligatoria | Descripción | Default |
|---|---|---|---|
| `LLM_PROVIDER` | No | Proveedor a usar: `openai` o `anthropic` | `openai` |
| `OPENAI_API_KEY` | Si usás OpenAI | API key de OpenAI | — |
| `ANTHROPIC_API_KEY` | Si usás Anthropic | API key de Anthropic | — |
| `OPENAI_MODEL` | No | Modelo de OpenAI | `gpt-4o-mini` |
| `ANTHROPIC_MODEL` | No | Modelo de Anthropic | `claude-haiku-4-5` |

Solo hace falta la API key del proveedor que vayas a usar.

## Ejecutar el script de prueba

```bash
# Proveedor por defecto (LLM_PROVIDER del .env, o openai)
python main.py --prompt "Explicame qué es el streaming en una frase"

# Elegir proveedor desde la línea de comandos
python main.py --prompt "Hola" --provider openai
python main.py --prompt "Hola" --provider anthropic

# Con parámetros
python main.py --prompt "Escribí un haiku" --temperature 0.7 --max-tokens 200 --system "Respondé en español"
```

| Opción | Descripción | Rango / default |
|---|---|---|
| `--prompt` | Texto de la consulta (obligatoria) | no vacío |
| `--provider` | `openai` o `anthropic` | `LLM_PROVIDER` |
| `--system` | Instrucción de sistema | — |
| `--temperature` | Temperatura | 0 a 2, default 1.0 |
| `--max-tokens` | Máximo de tokens de la respuesta | 1 a 1024, default 512 |
| `--top-p` | Nucleus sampling | mayor que 0 y hasta 1 |

Nota: Anthropic solo admite temperatura de 0 a 1; si pasás un valor mayor se limita a 1.0 para ese proveedor.

## Manejo de errores

La app no termina con un traceback: muestra un mensaje y devuelve un código de salida.

| Situación | Excepción | Código de salida |
|---|---|---|
| Parámetros inválidos (p. ej. `--temperature 3`) | `ValidationError` de pydantic | 2 |
| Falta la API key | `LLMConfigError` | 1 |
| Límite de tasa (rate limit, HTTP 429) | `LLMRateLimitError` | 1 |
| Error de red o timeout | `LLMConnectionError` | 1 |
| Otro error HTTP del proveedor (401, 500, ...) | `LLMError` | 1 |

## Pruebas

**Tests automáticos** (no usan la red ni API keys: reemplazan los SDK por fakes):

```bash
pip install -r requirements-dev.txt
pytest
```

Cubren las validaciones de pydantic (`tests/test_schemas.py`), la selección de proveedor y el camino feliz con streaming (`tests/test_manager.py`) y la traducción de errores de rate limit, red y HTTP, tanto al iniciar la llamada como en medio del streaming (`tests/test_errors.py`).

**Prueba de conexión real con OpenAI** (requiere `OPENAI_API_KEY` y consume créditos mínimos):

```bash
python test_openai_connection.py
python test_openai_connection.py --prompt "Contá hasta 5"
```

Imprime la respuesta en streaming y, al final, la cantidad de fragmentos y los tiempos. Termina con código 0 si funcionó y 1 si falló.

## Estructura

```
main.py                    Script de prueba (CLI)
test_openai_connection.py  Prueba manual de la conexión real con OpenAI
tests/                     Tests automáticos (pytest)
llm_manager.py     AsyncLLMManager: elige el proveedor y delega en su cliente
schemas.py         Esquemas pydantic (GenerationParams, Settings, Provider)
exceptions.py      Excepciones controladas
providers/
  base.py              BaseLLMClient (clase abstracta común)
  openai_client.py     OpenAIClient
  anthropic_client.py  AnthropicClient
```

`AsyncLLMManager.generate(params)` es un generador asíncrono (usa `yield`):

```python
manager = AsyncLLMManager(provider="openai")
async for chunk in manager.generate(GenerationParams(prompt="Hola")):
    print(chunk, end="", flush=True)
```

### Agregar un proveedor

1. Crear `providers/<nombre>_client.py` con una clase que herede de `BaseLLMClient` e implemente `build_client` y `chat`, y declare los errores de su SDK.
2. Agregarlo al enum `Provider` y a `Settings` en `schemas.py`.
3. Registrarlo en `AsyncLLMManager._select_client`.
