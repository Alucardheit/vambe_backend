# Vambe Backend

Backend FastAPI que analiza transcripciones de reuniones comerciales con LLMs (Gemini, OpenAI, Claude, o modelos locales en Hugging Face) y expone métricas agregadas + drill-down por cliente para un dashboard de ventas.

> El frontend Next.js que consume este backend vive en un repo separado: [vambe_frontend](../vambe_frontend).

---

## Documentación técnica

- [docs/architecture.md](docs/architecture.md) — visión del sistema, módulos, flujo de datos.
- [docs/decisions.md](docs/decisions.md) — decisiones técnicas clave (formato ADR).

---

## Variables de entorno

Crea un `.env` en la raíz. Solo necesitas las API keys de los providers que vayas a usar.

```env
# --- API keys de LLMs ---
GOOGLE_API_KEY=tu_key            # Gemini — https://aistudio.google.com/apikey
OPENAI_API_KEY=tu_key            # OpenAI
ANTHROPIC_API_KEY=tu_key         # Claude
HF_TOKEN=tu_token                # Hugging Face (opcional, mejora rate limits y permite modelos gated)

# --- Comportamiento (opcional) ---
LLM_MAX_CONCURRENCY=3            # Cap global de requests paralelos al LLM
LLM_MAX_RETRIES=3                # Retries en errores de rate limit

# --- CORS (opcional) ---
CORS_ALLOW_ORIGINS=["http://localhost:3000"]   # Frontend en dev
# CORS_ALLOW_METHODS, CORS_ALLOW_HEADERS, CORS_ALLOW_CREDENTIALS también disponibles

# --- API metadata (opcional) ---
# API_DEBUG=false
# API_TITLE=Suite
# API_DISABLE_DOCS=false         # true para ocultar /docs en producción
```

Ver [src/settings/](src/settings/) para la lista completa de settings.

---

## Instalación

```bash
uv sync
```

Esto instala todo, incluyendo `torch`, `transformers` y `accelerate`.

### Windows / Linux + GPU NVIDIA

El `pyproject.toml` ya tiene configurado el index de PyTorch con CUDA 12.8:

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

`uv sync` baja directamente `torch+cu128` — no hay paso manual. Verifica con:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# Esperado: 2.x.x+cu128 True
```

> **RTX 5090 (Blackwell)**: requiere CUDA 12.8. No bajar a `cu121` o anteriores — no funcionarán.

### macOS (Apple Silicon)

El wheel default de PyPI trae soporte **MPS** (Metal). El backend autodetecta MPS y lo usa. No requiere configuración.

### Sin GPU (CPU)

Funciona automáticamente. SmolLM2 1.7B en CPU corre a ~5-15 tokens/s — usable para batches pequeños, lento para 60+ filas. Para batches grandes en CPU, preferir un provider cloud (Gemini/OpenAI/Claude).

> El modelo Hugging Face se descarga la primera vez al cache de HF (`~/.cache/huggingface` en Linux/macOS, `%USERPROFILE%\.cache\huggingface` en Windows).

---

## Levantar el backend

```bash
uv run uvicorn src.main:app --reload --port 8000
```

- OpenAPI/Swagger: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Endpoints

### `POST /clients/analyze`

Analiza un CSV de transcripciones y devuelve las filas enriquecidas con campos extraídos por el LLM (industria, dolores, objeciones, próximos pasos, etc.).

**Query params:**

| Param | Default | Valores |
|---|---|---|
| `language` | `es` | `es` \| `en` |
| `format` | `json` | `json` (para UI) \| `csv` (descarga) |
| `provider` | `gemini` | `gemini` \| `openai` \| `claude` \| `huggingface` |
| `model` | (default por provider) | id del modelo |

**Body:** `multipart/form-data` con un archivo CSV (opcional). Sin file → usa `src/core/vambe_clients.csv`.

**Columnas requeridas en el CSV:** `Nombre`, `Vendedor asignado`, `closed`, `Transcripcion`.

### `POST /indicators/compute`

Acepta un CSV crudo (lo analiza internamente) **o** uno ya analizado, y devuelve el `IndicatorsResponse` con métricas agregadas listas para el dashboard:

- KPIs: `total_clients`, `closed_rate`, `analysis_errors`.
- Distribuciones: `by_industria`, `by_vendedor`, `by_fuente_lead`, `by_nivel_interes`, `by_probabilidad_cierre`.
- Tasas de cierre: `closed_rate_by_industria`, `closed_rate_by_vendedor`.
- Insights agregados: `top_puntos_positivos`, `top_puntos_negativos`, `top_objeciones`, `top_proximos_pasos`.
- `analyzed_rows`: filas crudas para drill-down por cliente.

Mismos query params que `/clients/analyze` (sin `format`).

---

## Modelos disponibles por provider

| Provider | Default | Otros disponibles |
|---|---|---|
| `gemini` | `gemini-2.5-flash` | cualquier modelo del SDK de google-genai |
| `openai` | `gpt-4o-mini` | `gpt-4o`, etc. |
| `claude` | `claude-haiku-4-5` | `claude-sonnet-4-6`, etc. |
| `huggingface` | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` |

Cualquier `model` válido del SDK del provider funciona — el campo es free-text.

---

## Notas operativas

### Calidad de modelos Hugging Face

- **SmolLM2 1.7B** (~3 GB VRAM): solo para validar el pipeline. Genera análisis genéricos y a veces ecoa palabras del prompt como respuesta. **No usar para análisis reales.**
- **Qwen2.5 3B** (~6 GB VRAM): balance razonable de calidad/velocidad. Buena en español.
- **Qwen2.5 7B** (~14 GB VRAM): calidad alta, cercana a modelos cloud. Recomendado para uso local serio.

Para análisis de máxima calidad, preferir **Gemini 2.5 Flash** (rápido, barato) o **Claude Haiku 4.5** (excelente extracción estructurada).

### Performance

Inferencia de 60 transcripciones (~300 tokens cada una):

| Provider | Tiempo aproximado |
|---|---|
| Gemini Flash / Claude Haiku / GPT-4o-mini | 30-90 s |
| Qwen2.5 7B en RTX 5090 | 1-2 min |
| SmolLM2 1.7B en RTX 5090 | 30-60 s |
| Cualquiera en CPU | 10+ min |

El Hugging Face local serializa requests (1 GPU = 1 inferencia a la vez); los providers cloud paralelizan según `LLM_MAX_CONCURRENCY`.
