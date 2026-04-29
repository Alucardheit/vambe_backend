# Vambe Backend

Backend FastAPI que analiza transcripciones de reuniones comerciales con varios LLMs:
**Gemini**, **OpenAI**, **Claude (Anthropic)** o **Hugging Face local** (SmolLM2).

---

## Variables de entorno

Crea un `.env` en la raíz:

```env
# --- Gemini (Google AI Studio) ---
# Obtén la key en https://aistudio.google.com/apikey
GOOGLE_API_KEY=tu_key
# Compatibilidad: API_KEY también funciona si ya lo tenías.

# --- OpenAI ---
OPENAI_API_KEY=tu_key

# --- Claude (Anthropic) ---
ANTHROPIC_API_KEY=tu_key

# --- Hugging Face (opcional, solo para modelos gated) ---
HF_TOKEN=tu_token

# --- Comportamiento (opcional) ---
LLM_MAX_CONCURRENCY=3
LLM_MAX_RETRIES=3
```

Solo necesitas las keys de los providers que vayas a usar.

---

## Instalación

```bash
uv sync
```

Esto instala todo, incluyendo `torch`, `transformers` y `accelerate` (~2 GB en
la primera descarga). Funciona en cualquier OS.

#### macOS (Apple Silicon)

El wheel de PyPI ya trae soporte **MPS** (Metal Performance Shaders). El backend
autodetecta MPS y lo usa.

#### Windows + NVIDIA (CUDA)

El wheel default de `torch` en PyPI es **CPU-only** en Windows. Para usar GPU NVIDIA,
después del `uv sync` reinstala torch con la rueda de CUDA:

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

(Reemplaza `cu121` por la versión de CUDA que tengas. Mira https://pytorch.org/get-started/locally/.)

#### Windows / Linux sin GPU, o macOS Intel

Funciona en CPU automáticamente. SmolLM2 1.7B en CPU corre a ~5–15 tokens/s,
suficiente para batches pequeños pero lento para 60+ filas.

> El modelo se descarga la primera vez al cache de HF
> (`~/.cache/huggingface` en mac/Linux, `%USERPROFILE%\.cache\huggingface` en Windows).

---

## Levantar el backend

```bash
uv run uvicorn src.main:app --reload --port 8000
```

OpenAPI/Swagger: http://localhost:8000/docs

---

## Endpoints

### `POST /clients/analyze`

Analiza un CSV de transcripciones. Query params:

| Param | Default | Valores |
|---|---|---|
| `language` | `es` | `es` \| `en` |
| `format` | `json` | `json` \| `csv` |
| `provider` | `gemini` | `gemini` \| `openai` \| `claude` \| `huggingface` |
| `model` | (default por provider) | id del modelo |

Defaults por provider:
- `gemini` → `gemini-2.5-flash`
- `openai` → `gpt-4o-mini`
- `claude` → `claude-haiku-4-5`
- `huggingface` → `HuggingFaceTB/SmolLM2-1.7B-Instruct`

```bash
# Gemini, JSON
curl -X POST "http://localhost:8000/clients/analyze?provider=gemini&language=es" \
  -F file=@vambe_clients.csv

# Claude Sonnet, descarga CSV
curl -X POST "http://localhost:8000/clients/analyze?provider=claude&model=claude-sonnet-4-6&format=csv" \
  -F file=@vambe_clients.csv -o salida.csv

# Hugging Face local (modelo SmolLM2 1.7B)
curl -X POST "http://localhost:8000/clients/analyze?provider=huggingface" \
  -F file=@vambe_clients.csv
```

### `POST /indicators/compute`

Calcula indicadores agregados (closed rate, distribuciones por industria/vendedor, gráfico
de área). Acepta CSV crudo (ejecuta análisis primero) o ya analizado. Mismos query params
que `/clients/analyze` excepto `format`.

---

## Estructura

```
src/
├── core/                # Recursos (CSV de entrada por defecto)
├── features/
│   ├── clients/
│   │   ├── analyzer.py         # Orquestador
│   │   ├── endpoints.py        # POST /clients/analyze
│   │   ├── schemas.py          # Pydantic (ES/EN)
│   │   └── providers/
│   │       ├── base.py                 # LLMProvider (ABC)
│   │       ├── gemini.py
│   │       ├── openai_provider.py
│   │       ├── anthropic_provider.py
│   │       └── huggingface_local.py    # MPS/CUDA/CPU autodetect
│   └── indicators/
│       ├── endpoints.py        # POST /indicators/compute
│       ├── service.py
│       └── schemas.py
├── settings/
│   └── providers_settings.py   # Multi-provider env vars
├── config.py
└── main.py
```

---

## Tests

```bash
pytest
```
