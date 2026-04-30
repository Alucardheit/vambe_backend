# Arquitectura

Documento de referencia de la arquitectura de Vambe Analytics. Audiencia: nuevos miembros del equipo que necesitan entender cómo está armado el sistema antes de tocar código.

---

## Visión general

El sistema toma un CSV de transcripciones de reuniones de ventas, las analiza con un LLM para extraer atributos estructurados (industria, dolores, objeciones, etc.), y expone un dashboard con métricas agregadas e insights por cliente.

```
                ┌──────────────────┐                ┌──────────────────┐
   CSV (input)  │   FastAPI        │   JSON         │   Next.js        │
  ─────────────▶│   backend        │ ─────────────▶ │   frontend       │
                │   (uvicorn)      │                │   (App Router)   │
                └──────────────────┘                └──────────────────┘
                        │
                        │ generate_structured()
                        ▼
                ┌──────────────────────────────────────────┐
                │  LLMProvider                             │
                │  ├── GeminiProvider (Google AI Studio)   │
                │  ├── OpenAIProvider                      │
                │  ├── AnthropicProvider (Claude)          │
                │  └── HuggingFaceLocalProvider (CUDA/CPU) │
                └──────────────────────────────────────────┘
```

Dos repositorios separados:

- **vambe_backend**: Python 3.11+, FastAPI, gestionado con `uv`.
- **vambe_frontend**: Next.js 15 (App Router), TypeScript, Tailwind.

---

## Backend

### Endpoints

| Método | Ruta | Propósito |
|---|---|---|
| `GET`  | `/` | Health + descubrimiento de docs |
| `POST` | `/clients/analyze` | CSV crudo → filas analizadas (JSON o CSV de salida) |
| `POST` | `/indicators/compute` | CSV (crudo o analizado) → indicadores agregados |

Ambos endpoints aceptan `provider` y `model` como query params para elegir LLM. Sin file → usan `src/core/vambe_clients.csv`.


### LLM Providers

Todos implementan la interfaz [LLMProvider](../src/features/clients/providers/base.py):

```python
class LLMProvider(Protocol):
    name: str
    recommended_concurrency: int

    async def generate_structured(
        self, *, system: str, user: str, schema: type[T], temperature: float
    ) -> T: ...
```

`get_provider(name, model)` devuelve la instancia correcta. La factory está en [providers/__init__.py](../src/features/clients/providers/__init__.py).

Para agregar un provider nuevo: implementar `LLMProvider`, registrarlo en `get_provider()`, agregarlo a `DEFAULT_MODELS`. Ver el archivo `huggingface_local.py` como referencia más completa (incluye carga lazy, device detection, JSON repair).

---

## Frontend

### Flujo de datos

```
analyze/page.tsx
    │ user selects file + provider
    ▼
api.computeIndicators(file, lang, {provider, model})
    │ POST /indicators/compute
    ▼
saveIndicators({data, meta})        # sessionStorage
    │
    ▼ navigate to /dashboard
dashboard/page.tsx
    └── Dashboard
        └── useSyncExternalStore(subscribeIndicators, ...)
            ↓
            renders StatCards / Charts / Insights / ClientsList
```

`useSyncExternalStore` permite que múltiples tabs reaccionen al storage event si la data cambia. Ver [decisions.md §7](decisions.md#7-state-del-frontend-via-sessionstorage).

### Drill-down

El `IndicatorsResponse` incluye `analyzed_rows` con TODOS los campos del LLM por cliente. La `ClientsList` permite filtrar/buscar y abrir un `ClientDrawer` que muestra el análisis completo (resumen, dolores, señales positivas, próximos pasos, etc.) sin requerir un fetch adicional.

---

## Configuración y secretos

Settings cargados via `pydantic-settings` desde `.env` y variables de entorno:

| Variable | Default | Uso |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini |
| `OPENAI_API_KEY` | — | OpenAI |
| `ANTHROPIC_API_KEY` | — | Claude |
| `HF_TOKEN` | opcional | Rate limits y modelos gated en HF Hub |
| `LLM_MAX_CONCURRENCY` | 3 | Cap global de requests paralelos al LLM |
| `LLM_MAX_RETRIES` | 3 | Retries en errores de rate limit |
| `CORS_ALLOW_ORIGINS` | `["*"]` | Orígenes del frontend |

Frontend usa `NEXT_PUBLIC_API_URL` (default `http://localhost:8000`).

---