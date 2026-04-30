# Decisiones técnicas clave

## 1. Abstracción de proveedores LLM detrás de una interfaz común

**Contexto.** Necesitamos analizar transcripciones con LLMs, pero ningún proveedor es ideal para todos los casos: Gemini es barato pero limitado en rate, Claude es robusto pero más caro, OpenAI es estándar de la industria, HuggingFace local es gratis pero requiere GPU. El usuario debería poder elegir según presupuesto/latencia/privacidad.

**Decisión.** Definir [LLMProvider](../src/features/clients/providers/base.py) como Protocol con un único método `generate_structured(system, user, schema, temperature) -> T`. Cada SDK (google-genai, openai, anthropic, transformers) se envuelve en una clase que implementa esa interfaz. La factory `get_provider(name, model)` resuelve por string.


---

## 2. Concurrencia con `asyncio.Semaphore` y `recommended_concurrency` por provider

**Contexto.** Procesar 60 transcripciones secuenciales toma minutos. Pero Gemini free tier rate-limita a ~10 RPM, OpenAI/Claude soportan más, y HuggingFace local en una sola GPU NO se beneficia de paralelismo (la GPU es el bottleneck — múltiples requests interleaved no aceleran nada y arriesgan OOM).

**Decisión.** Cada `LLMProvider` declara su `recommended_concurrency` (gemini=3, openai=5, anthropic=5, huggingface=1). El analyzer crea un `asyncio.Semaphore(min(settings.llm_max_concurrency, provider.recommended_concurrency))` y todos los `_analyze_row` lo respetan. Para HF, además hay un `asyncio.Lock` interno porque tokenizer/model no son thread-safe.

---

## 3. Output estructurado vía Pydantic

**Contexto.** Necesitamos campos tipados (industria, dolores, etc.) — no texto libre. Cada SDK tiene su propio mecanismo de "structured output" (Gemini: `response_schema`, OpenAI: `response_format`, Anthropic: tool use, HF: prompt + parser).

**Decisión.** Definir el contrato como Pydantic model ([ClientAnalysisES/EN](../src/features/clients/schemas.py)) y que cada provider lo traduzca a su mecanismo nativo. El analyzer no sabe nada del SDK — solo recibe una instancia validada.


---

## 4. HuggingFace local con detección automática de device + dtype

**Contexto.** El equipo tiene una RTX 5090 (32 GB VRAM, arquitectura Blackwell) y queremos correr modelos open-source ahí. Pero la misma codebase tiene que funcionar en MacBook M-series (MPS), CPU-only, y máquinas con CUDA convencional.

**Decisión.** [_detect_device()](../src/features/clients/providers/huggingface_local.py) selecciona en orden: `cuda` → `mps` (macOS) → `cpu`. [_select_dtype()](../src/features/clients/providers/huggingface_local.py) usa `float16` en GPU y `float32` en CPU (CPU no acelera fp16). El modelo carga con `device_map=device` y `dtype=dtype` directo en `from_pretrained` — no hay `.to(device)` posterior, lo que evita transferencias innecesarias.


---

## 5. JSON repair tolerante en HuggingFace

**Contexto.** Los modelos pequeños (SmolLM2 1.7B) no respetan instrucciones de "no markdown fences" y suelen emitir ` ```json {...} ``` `. Si max_new_tokens corta el output a la mitad, el fence queda sin cerrar y el JSON queda truncado.

**Decisión.** [_extract_first_json()](../src/features/clients/providers/huggingface_local.py) maneja tres casos:
1. Fence completo (` ```json {...} ``` `) → extrae contenido.
2. Fence sin cerrar (truncado) → strip del prefix solo.
3. Sin fence → usa el texto crudo.

Luego escanea char por char buscando un `{...}` balanceado, manejando strings escapados. Devuelve `None` si nada parsea — el caller registra error y deja la fila con marcador `error` (que `compute_indicators` filtra y cuenta en `analysis_errors`).

---

## 6. Clasificación canónica de texto libre (industria, fuente_lead)

**Contexto.** El LLM responde con texto libre: "Comercio electrónico de moda sostenible", "comercio electrónico", "Comercio Electrónico". Agrupar por string exacto fragmenta los gráficos en 50+ buckets para 60 clientes — el dashboard se vuelve inútil. Agregamos prompts más estrictos pero el LLM sigue siendo creativo, y queremos que el sistema sea robusto a CSVs nuevos sin re-prompting.

**Decisión.** Post-procesamiento en [service.py](../src/features/indicators/service.py): listas de patterns regex (`_INDUSTRY_PATTERNS`, `_LEAD_SOURCE_PATTERNS`) que mapean texto libre a buckets canónicos. `_classify(value, patterns)` devuelve el primer match, o `"otro"` si ningún pattern matchea, o `"no especificado"` si está vacío. Antes de matchear, `_strip_accents` normaliza tildes para que `búsqueda` y `busqueda` sean equivalentes.

Las **filas crudas** (`analyzed_rows` en la respuesta) preservan el texto original — solo las **agregaciones** usan los buckets. Así el drawer del frontend muestra "Comercio electrónico de moda sostenible" mientras el chart consolida en "moda".

---

## 7. Filtro de "schema echo" en agregaciones de texto libre

**Contexto.** SmolLM2-1.7B (y modelos pequeños en general) tienden a copiar las palabras del prompt como respuesta. Si el prompt incluye "señales como urgencia, presupuesto, autoridad", el modelo emite eso como `puntos_positivos` para muchos clientes, sin extraer realmente del transcript. El dashboard mostraba `urgencia ×33`, `presupuesto ×33`, `autoridad ×33` — falsos insights.

**Decisión.** [_aggregate_items()](../src/features/indicators/service.py) cuenta presencia por fila. Items que aparecen en >50% de las filas se descartan automáticamente — son casi siempre eco del schema, no insights reales. También aplicamos filtros de basura (placeholders `"str"`, números sueltos, items <5 chars).

---

## 8. State del frontend vía `sessionStorage`

**Contexto.** El usuario sube un CSV en `/analyze` y queremos llevarlo a `/dashboard` con la data lista, sin re-correr el análisis (que toma segundos a minutos). Tampoco queremos un backend con estado por usuario (no hay auth en este MVP).

**Decisión.** [indicators-storage.ts](../vambe_frontend/lib/indicators-storage.ts) guarda el `IndicatorsResponse` en `sessionStorage` después de un análisis exitoso. El `Dashboard` lo lee con `useSyncExternalStore`, lo que permite reactividad si el storage cambia (incluso desde otra tab).

---

## 9. Modal "ver más" para listas con overflow

**Contexto.** El dashboard tiene varios charts (industria, fuente, vendedor) que en producción tienen 10-50 categorías pero queremos mostrar solo top 5-6 para no saturar. 

**Decisión.** [CategoryListModal.tsx](../vambe_frontend/components/dashboard/CategoryListModal.tsx) es un modal reusable que recibe `data: CategoryStat[]` y muestra la lista completa con búsqueda + orden (por cantidad o alfabético). `CategoryChart` y `InsightsList` lo invocan en click del botón "Ver N más".
