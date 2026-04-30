import asyncio
import csv
import io
import logging
from pathlib import Path
from typing import Iterable

from src.config import settings

from .providers import DEFAULT_MODELS, LLMProvider, get_provider
from .schemas import (
    AnalyzedClientRowEN,
    AnalyzedClientRowES,
    ClientAnalysisEN,
    ClientAnalysisES,
    Language,
)

logger = logging.getLogger(__name__)

DEFAULT_CSV_PATH: Path = Path(__file__).resolve().parent.parent.parent / "core" / "vambe_clients.csv"

INPUT_REQUIRED_COLUMNS: tuple[str, ...] = ("Nombre", "Vendedor asignado", "closed", "Transcripcion")

OUTPUT_FIELDS_ES: tuple[str, ...] = (
    "nombre",
    "vendedor_asignado",
    "closed",
    "industria",
    "caso_de_uso",
    "volumen_interacciones",
    "necesidades_especificas",
    "fuente_lead",
    "puntos_positivos",
    "puntos_negativos",
    "objeciones_principales",
    "nivel_interes",
    "probabilidad_cierre",
    "proximos_pasos_sugeridos",
    "resumen",
)

OUTPUT_FIELDS_EN: tuple[str, ...] = (
    "name",
    "salesperson",
    "closed",
    "industry",
    "use_case",
    "interaction_volume",
    "specific_needs",
    "lead_source",
    "positive_points",
    "negative_points",
    "main_objections",
    "interest_level",
    "closing_probability",
    "suggested_next_steps",
    "summary",
)

ANALYSIS_FIELDS_ES: tuple[str, ...] = tuple(
    f for f in OUTPUT_FIELDS_ES if f not in {"nombre", "vendedor_asignado", "closed"}
)
ANALYSIS_FIELDS_EN: tuple[str, ...] = tuple(
    f for f in OUTPUT_FIELDS_EN if f not in {"name", "salesperson", "closed"}
)

OUTPUT_FIELDS_BY_LANG: dict[Language, tuple[str, ...]] = {
    "es": OUTPUT_FIELDS_ES,
    "en": OUTPUT_FIELDS_EN,
}

SYSTEM_PROMPTS: dict[Language, str] = {
    "es": (
        "Eres un analista comercial senior de Vambe (empresa de automatización de "
        "interacciones con clientes: chatbots, IA conversacional). Recibirás la transcripción "
        "de una reunión de ventas con un lead y un flag closed (1=cerró, 0=no cerró). Produce "
        "un análisis ESPECÍFICO y ACCIONABLE en español. CITA TEXTUAL del transcript cuando "
        "puedas — números, sectores, integraciones, frases reales del lead. Prohibido usar "
        "palabras genéricas o categorías abstractas como respuesta.\n\n"
        "Qué va en cada campo:\n"
        "- 'puntos_positivos': señales concretas de que ESTE lead va a comprar — extraídas "
        "del transcript. Ej: 'menciona presupuesto de 5M para Q2', 'urgencia por lanzamiento "
        "en marzo'. NO escribas categorías abstractas; usa las frases del cliente.\n"
        "- 'puntos_negativos': los DOLORES del cliente — qué problema concreto vive HOY que "
        "lo trajo a Vambe. Ej: '500 consultas diarias respondidas a mano', 'pierde leads "
        "fuera de horario'. NO son críticas a Vambe.\n"
        "- 'objeciones_principales': trabas EXPLÍCITAS que el lead mencionó contra contratar. "
        "Si el lead no expresó objeción, escribe 'ninguna'. NO inventes objeciones."
    ),
    "en": (
        "You are a senior sales analyst at Vambe (customer interaction automation company: "
        "chatbots, conversational AI). You will receive a sales meeting transcript with a "
        "lead and a closed flag (1=won, 0=lost). Produce a SPECIFIC and ACTIONABLE analysis "
        "in English. QUOTE the transcript when possible — numbers, sectors, integrations, "
        "real lead phrases. Forbidden to answer with generic words or abstract category names.\n\n"
        "What goes in each field:\n"
        "- 'positive_points': concrete signals THIS lead will buy — pulled from the transcript. "
        "Ex: 'mentioned 5M budget for Q2', 'urgency for March launch'. NEVER abstract "
        "categories; use the client's own phrases.\n"
        "- 'negative_points': client PAIN POINTS — what specific problem they have TODAY that "
        "led them to Vambe. Ex: '500 daily inquiries handled manually', 'losing after-hours "
        "leads'. NOT criticisms of Vambe.\n"
        "- 'main_objections': EXPLICIT blockers the lead voiced against hiring. If the lead "
        "raised no objection, write 'none'. NEVER invent objections."
    ),
}


def _build_prompt(language: Language, nombre: str, vendedor: str, closed: str, transcripcion: str) -> str:
    """Construye el prompt de usuario en el idioma indicado."""
    if language == "es":
        estado = "SE CERRÓ la venta" if str(closed).strip() == "1" else "NO se cerró la venta"
        return (
            f"Cliente: {nombre}\nVendedor asignado: {vendedor}\nResultado: {estado}\n\n"
            f"Transcripción:\n\"\"\"\n{transcripcion}\n\"\"\"\n\n"
            "Genera el análisis basándote SOLO en el transcript. Si un campo no se menciona, "
            "responde 'no especificado'. Para 'nivel_interes' usa exactamente: alto/medio/bajo. "
            "Para 'probabilidad_cierre' usa exactamente: alta/media/baja seguido de una "
            "justificación corta. En 'puntos_positivos', 'puntos_negativos' y "
            "'objeciones_principales' incluye 2-3 items separados por '; ' tomados "
            "TEXTUALMENTE del transcript (no inventes, no uses palabras de las instrucciones)."
        )
    estado = "deal CLOSED" if str(closed).strip() == "1" else "deal NOT closed"
    return (
        f"Client: {nombre}\nAssigned salesperson: {vendedor}\nOutcome: {estado}\n\n"
        f"Transcript:\n\"\"\"\n{transcripcion}\n\"\"\"\n\n"
        "Generate the analysis based ONLY on the transcript. If a field is not mentioned, "
        "answer 'not specified'. For 'interest_level' use exactly: high/medium/low. "
        "For 'closing_probability' use exactly: high/medium/low followed by a short "
        "justification. In 'positive_points', 'negative_points' and 'main_objections' "
        "include 2-3 items separated by '; ' taken VERBATIM from the transcript (do not "
        "invent, do not use words from the instructions)."
    )


def _empty_analysis(language: Language, reason: str) -> ClientAnalysisES | ClientAnalysisEN:
    """Marcadores de error para no romper el batch ante un fallo del provider."""
    if language == "es":
        return ClientAnalysisES(
            industria="error",
            caso_de_uso=reason,
            volumen_interacciones="error",
            necesidades_especificas="error",
            fuente_lead="error",
            puntos_positivos="error",
            puntos_negativos=reason,
            objeciones_principales="error",
            nivel_interes="error",
            probabilidad_cierre="error",
            proximos_pasos_sugeridos="reintentar análisis",
            resumen=f"Error en análisis: {reason}",
        )
    return ClientAnalysisEN(
        industry="error",
        use_case=reason,
        interaction_volume="error",
        specific_needs="error",
        lead_source="error",
        positive_points="error",
        negative_points=reason,
        main_objections="error",
        interest_level="error",
        closing_probability="error",
        suggested_next_steps="retry analysis",
        summary=f"Analysis error: {reason}",
    )


def _build_analyzed_row(
    language: Language,
    analysis: ClientAnalysisES | ClientAnalysisEN,
    nombre: str,
    vendedor: str,
    closed: str,
) -> AnalyzedClientRowES | AnalyzedClientRowEN:
    """Combina passthrough + análisis en un AnalyzedClientRow del idioma correspondiente."""
    if language == "es":
        assert isinstance(analysis, ClientAnalysisES)
        return AnalyzedClientRowES(
            nombre=nombre, vendedor_asignado=vendedor, closed=closed, **analysis.model_dump()
        )
    assert isinstance(analysis, ClientAnalysisEN)
    return AnalyzedClientRowEN(
        name=nombre, salesperson=vendedor, closed=closed, **analysis.model_dump()
    )


async def _analyze_row(
    provider: LLMProvider,
    semaphore: asyncio.Semaphore,
    row: dict[str, str],
    language: Language,
) -> AnalyzedClientRowES | AnalyzedClientRowEN:
    """Analiza una fila con el provider y devuelve la fila enriquecida."""
    nombre = row.get("Nombre", "").strip()
    vendedor = row.get("Vendedor asignado", "").strip()
    closed = row.get("closed", "").strip()
    transcripcion = row.get("Transcripcion", "").strip()

    schema_cls: type[ClientAnalysisES] | type[ClientAnalysisEN] = (
        ClientAnalysisES if language == "es" else ClientAnalysisEN
    )

    if not transcripcion:
        analysis = _empty_analysis(
            language, "transcripción vacía" if language == "es" else "empty transcript"
        )
    else:
        async with semaphore:
            try:
                analysis = await provider.generate_structured(
                    system=SYSTEM_PROMPTS[language],
                    user=_build_prompt(language, nombre, vendedor, closed, transcripcion),
                    schema=schema_cls,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception(
                    "Análisis falló para %s con provider=%s model=%s",
                    nombre,
                    provider.name,
                    getattr(provider, "model", getattr(provider, "model_id", "?")),
                )
                analysis = _empty_analysis(language, f"{type(exc).__name__}: {exc}")

    return _build_analyzed_row(language, analysis, nombre, vendedor, closed)


async def analyze_clients(
    rows: Iterable[dict[str, str]],
    *,
    language: Language = "es",
    provider: str = "gemini",
    model: str | None = None,
) -> list[AnalyzedClientRowES] | list[AnalyzedClientRowEN]:
    """
    Corre el análisis en paralelo con el provider/model elegidos.

    - `provider`: 'gemini' | 'openai' | 'claude' | 'huggingface'.
    - `model`: identificador del modelo; default por provider en `DEFAULT_MODELS`.
    - Concurrencia efectiva: `min(settings.llm_max_concurrency, provider.recommended_concurrency)`.

    Retorna: lista de `AnalyzedClientRowES` o `AnalyzedClientRowEN`.
    """
    llm = get_provider(provider, model)
    concurrency = max(1, min(settings.llm_max_concurrency, llm.recommended_concurrency))
    logger.info(
        "Analizando con provider=%s model=%s concurrency=%d",
        llm.name,
        model or DEFAULT_MODELS.get(provider, "?"),
        concurrency,
    )
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [_analyze_row(llm, semaphore, row, language) for row in rows]
    return await asyncio.gather(*tasks)


def read_csv_rows(content: bytes | str | Path) -> list[dict[str, str]]:
    """Lee un CSV (bytes/str/Path) y devuelve filas como dicts string→string."""
    if isinstance(content, Path):
        text = content.read_text(encoding="utf-8")
    elif isinstance(content, bytes):
        text = content.decode("utf-8")
    else:
        text = content
    reader = csv.DictReader(io.StringIO(text))
    return [{k: (v or "") for k, v in row.items()} for row in reader]


def detect_analysis_language(rows: list[dict[str, str]]) -> Language | None:
    """Devuelve 'es'/'en' si el CSV ya está analizado, o None si es crudo."""
    if not rows:
        return None
    fields = set(rows[0].keys())
    if all(f in fields for f in ANALYSIS_FIELDS_ES):
        return "es"
    if all(f in fields for f in ANALYSIS_FIELDS_EN):
        return "en"
    return None


def is_already_analyzed(rows: list[dict[str, str]]) -> bool:
    """True si las filas ya tienen las columnas de análisis en algún idioma soportado."""
    return detect_analysis_language(rows) is not None


def rows_to_csv(
    rows: list[AnalyzedClientRowES] | list[AnalyzedClientRowEN],
    language: Language = "es",
) -> str:
    """Serializa filas analizadas a CSV con los headers del idioma indicado."""
    fields = OUTPUT_FIELDS_BY_LANG[language]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(fields), quoting=csv.QUOTE_ALL)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.model_dump())
    return buffer.getvalue()
