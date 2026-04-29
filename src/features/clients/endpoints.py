import logging
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse

from .analyzer import (
    DEFAULT_CSV_PATH,
    INPUT_REQUIRED_COLUMNS,
    analyze_clients,
    read_csv_rows,
    rows_to_csv,
)
from .schemas import AnalyzeResponse, Language

logger = logging.getLogger(__name__)
router = APIRouter()

OutputFormat = Literal["json", "csv"]
ProviderName = Literal["gemini", "openai", "claude", "anthropic", "huggingface"]


@router.post(
    "/analyze",
    summary="Analiza transcripciones con el LLM elegido (Gemini/OpenAI/Claude/HF)",
    status_code=status.HTTP_200_OK,
    response_model=None,
    responses={
        status.HTTP_200_OK: {
            "description": "JSON con filas analizadas, o CSV descargable si format=csv.",
            "content": {"application/json": {}, "text/csv": {}},
        },
        status.HTTP_400_BAD_REQUEST: {"description": "El CSV está vacío o le faltan columnas."},
        status.HTTP_404_NOT_FOUND: {"description": "El CSV por defecto no existe."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Falla en el análisis."},
    },
)
async def analyze(
    file: Annotated[UploadFile | None, File(description="CSV de clientes; opcional.")] = None,
    language: Annotated[
        Language,
        Query(description="Idioma del análisis y headers/keys: 'es' (default) o 'en'."),
    ] = "es",
    format: Annotated[
        OutputFormat,
        Query(description="'json' (default, para mostrar en UI) o 'csv' (descarga directa)."),
    ] = "json",
    provider: Annotated[
        ProviderName,
        Query(description="LLM provider: gemini | openai | claude | huggingface."),
    ] = "gemini",
    model: Annotated[
        str | None,
        Query(description="Identificador del modelo. None → default por provider."),
    ] = None,
) -> AnalyzeResponse | StreamingResponse:
    """
    Analiza la columna `Transcripcion` con el provider/modelo elegidos y devuelve JSON o CSV.

    - Sin file → usa `src/core/vambe_clients.csv`.
    - El CSV de entrada debe tener: `Nombre`, `Vendedor asignado`, `closed`, `Transcripcion`.
    - Provider 'huggingface' requiere `uv sync --extra huggingface` y descarga el modelo a disco.

    **Retorna**: `AnalyzeResponse` (JSON) o `StreamingResponse` (CSV).
    """
    if file is not None:
        rows = read_csv_rows(await file.read())
    else:
        if not DEFAULT_CSV_PATH.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No existe {DEFAULT_CSV_PATH}",
            )
        rows = read_csv_rows(DEFAULT_CSV_PATH)

    if not rows:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="El CSV no contiene filas.")

    missing = [c for c in INPUT_REQUIRED_COLUMNS if c not in rows[0]]
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Faltan columnas en el CSV: {missing}",
        )

    logger.info(
        "Analizando %d clientes (lang=%s, format=%s, provider=%s, model=%s)",
        len(rows),
        language,
        format,
        provider,
        model or "default",
    )
    try:
        analyzed = await analyze_clients(rows, language=language, provider=provider, model=model)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    if format == "csv":
        csv_text = rows_to_csv(analyzed, language=language)
        filename = f"vambe_clients_analysis_{language}.csv"
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    return AnalyzeResponse(
        language=language,
        count=len(analyzed),
        rows=[r.model_dump() for r in analyzed],
    )
