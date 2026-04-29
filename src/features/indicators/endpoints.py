import logging
from typing import Annotated, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status

from src.features.clients.analyzer import (
    DEFAULT_CSV_PATH,
    INPUT_REQUIRED_COLUMNS,
    analyze_clients,
    detect_analysis_language,
    read_csv_rows,
)
from src.features.clients.schemas import Language

from .schemas import IndicatorsResponse
from .service import compute_indicators

logger = logging.getLogger(__name__)
router = APIRouter()

ProviderName = Literal["gemini", "openai", "claude", "anthropic", "huggingface"]


@router.post(
    "/compute",
    summary="Calcula indicadores comerciales (acepta CSV ES o EN, cualquier provider)",
    response_model=IndicatorsResponse,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_400_BAD_REQUEST: {"description": "CSV vacío o columnas faltantes."},
        status.HTTP_404_NOT_FOUND: {"description": "El CSV por defecto no existe."},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"description": "Falla en el análisis previo."},
    },
)
async def compute(
    file: Annotated[
        UploadFile | None,
        File(description="CSV crudo o ya analizado (ES o EN); opcional."),
    ] = None,
    language: Annotated[
        Language,
        Query(description="Idioma del análisis si el CSV es crudo: 'es' (default) o 'en'."),
    ] = "es",
    provider: Annotated[
        ProviderName,
        Query(description="LLM provider para CSV crudos: gemini | openai | claude | huggingface."),
    ] = "gemini",
    model: Annotated[
        str | None,
        Query(description="Identificador del modelo. None → default por provider."),
    ] = None,
) -> IndicatorsResponse:
    """
    Calcula indicadores agregados a partir de un CSV de clientes.

    - Si el CSV ya tiene columnas de análisis (ES o EN), las usa directamente.
    - Si es crudo, corre análisis con `provider`/`model` y luego computa.
    - Sin file → usa `src/core/vambe_clients.csv`.

    Retorna: `IndicatorsResponse` con tasas, distribuciones y datos para gráficos.
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

    detected = detect_analysis_language(rows)
    if detected is not None:
        logger.info("CSV ya analizado (lang=%s): %d filas", detected, len(rows))
        analyzed_rows = rows
    else:
        missing = [c for c in INPUT_REQUIRED_COLUMNS if c not in rows[0]]
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Faltan columnas en el CSV crudo: {missing}",
            )
        logger.info(
            "CSV crudo: corriendo %s/%s para %d filas (lang=%s)",
            provider,
            model or "default",
            len(rows),
            language,
        )
        try:
            analyzed = await analyze_clients(
                rows, language=language, provider=provider, model=model
            )
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)
            ) from exc
        analyzed_rows = [r.model_dump() for r in analyzed]

    return compute_indicators(analyzed_rows)
