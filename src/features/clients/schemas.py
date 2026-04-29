from typing import Literal

from pydantic import BaseModel, Field

Language = Literal["es", "en"]


class ClientAnalysisES(BaseModel):
    """Análisis estructurado producido por Gemini en español."""

    industria: str = Field(description="Sector o industria del cliente.")
    caso_de_uso: str = Field(description="Problema concreto que el cliente busca resolver con Vambe.")
    volumen_interacciones: str = Field(
        description="Volumen mencionado (ej. '500 semanales'). 'no especificado' si no aparece.",
    )
    necesidades_especificas: str = Field(
        description="Features, integraciones o requisitos específicos que pidió el cliente.",
    )
    fuente_lead: str = Field(description="Cómo conoció Vambe.")
    puntos_positivos: str = Field(description="Señales favorables del lead, separadas por '; '.")
    puntos_negativos: str = Field(description="Riesgos o señales desfavorables, separadas por '; '.")
    objeciones_principales: str = Field(description="Objeciones explícitas, separadas por '; '. 'ninguna' si no hay.")
    nivel_interes: str = Field(description="'alto', 'medio' o 'bajo'.")
    probabilidad_cierre: str = Field(description="'alta'/'media'/'baja' + justificación corta.")
    proximos_pasos_sugeridos: str = Field(description="Acciones recomendadas, separadas por '; '.")
    resumen: str = Field(description="Resumen de 1-2 líneas.")


class ClientAnalysisEN(BaseModel):
    """Structured analysis produced by Gemini in English."""

    industry: str = Field(description="Client industry or sector.")
    use_case: str = Field(description="Concrete problem the client wants to solve with Vambe.")
    interaction_volume: str = Field(
        description="Volume mentioned (e.g. '500 weekly'). 'not specified' if absent.",
    )
    specific_needs: str = Field(description="Specific features, integrations or requirements requested.")
    lead_source: str = Field(description="How the client heard about Vambe.")
    positive_points: str = Field(description="Favorable signals, separated by '; '.")
    negative_points: str = Field(description="Risks or unfavorable signals, separated by '; '.")
    main_objections: str = Field(description="Explicit objections, separated by '; '. 'none' if none.")
    interest_level: str = Field(description="'high', 'medium' or 'low'.")
    closing_probability: str = Field(description="'high'/'medium'/'low' + short justification.")
    suggested_next_steps: str = Field(description="Recommended actions, separated by '; '.")
    summary: str = Field(description="1-2 line summary.")


class AnalyzedClientRowES(ClientAnalysisES):
    nombre: str
    vendedor_asignado: str
    closed: str


class AnalyzedClientRowEN(ClientAnalysisEN):
    name: str
    salesperson: str
    closed: str


# Back-compat aliases (default language = Spanish)
ClientAnalysis = ClientAnalysisES
AnalyzedClientRow = AnalyzedClientRowES


class AnalyzeResponse(BaseModel):
    """Respuesta JSON del endpoint `/clients/analyze`."""

    language: Language = Field(description="Idioma de las filas: 'es' o 'en'.")
    count: int = Field(ge=0, description="Cantidad de filas analizadas.")
    rows: list[dict] = Field(
        description=(
            "Filas analizadas. Las claves siguen el idioma elegido: ES → 'nombre','industria',... | "
            "EN → 'name','industry',..."
        ),
    )
