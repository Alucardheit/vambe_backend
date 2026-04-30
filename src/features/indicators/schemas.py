from pydantic import BaseModel, Field


class CategoryStat(BaseModel):
    """Conteo y porcentaje de una categoría dentro del total."""

    label: str = Field(description="Nombre de la categoría (industria, vendedor, etc.).")
    count: int = Field(ge=0, description="Cantidad de filas en esta categoría.")
    percentage: float = Field(ge=0, le=100, description="Porcentaje sobre el total (0-100).")


class ClosedRate(BaseModel):
    """Tasa global de cierre."""

    total: int = Field(ge=0)
    closed: int = Field(ge=0)
    not_closed: int = Field(ge=0)
    closed_percentage: float = Field(ge=0, le=100)
    not_closed_percentage: float = Field(ge=0, le=100)


class ClosedRateByCategory(BaseModel):
    """Tasa de cierre desglosada por una categoría (industria, vendedor, ...)."""

    label: str
    total: int = Field(ge=0)
    closed: int = Field(ge=0)
    not_closed: int = Field(ge=0)
    closed_percentage: float = Field(ge=0, le=100)


class ChartSeries(BaseModel):
    """Serie de datos para un gráfico."""

    name: str = Field(description="Nombre de la serie, ej. 'closed' o 'not_closed'.")
    values: list[int] = Field(description="Valores alineados con `categories` del chart.")


class StackedAreaChart(BaseModel):
    """
    Datos listos para un gráfico de área apilado (closed vs not_closed) por categoría.
    El frontend mapea `categories` al eje X y `series` a las áreas.
    """

    categories: list[str] = Field(description="Etiquetas del eje X, p. ej. industrias.")
    series: list[ChartSeries] = Field(description="Series apiladas (típicamente 'closed' y 'not_closed').")


class IndicatorsResponse(BaseModel):
    """Respuesta agregada de indicadores comerciales."""

    total_clients: int = Field(ge=0, description="Total de filas analizadas (excluye errores de análisis).")
    analysis_errors: int = Field(ge=0, description="Cantidad de filas con análisis fallido.")
    closed_rate: ClosedRate
    by_industria: list[CategoryStat]
    by_nivel_interes: list[CategoryStat]
    by_probabilidad_cierre: list[CategoryStat]
    by_fuente_lead: list[CategoryStat]
    by_vendedor: list[CategoryStat]
    closed_rate_by_industria: list[ClosedRateByCategory]
    closed_rate_by_vendedor: list[ClosedRateByCategory]
    industria_area_chart: StackedAreaChart = Field(
        description="Datos para gráfico de área apilado: closed vs not_closed por industria."
    )
    top_puntos_positivos: list[CategoryStat] = Field(
        default_factory=list,
        description="Items más mencionados extraídos del campo puntos_positivos.",
    )
    top_puntos_negativos: list[CategoryStat] = Field(
        default_factory=list,
        description="Items más mencionados extraídos del campo puntos_negativos.",
    )
    top_objeciones: list[CategoryStat] = Field(
        default_factory=list,
        description="Objeciones más frecuentes extraídas del campo objeciones_principales.",
    )
    top_proximos_pasos: list[CategoryStat] = Field(
        default_factory=list,
        description="Próximos pasos sugeridos más mencionados.",
    )
    analyzed_rows: list[dict] = Field(
        default_factory=list,
        description="Filas analizadas completas para drill-down por cliente.",
    )
