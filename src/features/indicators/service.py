from collections import Counter, defaultdict
from typing import Iterable

from .schemas import (
    CategoryStat,
    ChartSeries,
    ClosedRate,
    ClosedRateByCategory,
    IndicatorsResponse,
    StackedAreaChart,
)

ERROR_MARKER: str = "error"

EN_TO_ES_KEYS: dict[str, str] = {
    "name": "nombre",
    "salesperson": "vendedor_asignado",
    "industry": "industria",
    "use_case": "caso_de_uso",
    "interaction_volume": "volumen_interacciones",
    "specific_needs": "necesidades_especificas",
    "lead_source": "fuente_lead",
    "positive_points": "puntos_positivos",
    "negative_points": "puntos_negativos",
    "main_objections": "objeciones_principales",
    "interest_level": "nivel_interes",
    "closing_probability": "probabilidad_cierre",
    "suggested_next_steps": "proximos_pasos_sugeridos",
    "summary": "resumen",
}


def normalize_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """
    Normaliza filas con headers en inglés a las claves canónicas en español
    para que el resto del cálculo opere sobre un único keyset.
    Las filas que ya están en español se devuelven tal cual.
    """
    if not rows:
        return rows
    sample = rows[0]
    if "industry" in sample and "industria" not in sample:
        return [{EN_TO_ES_KEYS.get(k, k): v for k, v in row.items()} for row in rows]
    return rows


def _is_closed(value: str) -> bool:
    """True si el valor representa una venta cerrada (closed=1)."""
    return str(value).strip() == "1"


def _percentage(part: int, total: int) -> float:
    """`part/total*100` redondeado a 2 decimales; 0 si `total==0`."""
    return round((part / total) * 100, 2) if total else 0.0


def _category_stats(values: Iterable[str], total: int) -> list[CategoryStat]:
    """Cuenta ocurrencias por valor y devuelve estadísticas ordenadas desc por count."""
    counter: Counter[str] = Counter(v.strip() or "no especificado" for v in values)
    return [
        CategoryStat(label=label, count=count, percentage=_percentage(count, total))
        for label, count in counter.most_common()
    ]


def _closed_rate_by_category(
    rows: list[dict[str, str]], category_field: str
) -> list[ClosedRateByCategory]:
    """Calcula tasa de cierre agrupada por una columna categórica."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        label = (row.get(category_field) or "").strip() or "no especificado"
        grouped[label].append(row)

    out: list[ClosedRateByCategory] = []
    for label, group in grouped.items():
        total = len(group)
        closed = sum(1 for r in group if _is_closed(r.get("closed", "")))
        out.append(
            ClosedRateByCategory(
                label=label,
                total=total,
                closed=closed,
                not_closed=total - closed,
                closed_percentage=_percentage(closed, total),
            )
        )
    out.sort(key=lambda x: x.total, reverse=True)
    return out


def _industria_area_chart(rows: list[dict[str, str]]) -> StackedAreaChart:
    """Genera datos de gráfico de área apilado closed/not_closed por industria."""
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: {"closed": 0, "not_closed": 0})
    for row in rows:
        industria = (row.get("industria") or "").strip() or "no especificado"
        bucket = "closed" if _is_closed(row.get("closed", "")) else "not_closed"
        grouped[industria][bucket] += 1

    sorted_items = sorted(
        grouped.items(), key=lambda kv: kv[1]["closed"] + kv[1]["not_closed"], reverse=True
    )
    categories = [k for k, _ in sorted_items]
    closed_values = [v["closed"] for _, v in sorted_items]
    not_closed_values = [v["not_closed"] for _, v in sorted_items]

    return StackedAreaChart(
        categories=categories,
        series=[
            ChartSeries(name="closed", values=closed_values),
            ChartSeries(name="not_closed", values=not_closed_values),
        ],
    )


def _first_word(text: str) -> str:
    """Primera palabra (lowercased) — 'Alta - tiene presupuesto' → 'alta'."""
    cleaned = text.strip().lower()
    return cleaned.split()[0].rstrip(".,;:") if cleaned else "no especificado"


def compute_indicators(rows: list[dict[str, str]]) -> IndicatorsResponse:
    """
    Calcula todos los indicadores a partir de filas analizadas (ES o EN — se normalizan).

    Las filas con `industria == "error"` se cuentan en `analysis_errors` y se excluyen
    del resto de cálculos.

    Retorna: `IndicatorsResponse` con tasas y series listas para graficar.
    """
    rows = normalize_rows(rows)
    errors = sum(1 for r in rows if (r.get("industria") or "").strip() == ERROR_MARKER)
    valid = [r for r in rows if (r.get("industria") or "").strip() != ERROR_MARKER]
    total = len(valid)

    closed_count = sum(1 for r in valid if _is_closed(r.get("closed", "")))
    closed_rate = ClosedRate(
        total=total,
        closed=closed_count,
        not_closed=total - closed_count,
        closed_percentage=_percentage(closed_count, total),
        not_closed_percentage=_percentage(total - closed_count, total),
    )

    return IndicatorsResponse(
        total_clients=total,
        analysis_errors=errors,
        closed_rate=closed_rate,
        by_industria=_category_stats((r.get("industria", "") for r in valid), total),
        by_nivel_interes=_category_stats((r.get("nivel_interes", "") for r in valid), total),
        by_probabilidad_cierre=_category_stats(
            (_first_word(r.get("probabilidad_cierre", "")) for r in valid), total
        ),
        by_fuente_lead=_category_stats((r.get("fuente_lead", "") for r in valid), total),
        by_vendedor=_category_stats((r.get("vendedor_asignado", "") for r in valid), total),
        closed_rate_by_industria=_closed_rate_by_category(valid, "industria"),
        closed_rate_by_vendedor=_closed_rate_by_category(valid, "vendedor_asignado"),
        industria_area_chart=_industria_area_chart(valid),
    )
