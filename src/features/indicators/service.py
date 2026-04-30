import re
import unicodedata
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

# Separadores típicos en los textos del LLM (ES + EN): comas, punto y coma,
# saltos, conectores ' y ' / ' and ', viñetas y guiones al inicio.
_ITEM_SPLIT_RE = re.compile(r"[,;\n]| y | and |^\s*[-*•·]\s*", re.IGNORECASE | re.MULTILINE)

# Items vacíos/triviales descartados tras el split.
_TRIVIAL_ITEMS = {
    "", "no especificado", "not specified", "n/a", "na",
    "ninguno", "ninguna", "none", "str", "string", "null",
}

# Filtra valores que parecen tipos/placeholders del schema o contadores numéricos sueltos.
_NUMERIC_ONLY = re.compile(r"^[\d.,%]+$")
_MIN_ITEM_LEN = 5  # 'str', 'n/a', etc. quedan fuera; frases reales pasan.

# Mapeos de normalización para campos categóricos con vocabulario fijo.
# Cubre ES, EN y errores comunes del LLM.
_INTEREST_BUCKETS: dict[str, str] = {
    "alto": "alto", "alta": "alto", "high": "alto",
    "medio": "medio", "media": "medio", "medium": "medio", "mid": "medio",
    "bajo": "bajo", "baja": "bajo", "low": "bajo",
}
_PROBABILITY_BUCKETS: dict[str, str] = _INTEREST_BUCKETS  # mismo vocabulario alta/media/baja

# Buckets canónicos para `industria`. El primer pattern que matchee gana, así que
# patterns más específicos deben ir antes que los genéricos. Los patterns operan
# sobre texto SIN acentos (ver `_classify` → `_strip_accents`).
_INDUSTRY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("salud", re.compile(r"salud|clinic|medic|paciente|hospital|farmac")),
    ("moda", re.compile(r"\bmoda\b|ropa|prenda|textil|vestir|fashion")),
    ("gastronomía", re.compile(r"restaurant|catering|gastronom|comida|alimento|\bbar\b|panader")),
    ("turismo/hospitalidad", re.compile(r"turismo|hotel|hospedaje|viaje|tour")),
    ("educación", re.compile(r"educac|colegio|universidad|curso|escuela|academ")),
    ("inmobiliaria", re.compile(r"inmobil|propiedad|arriend|real\s*estate|bienes\s*raices")),
    ("agricultura", re.compile(r"agric|cultivo|cosecha|granja|agro")),
    ("logística", re.compile(r"logist|envio|transport|delivery|courier")),
    ("manufactura", re.compile(r"fabrica|produccion|manufactur|industrial")),
    ("automotriz", re.compile(r"automov|vehicul|taller|mecanic")),
    ("construcción", re.compile(r"construc|obra|edific|arquitect")),
    ("seguridad", re.compile(r"seguridad|vigilanc|alarma")),
    ("deportes/fitness", re.compile(r"deport|gimnasio|fitness|atlet")),
    ("finanzas", re.compile(r"financ|banco|inversion|seguros|credit")),
    ("consultoría", re.compile(r"consultor|asesor")),
    ("tecnología/software", re.compile(r"software|tech|tecnolog|startup|saas|\bapp\b|\bit\b|digital")),
    ("comercio/retail", re.compile(r"tienda|comercio|e-?commerce|retail|venta")),
]

# Buckets canónicos para `fuente_lead`. Mismas reglas; texto sin acentos.
_LEAD_SOURCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("conferencia/evento", re.compile(r"conferenc|evento|networking|seminario|congreso|charla|expo|feria")),
    ("recomendación", re.compile(r"colega|recomend|mencion|compañer|amigo|conocido|colaborador|cliente\s+actual")),
    ("foro/comunidad", re.compile(r"\bforo\b|comunidad|grupo\s+de")),
    ("redes sociales", re.compile(r"linkedin|podcast|instagram|facebook|twitter|publicacion|articulo|blog|youtube|tiktok")),
    ("búsqueda online", re.compile(r"google|busqu|\bsearch\b|buscando|internet")),
    ("publicidad", re.compile(r"\bads?\b|publicidad|anuncio|campaña")),
]


def _strip_accents(text: str) -> str:
    """Remueve acentos para que regex `busqu` matchee 'búsqueda', 'tecnologia'≈'tecnología', etc."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _classify(value: str, patterns: list[tuple[str, re.Pattern[str]]]) -> str:
    """
    Mapea un string libre del LLM a un bucket canónico vía keyword matching.
    Normaliza acentos antes de matchear ('búsqueda' → 'busqueda').
    Devuelve 'no especificado' si está vacío, 'otro' si ningún pattern matchea.
    """
    raw = (value or "").strip().lower()
    if not raw or raw in _TRIVIAL_ITEMS:
        return "no especificado"
    text = _strip_accents(raw)
    for label, pattern in patterns:
        if pattern.search(text):
            return label
    return "otro"

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


def _bucketize(text: str, buckets: dict[str, str], fallback: str = "no especificado") -> str:
    """
    Normaliza un valor categórico al bucket canónico buscando la primera palabra
    de `text` en el diccionario. 'Alta - tiene presupuesto' → 'alto'.
    Valores que no caen en ningún bucket (basura, números sueltos) → `fallback`.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return fallback
    first = cleaned.split()[0].rstrip(".,;:()")
    return buckets.get(first, fallback)


def _split_items(text: str) -> list[str]:
    """
    Divide un texto libre en items normalizados y descarta basura común del LLM:
    placeholders ('str'/'string'/'null'), números sueltos ('0.6'), items <5 chars.
    """
    if not text:
        return []
    parts = _ITEM_SPLIT_RE.split(text)
    out: list[str] = []
    for p in parts:
        item = p.strip().lower().rstrip(".:;,()").lstrip("-*•·() ")
        if not item or item in _TRIVIAL_ITEMS:
            continue
        if len(item) < _MIN_ITEM_LEN:
            continue
        if _NUMERIC_ONLY.match(item):
            continue
        out.append(item)
    return out


def _aggregate_items(rows: list[dict[str, str]], field: str, top_n: int = 8) -> list[CategoryStat]:
    """
    Extrae items del campo de texto libre y devuelve los top_n más frecuentes.

    Filtra eco del schema: items que aparecen en >50% de las filas son casi siempre
    el LLM repitiendo palabras de las instrucciones, no insights reales.
    """
    n_rows = len(rows) or 1
    row_presence: Counter[str] = Counter()
    raw_counter: Counter[str] = Counter()
    for row in rows:
        items = _split_items(row.get(field, ""))
        for item in items:
            raw_counter[item] += 1
        for item in set(items):
            row_presence[item] += 1

    too_common_threshold = n_rows * 0.5
    filtered = [
        (item, count)
        for item, count in raw_counter.items()
        if row_presence[item] <= too_common_threshold
    ]
    filtered.sort(key=lambda kv: kv[1], reverse=True)

    total = sum(c for _, c in filtered) or 1
    return [
        CategoryStat(label=label, count=count, percentage=_percentage(count, total))
        for label, count in filtered[:top_n]
    ]


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

    # Filas enriquecidas con la industria/fuente clasificadas a buckets canónicos.
    # Se usan para todas las agregaciones, pero `analyzed_rows` mantiene el texto
    # original del LLM para que el drawer muestre el detalle textual.
    enriched = [
        {
            **r,
            "industria": _classify(r.get("industria", ""), _INDUSTRY_PATTERNS),
            "fuente_lead": _classify(r.get("fuente_lead", ""), _LEAD_SOURCE_PATTERNS),
        }
        for r in valid
    ]

    return IndicatorsResponse(
        total_clients=total,
        analysis_errors=errors,
        closed_rate=closed_rate,
        by_industria=_category_stats((r["industria"] for r in enriched), total),
        by_nivel_interes=_category_stats(
            (_bucketize(r.get("nivel_interes", ""), _INTEREST_BUCKETS) for r in enriched), total
        ),
        by_probabilidad_cierre=_category_stats(
            (_bucketize(r.get("probabilidad_cierre", ""), _PROBABILITY_BUCKETS) for r in enriched),
            total,
        ),
        by_fuente_lead=_category_stats((r["fuente_lead"] for r in enriched), total),
        by_vendedor=_category_stats((r.get("vendedor_asignado", "") for r in enriched), total),
        closed_rate_by_industria=_closed_rate_by_category(enriched, "industria"),
        closed_rate_by_vendedor=_closed_rate_by_category(enriched, "vendedor_asignado"),
        industria_area_chart=_industria_area_chart(enriched),
        top_puntos_positivos=_aggregate_items(valid, "puntos_positivos"),
        top_puntos_negativos=_aggregate_items(valid, "puntos_negativos"),
        top_objeciones=_aggregate_items(valid, "objeciones_principales"),
        top_proximos_pasos=_aggregate_items(valid, "proximos_pasos_sugeridos"),
        analyzed_rows=valid,
    )
