"""Tests para src/features/indicators/service.py — funciones puras de agregación."""

import pytest

from src.features.indicators.service import (
    _INDUSTRY_PATTERNS,
    _INTEREST_BUCKETS,
    _LEAD_SOURCE_PATTERNS,
    _aggregate_items,
    _bucketize,
    _category_stats,
    _classify,
    _is_closed,
    _percentage,
    _split_items,
    _strip_accents,
    compute_indicators,
    normalize_rows,
)


class TestPrimitives:
    def test_is_closed(self):
        assert _is_closed("1") is True
        assert _is_closed(" 1 ") is True
        assert _is_closed("0") is False
        assert _is_closed("") is False
        assert _is_closed("yes") is False

    def test_percentage(self):
        assert _percentage(50, 100) == 50.0
        assert _percentage(1, 3) == 33.33
        assert _percentage(0, 0) == 0.0  # división por cero → 0
        assert _percentage(0, 10) == 0.0

    def test_strip_accents(self):
        assert _strip_accents("búsqueda") == "busqueda"
        assert _strip_accents("Tecnología") == "Tecnologia"
        assert _strip_accents("Educación") == "Educacion"
        assert _strip_accents("sin acentos") == "sin acentos"


class TestBucketize:
    def test_normalizes_interest_es_en(self):
        assert _bucketize("alta", _INTEREST_BUCKETS) == "alto"
        assert _bucketize("alto", _INTEREST_BUCKETS) == "alto"
        assert _bucketize("high", _INTEREST_BUCKETS) == "alto"
        assert _bucketize("media", _INTEREST_BUCKETS) == "medio"
        assert _bucketize("medium", _INTEREST_BUCKETS) == "medio"
        assert _bucketize("baja", _INTEREST_BUCKETS) == "bajo"
        assert _bucketize("low", _INTEREST_BUCKETS) == "bajo"

    def test_handles_compound_values(self):
        # 'alta - tiene presupuesto' → primer token es 'alta'
        assert _bucketize("alta - tiene presupuesto", _INTEREST_BUCKETS) == "alto"
        assert _bucketize("high (decision maker)", _INTEREST_BUCKETS) == "alto"

    def test_garbage_falls_to_default(self):
        assert _bucketize("0.6", _INTEREST_BUCKETS) == "no especificado"
        assert _bucketize("str", _INTEREST_BUCKETS) == "no especificado"
        assert _bucketize("", _INTEREST_BUCKETS) == "no especificado"
        assert _bucketize("xyz", _INTEREST_BUCKETS) == "no especificado"


class TestSplitItems:
    def test_splits_by_common_separators(self):
        assert _split_items("rapport, conocimiento, capacidad de respuesta") == [
            "rapport",
            "conocimiento",
            "capacidad de respuesta",
        ]
        assert _split_items("urgencia; presupuesto; autoridad") == [
            "urgencia",
            "presupuesto",
            "autoridad",
        ]
        assert _split_items("rapport y experiencia previa") == ["rapport", "experiencia previa"]

    def test_filters_junk(self):
        assert _split_items("str") == []  # placeholder
        assert _split_items("0.6") == []  # numérico puro
        assert _split_items("n/a") == []  # short trivial
        assert _split_items("no especificado") == []  # marker conocido
        assert _split_items("") == []
        assert _split_items("a, bb, ccc") == []  # todos < 5 chars

    def test_strips_punctuation(self):
        # Bullets, paréntesis y puntos finales se limpian
        items = _split_items("- presupuesto disponible; - urgencia confirmada.")
        assert "presupuesto disponible" in items
        assert "urgencia confirmada" in items


class TestClassify:
    def test_industry_handles_case_and_accents(self):
        # 3 variantes del mismo concepto → mismo bucket
        assert _classify("Comercio electrónico", _INDUSTRY_PATTERNS) == "comercio/retail"
        assert _classify("comercio electrónico", _INDUSTRY_PATTERNS) == "comercio/retail"
        assert _classify("Comercio Electrónico", _INDUSTRY_PATTERNS) == "comercio/retail"

    def test_industry_picks_specific_first(self):
        # 'Software financiero' debe caer en 'finanzas' (más específico que 'tech')
        assert _classify("Software financiero", _INDUSTRY_PATTERNS) == "finanzas"
        # 'Moda sostenible' → 'moda'
        assert _classify("Comercio electrónico de moda sostenible", _INDUSTRY_PATTERNS) == "moda"

    def test_industry_common_buckets(self):
        cases = {
            "Clínica médica": "salud",
            "restaurante familiar": "gastronomía",
            "startup tecnológica": "tecnología/software",
            "consultoría ambiental": "consultoría",
            "agricultura sostenible": "agricultura",
            "empresa de logística": "logística",
            "taller mecánico": "automotriz",
            "construcción residencial": "construcción",
        }
        for value, expected in cases.items():
            assert _classify(value, _INDUSTRY_PATTERNS) == expected, f"falló: {value!r}"

    def test_lead_source_handles_variants(self):
        cases = {
            "colega": "recomendación",
            "colega en la industria": "recomendación",
            "Foro de comercio electrónico": "foro/comunidad",
            "conferencia de tecnología": "conferencia/evento",
            "Evento de Networking": "conferencia/evento",
            "publicación en LinkedIn": "redes sociales",
            "podcast de tecnología": "redes sociales",
            "búsqueda en Google": "búsqueda online",
            "búsqueda en internet": "búsqueda online",
            "campaña publicitaria": "publicidad",
        }
        for value, expected in cases.items():
            assert _classify(value, _LEAD_SOURCE_PATTERNS) == expected, f"falló: {value!r}"

    def test_unknown_falls_to_otro(self):
        assert _classify("xyzabc123", _INDUSTRY_PATTERNS) == "otro"
        assert _classify("xyzabc123", _LEAD_SOURCE_PATTERNS) == "otro"

    def test_empty_returns_no_especificado(self):
        assert _classify("", _INDUSTRY_PATTERNS) == "no especificado"
        assert _classify("   ", _INDUSTRY_PATTERNS) == "no especificado"
        assert _classify("no especificado", _INDUSTRY_PATTERNS) == "no especificado"


class TestAggregateItems:
    def test_basic_frequency_count(self):
        # 10 filas para que ningún item supere el 50% y caiga en el filtro de schema-echo
        rows = [
            {"campo": "rapport, urgencia"},
            {"campo": "rapport, presupuesto"},
            {"campo": "urgencia, autoridad"},
            {"campo": "fit con producto"},
            {"campo": "decision maker presente"},
            {"campo": "presupuesto confirmado"},
            {"campo": "timing apropiado"},
            {"campo": "cultura tecnologica"},
            {"campo": "experiencia previa"},
            {"campo": "interes genuino"},
        ]
        result = _aggregate_items(rows, "campo", top_n=10)
        labels = {s.label: s.count for s in result}
        # rapport aparece 2 veces (20%), bajo threshold
        assert labels["rapport"] == 2
        assert labels["urgencia"] == 2
        assert labels["presupuesto"] == 1
        assert labels["autoridad"] == 1

    def test_filters_schema_echo(self):
        # Item presente en >50% de las filas se descarta como eco del schema
        rows = [{"campo": "urgencia, presupuesto, autoridad"}] * 10
        rows.append({"campo": "frase realmente específica del cliente"})
        result = _aggregate_items(rows, "campo", top_n=10)
        labels = [s.label for s in result]
        assert "urgencia" not in labels
        assert "presupuesto" not in labels
        assert "autoridad" not in labels
        assert "frase realmente especifica del cliente" in [_strip_accents(l) for l in labels] or \
               "frase realmente específica del cliente" in labels

    def test_empty_rows(self):
        assert _aggregate_items([], "campo") == []
        assert _aggregate_items([{"campo": ""}], "campo") == []

    def test_respects_top_n(self):
        rows = [{"campo": f"item_{i}"} for i in range(20)]
        result = _aggregate_items(rows, "campo", top_n=5)
        assert len(result) <= 5


class TestCategoryStats:
    def test_counts_and_percentages(self):
        result = _category_stats(["a", "b", "a", "c", "a"], total=5)
        labels = {s.label: (s.count, s.percentage) for s in result}
        assert labels["a"] == (3, 60.0)
        assert labels["b"] == (1, 20.0)
        assert labels["c"] == (1, 20.0)

    def test_handles_empty_strings(self):
        result = _category_stats(["", "  ", "x"], total=3)
        labels = {s.label for s in result}
        assert "no especificado" in labels  # empty/whitespace bucketed
        assert "x" in labels


class TestNormalizeRows:
    def test_translates_english_keys_to_spanish(self):
        rows = [{"name": "Carlos", "industry": "tech", "closed": "1"}]
        normalized = normalize_rows(rows)
        assert normalized[0]["nombre"] == "Carlos"
        assert normalized[0]["industria"] == "tech"

    def test_passthrough_when_already_spanish(self):
        rows = [{"nombre": "Carlos", "industria": "tech"}]
        assert normalize_rows(rows) == rows

    def test_empty_list(self):
        assert normalize_rows([]) == []


class TestComputeIndicators:
    @pytest.fixture
    def synthetic_rows(self):
        """3 rows: 2 cerradas, 1 no cerrada, distintas industrias y vendedores."""
        return [
            {
                "nombre": "Cliente A",
                "vendedor_asignado": "Toro",
                "closed": "1",
                "industria": "Comercio electrónico",
                "fuente_lead": "colega",
                "nivel_interes": "alta",
                "probabilidad_cierre": "alta - tiene presupuesto",
                "puntos_positivos": "presupuesto confirmado, urgencia clara",
                "puntos_negativos": "gestión manual ineficiente",
                "objeciones_principales": "ninguna",
                "proximos_pasos_sugeridos": "enviar propuesta",
                "caso_de_uso": "automatización",
                "necesidades_especificas": "integración con CRM",
                "volumen_interacciones": "500 semanales",
                "resumen": "Lead caliente",
            },
            {
                "nombre": "Cliente B",
                "vendedor_asignado": "Puma",
                "closed": "0",
                "industria": "Clínica médica",
                "fuente_lead": "conferencia de tecnología",
                "nivel_interes": "medium",
                "probabilidad_cierre": "media",
                "puntos_positivos": "interés genuino",
                "puntos_negativos": "no tiene presupuesto Q2",
                "objeciones_principales": "precio alto",
                "proximos_pasos_sugeridos": "follow-up en Q3",
                "caso_de_uso": "atención pacientes",
                "necesidades_especificas": "HIPAA compliance",
                "volumen_interacciones": "100 diarias",
                "resumen": "Lead tibio",
            },
            {
                "nombre": "Cliente C",
                "vendedor_asignado": "Toro",
                "closed": "1",
                "industria": "Software financiero",
                "fuente_lead": "Foro de comercio electrónico",
                "nivel_interes": "high",
                "probabilidad_cierre": "high",
                "puntos_positivos": "decision maker presente",
                "puntos_negativos": "procesos manuales",
                "objeciones_principales": "timing de implementación",
                "proximos_pasos_sugeridos": "demo técnica",
                "caso_de_uso": "operaciones",
                "necesidades_especificas": "API REST",
                "volumen_interacciones": "1000 mensuales",
                "resumen": "Lead caliente",
            },
        ]

    def test_basic_counts(self, synthetic_rows):
        result = compute_indicators(synthetic_rows)
        assert result.total_clients == 3
        assert result.closed_rate.closed == 2
        assert result.closed_rate.not_closed == 1
        assert result.closed_rate.closed_percentage == pytest.approx(66.67, abs=0.1)

    def test_industry_classified_to_buckets(self, synthetic_rows):
        result = compute_indicators(synthetic_rows)
        labels = {s.label for s in result.by_industria}
        # 'Comercio electrónico' → comercio/retail
        # 'Clínica médica' → salud
        # 'Software financiero' → finanzas (NO 'tecnología' — orden importa)
        assert "comercio/retail" in labels
        assert "salud" in labels
        assert "finanzas" in labels

    def test_lead_source_classified(self, synthetic_rows):
        result = compute_indicators(synthetic_rows)
        labels = {s.label for s in result.by_fuente_lead}
        assert "recomendación" in labels  # 'colega'
        assert "conferencia/evento" in labels  # 'conferencia de tecnología'
        assert "foro/comunidad" in labels  # 'Foro de comercio electrónico'

    def test_interest_normalized(self, synthetic_rows):
        result = compute_indicators(synthetic_rows)
        labels = {s.label for s in result.by_nivel_interes}
        # 'alta', 'medium', 'high' → 'alto', 'medio', 'alto'
        assert "alto" in labels
        assert "medio" in labels

    def test_analyzed_rows_preserves_originals(self, synthetic_rows):
        # El drawer del frontend necesita el texto original, no los buckets
        result = compute_indicators(synthetic_rows)
        original_industries = {r["industria"] for r in result.analyzed_rows}
        assert "Comercio electrónico" in original_industries
        assert "Clínica médica" in original_industries
        assert "Software financiero" in original_industries

    def test_excludes_error_rows(self, synthetic_rows):
        synthetic_rows.append(
            {**synthetic_rows[0], "industria": "error", "nombre": "Cliente Error"}
        )
        result = compute_indicators(synthetic_rows)
        assert result.total_clients == 3  # los 3 válidos
        assert result.analysis_errors == 1
