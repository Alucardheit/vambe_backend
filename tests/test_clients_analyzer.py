"""Tests para src/features/clients/analyzer.py — helpers no-LLM."""

from src.features.clients.analyzer import (
    SYSTEM_PROMPTS,
    _build_prompt,
    _empty_analysis,
    detect_analysis_language,
    is_already_analyzed,
    read_csv_rows,
    rows_to_csv,
)
from src.features.clients.schemas import (
    AnalyzedClientRowES,
    ClientAnalysisEN,
    ClientAnalysisES,
)


class TestReadCsvRows:
    def test_parses_basic_csv(self):
        csv_text = "Nombre,closed,Transcripcion\nCarlos,1,hola mundo\nMaria,0,otro texto\n"
        rows = read_csv_rows(csv_text)
        assert len(rows) == 2
        assert rows[0]["Nombre"] == "Carlos"
        assert rows[0]["closed"] == "1"
        assert rows[1]["Transcripcion"] == "otro texto"

    def test_handles_bytes_input(self):
        csv_bytes = b"Nombre,closed\nCarlos,1\n"
        rows = read_csv_rows(csv_bytes)
        assert rows[0]["Nombre"] == "Carlos"

    def test_empty_cells_become_empty_string(self):
        csv_text = "a,b,c\n1,,3\n"
        rows = read_csv_rows(csv_text)
        assert rows[0]["b"] == ""


class TestDetectAnalysisLanguage:
    def test_detects_spanish_when_es_fields_present(self):
        rows = [
            {
                "industria": "tech",
                "caso_de_uso": "x",
                "volumen_interacciones": "x",
                "necesidades_especificas": "x",
                "fuente_lead": "x",
                "puntos_positivos": "x",
                "puntos_negativos": "x",
                "objeciones_principales": "x",
                "nivel_interes": "alto",
                "probabilidad_cierre": "alta",
                "proximos_pasos_sugeridos": "x",
                "resumen": "x",
            }
        ]
        assert detect_analysis_language(rows) == "es"

    def test_detects_english_when_en_fields_present(self):
        rows = [
            {
                "industry": "tech",
                "use_case": "x",
                "interaction_volume": "x",
                "specific_needs": "x",
                "lead_source": "x",
                "positive_points": "x",
                "negative_points": "x",
                "main_objections": "x",
                "interest_level": "high",
                "closing_probability": "high",
                "suggested_next_steps": "x",
                "summary": "x",
            }
        ]
        assert detect_analysis_language(rows) == "en"

    def test_returns_none_for_raw_csv(self):
        rows = [{"Nombre": "Carlos", "closed": "1", "Transcripcion": "x"}]
        assert detect_analysis_language(rows) is None

    def test_returns_none_for_empty(self):
        assert detect_analysis_language([]) is None


class TestIsAlreadyAnalyzed:
    def test_returns_true_for_analyzed(self):
        rows = [
            {
                "industria": "tech",
                "caso_de_uso": "x",
                "volumen_interacciones": "x",
                "necesidades_especificas": "x",
                "fuente_lead": "x",
                "puntos_positivos": "x",
                "puntos_negativos": "x",
                "objeciones_principales": "x",
                "nivel_interes": "alto",
                "probabilidad_cierre": "alta",
                "proximos_pasos_sugeridos": "x",
                "resumen": "x",
            }
        ]
        assert is_already_analyzed(rows) is True

    def test_returns_false_for_raw(self):
        rows = [{"Nombre": "Carlos", "closed": "1"}]
        assert is_already_analyzed(rows) is False


class TestBuildPrompt:
    def test_es_prompt_includes_closed_status(self):
        prompt = _build_prompt("es", "Carlos", "Toro", "1", "transcripción de prueba")
        assert "Carlos" in prompt
        assert "Toro" in prompt
        assert "SE CERRÓ" in prompt
        assert "transcripción de prueba" in prompt

    def test_es_prompt_indicates_not_closed(self):
        prompt = _build_prompt("es", "Maria", "Puma", "0", "x")
        assert "NO se cerró" in prompt

    def test_en_prompt(self):
        prompt = _build_prompt("en", "Carlos", "Toro", "1", "test transcript")
        assert "Carlos" in prompt
        assert "deal CLOSED" in prompt
        assert "test transcript" in prompt

    def test_system_prompts_include_definitions(self):
        # Validamos que el prompt explica los conceptos clave (anti-eco)
        assert "puntos_positivos" in SYSTEM_PROMPTS["es"]
        assert "DOLORES" in SYSTEM_PROMPTS["es"] or "dolor" in SYSTEM_PROMPTS["es"].lower()
        assert "positive_points" in SYSTEM_PROMPTS["en"]


class TestEmptyAnalysis:
    def test_es_returns_es_schema(self):
        empty = _empty_analysis("es", "transcripción vacía")
        assert isinstance(empty, ClientAnalysisES)
        assert empty.industria == "error"
        assert "transcripción vacía" in empty.resumen

    def test_en_returns_en_schema(self):
        empty = _empty_analysis("en", "empty transcript")
        assert isinstance(empty, ClientAnalysisEN)
        assert empty.industry == "error"
        assert "empty transcript" in empty.summary


class TestRowsToCsv:
    def test_serializes_es_rows_with_correct_headers(self):
        row = AnalyzedClientRowES(
            nombre="Carlos",
            vendedor_asignado="Toro",
            closed="1",
            industria="tech",
            caso_de_uso="x",
            volumen_interacciones="x",
            necesidades_especificas="x",
            fuente_lead="x",
            puntos_positivos="x",
            puntos_negativos="x",
            objeciones_principales="x",
            nivel_interes="alto",
            probabilidad_cierre="alta",
            proximos_pasos_sugeridos="x",
            resumen="x",
        )
        csv_text = rows_to_csv([row], language="es")
        lines = csv_text.strip().split("\n")
        # header en primera línea
        assert "nombre" in lines[0]
        assert "industria" in lines[0]
        # data en segunda
        assert "Carlos" in lines[1]
        assert "Toro" in lines[1]
