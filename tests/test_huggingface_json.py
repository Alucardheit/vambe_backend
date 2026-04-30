"""
Tests para el parser de JSON tolerante de huggingface_local.py.

Importante: importamos solo `_extract_first_json` (función pura) para evitar cargar
torch/transformers en CI. El módulo importa torch lazy dentro de las funciones que
lo necesitan, así que importar la función pura es seguro.
"""

from src.features.clients.providers.huggingface_local import _extract_first_json


class TestExtractFirstJson:
    def test_plain_json(self):
        text = '{"a": 1, "b": "x"}'
        assert _extract_first_json(text) == '{"a": 1, "b": "x"}'

    def test_json_with_full_fence(self):
        text = '```json\n{"a": 1}\n```'
        assert _extract_first_json(text) == '{"a": 1}'

    def test_json_with_fence_no_language(self):
        text = '```\n{"a": 1}\n```'
        assert _extract_first_json(text) == '{"a": 1}'

    def test_unclosed_fence_truncated_output(self):
        # Caso real: el modelo emite ```json {...} pero max_new_tokens corta antes del cierre
        text = '```json\n{"a": 1, "b": "complete"}'
        result = _extract_first_json(text)
        assert result == '{"a": 1, "b": "complete"}'

    def test_truncated_json_returns_none(self):
        # JSON genuinamente roto (sin cierre de '}') → None, no fragmento inválido
        text = '```json\n{"a": 1, "b": "incomplet'
        assert _extract_first_json(text) is None

    def test_handles_text_before_json(self):
        # El modelo a veces antepone "Here is the analysis:" o similar
        text = 'Here is the result:\n{"a": 1}'
        assert _extract_first_json(text) == '{"a": 1}'

    def test_handles_nested_braces(self):
        text = '{"outer": {"inner": "value"}, "other": 2}'
        assert _extract_first_json(text) == '{"outer": {"inner": "value"}, "other": 2}'

    def test_handles_braces_inside_strings(self):
        # Una llave dentro de un string no debe contar para el balance
        text = '{"texto": "hola { mundo", "n": 1}'
        assert _extract_first_json(text) == '{"texto": "hola { mundo", "n": 1}'

    def test_handles_escaped_quotes(self):
        text = '{"texto": "ella dijo \\"hola\\"", "n": 1}'
        result = _extract_first_json(text)
        assert result == '{"texto": "ella dijo \\"hola\\"", "n": 1}'

    def test_returns_first_json_when_multiple(self):
        # Si hay 2 objetos, devuelve el primero balanceado
        text = '{"a": 1} y luego {"b": 2}'
        assert _extract_first_json(text) == '{"a": 1}'

    def test_empty_returns_none(self):
        assert _extract_first_json("") is None

    def test_no_json_returns_none(self):
        assert _extract_first_json("solo texto sin estructura") is None
