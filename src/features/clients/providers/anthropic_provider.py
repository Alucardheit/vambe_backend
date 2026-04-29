import logging
from typing import TypeVar

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from .base import LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_TOOL_NAME = "respond_with_analysis"


class AnthropicProvider(LLMProvider):
    name = "anthropic"
    recommended_concurrency = 5

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY no está configurado.")
        self._client = AsyncAnthropic(api_key=api_key)
        self.model = model

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float = 0.2,
    ) -> T:
        # Forzar tool-use con un schema único garantiza JSON estructurado.
        response = await self._client.messages.create(
            model=self.model,
            max_tokens=4096,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
            tools=[
                {
                    "name": _TOOL_NAME,
                    "description": "Devuelve el análisis estructurado del cliente.",
                    "input_schema": schema.model_json_schema(),
                }
            ],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )
        for block in response.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                return schema.model_validate(block.input)
        raise RuntimeError("Anthropic no devolvió un bloque tool_use con el análisis.")
