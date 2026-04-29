from abc import ABC, abstractmethod
from typing import TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Interfaz async para LLMs que producen un objeto Pydantic estructurado."""

    name: str
    """Identificador del provider (gemini/openai/anthropic/huggingface)."""

    recommended_concurrency: int = 5
    """Tope de paralelismo razonable. La capa superior toma min(setting, este)."""

    @abstractmethod
    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float = 0.2,
    ) -> T:
        """
        Llama al LLM con `system`+`user` y devuelve una instancia validada de `schema`.
        Debe convertir las respuestas del provider al schema Pydantic.
        """
