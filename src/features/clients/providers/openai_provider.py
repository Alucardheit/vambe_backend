import json
import logging
from typing import TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from .base import LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(LLMProvider):
    name = "openai"
    recommended_concurrency = 5

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY no está configurado.")
        self._client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float = 0.2,
    ) -> T:
        # `chat.completions.parse` (openai>=1.40) acepta una clase Pydantic y
        # devuelve la instancia ya validada — el path más robusto.
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "strict": True,
                "schema": schema.model_json_schema(),
            },
        }
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_format,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        return schema.model_validate(json.loads(content))
