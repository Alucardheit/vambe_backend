import asyncio
import logging
import re
from typing import TypeVar

from google import genai
from google.genai import types
from pydantic import BaseModel

from .base import LLMProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s")


def _is_rate_limit_error(exc: Exception) -> bool:
    """True si el error es 429 RESOURCE_EXHAUSTED."""
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg


def _extract_retry_delay(exc: Exception) -> float | None:
    """Extrae el `retryDelay` (segundos) del payload del error 429 si aparece."""
    match = _RETRY_DELAY_RE.search(str(exc))
    return float(match.group(1)) if match else None


class GeminiProvider(LLMProvider):
    name = "gemini"
    # Free tier RPM ~10; mantener bajo para evitar 429.
    recommended_concurrency = 3

    def __init__(self, api_key: str, model: str, max_retries: int = 3) -> None:
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY (o API_KEY) no está configurado.")
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_retries = max_retries

    async def generate_structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        temperature: float = 0.2,
    ) -> T:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(
                        system_instruction=system,
                        response_mime_type="application/json",
                        response_schema=schema,
                        temperature=temperature,
                    ),
                )
                parsed = response.parsed
                return parsed if isinstance(parsed, schema) else schema.model_validate(parsed)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt >= self.max_retries or not _is_rate_limit_error(exc):
                    raise
                delay = _extract_retry_delay(exc) or (2**attempt)
                logger.warning(
                    "Gemini 429 (intento %d/%d), reintentando en %.1fs",
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)
        assert last_exc is not None  # unreachable
        raise last_exc
