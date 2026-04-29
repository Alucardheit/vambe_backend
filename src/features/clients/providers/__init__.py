from src.config import settings

from .anthropic_provider import AnthropicProvider
from .base import LLMProvider
from .gemini import GeminiProvider
from .openai_provider import OpenAIProvider

__all__ = [
    "LLMProvider",
    "ProviderName",
    "DEFAULT_MODELS",
    "get_provider",
]

ProviderName = str  # "gemini" | "openai" | "claude" | "anthropic" | "huggingface"

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-haiku-4-5",
    "anthropic": "claude-haiku-4-5",
    "huggingface": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
}


def get_provider(provider: str, model: str | None = None) -> LLMProvider:
    """
    Devuelve una instancia de `LLMProvider` para el `(provider, model)` pedido.

    - `provider`: 'gemini' | 'openai' | 'claude' | 'anthropic' | 'huggingface'
    - `model`: identificador del modelo. Si es None usa `DEFAULT_MODELS[provider]`.

    Lanza:
      - `ValueError` si el provider es desconocido.
      - `RuntimeError` si falta la API key del provider o las deps de HF.
    """
    provider_norm = provider.lower().strip()
    chosen_model = (model or "").strip() or DEFAULT_MODELS.get(provider_norm, "")
    if not chosen_model:
        raise ValueError(f"No hay modelo default para provider '{provider}'.")

    if provider_norm == "gemini":
        return GeminiProvider(
            api_key=settings.google_api_key,
            model=chosen_model,
            max_retries=settings.llm_max_retries,
        )
    if provider_norm == "openai":
        return OpenAIProvider(api_key=settings.openai_api_key, model=chosen_model)
    if provider_norm in {"claude", "anthropic"}:
        return AnthropicProvider(api_key=settings.anthropic_api_key, model=chosen_model)
    if provider_norm == "huggingface":
        # Import perezoso para que el backend levante sin torch/transformers instalados.
        try:
            from .huggingface_local import HuggingFaceLocalProvider
        except ImportError as exc:
            raise RuntimeError(
                "Faltan deps de Hugging Face. Instala con: `uv sync --extra huggingface`"
            ) from exc
        return HuggingFaceLocalProvider(
            model_id=chosen_model,
            token=settings.hf_token or None,
        )
    raise ValueError(f"Provider desconocido: '{provider}'.")
