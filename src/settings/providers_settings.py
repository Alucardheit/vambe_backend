from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class ProvidersSettings(BaseSettings):
    """API keys y comportamiento compartido entre proveedores LLM."""

    # Google / Gemini — `GOOGLE_API_KEY` preferido; `API_KEY` por compatibilidad.
    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "API_KEY"),
        description="API key de Google AI Studio (Gemini).",
    )

    # OpenAI
    openai_api_key: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY",
        description="API key de OpenAI.",
    )

    # Anthropic / Claude
    anthropic_api_key: str = Field(
        default="",
        validation_alias="ANTHROPIC_API_KEY",
        description="API key de Anthropic (Claude).",
    )

    # Hugging Face — opcional (modelos públicos no la requieren)
    hf_token: str = Field(
        default="",
        validation_alias=AliasChoices("HF_TOKEN", "HUGGINGFACE_TOKEN"),
        description="Token de Hugging Face. Solo necesario para modelos gated.",
    )

    # Comportamiento general
    llm_max_concurrency: int = Field(
        default=3,
        validation_alias=AliasChoices("LLM_MAX_CONCURRENCY", "GEMINI_MAX_CONCURRENCY"),
        description="Concurrencia máxima de llamadas al LLM (provider puede bajarla).",
    )
    llm_max_retries: int = Field(
        default=3,
        validation_alias=AliasChoices("LLM_MAX_RETRIES", "GEMINI_MAX_RETRIES"),
        description="Reintentos ante 429/rate-limit con backoff.",
    )
