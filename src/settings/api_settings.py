from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class APISettings(BaseSettings):
    """
    FastAPI application metadata and behavior.

    Mirrors the fields previously defined in src/config.py.
    """
    # https://fastapi.tiangolo.com/reference/fastapi/#fastapi.FastAPI
    api_debug: bool = Field(
        default=False,
        validation_alias="API_DEBUG",
        description="Boolean indicating if debug tracebacks should be returned on server errors.",
    )
    api_title: str = Field(
        default="Suite",
        validation_alias="API_TITLE",
        description="The title of the API.",
    )
    api_summary: Optional[str] = Field(
        default="Suite",
        validation_alias="API_SUMMARY",
        description="A short summary of the API.",
    )
    api_description: str = Field(
        default="",
        validation_alias="API_DESCRIPTION",
        description="A description of the API. Supports Markdown (using CommonMark syntax).",
    )
    api_version: str = Field(
        default="1.0.0",
        validation_alias="API_VERSION",
        description="The version of the API.",
    )
    api_openapi_url: Optional[str] = Field(
        default="/openapi.json",
        validation_alias="API_OPENAPI_URL",
        description="The URL where the OpenAPI schema will be served from.",
    )
    api_docs_url: Optional[str] = Field(
        default="/docs",
        validation_alias="API_DOCS_URL",
        description="The path to the automatic interactive API documentation.",
    )
    api_redoc_url: Optional[str] = Field(
        default="/redoc",
        validation_alias="API_REDOC_URL",
        description=(
            "The path to the alternative automatic interactive API documentation provided by ReDoc."
        ),
    )
    api_root_path: str = Field(
        default="",
        validation_alias="API_ROOT_PATH",
        description=(
            "A path prefix handled by a proxy that is not seen by the application but is seen by external clients, "
            "which affects things like Swagger UI."
        ),
    )
    api_disable_docs: bool = Field(
        default=False,
        validation_alias="API_DISABLE_DOCS",
        description="Boolean indicating if documentation should be disabled.",
    )
