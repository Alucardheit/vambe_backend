from typing import Optional, Sequence

from pydantic import Field
from pydantic_settings import BaseSettings


class CORSSettings(BaseSettings):
	"""
	Configuration for handling Cross-Origin Resource Sharing (CORS) in FastAPI.

	Reference: https://fastapi.tiangolo.com/tutorial/cors/
	"""

	cors_allow_origins: Sequence[str] = Field(
		default=["*"],
		validation_alias="CORS_ALLOW_ORIGINS",
		description="A list of origins that should be permitted to make cross-origin requests.",
	)
	cors_allow_origin_regex: Optional[str] = Field(
		default=None,
		validation_alias="CORS_ALLOW_ORIGIN_REGEX",
		description=(
			"A regex string to match against origins that should be permitted to make cross-origin requests."
		),
	)
	cors_allow_methods: Sequence[str] = Field(
		default=["*"],
		validation_alias="CORS_ALLOW_METHODS",
		description="A list of HTTP methods that should be allowed for cross-origin requests.",
	)
	cors_allow_headers: Sequence[str] = Field(
		default=["*"],
		validation_alias="CORS_ALLOW_HEADERS",
		description="A list of HTTP request headers that should be supported for cross-origin requests.",
	)
	cors_allow_credentials: bool = Field(
		default=False,
		validation_alias="CORS_ALLOW_CREDENTIALS",
		description="Indicate that cookies should be supported for cross-origin requests.",
	)
	cors_expose_headers: Sequence[str] = Field(
		default=[],
		validation_alias="CORS_EXPOSE_HEADERS",
		description="Indicate any response headers that should be made accessible to the browser.",
	)
	cors_max_age: int = Field(
		default=600,
		validation_alias="CORS_MAX_AGE",
		description="Sets a maximum time in seconds for browsers to cache CORS responses.",
	)

