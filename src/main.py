import logging

import uvicorn
from fastapi import FastAPI
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.middleware.cors import CORSMiddleware

from src.config import settings
from src.features.api import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

app = FastAPI(
    debug=settings.api_debug,
    title=settings.api_title,
    summary=settings.api_summary,
    description=settings.api_description,
    version=settings.api_version,
    openapi_url=settings.api_openapi_url if not settings.api_disable_docs else None,
    docs_url=settings.api_docs_url if not settings.api_disable_docs else None,
    redoc_url=settings.api_redoc_url if not settings.api_disable_docs else None,
    root_path=settings.api_root_path,
    swagger_ui_parameters={"operationsSorter": "alpha"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
    allow_credentials=settings.cors_allow_credentials,
    allow_origin_regex=settings.cors_allow_origin_regex,
    expose_headers=settings.cors_expose_headers,
    max_age=settings.cors_max_age,
)


@app.get("/")
async def read_main():
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "documentation": {
            "swagger": f"{settings.api_docs_url}",
            "redoc": f"{settings.api_redoc_url}",
            "openapi": f"{settings.api_openapi_url}",
        },
    }


app.include_router(router)

FastAPIInstrumentor.instrument_app(app)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000, log_level="debug")
