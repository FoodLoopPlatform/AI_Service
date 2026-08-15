from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.api.routes.health import router as health_router
from app.api.routes.monitoring import router as monitoring_router
from app.api.routes.pricing import router as pricing_router
from app.config.settings import settings
from app.config.validation import validate_production_settings
from app.embeddings.base import EmbeddingProviderError
from app.middleware.correlation import CorrelationIdMiddleware
from app.vector_store.base import VectorStoreError

logger = logging.getLogger("foodloop")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup initialization and production configuration validation
    logger.info(
        "Starting %s (v%s) in '%s' environment...",
        settings.APP_NAME,
        settings.APP_VERSION,
        settings.APP_ENV,
    )
    try:
        validate_production_settings(settings)
        logger.info("Production configuration validated successfully.")
    except Exception as e:
        logger.warning("Startup configuration warning: %s", e)

    yield

    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Middleware
app.add_middleware(CorrelationIdMiddleware)

# Include Routers
app.include_router(health_router)
app.include_router(monitoring_router)
app.include_router(pricing_router)


# Global Exception Handlers
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "validation_error",
            "message": str(exc),
        },
    )


@app.exception_handler(ValidationError)
@app.exception_handler(ValueError)
async def domain_validation_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "validation_error",
            "message": str(exc),
        },
    )


@app.exception_handler(VectorStoreError)
@app.exception_handler(EmbeddingProviderError)
async def infrastructure_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "retrieval_error",
            "message": "Retrieval infrastructure error occurred.",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": "An unexpected error occurred during request processing.",
        },
    )
