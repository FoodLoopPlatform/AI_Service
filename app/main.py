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


tags_metadata = [
    {
        "name": "health",
        "description": "Liveness and readiness health checks for service monitoring and load balancers.",
    },
    {
        "name": "monitoring",
        "description": "Inventory risk evaluation, shelf-life pressure assessment, and routing (`NO_ACTION` vs `PRICING`).",
    },
    {
        "name": "pricing",
        "description": "Batch dynamic pricing recommendation engine (0–15% discount) and historical knowledge ingestion.",
    },
]

app = FastAPI(
    title="FoodLoop AI Service API",
    description="""
# 🥦 FoodLoop AI Service API 🚀

Autonomous Multi-Agent Microservice powering real-time inventory risk monitoring and dynamic pricing optimization.

### 🌟 Key Features
* 🛡️ **Inventory Risk Monitoring**: Automated evaluation of shelf-life, sales velocity, and risk pressure.
* 🏷️ **Dynamic Pricing Agent**: Batch-optimized LLM and rule-based discount recommendations (0–15%).
* 📚 **RAG Knowledge Ingestion**: Embeds and indexes historical pricing outcomes into Qdrant vector store.
* ⚡ **High Availability**: Deterministic business fallbacks when external LLM gateways are unavailable.

---
* **Swagger UI Documentation**: `/docs`
* **ReDoc Documentation**: `/redoc`
* **OpenAPI Specification**: `/openapi.json`
""",
    version=settings.APP_VERSION,
    openapi_tags=tags_metadata,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
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
    logger.error("Request validation failed for %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=getattr(status, "HTTP_422_UNPROCESSABLE_CONTENT", 422),
        content={
            "error": "validation_error",
            "message": str(exc),
            "details": exc.errors() if hasattr(exc, "errors") else None,
        },
    )


@app.exception_handler(ValidationError)
@app.exception_handler(ValueError)
async def domain_validation_exception_handler(request: Request, exc: Exception):
    logger.error("Validation error for %s: %s", request.url.path, exc)
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
    logger.error("Infrastructure error for %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "retrieval_error",
            "message": f"Retrieval infrastructure error: {str(exc)}",
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception processing %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "internal_error",
            "message": f"An error occurred during request processing: {str(exc)}",
        },
    )
