from fastapi import APIRouter, HTTPException, status

from app.config.settings import settings
from app.config.validation import ConfigurationError, validate_production_settings
from app.vector_store.qdrant import check_qdrant_readiness

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Process liveness check",
    description="Returns 200 OK if the Python service process is alive.",
)
async def health_check():
    return {"status": "ok"}


@router.get(
    "/ready",
    summary="Service readiness check",
    description="Verifies configuration and external dependencies readiness before receiving traffic.",
)
async def readiness_check():
    checks: dict[str, str] = {}

    # 1. Configuration check
    try:
        validate_production_settings(settings)
        checks["configuration"] = "ok"
    except ConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service readiness configuration failure: {e}",
        )

    # 2. Vector store readiness check (if qdrant)
    if settings.VECTOR_STORE_PROVIDER == "qdrant":
        try:
            qdrant_status = check_qdrant_readiness()
            checks["qdrant_vector_store"] = qdrant_status["status"]
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service readiness Qdrant failure: {e}",
            )
    else:
        checks["vector_store_provider"] = settings.VECTOR_STORE_PROVIDER

    return {
        "status": "ready",
        "checks": checks,
    }


@router.get(
    "/version",
    summary="Service version information",
    description="Returns application version and configuration metadata without exposing secrets.",
)
async def version_info():
    return {
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
        "vector_store_provider": settings.VECTOR_STORE_PROVIDER,
    }
