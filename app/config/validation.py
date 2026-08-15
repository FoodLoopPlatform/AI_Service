import logging
from app.config.settings import Settings

logger = logging.getLogger(__name__)


class ConfigurationError(Exception):
    """Raised when application production configuration settings are invalid."""

    pass


def validate_production_settings(settings_obj: Settings) -> list[str]:
    """Validates configuration settings for production readiness.

    Returns validation details or raises ConfigurationError if invalid.
    Does NOT log or leak secret values.
    """
    errors: list[str] = []

    # App configuration
    if not settings_obj.APP_NAME:
        errors.append("APP_NAME must not be empty.")

    # Vector store provider check
    if settings_obj.VECTOR_STORE_PROVIDER not in ("memory", "qdrant"):
        errors.append(
            f"Invalid VECTOR_STORE_PROVIDER '{settings_obj.VECTOR_STORE_PROVIDER}'. Must be 'memory' or 'qdrant'."
        )

    if settings_obj.VECTOR_STORE_PROVIDER == "qdrant":
        if not settings_obj.QDRANT_URL:
            errors.append("QDRANT_URL must be configured when VECTOR_STORE_PROVIDER is 'qdrant'.")
        if settings_obj.QDRANT_VECTOR_SIZE != 1024:
            errors.append(
                f"QDRANT_VECTOR_SIZE must be 1024 for BGE-M3 embeddings, got {settings_obj.QDRANT_VECTOR_SIZE}."
            )
        if not settings_obj.QDRANT_COLLECTION_NAME:
            errors.append("QDRANT_COLLECTION_NAME must not be empty.")
        if settings_obj.QDRANT_TIMEOUT_SECONDS <= 0:
            errors.append("QDRANT_TIMEOUT_SECONDS must be greater than 0.")

    # Embedding provider check
    if settings_obj.EMBEDDING_PROVIDER not in ("local_bge_m3", "openai", "fake"):
        errors.append(
            f"Invalid EMBEDDING_PROVIDER '{settings_obj.EMBEDDING_PROVIDER}'. Must be 'local_bge_m3', 'openai', or 'fake'."
        )

    if settings_obj.EMBEDDING_VECTOR_SIZE != 1024:
        errors.append(
            f"EMBEDDING_VECTOR_SIZE must be 1024, got {settings_obj.EMBEDDING_VECTOR_SIZE}."
        )

    if settings_obj.EMBEDDING_DEVICE not in ("cpu", "cuda"):
        errors.append(
            f"Invalid EMBEDDING_DEVICE '{settings_obj.EMBEDDING_DEVICE}'. Must be 'cpu' or 'cuda'."
        )

    # Production-specific environment validation
    if settings_obj.APP_ENV == "production":
        if not settings_obj.OPENAI_API_KEY or settings_obj.OPENAI_API_KEY == "placeholder-api-key":
            errors.append("OPENAI_API_KEY must be configured in production environment.")
        if not settings_obj.OPENAI_MODEL:
            errors.append("OPENAI_MODEL must be configured.")
        if settings_obj.VECTOR_STORE_PROVIDER != "qdrant":
            errors.append("Production environment rejects 'memory' vector store. VECTOR_STORE_PROVIDER must be 'qdrant'.")
        if settings_obj.EMBEDDING_PROVIDER == "fake":
            errors.append("Production environment rejects 'fake' embedding provider. EMBEDDING_PROVIDER must be 'local_bge_m3'.")
        if settings_obj.WEATHER_PROVIDER == "mock":
            errors.append("Production environment rejects 'mock' weather provider. WEATHER_PROVIDER must be 'open_meteo'.")
        if settings_obj.EVENTS_PROVIDER == "mock":
            errors.append("Production environment rejects 'mock' events provider. EVENTS_PROVIDER must be 'nager_date' or 'nager'.")

    # Limits and timeouts
    if settings_obj.MAX_PRICING_BATCH_SIZE <= 0 or settings_obj.MAX_PRICING_BATCH_SIZE > 100:
        errors.append(
            f"MAX_PRICING_BATCH_SIZE must be between 1 and 100, got {settings_obj.MAX_PRICING_BATCH_SIZE}."
        )

    if (
        settings_obj.HISTORICAL_INGESTION_MAX_BATCH_SIZE <= 0
        or settings_obj.HISTORICAL_INGESTION_MAX_BATCH_SIZE > 500
    ):
        errors.append(
            f"HISTORICAL_INGESTION_MAX_BATCH_SIZE must be between 1 and 500, got {settings_obj.HISTORICAL_INGESTION_MAX_BATCH_SIZE}."
        )

    if settings_obj.PRICING_RETRIEVAL_TOP_K <= 0:
        errors.append(
            f"PRICING_RETRIEVAL_TOP_K must be greater than 0, got {settings_obj.PRICING_RETRIEVAL_TOP_K}."
        )

    if settings_obj.OPENAI_TIMEOUT_SECONDS <= 0:
        errors.append(
            f"OPENAI_TIMEOUT_SECONDS must be greater than 0, got {settings_obj.OPENAI_TIMEOUT_SECONDS}."
        )

    if settings_obj.WEATHER_API_TIMEOUT_SECONDS <= 0:
        errors.append("WEATHER_API_TIMEOUT_SECONDS must be greater than 0.")

    if settings_obj.HOLIDAY_API_TIMEOUT_SECONDS <= 0:
        errors.append("HOLIDAY_API_TIMEOUT_SECONDS must be greater than 0.")

    if errors:
        raise ConfigurationError("Production configuration validation failed: " + "; ".join(errors))

    return ["Configuration validated successfully."]
