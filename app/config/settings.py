from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central production settings and environment variables for FoodLoop AI Service."""

    APP_NAME: str = "FoodLoop AI Service"
    APP_ENV: str = "development"
    APP_VERSION: str = "1.0.0"

    # Batch limits
    MAX_PRICING_BATCH_SIZE: int = 50
    HISTORICAL_INGESTION_MAX_BATCH_SIZE: int = 100

    # OpenAI-compatible LLM configuration (SambaNova / Gemma 2 27B IT)
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.sambanova.ai/v1"
    OPENAI_MODEL: str = "gemma-2-27b-it"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_TIMEOUT_SECONDS: float = 30.0



    # Local BGE-M3 Multilingual Embedding Configuration (MVP Default)
    EMBEDDING_PROVIDER: str = "local_bge_m3"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_VECTOR_SIZE: int = 1024
    EMBEDDING_DEVICE: str = "cpu"

    # Vector store configuration (Aligned with BGE-M3 1024-d embeddings)
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str = ""
    QDRANT_COLLECTION_NAME: str = "foodloop_pricing_knowledge_bge_m3"
    QDRANT_VECTOR_SIZE: int = 1024
    QDRANT_TIMEOUT_SECONDS: float = 10.0

    VECTOR_STORE_PROVIDER: str = "memory"
    PRICING_RETRIEVAL_TOP_K: int = 5

    # External tool adapters configuration - Weather
    WEATHER_PROVIDER: str = "mock"
    WEATHER_API_BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
    WEATHER_API_KEY: str = ""
    WEATHER_API_TIMEOUT_SECONDS: float = 5.0

    # External tool adapters configuration - Holidays / Events
    EVENTS_PROVIDER: str = "mock"
    HOLIDAY_API_BASE_URL: str = "https://date.nager.at/api/v4"
    HOLIDAY_API_TIMEOUT_SECONDS: float = 5.0
    DEFAULT_COUNTRY_CODE: str = "EG"

    EVENTS_API_KEY: str = ""
    EVENTS_BASE_URL: str = ""
    EVENTS_API_TIMEOUT_SECONDS: float = 5.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
