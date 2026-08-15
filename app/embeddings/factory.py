from app.config.settings import settings
from app.embeddings.base import EmbeddingProvider, EmbeddingProviderError


def get_embedding_provider() -> EmbeddingProvider:
    """Factory function to instantiate the configured EmbeddingProvider based on application settings.
    
    Instantiation is deferred until execution time (not at import time).
    Supported values for settings.EMBEDDING_PROVIDER:
      - "fake": FakeEmbeddingProvider for fast deterministic testing
      - "local_bge_m3" (or "bge_m3"): LocalBGEEmbeddingProvider using BAAI/bge-m3
      - "openai": OpenAIEmbeddingProvider using OpenAI API
    """
    provider_name = (settings.EMBEDDING_PROVIDER or "local_bge_m3").lower().strip()

    if provider_name == "fake":
        from app.embeddings.fake import FakeEmbeddingProvider
        return FakeEmbeddingProvider()

    elif provider_name in ("local_bge_m3", "bge_m3", "bge-m3", "local"):
        from app.embeddings.bge_m3 import LocalBGEEmbeddingProvider
        return LocalBGEEmbeddingProvider()

    elif provider_name == "openai":
        from app.embeddings.openai import OpenAIEmbeddingProvider
        return OpenAIEmbeddingProvider(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
        )

    else:
        raise EmbeddingProviderError(
            f"Unsupported EMBEDDING_PROVIDER setting: '{provider_name}'. "
            f"Supported providers are ('fake', 'local_bge_m3', 'openai')."
        )
