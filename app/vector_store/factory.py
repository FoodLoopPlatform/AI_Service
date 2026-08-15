from app.config.settings import settings
from app.vector_store.base import VectorStore, VectorStoreError
from app.vector_store.in_memory import InMemoryVectorStore
from app.vector_store.qdrant import QdrantVectorStore

_vector_store_instance: VectorStore | None = None


def get_vector_store(provider: str | None = None) -> VectorStore:
    """Factory function to retrieve the configured VectorStore instance.
    
    Supports 'memory' (default) and 'qdrant' provider selection based on Settings or argument.
    Instantiation does not make network calls.
    """
    global _vector_store_instance
    if _vector_store_instance is not None and provider is None:
        return _vector_store_instance

    selected_provider = (provider or settings.VECTOR_STORE_PROVIDER or "memory").lower().strip()

    if selected_provider == "memory":
        store = InMemoryVectorStore()
    elif selected_provider == "qdrant":
        store = QdrantVectorStore()
    else:
        raise VectorStoreError(
            f"Unsupported VECTOR_STORE_PROVIDER: '{selected_provider}'. Expected 'memory' or 'qdrant'."
        )

    if provider is None:
        _vector_store_instance = store

    return store


def set_vector_store(store: VectorStore | None) -> None:
    """Sets or resets the singleton vector store instance (useful for testing)."""
    global _vector_store_instance
    _vector_store_instance = store
