import hashlib
import random

from app.config.settings import settings
from app.embeddings.base import (
    EmbeddingProvider,
    EmbeddingProviderError,
    validate_documents_embeddings,
    validate_query_embedding,
)


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic fake embedding provider for unit testing without external calls."""

    def __init__(self, dimension: int | None = None):
        dim = dimension if dimension is not None else settings.EMBEDDING_VECTOR_SIZE
        if dim <= 0:
            raise EmbeddingProviderError("Vector dimension must be greater than zero.")
        self.dimension = dim

    def _generate_deterministic_vector(self, text: str) -> list[float]:
        """Generates a pseudo-random, deterministic float vector from a SHA-256 hash of text."""
        seed_bytes = hashlib.sha256(text.encode("utf-8")).digest()
        seed_int = int.from_bytes(seed_bytes[:4], "big")
        rng = random.Random(seed_int)
        return [round(rng.uniform(-1.0, 1.0), 6) for _ in range(self.dimension)]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate deterministic vector embeddings for a list of document strings."""
        if not documents:
            return []

        embeddings = [self._generate_deterministic_vector(doc) for doc in documents]
        validate_documents_embeddings(documents, embeddings)
        return embeddings

    def embed_query(self, query: str) -> list[float]:
        """Generate a deterministic vector embedding for a query string."""
        embedding = self._generate_deterministic_vector(query)
        validate_query_embedding(query, embedding)
        return embedding

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Generate deterministic vector embeddings for a list of query strings in batch."""
        if not queries:
            return []
        embeddings = [self._generate_deterministic_vector(q) for q in queries]
        validate_documents_embeddings(queries, embeddings)
        return embeddings
