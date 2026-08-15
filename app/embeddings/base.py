from abc import ABC, abstractmethod


class EmbeddingProviderError(Exception):
    """Exception raised for errors during embedding generation or vector validation."""

    pass


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    @abstractmethod
    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of document strings."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Generate a vector embedding for a single query string."""
        pass

    @abstractmethod
    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Generate vector embeddings for a list of query strings in a single batch operation."""
        pass



def validate_documents_embeddings(
    documents: list[str],
    embeddings: list[list[float]],
) -> None:
    """Validates returned document embeddings against batch validation constraints."""
    if not documents:
        if embeddings:
            raise EmbeddingProviderError("Expected empty embeddings for empty documents input.")
        return

    if len(embeddings) != len(documents):
        raise EmbeddingProviderError(
            f"Embedding count mismatch: expected {len(documents)}, got {len(embeddings)}."
        )

    expected_dim = len(embeddings[0])
    if expected_dim == 0:
        raise EmbeddingProviderError("Returned empty vector in embedding batch.")

    for idx, vec in enumerate(embeddings):
        if len(vec) == 0:
            raise EmbeddingProviderError(f"Returned empty vector at index {idx} in embedding batch.")
        if len(vec) != expected_dim:
            raise EmbeddingProviderError(
                f"Inconsistent vector dimension at index {idx}: expected {expected_dim}, got {len(vec)}."
            )


def validate_query_embedding(query: str, embedding: list[float]) -> None:
    """Validates returned query embedding."""
    if not embedding or len(embedding) == 0:
        raise EmbeddingProviderError("Query embedding vector is empty.")
