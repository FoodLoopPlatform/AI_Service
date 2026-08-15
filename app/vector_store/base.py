from abc import ABC, abstractmethod

from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument


class VectorStoreError(Exception):
    """Exception raised for vector store infrastructure or input errors."""

    pass


class VectorStore(ABC):
    """Abstract provider-agnostic interface for vector store operations."""

    @abstractmethod
    def upsert(
        self,
        documents: list[PricingKnowledgeDocument],
        embeddings: list[list[float]],
    ) -> None:
        """Inserts or updates documents and their corresponding vector embeddings."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        *,
        store_id: str,
        product_id: str | None = None,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[PricingKnowledgeItem]:
        """Searches for relevant knowledge items using vector similarity and metadata filtering."""
        pass

    @abstractmethod
    def delete(
        self,
        document_ids: list[str],
    ) -> None:
        """Deletes documents matching the specified document IDs."""
        pass


def validate_upsert_inputs(
    documents: list[PricingKnowledgeDocument],
    embeddings: list[list[float]],
) -> None:
    """Validates inputs for upsert operations."""
    if len(documents) != len(embeddings):
        raise VectorStoreError(
            f"Document count ({len(documents)}) does not match embedding count ({len(embeddings)})."
        )

    if not documents:
        return

    expected_dim = len(embeddings[0])

    for idx, (doc, vec) in enumerate(zip(documents, embeddings)):
        if not doc.document_id or not isinstance(doc.document_id, str) or not doc.document_id.strip():
            raise VectorStoreError(f"Document at index {idx} has an empty document_id.")
        if len(vec) == 0:
            raise VectorStoreError(f"Embedding vector at index {idx} is empty.")
        if len(vec) != expected_dim:
            raise VectorStoreError(
                f"Inconsistent vector dimension at index {idx}: expected {expected_dim}, got {len(vec)}."
            )



def validate_search_inputs(
    query_embedding: list[float],
    store_id: str,
    top_k: int,
) -> None:
    """Validates inputs for search operations."""
    if not query_embedding or len(query_embedding) == 0:
        raise VectorStoreError("Query embedding vector cannot be empty.")

    if not store_id or not isinstance(store_id, str) or not store_id.strip():
        raise VectorStoreError("store_id is mandatory and must be a non-empty string.")

    if top_k < 1:
        raise VectorStoreError(f"top_k must be at least 1, got {top_k}.")
