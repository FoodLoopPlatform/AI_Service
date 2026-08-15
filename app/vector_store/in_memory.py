import math

from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store.base import (
    VectorStore,
    VectorStoreError,
    validate_search_inputs,
    validate_upsert_inputs,
)


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculates normalized cosine similarity score between 0.0 and 1.0."""
    if len(vec1) != len(vec2):
        raise VectorStoreError(
            f"Vector dimension mismatch during similarity calculation: {len(vec1)} vs {len(vec2)}."
        )

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm_a = math.sqrt(sum(a * a for a in vec1))
    norm_b = math.sqrt(sum(b * b for b in vec2))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    raw_cos = dot_product / (norm_a * norm_b)
    clamped_cos = max(-1.0, min(1.0, raw_cos))
    score = (clamped_cos + 1.0) / 2.0
    return max(0.0, min(1.0, score))


class InMemoryVectorStore(VectorStore):
    """In-memory deterministic vector store implementation for testing."""

    def __init__(self):
        self._storage: dict[str, tuple[PricingKnowledgeDocument, list[float]]] = {}

    def upsert(
        self,
        documents: list[PricingKnowledgeDocument],
        embeddings: list[list[float]],
    ) -> None:
        """Inserts or updates documents and embeddings idempotently by document_id."""
        validate_upsert_inputs(documents, embeddings)
        for doc, vec in zip(documents, embeddings):
            self._storage[doc.document_id] = (doc, vec)

    def search(
        self,
        query_embedding: list[float],
        *,
        store_id: str,
        product_id: str | None = None,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[PricingKnowledgeItem]:
        """Searches for documents using vector similarity and metadata filtering."""
        validate_search_inputs(query_embedding, store_id=store_id, top_k=top_k)

        if not self._storage:
            return []

        scored_records = []
        for doc_id, (doc, vec) in self._storage.items():
            # 1. Filter by store_id
            if doc.store_id != store_id:
                continue

            # 2. Filter by product_id (if provided)
            if product_id is not None and doc.product_id != product_id:
                continue

            # 3. Filter by category (if provided)
            if category is not None:
                doc_cat = doc.metadata.get("category")
                if doc_cat != category:
                    continue

            # 4. Calculate similarity score
            score = _cosine_similarity(query_embedding, vec)
            scored_records.append((score, doc))

        if not scored_records:
            return []

        # 5. Sort descending by score, ties broken deterministically by document_id
        scored_records.sort(key=lambda x: (x[0], x[1].document_id), reverse=True)

        # 6. Limit to top_k and map to PricingKnowledgeItem
        results = []
        for score, doc in scored_records[:top_k]:
            item = PricingKnowledgeItem(
                product_id=doc.product_id,
                store_id=doc.store_id,
                content=doc.content,
                metadata=doc.metadata,
                relevance_score=score,
            )
            results.append(item)

        return results

    def delete(
        self,
        document_ids: list[str],
    ) -> None:
        """Deletes documents matching the specified document IDs."""
        if not document_ids:
            return
        for doc_id in document_ids:
            self._storage.pop(doc_id, None)
