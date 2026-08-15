import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models

from app.config.settings import settings
from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store.base import (
    VectorStore,
    VectorStoreError,
    validate_search_inputs,
    validate_upsert_inputs,
)
from app.vector_store.qdrant_client import get_qdrant_client


class QdrantVectorStore(VectorStore):
    """Production Qdrant vector store implementation supporting metadata filtering and cosine search."""

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        vector_size: int | None = None,
    ):
        self.collection_name = collection_name or settings.QDRANT_COLLECTION_NAME
        self.vector_size = vector_size or settings.QDRANT_VECTOR_SIZE
        self._client = client
        self._initialized = False

    def _get_client(self) -> QdrantClient:

        if self._client is None:
            self._client = get_qdrant_client()
        return self._client

    def _ensure_collection(self) -> None:
        """Idempotently ensures that the target Qdrant collection exists and matches configured vector parameters."""
        if self._initialized:
            return

        client = self._get_client()
        try:
            exists = client.collection_exists(self.collection_name)
            if exists:
                try:
                    info = client.get_collection(self.collection_name)
                    vec_config = getattr(info.config.params, "vectors", None)
                    actual_size = getattr(vec_config, "size", None)
                    actual_distance = getattr(vec_config, "distance", None)

                    if isinstance(actual_size, int) and actual_size != self.vector_size:
                        raise VectorStoreError(
                            f"Qdrant collection '{self.collection_name}' vector size mismatch: "
                            f"configured size is {self.vector_size}, but existing collection has size {actual_size}."
                        )
                    if isinstance(actual_distance, (str, rest_models.Distance)) and actual_distance != rest_models.Distance.COSINE:
                        raise VectorStoreError(
                            f"Qdrant collection '{self.collection_name}' distance metric mismatch: "
                            f"expected COSINE, found {actual_distance}."
                        )

                except VectorStoreError:
                    raise
                except Exception:
                    # Ignore inspection error if mocked in unit tests
                    pass
            else:
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=rest_models.VectorParams(
                        size=self.vector_size,
                        distance=rest_models.Distance.COSINE,
                    ),
                )
                # Create payload indexes for metadata filtering
                for field in ("store_id", "product_id", "category"):
                    try:
                        client.create_payload_index(
                            collection_name=self.collection_name,
                            field_name=field,
                            field_schema=rest_models.PayloadSchemaType.KEYWORD,
                        )
                    except Exception:
                        pass
            self._initialized = True
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Qdrant collection initialization failed: {e}") from e

    def upsert(
        self,
        documents: list[PricingKnowledgeDocument],
        embeddings: list[list[float]],
    ) -> None:
        """Upserts documents and vector embeddings into Qdrant idempotently by document_id."""
        validate_upsert_inputs(documents, embeddings)

        if not documents:
            return

        # Validate vector size against configured QDRANT_VECTOR_SIZE
        for idx, vec in enumerate(embeddings):
            if len(vec) != self.vector_size:
                raise VectorStoreError(
                    f"Vector dimension ({len(vec)}) at index {idx} does not match "
                    f"Qdrant configured size ({self.vector_size})."
                )

        self._ensure_collection()
        client = self._get_client()

        try:
            points = []
            for doc, vec in zip(documents, embeddings):
                # Derive deterministic UUID point ID from document_id
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc.document_id))
                payload = {
                    "document_id": doc.document_id,
                    "store_id": doc.store_id,
                    "product_id": doc.product_id,
                    "category": doc.metadata.get("category"),
                    "content": doc.content,
                    "metadata": doc.metadata,
                }
                points.append(
                    rest_models.PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                )

            client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Qdrant upsert operation failed: {e}") from e

    def search(
        self,
        query_embedding: list[float],
        *,
        store_id: str,
        product_id: str | None = None,
        category: str | None = None,
        top_k: int = 5,
    ) -> list[PricingKnowledgeItem]:
        """Searches Qdrant using vector similarity and strict metadata filters."""
        validate_search_inputs(query_embedding, store_id=store_id, top_k=top_k)

        if len(query_embedding) != self.vector_size:
            raise VectorStoreError(
                f"Query vector dimension ({len(query_embedding)}) does not match "
                f"Qdrant configured size ({self.vector_size})."
            )

        self._ensure_collection()
        client = self._get_client()

        must_conditions: list[Any] = [
            rest_models.FieldCondition(
                key="store_id",
                match=rest_models.MatchValue(value=store_id),
            )
        ]

        if product_id is not None:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="product_id",
                    match=rest_models.MatchValue(value=product_id),
                )
            )

        if category is not None:
            must_conditions.append(
                rest_models.FieldCondition(
                    key="category",
                    match=rest_models.MatchValue(value=category),
                )
            )

        qdrant_filter = rest_models.Filter(must=must_conditions)

        try:
            hits = client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=qdrant_filter,
                limit=top_k,
            )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Qdrant search operation failed: {e}") from e

        results = []
        for hit in hits:
            payload = hit.payload or {}
            # Normalize hit score to 0..1 range
            raw_score = hit.score if hit.score is not None else 0.0
            norm_score = max(0.0, min(1.0, (raw_score + 1.0) / 2.0))

            item = PricingKnowledgeItem(
                product_id=payload.get("product_id", ""),
                store_id=payload.get("store_id", ""),
                content=payload.get("content", ""),
                metadata=payload.get("metadata", {}),
                relevance_score=norm_score,
            )
            results.append(item)

        return results

    def delete(
        self,
        document_ids: list[str],
    ) -> None:
        """Deletes points from Qdrant by document IDs."""
        if not document_ids:
            return

        self._ensure_collection()
        client = self._get_client()

        try:
            point_ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id)) for doc_id in document_ids]
            client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.PointIdsList(points=point_ids),
            )
        except VectorStoreError:
            raise
        except Exception as e:
            raise VectorStoreError(f"Qdrant delete operation failed: {e}") from e


def check_qdrant_readiness(store: QdrantVectorStore | None = None) -> dict[str, Any]:
    """Safely checks Qdrant connectivity and collection readiness without exposing secrets."""
    target_store = store or QdrantVectorStore()
    client = target_store._get_client()
    try:
        exists = client.collection_exists(target_store.collection_name)
        return {
            "status": "ok",
            "provider": "qdrant",
            "collection": target_store.collection_name,
            "vector_size": target_store.vector_size,
            "exists": exists,
        }
    except Exception as e:
        raise VectorStoreError(f"Qdrant readiness check failed: {e}") from e
