import os
import uuid
import pytest

from app.config.settings import settings
from app.embeddings.bge_m3 import LocalBGEEmbeddingProvider
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store.qdrant import QdrantVectorStore


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS") != "true"
    or settings.VECTOR_STORE_PROVIDER != "qdrant",
    reason="Live historical ingestion test requires RUN_EXTERNAL_INTEGRATION_TESTS=true and VECTOR_STORE_PROVIDER=qdrant",
)
def test_arabic_doc_english_query_live_qdrant_retrieval():
    """Opt-in live integration test validating multilingual retrieval in Qdrant with local BGE-M3."""
    test_collection = f"foodloop_bge_live_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(collection_name=test_collection, vector_size=1024)
    provider = LocalBGEEmbeddingProvider()

    try:
        # 1. Prepare Arabic historical pricing document
        arabic_content = (
            "تم بيع حليب جهينة كامل الدسم سعة 1 لتر بتخفيض 25% وكان الإقبال مرتفعاً جداً"
        )
        doc_id = f"doc_ar_{uuid.uuid4().hex[:8]}"
        doc = PricingKnowledgeDocument(
            document_id=doc_id,
            store_id="store_cairo_01",
            product_id="prod_juhayna_1l",
            content=arabic_content,
            metadata={"category": "dairy", "discount": 0.25},
        )

        # 2. Embed Arabic document using BGE-M3
        doc_embeddings = provider.embed_documents([doc.content])
        store.upsert([doc], doc_embeddings)

        # 3. Query in English
        english_query = "High demand Juhayna milk 25 percent discount"
        query_embedding = provider.embed_query(english_query)

        # 4. Perform vector retrieval
        hits = store.search(query_embedding, store_id="store_cairo_01", top_k=5)
        assert len(hits) >= 1
        assert hits[0].product_id == "prod_juhayna_1l"
        assert hits[0].store_id == "store_cairo_01"
        assert hits[0].relevance_score > 0.4

        # 5. Cleanup
        store.delete([doc_id])
    finally:
        try:
            client = store._get_client()
            client.delete_collection(test_collection)
        except Exception:
            pass
