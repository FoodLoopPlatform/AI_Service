import os
import uuid
import pytest

from app.config.settings import settings
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store.qdrant import QdrantVectorStore, check_qdrant_readiness


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS") != "true"
    or settings.VECTOR_STORE_PROVIDER != "qdrant",
    reason="Live Qdrant test requires RUN_EXTERNAL_INTEGRATION_TESTS=true and VECTOR_STORE_PROVIDER=qdrant",
)
def test_real_qdrant_live_integration():
    """Opt-in live integration test against Qdrant using a dedicated test collection."""
    test_collection = f"foodloop_test_{uuid.uuid4().hex[:8]}"
    store = QdrantVectorStore(collection_name=test_collection, vector_size=1024)

    try:
        # 1. Readiness check
        readiness = check_qdrant_readiness(store)
        assert readiness["status"] == "ok"

        # 2. Upsert document
        doc_id = f"doc_{uuid.uuid4().hex[:8]}"
        doc = PricingKnowledgeDocument(
            document_id=doc_id,
            store_id="store_live_123",
            product_id="prod_live_456",
            content="Historical discount test record",
            metadata={"category": "dairy"},
        )
        fake_vector = [0.1] * 1024
        store.upsert([doc], [fake_vector])

        # 3. Search document
        results = store.search(fake_vector, store_id="store_live_123", top_k=5)
        assert len(results) >= 1
        assert results[0].product_id == "prod_live_456"

        # 4. Delete document
        store.delete([doc_id])
    finally:
        # Cleanup test collection
        try:
            client = store._get_client()
            client.delete_collection(test_collection)
        except Exception:
            pass
