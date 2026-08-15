from datetime import datetime, timezone
import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.config.settings import settings
from app.embeddings import EmbeddingProviderError, FakeEmbeddingProvider
from app.main import app
from app.agents.pricing.knowledge_builder import build_pricing_knowledge_document
from app.agents.pricing.retriever import VectorPricingKnowledgeRetriever
from app.schemas.historical_pricing import HistoricalPricingEvent, Outcome
from app.schemas.pricing import PricingProductContext
from app.schemas.pricing_knowledge_ingestion import (
    HistoricalPricingIngestionRequest,
    HistoricalPricingIngestionResponse,
)
from app.services.pricing_knowledge_ingestion import HistoricalPricingIngestionService
from app.vector_store import VectorStoreError
from app.vector_store.in_memory import InMemoryVectorStore

client = TestClient(app)


def _make_event(
    event_id: str = "evt-001",
    store_id: str = "store-01",
    product_id: str = "prod-100",
    discount: float = 10.0,
    outcome: Outcome = Outcome.SOLD_OUT,
) -> HistoricalPricingEvent:
    return HistoricalPricingEvent(
        event_id=event_id,
        store_id=store_id,
        product_id=product_id,
        category="Dairy",
        recorded_at=datetime(2026, 8, 10, 12, 0, tzinfo=timezone.utc),
        quantity=20.0,
        current_price=40.0,
        original_price=40.0,
        price_floor=28.0,
        sales_velocity=2.0,
        historical_average_daily_sales=5.0,
        hours_remaining=30.0,
        discount_percentage=discount,
        units_sold_after_discount=18.0,
        sell_through_rate=0.90,
        outcome=outcome,
    )


def _make_product_ctx(store_id: str, product_id: str) -> PricingProductContext:
    return PricingProductContext(
        product_id=product_id,
        store_id=store_id,
        category="Dairy",
        inventory={"quantity": 20.0, "original_price": 40.0, "current_price": 40.0, "price_floor": 28.0},
        demand={"sales_velocity": 2.0, "historical_sales": {"average_daily_sales": 5.0}},
        expiry={"expires_at": datetime.now(timezone.utc), "hours_remaining": 30.0},
        risk_assessment={"risk_level": "HIGH", "reason": "Expiry risk", "confidence": 0.9},
    )


def test_1_valid_single_event_ingestion():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt = _make_event("evt-001")
    resp = service.ingest([evt])

    assert isinstance(resp, HistoricalPricingIngestionResponse)
    assert resp.accepted_count == 1
    assert resp.upserted_count == 1
    assert resp.failed_count == 0
    assert resp.document_ids == ["doc-evt-001"]


def test_2_valid_multi_event_batch():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    events = [_make_event("evt-001"), _make_event("evt-002")]
    resp = service.ingest(events)

    assert resp.accepted_count == 2
    assert resp.upserted_count == 2
    assert resp.document_ids == ["doc-evt-001", "doc-evt-002"]


def test_3_empty_events_list_rejected():
    with pytest.raises(ValidationError, match="at least one event"):
        HistoricalPricingIngestionRequest(events=[])


def test_4_oversized_batch_rejected():
    max_batch = settings.HISTORICAL_INGESTION_MAX_BATCH_SIZE
    events = [_make_event(f"evt-{i}") for i in range(max_batch + 1)]

    with pytest.raises(ValidationError, match="exceeds configured maximum limit"):
        HistoricalPricingIngestionRequest(events=events)


def test_5_invalid_historical_event_rejected():
    with pytest.raises(ValidationError):
        _make_event(discount=25.0)  # > 15.0 max allowed discount


def test_6_builder_called_for_every_valid_event():
    mock_builder = MagicMock(side_effect=build_pricing_knowledge_document)
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(
        embedding_provider=emb, vector_store=store, knowledge_builder=mock_builder
    )

    events = [_make_event("evt-001"), _make_event("evt-002")]
    service.ingest(events)

    assert mock_builder.call_count == 2


def test_7_batch_embedding_called_once_for_n_events():
    mock_emb = MagicMock(spec=FakeEmbeddingProvider)
    mock_emb.embed_documents.return_value = [[0.1] * 1536, [0.2] * 1536]
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=mock_emb, vector_store=store)

    events = [_make_event("evt-001"), _make_event("evt-002")]
    service.ingest(events)

    assert mock_emb.embed_documents.call_count == 1
    texts_arg = mock_emb.embed_documents.call_args[0][0]
    assert len(texts_arg) == 2


def test_8_number_of_embeddings_matches_documents():
    mock_emb = MagicMock(spec=FakeEmbeddingProvider)
    mock_emb.embed_documents.return_value = [[0.1] * 1536]  # 1 vector for 2 docs
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=mock_emb, vector_store=store)

    events = [_make_event("evt-001"), _make_event("evt-002")]
    with pytest.raises(EmbeddingProviderError, match="mismatch"):
        service.ingest(events)


def test_9_vector_store_upsert_receives_n_docs_and_n_embeddings():
    mock_store = MagicMock(spec=InMemoryVectorStore)
    emb = FakeEmbeddingProvider()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=mock_store)

    events = [_make_event("evt-001"), _make_event("evt-002")]
    service.ingest(events)

    assert mock_store.upsert.call_count == 1
    docs_arg, vecs_arg = mock_store.upsert.call_args[0]
    assert len(docs_arg) == 2
    assert len(vecs_arg) == 2


def test_10_event_id_becomes_deterministic_document_id():
    evt = _make_event("evt-999")
    doc = build_pricing_knowledge_document(evt)
    assert doc.document_id == "doc-evt-999"


def test_11_reingesting_same_event_does_not_duplicate():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt = _make_event("evt-001")
    service.ingest([evt])
    assert len(store._storage) == 1

    # Re-ingest same event
    service.ingest([evt])
    assert len(store._storage) == 1


def test_12_reingesting_corrected_event_updates_existing_knowledge():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt_v1 = _make_event("evt-001", discount=5.0)
    service.ingest([evt_v1])
    doc_v1, _ = store._storage["doc-evt-001"]
    assert "5.0%" in doc_v1.content

    # Re-ingest corrected event with 10.0% discount
    evt_v2 = _make_event("evt-001", discount=10.0)
    service.ingest([evt_v2])
    doc_v2, _ = store._storage["doc-evt-001"]
    assert len(store._storage) == 1
    assert "10.0%" in doc_v2.content


def test_13_store_isolation_preserved():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt = _make_event("evt-001", store_id="store-ABC", product_id="prod-XYZ")
    service.ingest([evt])

    doc, _ = store._storage["doc-evt-001"]
    assert doc.store_id == "store-ABC"
    assert doc.metadata["store_id"] == "store-ABC"


def test_14_product_isolation_preserved():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt = _make_event("evt-001", store_id="store-ABC", product_id="prod-XYZ")
    service.ingest([evt])

    doc, _ = store._storage["doc-evt-001"]
    assert doc.product_id == "prod-XYZ"
    assert doc.metadata["product_id"] == "prod-XYZ"


def test_15_embedding_provider_error_propagates():
    mock_emb = MagicMock(spec=FakeEmbeddingProvider)
    mock_emb.embed_documents.side_effect = EmbeddingProviderError("OpenAI connection error")
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=mock_emb, vector_store=store)

    with pytest.raises(EmbeddingProviderError, match="OpenAI connection error"):
        service.ingest([_make_event()])


def test_16_vector_store_error_propagates():
    emb = FakeEmbeddingProvider()
    mock_store = MagicMock(spec=InMemoryVectorStore)
    mock_store.upsert.side_effect = VectorStoreError("Qdrant write failed")
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=mock_store)

    with pytest.raises(VectorStoreError, match="Qdrant write failed"):
        service.ingest([_make_event()])


def test_17_no_fake_fallback_knowledge_on_failure():
    mock_emb = MagicMock(spec=FakeEmbeddingProvider)
    mock_emb.embed_documents.side_effect = RuntimeError("Fatal API Failure")
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=mock_emb, vector_store=store)

    with pytest.raises(EmbeddingProviderError):
        service.ingest([_make_event()])

    assert len(store._storage) == 0


def test_18_no_fake_fallback_embedding_on_failure():
    emb = FakeEmbeddingProvider()
    mock_store = MagicMock(spec=InMemoryVectorStore)
    mock_store.upsert.side_effect = VectorStoreError("Storage offline")
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=mock_store)

    with pytest.raises(VectorStoreError):
        service.ingest([_make_event()])


def test_19_in_memory_vector_store_end_to_end_ingestion():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt = _make_event("evt-001")
    resp = service.ingest([evt])

    assert resp.accepted_count == 1
    assert "doc-evt-001" in store._storage


def test_20_ingested_knowledge_retrievable_via_retriever():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt = _make_event("evt-001", store_id="store-99", product_id="prod-88")
    service.ingest([evt])

    retriever = VectorPricingKnowledgeRetriever(embedding_provider=emb, vector_store=store)
    p_ctx = _make_product_ctx(store_id="store-99", product_id="prod-88")
    items = retriever.retrieve(store_id="store-99", products=[p_ctx])

    assert len(items) == 1
    assert items[0].store_id == "store-99"
    assert items[0].product_id == "prod-88"
    assert "Historical pricing event" in items[0].content


def test_21_correlation_id_propagation_via_api():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    mock_service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt_dict = _make_event("evt-001").model_dump(mode="json")
    payload = {"events": [evt_dict]}

    with patch("app.api.routes.pricing.get_ingestion_service", return_value=mock_service):
        response = client.post(
            "/api/v1/pricing/knowledge/ingest",
            json=payload,
            headers={"X-Request-ID": "test-correlation-1234"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["accepted_count"] == 1
    assert data["upserted_count"] == 1
    assert data["document_ids"] == ["doc-evt-001"]
    assert response.headers.get("X-Request-ID") == "test-correlation-1234"


def test_22_end_to_end_store_isolation_no_cross_leakage():
    emb = FakeEmbeddingProvider()
    store = InMemoryVectorStore()
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt_store_A = _make_event("evt-A", store_id="store-A", product_id="prod-X")
    evt_store_B = _make_event("evt-B", store_id="store-B", product_id="prod-X")
    service.ingest([evt_store_A, evt_store_B])

    retriever = VectorPricingKnowledgeRetriever(embedding_provider=emb, vector_store=store)

    p_A = _make_product_ctx(store_id="store-A", product_id="prod-X")
    items_A = retriever.retrieve(store_id="store-A", products=[p_A])
    assert len(items_A) == 1
    assert items_A[0].store_id == "store-A"

    p_B = _make_product_ctx(store_id="store-B", product_id="prod-X")
    items_B = retriever.retrieve(store_id="store-B", products=[p_B])
    assert len(items_B) == 1
    assert items_B[0].store_id == "store-B"


@pytest.mark.skipif(
    os.getenv("RUN_EXTERNAL_INTEGRATION_TESTS") != "true"
    or os.getenv("VECTOR_STORE_PROVIDER") != "qdrant",
    reason="Opt-in live Qdrant ingestion test requires RUN_EXTERNAL_INTEGRATION_TESTS=true and VECTOR_STORE_PROVIDER=qdrant",
)
def test_live_qdrant_historical_ingestion():
    """Opt-in live integration test against real Qdrant vector store."""
    from app.vector_store.qdrant import QdrantVectorStore
    emb = FakeEmbeddingProvider()
    store = QdrantVectorStore(collection_name="test_foodloop_ingestion")
    service = HistoricalPricingIngestionService(embedding_provider=emb, vector_store=store)

    evt = _make_event("live-evt-001", store_id="live-store-01", product_id="live-prod-100")
    resp = service.ingest([evt])

    assert resp.accepted_count == 1
    assert resp.upserted_count == 1
    assert resp.document_ids == ["doc-live-evt-001"]

    # Verify retrieval against live Qdrant collection
    retriever = VectorPricingKnowledgeRetriever(embedding_provider=emb, vector_store=store)
    p_live = _make_product_ctx(store_id="live-store-01", product_id="live-prod-100")
    items = retriever.retrieve(store_id="live-store-01", products=[p_live])
    assert len(items) == 1
    assert items[0].store_id == "live-store-01"

    # Cleanup
    store.delete(["doc-live-evt-001"])
