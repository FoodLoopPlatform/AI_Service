import inspect
import sys
from unittest.mock import MagicMock, patch
import pytest

from app.config.settings import settings
from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store import (
    InMemoryVectorStore,
    QdrantVectorStore,
    VectorStoreError,
    get_qdrant_client,
    get_vector_store,
    set_vector_store,
)


def make_sample_doc(
    doc_id: str = "doc-q-001",
    store_id: str = "store-cairo-01",
    product_id: str = "p-100",
    category: str = "Dairy",
    content: str = "Historical milk pricing facts",
) -> PricingKnowledgeDocument:
    return PricingKnowledgeDocument(
        document_id=doc_id,
        store_id=store_id,
        product_id=product_id,
        content=content,
        metadata={
            "store_id": store_id,
            "product_id": product_id,
            "category": category,
        },
    )


def test_qdrant_client_construction():
    """Requirement 13.1: Client construction succeeds without network calls."""
    client = get_qdrant_client(url="http://localhost:6333", api_key="")
    assert client is not None


def test_qdrant_configuration_loading():
    """Requirement 13.2: QdrantVectorStore loads configuration from Settings."""
    store = QdrantVectorStore()
    assert store.collection_name == settings.QDRANT_COLLECTION_NAME
    assert store.vector_size == settings.QDRANT_VECTOR_SIZE


def test_collection_creation_when_not_exists():
    """Requirement 13.3: _ensure_collection creates collection if missing."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False

    store = QdrantVectorStore(client=mock_client, vector_size=1536)
    store._ensure_collection()

    mock_client.collection_exists.assert_called_once_with(store.collection_name)
    mock_client.create_collection.assert_called_once()


def test_existing_collection_handling():
    """Requirement 13.4: _ensure_collection skips creation if collection exists."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    store = QdrantVectorStore(client=mock_client)
    store._ensure_collection()

    mock_client.collection_exists.assert_called_once_with(store.collection_name)
    mock_client.create_collection.assert_not_called()


def test_upsert_success():
    """Requirement 13.5: Upsert formats points and calls Qdrant client."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    doc = make_sample_doc(doc_id="doc-1")
    vec = [0.1, 0.2, 0.3, 0.4]

    store.upsert([doc], [vec])

    mock_client.upsert.assert_called_once()
    kwargs = mock_client.upsert.call_args.kwargs
    assert kwargs["collection_name"] == store.collection_name
    assert len(kwargs["points"]) == 1
    assert kwargs["points"][0].payload["document_id"] == "doc-1"


def test_idempotent_upsert_uses_stable_point_id():
    """Requirement 13.6: Upsert uses deterministic point IDs derived from document_id."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    doc1 = make_sample_doc(doc_id="doc-dup")
    doc2 = make_sample_doc(doc_id="doc-dup", content="Updated text")

    store.upsert([doc1], [[0.1, 0.2, 0.3, 0.4]])
    point_id_1 = mock_client.upsert.call_args.kwargs["points"][0].id

    store.upsert([doc2], [[0.5, 0.6, 0.7, 0.8]])
    point_id_2 = mock_client.upsert.call_args.kwargs["points"][0].id

    assert point_id_1 == point_id_2


def test_empty_upsert_safely_does_nothing():
    """Requirement 13.7: Empty upsert does not trigger network or client calls."""
    mock_client = MagicMock()
    store = QdrantVectorStore(client=mock_client)
    store.upsert([], [])
    mock_client.upsert.assert_not_called()


def test_invalid_vector_dimensions_rejected():
    """Requirement 13.8: Vector dimension mismatch raises VectorStoreError."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    store = QdrantVectorStore(client=mock_client, vector_size=4)

    doc = make_sample_doc()
    with pytest.raises(VectorStoreError) as exc_info:
        store.upsert([doc], [[0.1, 0.2]])  # expected 4, got 2
    assert "does not match Qdrant configured size" in str(exc_info.value)


def test_mismatched_document_vector_count_rejected():
    """Requirement 13.9: Mismatched counts raise VectorStoreError."""
    store = QdrantVectorStore()
    doc1 = make_sample_doc(doc_id="d1")
    doc2 = make_sample_doc(doc_id="d2")
    with pytest.raises(VectorStoreError) as exc_info:
        store.upsert([doc1, doc2], [[0.1, 0.2, 0.3, 0.4]])
    assert "Document count (2) does not match embedding count (1)" in str(exc_info.value)


def test_search_and_result_mapping():
    """Requirement 13.10 & 13.15: Search maps Qdrant ScoredPoint into PricingKnowledgeItem."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    # Mock Qdrant ScoredPoint response
    hit = MagicMock()
    hit.score = 0.85
    hit.payload = {
        "document_id": "doc-1",
        "store_id": "store-cairo-01",
        "product_id": "p-100",
        "content": "Milk fact",
        "metadata": {"category": "Dairy"},
    }
    mock_client.search.return_value = [hit]

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    results = store.search([0.1, 0.2, 0.3, 0.4], store_id="store-cairo-01")

    assert len(results) == 1
    assert isinstance(results[0], PricingKnowledgeItem)
    assert results[0].product_id == "p-100"
    assert results[0].store_id == "store-cairo-01"
    assert results[0].content == "Milk fact"
    assert 0.0 <= results[0].relevance_score <= 1.0


def test_mandatory_store_id_filtering():
    """Requirement 13.11: Qdrant search filter includes mandatory store_id."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.search.return_value = []

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    store.search([0.1, 0.2, 0.3, 0.4], store_id="store-alex-02")

    mock_client.search.assert_called_once()
    qdrant_filter = mock_client.search.call_args.kwargs["query_filter"]
    assert qdrant_filter is not None
    conditions = qdrant_filter.must
    assert any(c.key == "store_id" and c.match.value == "store-alex-02" for c in conditions)


def test_product_id_filtering():
    """Requirement 13.12: Qdrant search filter includes optional product_id."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.search.return_value = []

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    store.search([0.1, 0.2, 0.3, 0.4], store_id="store-01", product_id="prod-milk-99")

    qdrant_filter = mock_client.search.call_args.kwargs["query_filter"]
    conditions = qdrant_filter.must
    assert any(c.key == "product_id" and c.match.value == "prod-milk-99" for c in conditions)


def test_category_filtering():
    """Requirement 13.13: Qdrant search filter includes optional category."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.search.return_value = []

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    store.search([0.1, 0.2, 0.3, 0.4], store_id="store-01", category="Bakery")

    qdrant_filter = mock_client.search.call_args.kwargs["query_filter"]
    conditions = qdrant_filter.must
    assert any(c.key == "category" and c.match.value == "Bakery" for c in conditions)


def test_top_k_parameter_passed_to_search():
    """Requirement 13.14: top_k parameter passed to Qdrant client limit."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.search.return_value = []

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    store.search([0.1, 0.2, 0.3, 0.4], store_id="store-01", top_k=7)

    assert mock_client.search.call_args.kwargs["limit"] == 7


def test_delete_operation():
    """Requirement 13.16: Delete removes points by document ID."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    store = QdrantVectorStore(client=mock_client)
    store.delete(["doc-1", "doc-2"])

    mock_client.delete.assert_called_once()
    kwargs = mock_client.delete.call_args.kwargs
    assert kwargs["collection_name"] == store.collection_name
    assert len(kwargs["points_selector"].points) == 2


def test_deleting_unknown_ids_is_safe():
    """Requirement 13.17: Deleting unknown IDs completes safely."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    store = QdrantVectorStore(client=mock_client)
    store.delete([])
    mock_client.delete.assert_not_called()


def test_qdrant_exception_maps_to_vector_store_error():
    """Requirement 13.18: Client exceptions mapped to VectorStoreError."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_client.search.side_effect = RuntimeError("Qdrant gRPC connection refused")

    store = QdrantVectorStore(client=mock_client, vector_size=4)
    with pytest.raises(VectorStoreError) as exc_info:
        store.search([0.1, 0.2, 0.3, 0.4], store_id="store-01")
    assert "Qdrant search operation failed" in str(exc_info.value)
    assert "connection refused" in str(exc_info.value)


def test_factory_provider_selection():
    """Requirement 13.20: Factory selects InMemoryVectorStore for 'memory' and QdrantVectorStore for 'qdrant'."""
    set_vector_store(None)
    mem_store = get_vector_store(provider="memory")
    assert isinstance(mem_store, InMemoryVectorStore)

    set_vector_store(None)
    qdrant_store = get_vector_store(provider="qdrant")
    assert isinstance(qdrant_store, QdrantVectorStore)

    set_vector_store(None)  # reset for subsequent test isolation


def test_architectural_qdrant_boundary():
    """Requirement 12: Ensure no Qdrant imports leak into agents, schemas, or embeddings."""
    import pkgutil
    import app.agents
    import app.schemas
    import app.embeddings

    def check_dir(module_pkg):
        for _, name, is_pkg in pkgutil.walk_packages(module_pkg.__path__, module_pkg.__name__ + "."):
            if name in sys.modules:
                mod = sys.modules[name]
                src = inspect.getsource(mod).lower()
                assert "qdrant" not in src, f"Forbidden 'qdrant' import found in module {name}"

    check_dir(app.agents)
    check_dir(app.schemas)
    check_dir(app.embeddings)
