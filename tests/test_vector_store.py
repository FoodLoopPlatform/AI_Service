import inspect
import sys
import pytest

from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store import (
    InMemoryVectorStore,
    VectorStore,
    VectorStoreError,
    get_vector_store,
)


def make_sample_document(
    document_id: str = "doc-001",
    store_id: str = "store-cairo-01",
    product_id: str = "p-100",
    category: str = "Dairy",
    content: str = "Historical pricing fact for Milk",
) -> PricingKnowledgeDocument:
    """Helper to create a PricingKnowledgeDocument for testing."""
    return PricingKnowledgeDocument(
        document_id=document_id,
        store_id=store_id,
        product_id=product_id,
        content=content,
        metadata={
            "store_id": store_id,
            "product_id": product_id,
            "category": category,
        },
    )


def test_empty_store_search_returns_empty_list():
    """Requirement 9.1: Search on an empty store returns []."""
    store = InMemoryVectorStore()
    results = store.search([1.0, 0.0], store_id="store-cairo-01")
    assert results == []


def test_upsert_one_document_and_search_match():
    """Requirement 9.2 & 9.3: Upserting one document and searching returns matching document."""
    store = InMemoryVectorStore()
    doc = make_sample_document(document_id="doc-1", store_id="store-01", product_id="p-100")
    store.upsert([doc], [[1.0, 0.0]])

    results = store.search([1.0, 0.0], store_id="store-01")
    assert len(results) == 1
    assert results[0].product_id == "p-100"
    assert results[0].store_id == "store-01"
    assert results[0].content == doc.content


def test_search_ranks_more_similar_documents_first():
    """Requirement 9.4: Search ranks more similar documents first."""
    store = InMemoryVectorStore()
    doc_high = make_sample_document(document_id="doc-high", content="High similarity doc")
    doc_low = make_sample_document(document_id="doc-low", content="Low similarity doc")

    # Query vector is [1.0, 0.0]
    # doc_high vector is [0.95, 0.05] (closer)
    # doc_low vector is [0.1, 0.9] (further)
    store.upsert([doc_high, doc_low], [[0.95, 0.05], [0.1, 0.9]])

    results = store.search([1.0, 0.0], store_id="store-cairo-01")
    assert len(results) == 2
    assert results[0].content == "High similarity doc"
    assert results[1].content == "Low similarity doc"
    assert results[0].relevance_score > results[1].relevance_score


def test_top_k_limits_results():
    """Requirement 9.5: top_k limits returned result list size."""
    store = InMemoryVectorStore()
    docs = [make_sample_document(document_id=f"doc-{i}") for i in range(10)]
    vecs = [[1.0, float(i) * 0.1] for i in range(10)]
    store.upsert(docs, vecs)

    results = store.search([1.0, 0.0], store_id="store-cairo-01", top_k=3)
    assert len(results) == 3


def test_store_id_filtering_works():
    """Requirement 9.6: Search filters strictly by store_id."""
    store = InMemoryVectorStore()
    doc_cairo = make_sample_document(document_id="doc-1", store_id="store-cairo")
    doc_alex = make_sample_document(document_id="doc-2", store_id="store-alex")
    store.upsert([doc_cairo, doc_alex], [[1.0, 0.0], [1.0, 0.0]])

    results_cairo = store.search([1.0, 0.0], store_id="store-cairo")
    assert len(results_cairo) == 1
    assert results_cairo[0].store_id == "store-cairo"


def test_product_id_filtering_works():
    """Requirement 9.7: Search filters by product_id when provided."""
    store = InMemoryVectorStore()
    doc_milk = make_sample_document(document_id="doc-1", product_id="prod-milk")
    doc_bread = make_sample_document(document_id="doc-2", product_id="prod-bread")
    store.upsert([doc_milk, doc_bread], [[1.0, 0.0], [1.0, 0.0]])

    results = store.search([1.0, 0.0], store_id="store-cairo-01", product_id="prod-milk")
    assert len(results) == 1
    assert results[0].product_id == "prod-milk"


def test_category_filtering_works():
    """Requirement 9.8: Search filters by category when provided."""
    store = InMemoryVectorStore()
    doc_dairy = make_sample_document(document_id="doc-1", category="Dairy")
    doc_bakery = make_sample_document(document_id="doc-2", category="Bakery")
    store.upsert([doc_dairy, doc_bakery], [[1.0, 0.0], [1.0, 0.0]])

    results = store.search([1.0, 0.0], store_id="store-cairo-01", category="Dairy")
    assert len(results) == 1
    assert results[0].metadata["category"] == "Dairy"


def test_cross_store_knowledge_never_appears():
    """Requirement 9.9: Knowledge from another store NEVER appears in search results."""
    store = InMemoryVectorStore()
    doc_giza = make_sample_document(document_id="doc-giza", store_id="store-giza", product_id="p-100")
    store.upsert([doc_giza], [[1.0, 0.0]])

    results = store.search([1.0, 0.0], store_id="store-cairo-01")
    assert results == []


def test_same_product_in_different_stores_remains_isolated():
    """Requirement 9.10: Same product ID in different stores remains isolated."""
    store = InMemoryVectorStore()
    doc_s1 = make_sample_document(document_id="doc-s1", store_id="store-01", product_id="p-100", content="Store 1 Milk")
    doc_s2 = make_sample_document(document_id="doc-s2", store_id="store-02", product_id="p-100", content="Store 2 Milk")
    store.upsert([doc_s1, doc_s2], [[1.0, 0.0], [1.0, 0.0]])

    res_s1 = store.search([1.0, 0.0], store_id="store-01", product_id="p-100")
    assert len(res_s1) == 1
    assert res_s1[0].content == "Store 1 Milk"


def test_same_store_with_different_products_remains_isolated():
    """Requirement 9.11: Same store with different product IDs remains isolated when product_id is filtered."""
    store = InMemoryVectorStore()
    doc_p1 = make_sample_document(document_id="doc-p1", store_id="store-01", product_id="p-100")
    doc_p2 = make_sample_document(document_id="doc-p2", store_id="store-01", product_id="p-101")
    store.upsert([doc_p1, doc_p2], [[1.0, 0.0], [1.0, 0.0]])

    res_p1 = store.search([1.0, 0.0], store_id="store-01", product_id="p-100")
    assert len(res_p1) == 1
    assert res_p1[0].product_id == "p-100"


def test_upsert_is_idempotent_by_document_id():
    """Requirement 9.12 & 9.13: Upserting the same document_id replaces existing record without duplicates."""
    store = InMemoryVectorStore()
    doc_orig = make_sample_document(document_id="doc-dup", content="Original content")
    doc_updated = make_sample_document(document_id="doc-dup", content="Updated content")

    store.upsert([doc_orig], [[1.0, 0.0]])
    assert len(store.search([1.0, 0.0], store_id="store-cairo-01")) == 1

    store.upsert([doc_updated], [[0.9, 0.1]])
    results = store.search([1.0, 0.0], store_id="store-cairo-01")

    assert len(results) == 1
    assert results[0].content == "Updated content"


def test_delete_removes_documents():
    """Requirement 9.14: Delete removes matching document from store."""
    store = InMemoryVectorStore()
    doc = make_sample_document(document_id="doc-del")
    store.upsert([doc], [[1.0, 0.0]])

    assert len(store.search([1.0, 0.0], store_id="store-cairo-01")) == 1

    store.delete(["doc-del"])
    assert store.search([1.0, 0.0], store_id="store-cairo-01") == []


def test_delete_unknown_ids_is_safe():
    """Requirement 9.15: Deleting non-existent document IDs is safe and does not error."""
    store = InMemoryVectorStore()
    store.delete(["unknown-doc-id-123"])


def test_empty_query_vector_rejected():
    """Requirement 9.16: Empty query vector raises VectorStoreError."""
    store = InMemoryVectorStore()
    with pytest.raises(VectorStoreError) as exc_info:
        store.search([], store_id="store-01")
    assert "Query embedding vector cannot be empty" in str(exc_info.value)


def test_empty_document_embedding_rejected():
    """Requirement 9.17: Empty document vector in upsert raises VectorStoreError."""
    store = InMemoryVectorStore()
    doc = make_sample_document()
    with pytest.raises(VectorStoreError) as exc_info:
        store.upsert([doc], [[]])
    assert "Embedding vector at index 0 is empty" in str(exc_info.value)


def test_mismatched_document_and_embedding_counts_rejected():
    """Requirement 9.18: Mismatched counts raise VectorStoreError."""
    store = InMemoryVectorStore()
    doc1 = make_sample_document(document_id="d1")
    doc2 = make_sample_document(document_id="d2")
    with pytest.raises(VectorStoreError) as exc_info:
        store.upsert([doc1, doc2], [[1.0, 0.0]])
    assert "Document count (2) does not match embedding count (1)" in str(exc_info.value)


def test_inconsistent_embedding_dimensions_rejected():
    """Requirement 9.19: Inconsistent vector dimensions raise VectorStoreError."""
    store = InMemoryVectorStore()
    doc1 = make_sample_document(document_id="d1")
    doc2 = make_sample_document(document_id="d2")
    with pytest.raises(VectorStoreError) as exc_info:
        store.upsert([doc1, doc2], [[1.0, 0.0], [1.0, 0.0, 0.5]])
    assert "Inconsistent vector dimension" in str(exc_info.value)


def test_invalid_top_k_rejected():
    """Requirement 9.20: top_k < 1 raises VectorStoreError."""
    store = InMemoryVectorStore()
    with pytest.raises(VectorStoreError) as exc_info:
        store.search([1.0, 0.0], store_id="store-01", top_k=0)
    assert "top_k must be at least 1" in str(exc_info.value)


def test_deterministic_ranking():
    """Requirement 9.21: Ranking is deterministic across repeated executions."""
    store = InMemoryVectorStore()
    doc1 = make_sample_document(document_id="d1", content="Content 1")
    doc2 = make_sample_document(document_id="d2", content="Content 2")
    store.upsert([doc1, doc2], [[0.8, 0.2], [0.6, 0.4]])

    res1 = store.search([1.0, 0.0], store_id="store-cairo-01")
    res2 = store.search([1.0, 0.0], store_id="store-cairo-01")

    assert [r.content for r in res1] == [r.content for r in res2]


def test_relevance_score_boundary_constraints():
    """Requirement 9.22: Relevance score is always constrained between 0.0 and 1.0."""
    store = InMemoryVectorStore()
    doc_pos = make_sample_document(document_id="pos", content="Positive")
    doc_neg = make_sample_document(document_id="neg", content="Negative")
    store.upsert([doc_pos, doc_neg], [[1.0, 0.0], [-1.0, 0.0]])

    results = store.search([1.0, 0.0], store_id="store-cairo-01")
    for r in results:
        assert 0.0 <= r.relevance_score <= 1.0


def test_vector_store_error_raised_for_invalid_operations():
    """Requirement 9.23: VectorStoreError is raised for invalid infrastructure operations."""
    store = InMemoryVectorStore()
    with pytest.raises(VectorStoreError):
        store.search([1.0, 0.0], store_id="")


def test_factory_returns_vector_store():
    """Requirement 8: Factory returns InMemoryVectorStore instance."""
    store = get_vector_store()
    assert isinstance(store, VectorStore)
    assert isinstance(store, InMemoryVectorStore)


def test_architectural_vector_store_isolation():
    """Requirement 10: VectorStore base and in_memory modules do not depend on external frameworks."""
    import app.vector_store.base as base_mod
    import app.vector_store.in_memory as mem_mod

    source_base = inspect.getsource(base_mod)
    source_mem = inspect.getsource(mem_mod)
    combined = (source_base + "\n" + source_mem).lower()

    prohibited_keywords = [
        "qdrant",
        "openai",
        "httpx",
        "requests",
        "langgraph",
        "langchain",
        "numpy",
        "scipy",
    ]

    for kw in prohibited_keywords:
        assert kw not in combined, f"Prohibited dependency '{kw}' found in vector_store module"
