import inspect
import sys
from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from app.agents.pricing.prompts import format_pricing_user_prompt
from app.agents.pricing.retriever import (
    DefaultPricingKnowledgeRetriever,
    InMemoryPricingKnowledgeRetriever,
    PricingKnowledgeRetriever,
    VectorPricingKnowledgeRetriever,
    build_product_query_text,
    get_production_pricing_knowledge_retriever,
    group_knowledge_by_product,
)
from app.config.settings import settings
from app.embeddings.base import EmbeddingProvider, EmbeddingProviderError
from app.embeddings.fake import FakeEmbeddingProvider
from app.schemas.pricing import PricingProductContext
from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.vector_store.base import VectorStore, VectorStoreError
from app.vector_store.in_memory import InMemoryVectorStore
from tests.test_pricing_agent import create_sample_product_context


def test_pricing_knowledge_item_valid():
    """Test valid PricingKnowledgeItem creation."""
    item = PricingKnowledgeItem(
        product_id="p-100",
        store_id="store-cairo-01",
        content="Milk nearing expiry should be discounted 10-15%.",
        metadata={"category": "Dairy"},
        relevance_score=0.95,
    )
    assert item.product_id == "p-100"
    assert item.store_id == "store-cairo-01"
    assert item.relevance_score == 0.95


def test_pricing_knowledge_item_invalid_relevance_score():
    """Requirement Test: Invalid relevance score (< 0 or > 1) rejected."""
    with pytest.raises(ValidationError) as exc_info_neg:
        PricingKnowledgeItem(
            product_id="p-100",
            store_id="store-cairo-01",
            content="Test content",
            relevance_score=-0.1,
        )
    assert "relevance_score" in str(exc_info_neg.value)

    with pytest.raises(ValidationError) as exc_info_high:
        PricingKnowledgeItem(
            product_id="p-100",
            store_id="store-cairo-01",
            content="Test content",
            relevance_score=1.1,
        )
    assert "relevance_score" in str(exc_info_high.value)


def test_single_product_retrieval():
    """Requirement 12.1: Single product retrieval via VectorPricingKnowledgeRetriever."""
    embedder = FakeEmbeddingProvider(dimension=4)
    store = InMemoryVectorStore()
    retriever = VectorPricingKnowledgeRetriever(embedder, store)

    p100 = create_sample_product_context("p-100")
    results = retriever.retrieve(store_id="store-01", products=[p100])
    assert results == []


def test_multiple_product_retrieval():
    """Requirement 12.2: Multiple product retrieval processes all products."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.embed_queries.return_value = [[0.1] * 4, [0.2] * 4]

    item1 = PricingKnowledgeItem(
        product_id="p-100", store_id="store-01", content="Fact 1", relevance_score=0.9
    )
    item2 = PricingKnowledgeItem(
        product_id="p-101", store_id="store-01", content="Fact 2", relevance_score=0.8
    )

    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.side_effect = [[item1], [item2]]

    retriever = VectorPricingKnowledgeRetriever(mock_embedder, mock_store)
    products = [create_sample_product_context("p-100"), create_sample_product_context("p-101")]

    results = retriever.retrieve(store_id="store-01", products=products)
    assert len(results) == 2
    assert results[0].product_id == "p-100"
    assert results[1].product_id == "p-101"


def test_empty_product_list_retrieval():
    """Requirement 12.3: Empty product list returns []."""
    embedder = FakeEmbeddingProvider(dimension=4)
    store = InMemoryVectorStore()
    retriever = VectorPricingKnowledgeRetriever(embedder, store)

    assert retriever.retrieve(store_id="store-01", products=[]) == []


def test_correct_store_id_product_id_category_propagation():
    """Requirement 12.4, 12.5, 12.6: store_id, product_id, category passed to search."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.embed_queries.return_value = [[0.1, 0.2, 0.3, 0.4]]

    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.return_value = []

    retriever = VectorPricingKnowledgeRetriever(mock_embedder, mock_store)
    p100 = create_sample_product_context("p-100")
    p100.category = "Dairy"

    retriever.retrieve(store_id="store-cairo-01", products=[p100])

    mock_store.search.assert_called_once()
    kwargs = mock_store.search.call_args.kwargs
    assert kwargs["store_id"] == "store-cairo-01"
    assert kwargs["product_id"] == "p-100"
    assert kwargs["category"] == "Dairy"


def test_product_and_store_isolation():
    """Requirement 12.7 & 12.8: Strict product and store isolation enforced."""
    embedder = FakeEmbeddingProvider(dimension=4)
    store = InMemoryVectorStore()
    retriever = VectorPricingKnowledgeRetriever(embedder, store)

    p100_s1 = create_sample_product_context("p-100")
    results = retriever.retrieve(store_id="store-01", products=[p100_s1])
    assert results == []


def test_no_historical_knowledge_case():
    """Requirement 12.9 & 12.14: No historical knowledge produces [] without fabrication."""
    embedder = FakeEmbeddingProvider(dimension=4)
    store = InMemoryVectorStore()
    retriever = VectorPricingKnowledgeRetriever(embedder, store)

    p100 = create_sample_product_context("p-100")
    results = retriever.retrieve(store_id="store-01", products=[p100])

    grouped = group_knowledge_by_product([p100], results)
    assert grouped == {"p-100": []}


def test_multiple_results_per_product():
    """Requirement 12.10: Vector store returning multiple items per product."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.embed_queries.return_value = [[0.1] * 4]

    items = [
        PricingKnowledgeItem(product_id="p-100", store_id="s1", content="Fact A", relevance_score=0.95),
        PricingKnowledgeItem(product_id="p-100", store_id="s1", content="Fact B", relevance_score=0.85),
    ]
    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.return_value = items

    retriever = VectorPricingKnowledgeRetriever(mock_embedder, mock_store)
    p100 = create_sample_product_context("p-100")
    results = retriever.retrieve(store_id="s1", products=[p100])

    assert len(results) == 2
    assert results[0].content == "Fact A"
    assert results[1].content == "Fact B"


def test_top_k_configuration():
    """Requirement 12.11: top_k defaults to settings.PRICING_RETRIEVAL_TOP_K or explicit value."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.embed_queries.return_value = [[0.1] * 4]
    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.return_value = []

    retriever_default = VectorPricingKnowledgeRetriever(mock_embedder, mock_store)
    assert retriever_default.top_k == settings.PRICING_RETRIEVAL_TOP_K

    retriever_custom = VectorPricingKnowledgeRetriever(mock_embedder, mock_store, top_k=10)
    assert retriever_custom.top_k == 10

    with pytest.raises(ValueError) as exc_info:
        VectorPricingKnowledgeRetriever(mock_embedder, mock_store, top_k=0)
    assert "top_k must be at least 1" in str(exc_info.value)


def test_embedding_failure_propagation():
    """Requirement 12.12 & Requirement 15: EmbeddingProviderError propagates cleanly."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.embed_queries.side_effect = EmbeddingProviderError("OpenAI API rate limit exceeded")

    mock_store = MagicMock(spec=VectorStore)
    retriever = VectorPricingKnowledgeRetriever(mock_embedder, mock_store)

    p100 = create_sample_product_context("p-100")
    with pytest.raises(EmbeddingProviderError) as exc_info:
        retriever.retrieve(store_id="store-01", products=[p100])
    assert "rate limit exceeded" in str(exc_info.value)


def test_vector_store_failure_propagation():
    """Requirement 12.13 & Requirement 15: VectorStoreError propagates cleanly."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.embed_queries.return_value = [[0.1] * 4]

    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.side_effect = VectorStoreError("Qdrant cluster unavailable")

    retriever = VectorPricingKnowledgeRetriever(mock_embedder, mock_store)
    p100 = create_sample_product_context("p-100")
    with pytest.raises(VectorStoreError) as exc_info:
        retriever.retrieve(store_id="store-01", products=[p100])
    assert "cluster unavailable" in str(exc_info.value)


def test_batch_embedding_single_api_call_per_batch():
    """Requirement 12.15 & 11: N products generate 1 single batch embed_queries call."""
    mock_embedder = MagicMock(spec=EmbeddingProvider)
    mock_embedder.embed_queries.return_value = [[0.1] * 4, [0.2] * 4, [0.3] * 4]

    mock_store = MagicMock(spec=VectorStore)
    mock_store.search.return_value = []

    retriever = VectorPricingKnowledgeRetriever(mock_embedder, mock_store)
    products = [
        create_sample_product_context("p-100"),
        create_sample_product_context("p-101"),
        create_sample_product_context("p-102"),
    ]

    retriever.retrieve(store_id="store-01", products=products)

    assert mock_embedder.embed_queries.call_count == 1
    query_texts = mock_embedder.embed_queries.call_args[0][0]
    assert len(query_texts) == 3


def test_query_text_contains_factual_operational_context():
    """Requirement 12.16: Query text contains factual current operational state."""
    product = create_sample_product_context("p-100")
    query_text = build_product_query_text(product)

    assert "Product ID: p-100" in query_text
    assert "Current inventory quantity:" in query_text
    assert "Current price:" in query_text
    assert "Sales velocity:" in query_text
    assert "Hours remaining:" in query_text
    assert "Risk level:" in query_text


def test_query_text_contains_no_llm_instructions_or_recommendation_requests():
    """Requirement 12.17: Query text contains NO instructions, discount requests, or prompt words."""
    product = create_sample_product_context("p-100")
    query_text = build_product_query_text(product)

    prohibited = [
        "recommend",
        "discount",
        "system prompt",
        "you are an AI",
        "calculate",
        "instruction",
    ]
    query_text_lower = query_text.lower()
    for word in prohibited:
        assert word not in query_text_lower, f"Forbidden word '{word}' found in retrieval query text"


def test_retriever_dependency_injection():
    """Requirement 12.19: VectorPricingKnowledgeRetriever receives dependencies via constructor."""
    embedder = FakeEmbeddingProvider(dimension=4)
    store = InMemoryVectorStore()
    retriever = VectorPricingKnowledgeRetriever(embedding_provider=embedder, vector_store=store)

    assert retriever.embedding_provider is embedder
    assert retriever.vector_store is store


def test_no_direct_qdrant_or_openai_imports_in_retriever():
    """Requirement 12.20 & 12.21: Retriever module has no direct Qdrant or OpenAI SDK imports."""
    import app.agents.pricing.retriever as ret_mod

    source = inspect.getsource(ret_mod).lower()
    assert "qdrant" not in source
    assert "openai" not in source


def test_group_knowledge_by_product():
    """Requirement Test: group_knowledge_by_product creates dict with entry for every product_id."""
    products = [
        create_sample_product_context("p-100"),
        create_sample_product_context("p-101"),
        create_sample_product_context("p-102"),
    ]

    knowledge = [
        PricingKnowledgeItem(
            product_id="p-100",
            store_id="store-01",
            content="Guideline P100",
            relevance_score=0.95,
        )
    ]

    grouped = group_knowledge_by_product(products, knowledge)

    assert set(grouped.keys()) == {"p-100", "p-101", "p-102"}
    assert len(grouped["p-100"]) == 1
    assert grouped["p-100"][0].content == "Guideline P100"
    assert grouped["p-101"] == []
    assert grouped["p-102"] == []


def test_prompt_isolation():
    """Requirement Test: Prompt formatter provides each product with ONLY its own retrieved knowledge."""
    products = [
        create_sample_product_context("p-100"),
        create_sample_product_context("p-101"),
        create_sample_product_context("p-102"),
    ]

    k_100 = PricingKnowledgeItem(
        product_id="p-100",
        store_id="store-01",
        content="Unique rule for Milk p-100",
        relevance_score=0.92,
    )
    k_101 = PricingKnowledgeItem(
        product_id="p-101",
        store_id="store-01",
        content="Unique rule for Yogurt p-101",
        relevance_score=0.88,
    )

    grouped_knowledge = group_knowledge_by_product(products, [k_100, k_101])

    prompt = format_pricing_user_prompt(
        store_id="store-01",
        products=products,
        knowledge_by_product=grouped_knowledge,
    )

    sections = prompt.split("PRODUCT ")

    sec_p100 = sections[1]
    sec_p101 = sections[2]
    sec_p102 = sections[3]

    # p-100 section assertions
    assert "Product ID: p-100" in sec_p100
    assert "Unique rule for Milk p-100" in sec_p100
    assert "Unique rule for Yogurt p-101" not in sec_p100

    # p-101 section assertions
    assert "Product ID: p-101" in sec_p101
    assert "Unique rule for Yogurt p-101" in sec_p101
    assert "Unique rule for Milk p-100" not in sec_p101

    # p-102 section assertions (no knowledge)
    assert "Product ID: p-102" in sec_p102
    assert "Historical Knowledge:\n  None provided" in sec_p102
    assert "Unique rule for Milk p-100" not in sec_p102
    assert "Unique rule for Yogurt p-101" not in sec_p102
