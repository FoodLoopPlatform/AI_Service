from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.agents.pricing import get_pricing_graph, run_pricing_agent
from app.agents.pricing.nodes import (
    get_pricing_knowledge_retriever,
    set_pricing_knowledge_retriever,
)
from app.agents.pricing.prompts import format_pricing_user_prompt
from app.agents.pricing.retriever import (
    DefaultPricingKnowledgeRetriever,
    PricingKnowledgeRetriever,
)
from app.schemas.monitoring import (
    DemandContext,
    ExpiryContext,
    HistoricalSales,
    InventoryMetrics,
    RiskLevel,
)
from app.schemas.pricing import (
    PricingBatchLLMResult,
    PricingBatchRequest,
    PricingBatchResponse,
    PricingDecision,
    PricingProductContext,
)
from app.schemas.risk_assessment import RiskAssessmentResult


def create_sample_product_context(product_id: str) -> PricingProductContext:
    """Helper to create a single product context item."""
    return PricingProductContext(
        product_id=product_id,
        product_name=f"Product {product_id}",
        category="Dairy",
        inventory=InventoryMetrics(
            quantity=10,
            original_price=40.0,
            current_price=40.0,
            price_floor=28.0,
        ),
        demand=DemandContext(
            sales_velocity=0.5,
            historical_sales=HistoricalSales(
                average_daily_sales=5.0,
                weekday_average=5.0,
                weekend_average=5.0,
            ),
        ),
        expiry=ExpiryContext(
            expires_at=datetime.fromisoformat("2026-08-16T12:00:00"),
            hours_remaining=18.0,
        ),
        risk_assessment=RiskAssessmentResult(
            risk_level=RiskLevel.HIGH,
            reason="High risk near expiry",
            confidence=0.9,
        ),
    )


def create_sample_batch_request(
    store_id: str = "store-cairo-01",
    product_ids: list[str] | None = None,
) -> PricingBatchRequest:
    """Helper to create a PricingBatchRequest."""
    p_ids = product_ids or ["p-100"]
    return PricingBatchRequest(
        store_id=store_id,
        products=[create_sample_product_context(pid) for pid in p_ids],
    )


@pytest.fixture(autouse=True)
def reset_retriever():
    """Ensure default retriever is restored after every test."""
    set_pricing_knowledge_retriever(DefaultPricingKnowledgeRetriever())
    yield
    set_pricing_knowledge_retriever(DefaultPricingKnowledgeRetriever())


def test_one_product_batch_execution():
    """Requirement Test: One product batch recommendation success."""
    req = create_sample_batch_request(product_ids=["p-100"])
    expected_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(
                product_id="p-100",
                discount_percentage=15.0,
                reason="High risk near expiry",
                confidence=0.92,
            )
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = expected_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert isinstance(resp, PricingBatchResponse)
    assert resp.store_id == "store-cairo-01"
    assert len(resp.decisions) == 1
    assert resp.decisions[0].product_id == "p-100"
    assert resp.decisions[0].discount_percentage == 15.0
    mock_base_llm.with_structured_output.assert_called_once_with(PricingBatchLLMResult)


def test_multiple_products_batch_execution():
    """Requirement Test: Multiple products batch recommendation success."""
    req = create_sample_batch_request(product_ids=["p-100", "p-101", "p-102"])
    expected_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p-100", discount_percentage=10.0, reason="Reason 1", confidence=0.9),
            PricingDecision(product_id="p-101", discount_percentage=5.0, reason="Reason 2", confidence=0.85),
            PricingDecision(product_id="p-102", discount_percentage=0.0, reason="Reason 3", confidence=0.95),
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = expected_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert len(resp.decisions) == 3
    assert [d.product_id for d in resp.decisions] == ["p-100", "p-101", "p-102"]
    assert [d.discount_percentage for d in resp.decisions] == [10.0, 5.0, 0.0]


def test_product_id_preservation():
    """Requirement Test: Product IDs are preserved exactly in batch output decisions."""
    req = create_sample_batch_request(product_ids=["prod-alpha", "prod-beta"])
    expected_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="prod-alpha", discount_percentage=12.0, reason="Alpha reason", confidence=0.9),
            PricingDecision(product_id="prod-beta", discount_percentage=8.0, reason="Beta reason", confidence=0.88),
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = expected_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert resp.decisions[0].product_id == "prod-alpha"
    assert resp.decisions[1].product_id == "prod-beta"


def test_duplicate_product_id_rejected():
    """Requirement Test: Duplicate product_id in LLM decision list triggers fallback."""
    req = create_sample_batch_request(product_ids=["p-100", "p-101"])
    # LLM returns decision for p-100 twice instead of p-101
    invalid_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p-100", discount_percentage=10.0, reason="Reason 1", confidence=0.9),
            PricingDecision(product_id="p-100", discount_percentage=5.0, reason="Reason 2", confidence=0.85),
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = invalid_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert "rule-based fallback" in resp.decisions[0].reason


def test_unknown_product_id_rejected():
    """Requirement Test: Unknown product_id in LLM decision list triggers fallback."""
    req = create_sample_batch_request(product_ids=["p-100"])
    invalid_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p-UNKNOWN", discount_percentage=10.0, reason="Unknown pid", confidence=0.9)
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = invalid_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert "rule-based fallback" in resp.decisions[0].reason


def test_missing_product_decision_rejected():
    """Requirement Test: Missing decision for an input product triggers fallback."""
    req = create_sample_batch_request(product_ids=["p-100", "p-101"])
    # LLM only returns decision for p-100, omitting p-101
    invalid_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p-100", discount_percentage=10.0, reason="Reason 1", confidence=0.9)
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = invalid_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert "rule-based fallback" in resp.decisions[0].reason


from app.schemas.pricing_knowledge import PricingKnowledgeItem


def test_store_id_and_products_passed_to_retriever():
    """Requirement Test: store_id and all products are passed to the retriever."""
    recorded_calls = []

    class MockStoreRetriever(PricingKnowledgeRetriever):
        def retrieve(
            self,
            store_id: str,
            products: list[PricingProductContext],
        ) -> list[PricingKnowledgeItem]:
            recorded_calls.append((store_id, [p.product_id for p in products]))
            return [
                PricingKnowledgeItem(
                    product_id="p-201",
                    store_id=store_id,
                    content="Store historical guideline",
                    relevance_score=0.9,
                )
            ]

    set_pricing_knowledge_retriever(MockStoreRetriever())

    req = create_sample_batch_request(store_id="store-alex-02", product_ids=["p-201", "p-202"])
    expected_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p-201", discount_percentage=10.0, reason="Reason 1", confidence=0.9),
            PricingDecision(product_id="p-202", discount_percentage=5.0, reason="Reason 2", confidence=0.85),
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = expected_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert len(recorded_calls) == 1
    assert recorded_calls[0][0] == "store-alex-02"
    assert recorded_calls[0][1] == ["p-201", "p-202"]


def test_llm_failure_propagation():
    """Requirement Test: LLM invocation failure triggers fallback."""
    req = create_sample_batch_request()

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.side_effect = RuntimeError("LLM API failure")
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert "rule-based fallback" in resp.decisions[0].reason


def test_retriever_failure_propagation():
    """Requirement Test: Knowledge retriever failure propagates explicitly."""
    class FailingRetriever(PricingKnowledgeRetriever):
        def retrieve(self, store_id: str, products: list[PricingProductContext]) -> list[PricingKnowledgeItem]:
            raise ValueError("Store retriever DB connection failed")

    set_pricing_knowledge_retriever(FailingRetriever())
    req = create_sample_batch_request()

    with pytest.raises(ValueError) as exc_info:
        run_pricing_agent(req)

    assert "Store retriever DB connection failed" in str(exc_info.value)


def test_no_real_openai_calls_during_tests():
    """Requirement Test: Guarantee no real OpenAI calls are made during test execution."""
    req = create_sample_batch_request()

    with patch("langchain_openai.ChatOpenAI.invoke", side_effect=AssertionError("Real OpenAI call attempted!")):
        mock_llm_structured = MagicMock()
        mock_llm_structured.invoke.return_value = PricingBatchLLMResult(
            decisions=[
                PricingDecision(product_id="p-100", discount_percentage=10.0, reason="Mocked", confidence=0.9)
            ]
        )
        mock_base_llm = MagicMock()
        mock_base_llm.with_structured_output.return_value = mock_llm_structured

        with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
            resp = run_pricing_agent(req)
            assert resp.decisions[0].discount_percentage == 10.0


from app.agents.pricing.retriever import VectorPricingKnowledgeRetriever, build_product_query_text
from app.embeddings.fake import FakeEmbeddingProvider
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument
from app.vector_store.in_memory import InMemoryVectorStore


def test_pricing_agent_execution_with_vector_retriever():
    """Requirement 13: Pricing agent executes cleanly using VectorPricingKnowledgeRetriever."""
    embedder = FakeEmbeddingProvider(dimension=4)
    store = InMemoryVectorStore()

    p100_ctx = create_sample_product_context("p-100")
    query_text = build_product_query_text(p100_ctx)
    doc_vec = embedder.embed_query(query_text)

    doc = PricingKnowledgeDocument(
        document_id="doc-100",
        store_id="store-cairo-01",
        product_id="p-100",
        content="Milk nearing expiry should receive 10-15% discount.",
        metadata={"store_id": "store-cairo-01", "product_id": "p-100", "category": "Dairy"},
    )
    store.upsert([doc], [doc_vec])

    retriever = VectorPricingKnowledgeRetriever(embedder, store)
    set_pricing_knowledge_retriever(retriever)

    req = create_sample_batch_request(store_id="store-cairo-01", product_ids=["p-100"])
    expected_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p-100", discount_percentage=15.0, reason="High risk near expiry", confidence=0.9)
        ]
    )

    mock_llm_structured = MagicMock()
    mock_llm_structured.invoke.return_value = expected_result
    mock_base_llm = MagicMock()
    mock_base_llm.with_structured_output.return_value = mock_llm_structured

    with patch("app.agents.pricing.nodes.get_llm", return_value=mock_base_llm):
        resp = run_pricing_agent(req)

    assert resp.decisions[0].product_id == "p-100"
    assert resp.decisions[0].discount_percentage == 15.0

