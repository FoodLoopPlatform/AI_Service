from datetime import datetime
import sys
import pytest
from pydantic import ValidationError

from app.agents.pricing.knowledge_builder import build_pricing_knowledge_document
from app.schemas.historical_pricing import HistoricalPricingEvent, Outcome
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument


def make_valid_event_dict(**overrides):
    """Helper to create a valid dictionary for HistoricalPricingEvent."""
    base = {
        "event_id": "event-001",
        "store_id": "store-cairo-01",
        "product_id": "p-100",
        "category": "Dairy",
        "recorded_at": datetime.fromisoformat("2026-08-15T12:00:00"),
        "quantity": 10.0,
        "current_price": 40.0,
        "original_price": 40.0,
        "price_floor": 28.0,
        "sales_velocity": 0.5,
        "historical_average_daily_sales": 5.0,
        "hours_remaining": 18.0,
        "discount_percentage": 10.0,
        "units_sold_after_discount": 8.0,
        "sell_through_rate": 0.8,
        "outcome": Outcome.PARTIALLY_SOLD,
    }
    base.update(overrides)
    return base


def test_valid_historical_pricing_event():
    """Requirement 8.1: Valid historical event accepted."""
    event = HistoricalPricingEvent(**make_valid_event_dict())
    assert event.event_id == "event-001"
    assert event.store_id == "store-cairo-01"
    assert event.product_id == "p-100"
    assert event.discount_percentage == 10.0
    assert event.outcome == Outcome.PARTIALLY_SOLD


def test_negative_quantity_rejected():
    """Requirement 8.2: Negative quantity rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(quantity=-1.0))
    assert "quantity" in str(exc_info.value)


def test_negative_prices_rejected():
    """Requirement 8.3: Negative prices rejected."""
    with pytest.raises(ValidationError) as exc_curr:
        HistoricalPricingEvent(**make_valid_event_dict(current_price=-5.0))
    assert "current_price" in str(exc_curr.value)

    with pytest.raises(ValidationError) as exc_orig:
        HistoricalPricingEvent(**make_valid_event_dict(original_price=-10.0))
    assert "original_price" in str(exc_orig.value)

    with pytest.raises(ValidationError) as exc_floor:
        HistoricalPricingEvent(**make_valid_event_dict(price_floor=-1.0))
    assert "price_floor" in str(exc_floor.value)


def test_negative_sales_velocity_rejected():
    """Requirement 8.4: Negative sales velocity rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(sales_velocity=-0.5))
    assert "sales_velocity" in str(exc_info.value)


def test_negative_historical_average_sales_rejected():
    """Requirement 8.5: Negative historical average sales rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(historical_average_daily_sales=-1.0))
    assert "historical_average_daily_sales" in str(exc_info.value)


def test_negative_hours_remaining_rejected():
    """Requirement 8.6: Negative hours_remaining rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(hours_remaining=-2.0))
    assert "hours_remaining" in str(exc_info.value)


def test_discount_greater_than_15_rejected():
    """Requirement 8.7: Discount > 15 rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(discount_percentage=15.1))
    assert "discount_percentage" in str(exc_info.value)


def test_discount_less_than_0_rejected():
    """Requirement 8.8: Discount < 0 rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(discount_percentage=-1.0))
    assert "discount_percentage" in str(exc_info.value)


def test_sell_through_rate_greater_than_1_rejected():
    """Requirement 8.9: sell_through_rate > 1 rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(sell_through_rate=1.05))
    assert "sell_through_rate" in str(exc_info.value)


def test_sell_through_rate_less_than_0_rejected():
    """Requirement 8.10: sell_through_rate < 0 rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(sell_through_rate=-0.1))
    assert "sell_through_rate" in str(exc_info.value)


def test_invalid_outcome_rejected():
    """Requirement 8.11: Invalid outcome rejected."""
    with pytest.raises(ValidationError) as exc_info:
        HistoricalPricingEvent(**make_valid_event_dict(outcome="INVALID_OUTCOME"))
    assert "outcome" in str(exc_info.value)


def test_document_builder_creates_correct_document():
    """Requirement 8.12: Document builder creates correct PricingKnowledgeDocument."""
    event = HistoricalPricingEvent(**make_valid_event_dict())
    doc = build_pricing_knowledge_document(event)

    assert isinstance(doc, PricingKnowledgeDocument)
    assert doc.document_id == "doc-event-001"
    assert "Historical pricing event for product p-100" in doc.content
    assert "PARTIALLY_SOLD" in doc.content


def test_document_contains_store_id():
    """Requirement 8.13: Document contains store_id as top-level field and in content."""
    event = HistoricalPricingEvent(**make_valid_event_dict(store_id="store-alex-02"))
    doc = build_pricing_knowledge_document(event)

    assert doc.store_id == "store-alex-02"
    assert "store-alex-02" in doc.content


def test_document_contains_product_id():
    """Requirement 8.14: Document contains product_id as top-level field and in content."""
    event = HistoricalPricingEvent(**make_valid_event_dict(product_id="prod-milk-99"))
    doc = build_pricing_knowledge_document(event)

    assert doc.product_id == "prod-milk-99"
    assert "prod-milk-99" in doc.content


def test_metadata_contains_store_id_and_product_id():
    """Requirement 8.15: Metadata contains store_id, product_id, category, outcome, discount_percentage."""
    event = HistoricalPricingEvent(**make_valid_event_dict(
        store_id="store-giza-03",
        product_id="prod-bread-05",
        category="Bakery",
        discount_percentage=12.0,
        outcome=Outcome.SOLD_OUT,
    ))
    doc = build_pricing_knowledge_document(event)

    assert doc.metadata["store_id"] == "store-giza-03"
    assert doc.metadata["product_id"] == "prod-bread-05"
    assert doc.metadata["category"] == "Bakery"
    assert doc.metadata["outcome"] == "SOLD_OUT"
    assert doc.metadata["discount_percentage"] == 12.0


def test_document_contains_historical_discount_as_fact():
    """Requirement 8.16: Document contains historical discount as a FACT."""
    event = HistoricalPricingEvent(**make_valid_event_dict(discount_percentage=10.0))
    doc = build_pricing_knowledge_document(event)

    assert "applied historical discount was 10.0%" in doc.content


def test_document_builder_does_not_generate_recommendation_language():
    """Requirement 8.17: Document builder does NOT generate recommendation language."""
    event = HistoricalPricingEvent(**make_valid_event_dict())
    doc = build_pricing_knowledge_document(event)

    forbidden_terms = ["recommend", "should", "optimal", "advice", "suggest", "floor adjustment"]
    content_lower = doc.content.lower()

    for term in forbidden_terms:
        assert term not in content_lower, f"Forbidden recommendation term '{term}' found in document content"


def test_same_input_produces_deterministic_output():
    """Requirement 8.18: Same input produces deterministic output."""
    event1 = HistoricalPricingEvent(**make_valid_event_dict())
    event2 = HistoricalPricingEvent(**make_valid_event_dict())

    doc1 = build_pricing_knowledge_document(event1)
    doc2 = build_pricing_knowledge_document(event2)

    assert doc1.document_id == doc2.document_id
    assert doc1.store_id == doc2.store_id
    assert doc1.product_id == doc2.product_id
    assert doc1.content == doc2.content
    assert doc1.metadata == doc2.metadata


def test_different_product_ids_produce_isolated_documents():
    """Requirement 8.19: Different product IDs produce isolated documents."""
    event_a = HistoricalPricingEvent(**make_valid_event_dict(product_id="product-A"))
    event_b = HistoricalPricingEvent(**make_valid_event_dict(product_id="product-B"))

    doc_a = build_pricing_knowledge_document(event_a)
    doc_b = build_pricing_knowledge_document(event_b)

    assert doc_a.product_id == "product-A"
    assert doc_b.product_id == "product-B"
    assert doc_a.metadata["product_id"] == "product-A"
    assert doc_b.metadata["product_id"] == "product-B"
    assert "product-A" in doc_a.content
    assert "product-B" not in doc_a.content
    assert "product-B" in doc_b.content
    assert "product-A" not in doc_b.content


def test_different_store_ids_produce_isolated_documents():
    """Requirement 8.20: Different store IDs produce isolated documents."""
    event_s1 = HistoricalPricingEvent(**make_valid_event_dict(store_id="store-01"))
    event_s2 = HistoricalPricingEvent(**make_valid_event_dict(store_id="store-02"))

    doc_s1 = build_pricing_knowledge_document(event_s1)
    doc_s2 = build_pricing_knowledge_document(event_s2)

    assert doc_s1.store_id == "store-01"
    assert doc_s2.store_id == "store-02"
    assert doc_s1.metadata["store_id"] == "store-01"
    assert doc_s2.metadata["store_id"] == "store-02"
    assert "store-01" in doc_s1.content
    assert "store-02" not in doc_s1.content
    assert "store-02" in doc_s2.content
    assert "store-01" not in doc_s2.content


def test_architectural_builder_isolation():
    """Requirement 9: Builder function does not depend on LLM, LangGraph, vector DB, HTTP client, or OpenAI."""
    from app.agents.pricing import knowledge_builder

    imported_modules = sys.modules

    # Ensure module source code does not reference prohibited frameworks
    import inspect
    source = inspect.getsource(knowledge_builder)

    prohibited_keywords = [
        "langgraph",
        "langchain",
        "openai",
        "qdrant",
        "chroma",
        "pinecone",
        "httpx",
        "requests",
        "get_llm",
    ]

    for kw in prohibited_keywords:
        assert kw not in source.lower(), f"Prohibited framework/client '{kw}' found in knowledge_builder.py"
