from datetime import datetime
import pytest
from pydantic import ValidationError

from app.schemas.pricing import (
    PricingBatchLLMResult,
    PricingBatchRequest,
    PricingBatchResponse,
    PricingDecision,
    PricingProductContext,
)


def make_valid_product_context_dict(product_id: str = "p-100"):
    """Helper to construct a valid dictionary for PricingProductContext."""
    return {
        "product_id": product_id,
        "product_name": "Fresh Milk 1L",
        "category": "Dairy",
        "inventory": {
            "quantity": 10,
            "original_price": 40.0,
            "current_price": 40.0,
            "price_floor": 28.0,
        },
        "demand": {
            "sales_velocity": 0.5,
            "historical_sales": {
                "average_daily_sales": 5.0,
                "weekday_average": 5.0,
                "weekend_average": 5.0,
            },
        },
        "expiry": {
            "expires_at": datetime.now().isoformat(),
            "hours_remaining": 12.0,
        },
        "risk_assessment": {
            "risk_level": "HIGH",
            "reason": "Limited shelf life remaining with moderate inventory.",
            "confidence": 0.9,
        },
    }


def make_valid_batch_request_dict(store_id: str = "store-cairo-01", product_ids: list[str] | None = None):
    """Helper to construct a valid dictionary for PricingBatchRequest."""
    p_ids = product_ids or ["p-100"]
    return {
        "store_id": store_id,
        "products": [make_valid_product_context_dict(pid) for pid in p_ids],
    }


def test_valid_pricing_decision_0_percent():
    """Requirement Test: Valid PricingDecision with 0%."""
    decision = PricingDecision(
        product_id="p-100",
        discount_percentage=0,
        reason="Product has sufficient shelf life and strong demand.",
        confidence=0.95,
    )
    assert decision.product_id == "p-100"
    assert decision.discount_percentage == 0.0
    assert decision.confidence == 0.95


def test_valid_pricing_decision_15_percent():
    """Requirement Test: Valid PricingDecision with 15%."""
    decision = PricingDecision(
        product_id="p-100",
        discount_percentage=15,
        reason="Product is high risk near expiry with weak sales velocity.",
        confidence=0.91,
    )
    assert decision.discount_percentage == 15.0
    assert decision.confidence == 0.91


def test_invalid_discount_less_than_0():
    """Requirement Test: Invalid discount < 0."""
    with pytest.raises(ValidationError) as exc_info:
        PricingDecision(
            product_id="p-100",
            discount_percentage=-1,
            reason="Invalid negative discount",
            confidence=0.8,
        )
    assert "discount_percentage" in str(exc_info.value)


def test_invalid_discount_greater_than_15():
    """Requirement Test: Invalid discount > 15."""
    with pytest.raises(ValidationError) as exc_info_15_1:
        PricingDecision(
            product_id="p-100",
            discount_percentage=15.1,
            reason="Invalid discount above max 15%",
            confidence=0.8,
        )
    assert "discount_percentage" in str(exc_info_15_1.value)

    with pytest.raises(ValidationError) as exc_info_20:
        PricingDecision(
            product_id="p-100",
            discount_percentage=20,
            reason="Invalid discount above max 15%",
            confidence=0.8,
        )
    assert "discount_percentage" in str(exc_info_20.value)


def test_invalid_confidence_less_than_0():
    """Requirement Test: Invalid confidence < 0."""
    with pytest.raises(ValidationError) as exc_info:
        PricingDecision(
            product_id="p-100",
            discount_percentage=10,
            reason="Invalid confidence",
            confidence=-0.1,
        )
    assert "confidence" in str(exc_info.value)


def test_invalid_confidence_greater_than_1():
    """Requirement Test: Invalid confidence > 1."""
    with pytest.raises(ValidationError) as exc_info:
        PricingDecision(
            product_id="p-100",
            discount_percentage=10,
            reason="Invalid confidence",
            confidence=1.1,
        )
    assert "confidence" in str(exc_info.value)


def test_forbidden_fields_in_pricing_decision():
    """Requirement Test: PricingDecision forbids extra financial execution fields."""
    forbidden_fields = [
        ("recommended_price", 35.0),
        ("final_price", 34.0),
        ("price_after_discount", 34.0),
        ("price_floor_adjustment", -2.0),
        ("donation_decision", True),
        ("automation_decision", "AUTO"),
    ]

    for field_name, field_val in forbidden_fields:
        kwargs = {
            "product_id": "p-100",
            "discount_percentage": 10,
            "reason": "Test reason",
            "confidence": 0.9,
            field_name: field_val,
        }
        with pytest.raises(ValidationError) as exc_info:
            PricingDecision(**kwargs)
        assert "Extra inputs are not permitted" in str(exc_info.value) or field_name in str(exc_info.value)


def test_one_product_batch_request():
    """Requirement Test: Valid one product batch request."""
    data = make_valid_batch_request_dict(store_id="store-01", product_ids=["p-100"])
    req = PricingBatchRequest(**data)
    assert req.store_id == "store-01"
    assert len(req.products) == 1
    assert req.products[0].product_id == "p-100"


def test_multiple_products_batch_request():
    """Requirement Test: Valid multiple products batch request."""
    data = make_valid_batch_request_dict(store_id="store-01", product_ids=["p-100", "p-101", "p-102"])
    req = PricingBatchRequest(**data)
    assert req.store_id == "store-01"
    assert len(req.products) == 3
    assert [p.product_id for p in req.products] == ["p-100", "p-101", "p-102"]


def test_empty_products_rejected():
    """Requirement Test: Empty products list is rejected."""
    with pytest.raises(ValidationError) as exc_info:
        PricingBatchRequest(store_id="store-01", products=[])
    assert "products" in str(exc_info.value)


def test_store_id_required():
    """Requirement Test: store_id is required and cannot be empty."""
    data = make_valid_batch_request_dict(store_id="")
    with pytest.raises(ValidationError) as exc_info:
        PricingBatchRequest(**data)
    assert "store_id" in str(exc_info.value)
