import pytest
from pydantic import ValidationError

from app.schemas.monitoring import (
    MonitoringRequest,
    MonitoringResponse,
    RiskLevel,
    Route,
)


def get_valid_request_dict():
    return {
        "product": {
            "id": "prod-123",
            "name": "Organic Milk",
            "category": "Dairy",
        },
        "inventory": {
            "quantity": 10,
            "original_price": 5.0,
            "current_price": 4.5,
            "price_floor": 2.0,
        },
        "demand": {
            "sales_velocity": 2.5,
            "historical_sales": {
                "average_daily_sales": 3.0,
                "weekday_average": 2.8,
                "weekend_average": 3.5,
            },
        },
        "expiry": {
            "expires_at": "2026-08-16T12:00:00Z",
            "hours_remaining": 40.5,
        },
        "location": {
            "latitude": 37.7749,
            "longitude": -122.4194,
            "store_id": "store-001",
        },
        "timestamp": "2026-08-14T20:00:00Z",
    }


def test_valid_monitoring_request():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    assert request.product.id == "prod-123"
    assert request.inventory.quantity == 10
    assert request.demand.sales_velocity == 2.5
    assert request.expiry.hours_remaining == 40.5
    assert request.location.store_id == "store-001"


def test_missing_required_product_fails():
    payload = get_valid_request_dict()
    del payload["product"]
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_missing_required_quantity_fails():
    payload = get_valid_request_dict()
    del payload["inventory"]["quantity"]
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_missing_required_current_price_fails():
    payload = get_valid_request_dict()
    del payload["inventory"]["current_price"]
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_missing_required_expiry_fails():
    payload = get_valid_request_dict()
    del payload["expiry"]
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_missing_required_sales_velocity_fails():
    payload = get_valid_request_dict()
    del payload["demand"]["sales_velocity"]
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_missing_required_location_fails():
    payload = get_valid_request_dict()
    del payload["location"]
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_invalid_inventory_negative_quantity():
    payload = get_valid_request_dict()
    payload["inventory"]["quantity"] = -1
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_invalid_inventory_negative_price():
    payload = get_valid_request_dict()
    payload["inventory"]["original_price"] = -5.0
    with pytest.raises(ValidationError):
        MonitoringRequest(**payload)


def test_invalid_confidence_above_one():
    with pytest.raises(ValidationError):
        MonitoringResponse(
            route=Route.NO_ACTION,
            risk_level=RiskLevel.LOW,
            reason="Test reason",
            confidence=1.5,
        )


def test_invalid_confidence_below_zero():
    with pytest.raises(ValidationError):
        MonitoringResponse(
            route=Route.NO_ACTION,
            risk_level=RiskLevel.LOW,
            reason="Test reason",
            confidence=-0.1,
        )


def test_valid_confidence_boundaries():
    res_zero = MonitoringResponse(
        route=Route.PRICING,
        risk_level=RiskLevel.HIGH,
        reason="Boundary test zero",
        confidence=0.0,
    )
    assert res_zero.confidence == 0.0

    res_one = MonitoringResponse(
        route=Route.NO_ACTION,
        risk_level=RiskLevel.CRITICAL,
        reason="Boundary test one",
        confidence=1.0,
    )
    assert res_one.confidence == 1.0
