from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.pricing import PricingBatchLLMResult, PricingDecision
from tests.test_pricing_schema import make_valid_batch_request_dict

client = TestClient(app)


def test_pricing_api_batch_endpoint_success():
    """Test successful batch recommendation response via FastAPI endpoint."""
    request_data = make_valid_batch_request_dict(store_id="store-cairo-01", product_ids=["p-100", "p-101"])
    mock_llm_result = PricingBatchLLMResult(
        decisions=[
            PricingDecision(product_id="p-100", discount_percentage=15.0, reason="High risk near expiry", confidence=0.92),
            PricingDecision(product_id="p-101", discount_percentage=5.0, reason="Moderate velocity", confidence=0.88),
        ]
    )

    with patch("app.agents.pricing.nodes.get_llm") as mock_get_llm:
        mock_llm_structured = MagicMock()
        mock_llm_structured.invoke.return_value = mock_llm_result
        mock_get_llm.return_value.with_structured_output.return_value = mock_llm_structured

        response = client.post("/api/v1/pricing/recommend", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["store_id"] == "store-cairo-01"
    assert len(data["decisions"]) == 2
    assert data["decisions"][0]["product_id"] == "p-100"
    assert data["decisions"][0]["discount_percentage"] == 15.0
    assert data["decisions"][1]["product_id"] == "p-101"
    assert data["decisions"][1]["discount_percentage"] == 5.0

    # Verify no extra financial fields are present
    assert "recommended_price" not in data
    assert "final_price" not in data


def test_pricing_api_batch_endpoint_validation_error():
    """Test FastAPI returns 422 Unprocessable Entity when products list is empty."""
    request_data = make_valid_batch_request_dict(store_id="store-cairo-01")
    request_data["products"] = []  # Invalid empty products list

    response = client.post("/api/v1/pricing/recommend", json=request_data)
    assert response.status_code == 422
