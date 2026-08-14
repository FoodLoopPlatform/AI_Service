from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from app.main import app
from app.schemas.context_analysis import ContextAnalysisResult
from app.schemas.monitoring import RiskLevel
from app.schemas.risk_assessment import RiskAssessmentResult
from tests.test_monitoring_schema import get_valid_request_dict

client = TestClient(app)


def make_mock_llm():
    context_result = ContextAnalysisResult(
        is_sufficient=True,
        missing_context=[],
        reason="Context is sufficient.",
        confidence=0.95,
    )
    risk_result = RiskAssessmentResult(
        risk_level=RiskLevel.LOW,
        reason="Risk is low based on normal metrics.",
        confidence=0.90,
    )

    mock_llm = MagicMock()

    def side_effect(schema):
        mock_struct = MagicMock()
        if schema == ContextAnalysisResult:
            mock_struct.invoke.return_value = context_result
        elif schema == RiskAssessmentResult:
            mock_struct.invoke.return_value = risk_result
        return mock_struct

    mock_llm.with_structured_output.side_effect = side_effect
    return mock_llm


def test_analyze_inventory_valid_payload():
    payload = get_valid_request_dict()

    mock_llm = make_mock_llm()

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        response = client.post("/api/v1/monitoring/analyze", json=payload)

    assert response.status_code == 200
    assert response.json() == {
        "route": "NO_ACTION",
        "risk_level": "LOW",
        "reason": "Risk is low based on normal metrics.",
        "confidence": 0.90,
    }


def test_analyze_inventory_invalid_payload_returns_422():
    payload = get_valid_request_dict()
    payload["inventory"]["quantity"] = -10  # Negative quantity is invalid
    response = client.post("/api/v1/monitoring/analyze", json=payload)
    assert response.status_code == 422
