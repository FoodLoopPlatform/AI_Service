from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from app.agents.monitoring.nodes import assess_risk
from app.agents.monitoring.state import MonitoringAgentState
from app.schemas.monitoring import MonitoringRequest, RiskLevel
from app.schemas.risk_assessment import RiskAssessmentResult
from tests.test_monitoring_schema import get_valid_request_dict


def test_risk_assessment_schema_validation():
    valid = RiskAssessmentResult(
        risk_level=RiskLevel.HIGH,
        reason="High risk due to short shelf life.",
        confidence=0.94,
    )
    assert valid.risk_level == RiskLevel.HIGH
    assert valid.confidence == 0.94

    with pytest.raises(ValidationError):
        RiskAssessmentResult(
            risk_level=RiskLevel.HIGH,
            reason="Invalid confidence high",
            confidence=1.2,
        )

    with pytest.raises(ValidationError):
        RiskAssessmentResult(
            risk_level=RiskLevel.HIGH,
            reason="Invalid confidence low",
            confidence=-0.5,
        )


def test_assess_risk_node_high_risk():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    state: MonitoringAgentState = {
        "request": request,
        "risk_signals": None,
        "risk_level": None,
        "missing_context": None,
        "is_context_sufficient": None,
        "context_analysis_reason": None,
        "context_analysis_confidence": None,
        "weather_context": None,
        "events_context": None,
        "route": None,
        "reason": None,
        "confidence": None,
    }

    mock_result = RiskAssessmentResult(
        risk_level=RiskLevel.HIGH,
        reason="Inventory significantly exceeds expected near-term sales and product is near expiry.",
        confidence=0.94,
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        node_output = assess_risk(state)

    assert node_output["risk_level"] == RiskLevel.HIGH
    assert node_output["confidence"] == 0.94
    assert "Inventory significantly exceeds" in node_output["reason"]
    assert node_output["risk_signals"] is not None


def test_assess_risk_node_medium_risk():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    state: MonitoringAgentState = {
        "request": request,
        "risk_signals": None,
        "risk_level": None,
        "missing_context": None,
        "is_context_sufficient": None,
        "context_analysis_reason": None,
        "context_analysis_confidence": None,
        "weather_context": None,
        "events_context": None,
        "route": None,
        "reason": None,
        "confidence": None,
    }

    mock_result = RiskAssessmentResult(
        risk_level=RiskLevel.MEDIUM,
        reason="Moderate sell-through risk based on demand pressure.",
        confidence=0.85,
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        node_output = assess_risk(state)

    assert node_output["risk_level"] == RiskLevel.MEDIUM
    assert node_output["confidence"] == 0.85


def test_assess_risk_node_low_risk():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    state: MonitoringAgentState = {
        "request": request,
        "risk_signals": None,
        "risk_level": None,
        "missing_context": None,
        "is_context_sufficient": None,
        "context_analysis_reason": None,
        "context_analysis_confidence": None,
        "weather_context": None,
        "events_context": None,
        "route": None,
        "reason": None,
        "confidence": None,
    }

    mock_result = RiskAssessmentResult(
        risk_level=RiskLevel.LOW,
        reason="Inventory level is aligned with normal sales velocity.",
        confidence=0.98,
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        node_output = assess_risk(state)

    assert node_output["risk_level"] == RiskLevel.LOW
    assert node_output["confidence"] == 0.98


def test_assess_risk_node_llm_failure_propagates():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    state: MonitoringAgentState = {
        "request": request,
        "risk_signals": None,
        "risk_level": None,
        "missing_context": None,
        "is_context_sufficient": None,
        "context_analysis_reason": None,
        "context_analysis_confidence": None,
        "weather_context": None,
        "events_context": None,
        "route": None,
        "reason": None,
        "confidence": None,
    }

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.side_effect = RuntimeError("LLM risk assessment service timeout")
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        with pytest.raises(RuntimeError, match="LLM risk assessment service timeout"):
            assess_risk(state)
