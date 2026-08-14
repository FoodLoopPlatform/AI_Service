import pytest

from app.agents.monitoring.nodes import RiskAssessmentMissingError, determine_route
from app.agents.monitoring.state import MonitoringAgentState
from app.schemas.monitoring import MonitoringRequest, RiskLevel, Route
from tests.test_monitoring_schema import get_valid_request_dict


def get_base_state(risk_level: RiskLevel | None) -> MonitoringAgentState:
    request = MonitoringRequest(**get_valid_request_dict())
    return {
        "request": request,
        "risk_signals": None,
        "risk_level": risk_level,
        "missing_context": None,
        "is_context_sufficient": None,
        "context_analysis_reason": None,
        "context_analysis_confidence": None,
        "weather_context": None,
        "events_context": None,
        "route": None,
        "reason": "Test risk assessment rationale.",
        "confidence": 0.93,
    }


def test_determine_route_low_risk():
    state = get_base_state(RiskLevel.LOW)
    output = determine_route(state)
    assert output == {"route": Route.NO_ACTION}


def test_determine_route_medium_risk():
    state = get_base_state(RiskLevel.MEDIUM)
    output = determine_route(state)
    assert output == {"route": Route.PRICING}


def test_determine_route_high_risk():
    state = get_base_state(RiskLevel.HIGH)
    output = determine_route(state)
    assert output == {"route": Route.PRICING}


def test_determine_route_critical_risk():
    state = get_base_state(RiskLevel.CRITICAL)
    output = determine_route(state)
    assert output == {"route": Route.PRICING}


def test_determine_route_missing_risk_raises_exception():
    state = get_base_state(None)
    with pytest.raises(RiskAssessmentMissingError, match="risk_level is missing"):
        determine_route(state)


def test_determine_route_preserves_rationale_and_confidence():
    state = get_base_state(RiskLevel.HIGH)
    output = determine_route(state)

    # Node output only contains route
    assert output == {"route": Route.PRICING}

    # Verify input state fields remain intact
    assert state["risk_level"] == RiskLevel.HIGH
    assert state["reason"] == "Test risk assessment rationale."
    assert state["confidence"] == 0.93
