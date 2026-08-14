from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.agents.monitoring import get_monitoring_graph, run_monitoring_agent
from app.agents.monitoring.nodes import RiskAssessmentMissingError, determine_route
from app.agents.monitoring.risk_signals import (
    SignalLevel,
    calculate_demand_pressure,
    calculate_inventory_pressure,
)
from app.main import app
from app.schemas.context_analysis import AllowedContext, ContextAnalysisResult
from app.schemas.monitoring import MonitoringRequest, RiskLevel, Route
from app.schemas.risk_assessment import RiskAssessmentResult
from app.tools.events import LocalEventsContext, LocalEventsToolError
from app.tools.weather import WeatherContext, WeatherToolError
from tests.test_monitoring_schema import get_valid_request_dict

client = TestClient(app)


def make_mock_llm(missing_context=None, risk_level=RiskLevel.LOW, reason="Test reason", confidence=0.90):
    if missing_context is None:
        missing_context = []

    context_result = ContextAnalysisResult(
        is_sufficient=(len(missing_context) == 0),
        missing_context=missing_context,
        reason="Context evaluation.",
        confidence=0.95,
    )

    risk_result = RiskAssessmentResult(
        risk_level=risk_level,
        reason=reason,
        confidence=confidence,
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


def test_scenario_1_valid_request_no_extra_context_low_risk():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    mock_llm = make_mock_llm(missing_context=[], risk_level=RiskLevel.LOW, reason="Low risk", confidence=0.95)

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        response = run_monitoring_agent(request)

    assert response.route == Route.NO_ACTION
    assert response.risk_level == RiskLevel.LOW
    assert response.reason == "Low risk"
    assert response.confidence == 0.95


def test_scenario_2_valid_request_weather_required_high_risk():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    mock_llm = make_mock_llm(missing_context=[AllowedContext.WEATHER], risk_level=RiskLevel.HIGH, reason="High risk with rain", confidence=0.92)

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_weather_forecast", return_value=WeatherContext(forecast=[])) as mock_weather:
        response = run_monitoring_agent(request)

    assert mock_weather.call_count == 1
    assert response.route == Route.PRICING
    assert response.risk_level == RiskLevel.HIGH


def test_scenario_3_valid_request_events_required_medium_risk():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    mock_llm = make_mock_llm(missing_context=[AllowedContext.LOCAL_EVENTS], risk_level=RiskLevel.MEDIUM, reason="Medium risk with event", confidence=0.88)

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_local_events", return_value=LocalEventsContext(events=[])) as mock_events:
        response = run_monitoring_agent(request)

    assert mock_events.call_count == 1
    assert response.route == Route.PRICING
    assert response.risk_level == RiskLevel.MEDIUM


def test_scenario_4_valid_request_both_contexts_required_critical_risk():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    mock_llm = make_mock_llm(missing_context=[AllowedContext.WEATHER, AllowedContext.LOCAL_EVENTS], risk_level=RiskLevel.CRITICAL, reason="Critical risk", confidence=0.98)

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_weather_forecast", return_value=WeatherContext(forecast=[])) as mock_weather, \
         patch("app.agents.monitoring.nodes.get_local_events", return_value=LocalEventsContext(events=[])) as mock_events:
        response = run_monitoring_agent(request)

    assert mock_weather.call_count == 1
    assert mock_events.call_count == 1
    assert response.route == Route.PRICING
    assert response.risk_level == RiskLevel.CRITICAL


def test_scenario_5_missing_required_request_field_http_422():
    payload = get_valid_request_dict()
    del payload["inventory"]["quantity"]  # Missing required field

    mock_llm = make_mock_llm()
    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        response = client.post("/api/v1/monitoring/analyze", json=payload)

    assert response.status_code == 422
    mock_llm.assert_not_called()  # Workflow does NOT execute


def test_scenario_6_unsupported_optional_context_validation_failure():
    with pytest.raises(ValidationError):
        ContextAnalysisResult(
            is_sufficient=False,
            missing_context=["traffic"],  # Unsupported
            reason="Traffic needed.",
            confidence=0.8,
        )


def test_scenario_7_weather_tool_failure_propagates():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    mock_llm = make_mock_llm(missing_context=[AllowedContext.WEATHER])

    initial_state = {
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

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_weather_forecast", side_effect=WeatherToolError("Weather service down")):
        graph = get_monitoring_graph()
        with pytest.raises(WeatherToolError, match="Weather service down"):
            graph.invoke(initial_state)


def test_scenario_8_events_tool_failure_propagates():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    mock_llm = make_mock_llm(missing_context=[AllowedContext.LOCAL_EVENTS])

    initial_state = {
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

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_local_events", side_effect=LocalEventsToolError("Events service down")):
        graph = get_monitoring_graph()
        with pytest.raises(LocalEventsToolError, match="Events service down"):
            graph.invoke(initial_state)


def test_scenario_9_context_analysis_llm_failure_propagates():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)

    initial_state = {
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
    mock_struct = MagicMock()
    mock_struct.invoke.side_effect = RuntimeError("Context analysis LLM timeout")
    mock_llm.with_structured_output.return_value = mock_struct

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        graph = get_monitoring_graph()
        with pytest.raises(RuntimeError, match="Context analysis LLM timeout"):
            graph.invoke(initial_state)


def test_scenario_10_risk_assessment_llm_failure_propagates():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)

    initial_state = {
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

    context_result = ContextAnalysisResult(
        is_sufficient=True,
        missing_context=[],
        reason="Sufficient",
        confidence=0.9,
    )

    mock_llm = MagicMock()

    def side_effect(schema):
        mock_struct = MagicMock()
        if schema == ContextAnalysisResult:
            mock_struct.invoke.return_value = context_result
        elif schema == RiskAssessmentResult:
            mock_struct.invoke.side_effect = RuntimeError("Risk assessment LLM failure")
        return mock_struct

    mock_llm.with_structured_output.side_effect = side_effect

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        graph = get_monitoring_graph()
        with pytest.raises(RuntimeError, match="Risk assessment LLM failure"):
            graph.invoke(initial_state)


def test_scenario_11_missing_risk_level_before_routing_raises_error():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)
    state = {
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

    with pytest.raises(RiskAssessmentMissingError, match="risk_level is missing"):
        determine_route(state)


def test_scenario_12_zero_sales_velocity_no_division_by_zero():
    pressure = calculate_inventory_pressure(quantity=50, sales_velocity=0.0, hours_remaining=24.0)
    assert pressure == SignalLevel.HIGH


def test_scenario_13_zero_historical_average_sales_no_division_by_zero():
    pressure = calculate_demand_pressure(sales_velocity=5.0, historical_average_daily_sales=0.0)
    assert pressure == SignalLevel.LOW
