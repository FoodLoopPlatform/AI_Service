from unittest.mock import MagicMock, patch

from app.agents.monitoring import get_monitoring_graph, run_monitoring_agent
from app.schemas.context_analysis import AllowedContext, ContextAnalysisResult
from app.schemas.monitoring import MonitoringRequest, RiskLevel, Route
from app.schemas.risk_assessment import RiskAssessmentResult
from app.tools.events import LocalEventsContext
from app.tools.weather import WeatherContext
from tests.test_monitoring_schema import get_valid_request_dict


def make_mock_llm(missing_context=None, risk_level=RiskLevel.LOW, reason="Risk assessment test.", confidence=0.90):
    if missing_context is None:
        missing_context = []

    context_result = ContextAnalysisResult(
        is_sufficient=(len(missing_context) == 0),
        missing_context=missing_context,
        reason="Context evaluation test.",
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


def test_monitoring_agent_low_risk_scenario():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)

    mock_llm = make_mock_llm(missing_context=[], risk_level=RiskLevel.LOW, reason="Sufficient sales velocity and long shelf life.", confidence=0.95)

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        response = run_monitoring_agent(request)

    assert response.route == Route.NO_ACTION
    assert response.risk_level == RiskLevel.LOW
    assert response.reason == "Sufficient sales velocity and long shelf life."
    assert response.confidence == 0.95


def test_monitoring_agent_high_risk_scenario():
    payload = get_valid_request_dict()
    request = MonitoringRequest(**payload)

    mock_llm = make_mock_llm(missing_context=[], risk_level=RiskLevel.HIGH, reason="Imminent expiry with high inventory volume.", confidence=0.93)

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        response = run_monitoring_agent(request)

    assert response.route == Route.PRICING
    assert response.risk_level == RiskLevel.HIGH
    assert response.reason == "Imminent expiry with high inventory volume."
    assert response.confidence == 0.93


def test_routing_case_1_no_missing_context():
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

    mock_llm = make_mock_llm([])

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_weather_forecast") as mock_weather, \
         patch("app.agents.monitoring.nodes.get_local_events") as mock_events:

        graph = get_monitoring_graph()
        final_state = graph.invoke(initial_state)

        mock_weather.assert_not_called()
        mock_events.assert_not_called()
        assert final_state["missing_context"] == []
        assert final_state["risk_level"] == RiskLevel.LOW
        assert final_state["route"] == Route.NO_ACTION


def test_routing_case_2_weather_missing():
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

    mock_llm = make_mock_llm([AllowedContext.WEATHER], risk_level=RiskLevel.MEDIUM)

    mock_weather_res = WeatherContext(forecast=[])

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_weather_forecast", return_value=mock_weather_res) as mock_weather, \
         patch("app.agents.monitoring.nodes.get_local_events") as mock_events:

        graph = get_monitoring_graph()
        final_state = graph.invoke(initial_state)

        assert mock_weather.call_count == 1
        mock_events.assert_not_called()
        assert final_state["weather_context"] == mock_weather_res
        assert final_state["events_context"] is None
        assert final_state["risk_level"] == RiskLevel.MEDIUM
        assert final_state["route"] == Route.PRICING


def test_routing_case_3_events_missing():
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

    mock_llm = make_mock_llm([AllowedContext.LOCAL_EVENTS], risk_level=RiskLevel.HIGH)

    mock_events_res = LocalEventsContext(events=[])

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_weather_forecast") as mock_weather, \
         patch("app.agents.monitoring.nodes.get_local_events", return_value=mock_events_res) as mock_events:

        graph = get_monitoring_graph()
        final_state = graph.invoke(initial_state)

        mock_weather.assert_not_called()
        assert mock_events.call_count == 1
        assert final_state["weather_context"] is None
        assert final_state["events_context"] == mock_events_res
        assert final_state["risk_level"] == RiskLevel.HIGH
        assert final_state["route"] == Route.PRICING


def test_routing_case_4_both_weather_and_events_missing():
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

    mock_llm = make_mock_llm([AllowedContext.WEATHER, AllowedContext.LOCAL_EVENTS], risk_level=RiskLevel.LOW)

    mock_weather_res = WeatherContext(forecast=[])
    mock_events_res = LocalEventsContext(events=[])

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm), \
         patch("app.agents.monitoring.nodes.get_weather_forecast", return_value=mock_weather_res) as mock_weather, \
         patch("app.agents.monitoring.nodes.get_local_events", return_value=mock_events_res) as mock_events:

        graph = get_monitoring_graph()
        final_state = graph.invoke(initial_state)

        assert mock_weather.call_count == 1
        assert mock_events.call_count == 1
        assert final_state["weather_context"] == mock_weather_res
        assert final_state["events_context"] == mock_events_res
        assert final_state["risk_level"] == RiskLevel.LOW
        assert final_state["route"] == Route.NO_ACTION
