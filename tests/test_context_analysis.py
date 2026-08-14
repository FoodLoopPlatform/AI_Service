from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError

from app.agents.monitoring.nodes import analyze_context
from app.agents.monitoring.state import MonitoringAgentState
from app.schemas.context_analysis import AllowedContext, ContextAnalysisResult
from app.schemas.monitoring import MonitoringRequest
from tests.test_monitoring_schema import get_valid_request_dict


def test_context_analysis_schema_valid_sufficient():
    result = ContextAnalysisResult(
        is_sufficient=True,
        missing_context=[],
        reason="The provided inventory, demand, expiry, and location data are sufficient.",
        confidence=0.95,
    )
    assert result.is_sufficient is True
    assert result.missing_context == []
    assert result.confidence == 0.95


def test_context_analysis_schema_valid_weather_required():
    result = ContextAnalysisResult(
        is_sufficient=False,
        missing_context=[AllowedContext.WEATHER],
        reason="Weather conditions may materially affect near-term foot traffic.",
        confidence=0.87,
    )
    assert result.is_sufficient is False
    assert result.missing_context == [AllowedContext.WEATHER]
    assert result.missing_context == ["weather"]


def test_context_analysis_schema_multiple_contexts():
    result = ContextAnalysisResult(
        is_sufficient=False,
        missing_context=[AllowedContext.WEATHER, AllowedContext.LOCAL_EVENTS],
        reason="Weather and local events are needed due to storm and festival.",
        confidence=0.90,
    )
    assert set(result.missing_context) == {"weather", "local_events"}


@pytest.mark.parametrize("invalid_ctx", ["traffic", "quantity", "price", "expiry"])
def test_context_analysis_schema_invalid_context_types(invalid_ctx):
    with pytest.raises(ValidationError):
        ContextAnalysisResult(
            is_sufficient=False,
            missing_context=[invalid_ctx],  # Unsupported value
            reason="Invalid context requested.",
            confidence=0.8,
        )


def test_context_analysis_schema_invalid_confidence():
    with pytest.raises(ValidationError):
        ContextAnalysisResult(
            is_sufficient=True,
            missing_context=[],
            reason="Invalid confidence high",
            confidence=1.5,
        )

    with pytest.raises(ValidationError):
        ContextAnalysisResult(
            is_sufficient=True,
            missing_context=[],
            reason="Invalid confidence low",
            confidence=-0.1,
        )


def test_analyze_context_node_sufficient():
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

    mock_result = ContextAnalysisResult(
        is_sufficient=True,
        missing_context=[],
        reason="Context is complete.",
        confidence=0.95,
    )

    mock_llm = MagicMock()
    mock_structured_llm = MagicMock()
    mock_structured_llm.invoke.return_value = mock_result
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        node_output = analyze_context(state)

    assert node_output["missing_context"] == []
    assert node_output["is_context_sufficient"] is True
    assert node_output["context_analysis_reason"] == "Context is complete."
    assert node_output["context_analysis_confidence"] == 0.95


def test_analyze_context_node_llm_failure_raises():
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
    mock_structured_llm.invoke.side_effect = RuntimeError("OpenAI API unreachable")
    mock_llm.with_structured_output.return_value = mock_structured_llm

    with patch("app.agents.monitoring.nodes.get_llm", return_value=mock_llm):
        with pytest.raises(RuntimeError, match="OpenAI API unreachable"):
            analyze_context(state)
