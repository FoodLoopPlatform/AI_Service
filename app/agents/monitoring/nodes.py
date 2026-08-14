from app.agents.monitoring.prompts import (
    CONTEXT_ANALYSIS_PROMPT,
    RISK_ASSESSMENT_PROMPT,
)
from app.agents.monitoring.risk_signals import calculate_risk_signals
from app.agents.monitoring.state import MonitoringAgentState
from app.llm import get_llm
from app.schemas.context_analysis import ContextAnalysisResult
from app.schemas.monitoring import RiskLevel, Route
from app.schemas.risk_assessment import RiskAssessmentResult
from app.tools.events import get_local_events
from app.tools.weather import get_weather_forecast


class RiskAssessmentMissingError(Exception):
    """Raised when determine_route is executed without a valid risk_level in state."""

    pass


class StateInvalidError(Exception):
    """Raised when an internal agent node encounters an incomplete or invalid state."""

    pass


def analyze_context(state: MonitoringAgentState) -> dict:
    """Node: Evaluates monitoring request context sufficiency using LLM structured output."""
    request = state.get("request")
    if request is None:
        raise StateInvalidError("Cannot analyze context: MonitoringRequest is missing from state.")

    request_json = request.model_dump_json(indent=2)

    llm = get_llm()
    structured_llm = llm.with_structured_output(ContextAnalysisResult)
    messages = CONTEXT_ANALYSIS_PROMPT.format_messages(request_json=request_json)
    result: ContextAnalysisResult = structured_llm.invoke(messages)

    if result is None:
        raise StateInvalidError("Context analysis LLM returned no result.")

    return {
        "missing_context": result.missing_context,
        "is_context_sufficient": result.is_sufficient,
        "context_analysis_reason": result.reason,
        "context_analysis_confidence": result.confidence,
    }


def fetch_weather_context(state: MonitoringAgentState) -> dict:
    """Node: Fetches external weather forecast context for the request time window."""
    request = state.get("request")
    if request is None:
        raise StateInvalidError("Cannot fetch weather: MonitoringRequest is missing from state.")

    weather_context = get_weather_forecast(
        latitude=request.location.latitude,
        longitude=request.location.longitude,
        from_time=request.timestamp,
        to_time=request.expiry.expires_at,
    )
    return {"weather_context": weather_context}


def fetch_local_events_context(state: MonitoringAgentState) -> dict:
    """Node: Fetches external local events context for the request time window."""
    request = state.get("request")
    if request is None:
        raise StateInvalidError("Cannot fetch local events: MonitoringRequest is missing from state.")

    events_context = get_local_events(
        latitude=request.location.latitude,
        longitude=request.location.longitude,
        from_time=request.timestamp,
        to_time=request.expiry.expires_at,
    )
    return {"events_context": events_context}


def assess_risk(state: MonitoringAgentState) -> dict:
    """Node: Assesses overall inventory risk using deterministic signals and LLM structured output."""
    request = state.get("request")
    if request is None:
        raise StateInvalidError("Cannot assess risk: MonitoringRequest is missing from state.")

    request_json = request.model_dump_json(indent=2)

    risk_signals = calculate_risk_signals(request)
    risk_signals_json = risk_signals.model_dump_json(indent=2)

    weather_context = state.get("weather_context")
    weather_context_json = (
        weather_context.model_dump_json(indent=2)
        if weather_context is not None
        else "None"
    )

    events_context = state.get("events_context")
    events_context_json = (
        events_context.model_dump_json(indent=2)
        if events_context is not None
        else "None"
    )

    llm = get_llm()
    structured_llm = llm.with_structured_output(RiskAssessmentResult)
    messages = RISK_ASSESSMENT_PROMPT.format_messages(
        request_json=request_json,
        risk_signals_json=risk_signals_json,
        weather_context_json=weather_context_json,
        events_context_json=events_context_json,
    )
    result: RiskAssessmentResult = structured_llm.invoke(messages)

    if result is None or result.risk_level is None:
        raise StateInvalidError("Risk assessment output is incomplete.")

    return {
        "risk_signals": risk_signals,
        "risk_level": result.risk_level,
        "reason": result.reason,
        "confidence": result.confidence,
    }


def determine_route(state: MonitoringAgentState) -> dict:
    """Node: Deterministically routes request based on assessed risk level."""
    request = state.get("request")
    if request is None:
        raise StateInvalidError("Cannot determine route: MonitoringRequest is missing from state.")

    risk_level = state.get("risk_level")
    if risk_level is None:
        raise RiskAssessmentMissingError(
            "Cannot determine route: risk_level is missing from agent state."
        )

    if risk_level == RiskLevel.LOW:
        return {"route": Route.NO_ACTION}

    return {"route": Route.PRICING}
