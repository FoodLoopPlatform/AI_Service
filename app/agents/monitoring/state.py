from typing import TypedDict

from app.agents.monitoring.risk_signals import RiskSignals
from app.schemas.context_analysis import AllowedContext
from app.schemas.monitoring import MonitoringRequest, RiskLevel, Route
from app.tools.events import LocalEventsContext
from app.tools.weather import WeatherContext


class MonitoringAgentState(TypedDict):
    """LangGraph state schema for the Inventory Monitoring Agent workflow."""

    request: MonitoringRequest
    risk_signals: RiskSignals | None
    risk_level: RiskLevel | None
    missing_context: list[AllowedContext] | None
    is_context_sufficient: bool | None
    context_analysis_reason: str | None
    context_analysis_confidence: float | None
    weather_context: WeatherContext | None
    events_context: LocalEventsContext | None
    route: Route | None
    reason: str | None
    confidence: float | None
