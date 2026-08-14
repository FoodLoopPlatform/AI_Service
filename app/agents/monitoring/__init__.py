from app.agents.monitoring.graph import get_monitoring_graph
from app.agents.monitoring.state import MonitoringAgentState
from app.schemas.monitoring import MonitoringRequest, MonitoringResponse


def run_monitoring_agent(request: MonitoringRequest) -> MonitoringResponse:
    """Entry point to execute the Inventory Monitoring Agent workflow."""
    initial_state: MonitoringAgentState = {
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

    graph = get_monitoring_graph()
    final_state = graph.invoke(initial_state)

    return MonitoringResponse(
        route=final_state["route"],
        risk_level=final_state["risk_level"],
        reason=final_state["reason"],
        confidence=final_state["confidence"],
    )


__all__ = ["run_monitoring_agent", "get_monitoring_graph"]
