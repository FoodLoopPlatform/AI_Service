from langgraph.graph import END, START, StateGraph

from app.agents.monitoring.nodes import (
    analyze_context,
    assess_risk,
    determine_route,
    fetch_local_events_context,
    fetch_weather_context,
)
from app.agents.monitoring.state import MonitoringAgentState


def route_after_context_analysis(state: MonitoringAgentState) -> str:
    missing = state.get("missing_context") or []
    missing_set = {
        item.value if hasattr(item, "value") else str(item) for item in missing
    }
    if "weather" in missing_set:
        return "fetch_weather_context"
    if "local_events" in missing_set:
        return "fetch_local_events_context"
    return "assess_risk"


def route_after_weather(state: MonitoringAgentState) -> str:
    missing = state.get("missing_context") or []
    missing_set = {
        item.value if hasattr(item, "value") else str(item) for item in missing
    }
    if "local_events" in missing_set:
        return "fetch_local_events_context"
    return "assess_risk"


def get_monitoring_graph():
    """Constructs and returns the compiled LangGraph workflow for the Monitoring Agent."""
    workflow = StateGraph(MonitoringAgentState)

    workflow.add_node("analyze_context", analyze_context)
    workflow.add_node("fetch_weather_context", fetch_weather_context)
    workflow.add_node("fetch_local_events_context", fetch_local_events_context)
    workflow.add_node("assess_risk", assess_risk)
    workflow.add_node("determine_route", determine_route)

    workflow.add_edge(START, "analyze_context")

    workflow.add_conditional_edges(
        "analyze_context",
        route_after_context_analysis,
        {
            "fetch_weather_context": "fetch_weather_context",
            "fetch_local_events_context": "fetch_local_events_context",
            "assess_risk": "assess_risk",
        },
    )

    workflow.add_conditional_edges(
        "fetch_weather_context",
        route_after_weather,
        {
            "fetch_local_events_context": "fetch_local_events_context",
            "assess_risk": "assess_risk",
        },
    )

    workflow.add_edge("fetch_local_events_context", "assess_risk")
    workflow.add_edge("assess_risk", "determine_route")
    workflow.add_edge("determine_route", END)

    return workflow.compile()
