from langgraph.graph import END, START, StateGraph

from app.agents.pricing.nodes import (
    pricing_recommendation,
    retrieve_pricing_knowledge,
)
from app.agents.pricing.state import PricingAgentState


def get_pricing_graph():
    """Constructs and returns the compiled LangGraph workflow for the Pricing Agent."""
    workflow = StateGraph(PricingAgentState)

    workflow.add_node("retrieve_pricing_knowledge", retrieve_pricing_knowledge)
    workflow.add_node("pricing_recommendation", pricing_recommendation)

    workflow.add_edge(START, "retrieve_pricing_knowledge")
    workflow.add_edge("retrieve_pricing_knowledge", "pricing_recommendation")
    workflow.add_edge("pricing_recommendation", END)

    return workflow.compile()
