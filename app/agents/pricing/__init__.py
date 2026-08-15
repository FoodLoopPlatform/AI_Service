from app.agents.pricing.graph import get_pricing_graph
from app.agents.pricing.state import PricingAgentState
from app.policies.store_policy import get_action_reason, get_action_requirement
from app.schemas.pricing import PricingBatchRequest, PricingBatchResponse, PricingDecision


def run_pricing_agent(request: PricingBatchRequest) -> PricingBatchResponse:
    """Entry point to execute the batch Pricing Agent workflow."""
    initial_state: PricingAgentState = {
        "store_id": request.store_id,
        "store_policy": request.store_policy,
        "products": request.products,
        "knowledge": None,
        "knowledge_by_product": None,
        "pricing_decisions": None,
    }

    graph = get_pricing_graph()
    final_state = graph.invoke(initial_state)

    raw_decisions = final_state.get("pricing_decisions")
    if raw_decisions is None:
        raise ValueError("Pricing Agent failed to produce valid pricing decisions.")

    # Deterministically derive policy interpretation if store_policy is present
    action_req = get_action_requirement(request.store_policy) if request.store_policy else None
    action_reas = get_action_reason(request.store_policy) if request.store_policy else None

    final_decisions = []
    for dec in raw_decisions:
        final_decisions.append(
            PricingDecision(
                product_id=dec.product_id,
                discount_percentage=dec.discount_percentage,
                reason=dec.reason,
                confidence=dec.confidence,
                action_requirement=action_req,
                action_reason=action_reas,
            )
        )

    return PricingBatchResponse(
        store_id=request.store_id,
        decisions=final_decisions,
    )


__all__ = ["run_pricing_agent", "get_pricing_graph"]
