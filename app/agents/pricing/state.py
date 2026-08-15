from typing import TypedDict

from app.schemas.pricing import PricingDecision, PricingProductContext
from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.schemas.store_policy import StorePolicy


class PricingAgentState(TypedDict):
    """LangGraph state schema for the batch Pricing Agent workflow."""

    store_id: str
    store_policy: StorePolicy | None
    products: list[PricingProductContext]
    knowledge: list[PricingKnowledgeItem] | None
    knowledge_by_product: dict[str, list[PricingKnowledgeItem]] | None
    pricing_decisions: list[PricingDecision] | None

