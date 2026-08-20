from app.agents.pricing.prompts import PRICING_SYSTEM_PROMPT, format_pricing_user_prompt
from app.agents.pricing.retriever import (
    PricingKnowledgeRetriever,
    get_production_pricing_knowledge_retriever,
    group_knowledge_by_product,
)
from app.agents.pricing.state import PricingAgentState
from app.llm.factory import get_llm
from app.schemas.pricing import PricingBatchLLMResult

_retriever_instance: PricingKnowledgeRetriever | None = None


def get_pricing_knowledge_retriever() -> PricingKnowledgeRetriever:
    """Returns the current pricing knowledge retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = get_production_pricing_knowledge_retriever()
    return _retriever_instance


def set_pricing_knowledge_retriever(retriever: PricingKnowledgeRetriever | None) -> None:
    """Sets a custom pricing knowledge retriever instance."""
    global _retriever_instance
    _retriever_instance = retriever



def retrieve_pricing_knowledge(state: PricingAgentState) -> dict:
    """Graph node: Retrieves store-aware knowledge items and groups them per product_id."""
    store_id = state.get("store_id")
    if not store_id or not isinstance(store_id, str) or not store_id.strip():
        raise ValueError("store_id is mandatory and must be a non-empty string.")

    products = state.get("products") or []
    retriever = get_pricing_knowledge_retriever()

    knowledge = retriever.retrieve(store_id=store_id, products=products)
    knowledge_by_product = group_knowledge_by_product(products=products, knowledge=knowledge)

    return {
        "knowledge": knowledge,
        "knowledge_by_product": knowledge_by_product,
    }


import logging

logger = logging.getLogger(__name__)


def pricing_recommendation(state: PricingAgentState) -> dict:
    """Graph node: Processes batch request via LLM structured output with deterministic fallback on failure."""
    store_id = state["store_id"]
    products = state["products"]
    knowledge_by_product = state.get("knowledge_by_product")

    try:
        user_prompt = format_pricing_user_prompt(
            store_id=store_id,
            products=products,
            knowledge_by_product=knowledge_by_product,
        )

        llm = get_llm().with_structured_output(PricingBatchLLMResult)

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=PRICING_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        llm_result: PricingBatchLLMResult = llm.invoke(messages)

        # --- Strict Batch Mapping Validation ---
        expected_product_ids = [p.product_id for p in products]
        expected_set = set(expected_product_ids)
        returned_product_ids = [d.product_id for d in llm_result.decisions]

        # Check 1: Unknown product IDs
        unknown_ids = set(returned_product_ids) - expected_set
        if unknown_ids:
            raise ValueError(
                f"LLM returned pricing decisions for unknown product_id(s): {sorted(list(unknown_ids))}"
            )

        # Check 2: Duplicate product IDs
        seen = set()
        duplicates = set()
        for pid in returned_product_ids:
            if pid in seen:
                duplicates.add(pid)
            seen.add(pid)
        if duplicates:
            raise ValueError(
                f"LLM returned duplicate pricing decisions for product_id(s): {sorted(list(duplicates))}"
            )

        # Check 3: Missing product decisions
        missing_ids = expected_set - set(returned_product_ids)
        if missing_ids:
            raise ValueError(
                f"LLM failed to return pricing decision(s) for product_id(s): {sorted(list(missing_ids))}"
            )

        return {"pricing_decisions": llm_result.decisions}

    except Exception as e:
        logger.warning(
            "Pricing recommendation LLM failed (%s), falling back to deterministic business pricing logic.",
            e,
        )

    # --- Deterministic Fallback Pricing Logic ---
    from app.schemas.pricing import PricingDecision

    fallback_decisions: list[PricingDecision] = []
    for p in products:
        risk_level_str = "MEDIUM"
        if p.risk_assessment and p.risk_assessment.risk_level:
            rl = p.risk_assessment.risk_level
            risk_level_str = str(getattr(rl, "value", rl)).upper()

        hours = p.expiry.hours_remaining

        if risk_level_str == "CRITICAL" or hours <= 12:
            discount = 15.0
            reason = f"Critical risk / {hours:.1f}h remaining: max discount applied (rule-based fallback)."
        elif risk_level_str == "HIGH" or hours <= 24:
            discount = 10.0
            reason = f"High risk / {hours:.1f}h remaining: accelerated discount applied (rule-based fallback)."
        elif risk_level_str == "MEDIUM" or hours <= 48:
            discount = 5.0
            reason = f"Moderate risk / {hours:.1f}h remaining: standard discount applied (rule-based fallback)."
        else:
            discount = 0.0
            reason = "Low risk / stable shelf-life: no discount needed (rule-based fallback)."

        fallback_decisions.append(
            PricingDecision(
                product_id=p.product_id,
                discount_percentage=discount,
                reason=reason,
                confidence=0.85,
            )
        )

    return {"pricing_decisions": fallback_decisions}
