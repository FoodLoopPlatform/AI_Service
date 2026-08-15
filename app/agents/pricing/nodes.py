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


def pricing_recommendation(state: PricingAgentState) -> dict:
    """Graph node: Processes batch request via LLM structured output and validates product mapping."""
    store_id = state["store_id"]
    products = state["products"]
    knowledge_by_product = state.get("knowledge_by_product")

    user_prompt = format_pricing_user_prompt(
        store_id=store_id,
        products=products,
        knowledge_by_product=knowledge_by_product,
    )

    llm = get_llm().with_structured_output(PricingBatchLLMResult)

    messages = [
        {"role": "system", "content": PRICING_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
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
