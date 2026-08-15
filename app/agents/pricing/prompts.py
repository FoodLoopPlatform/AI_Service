from app.schemas.pricing import PricingProductContext
from app.schemas.pricing_knowledge import PricingKnowledgeItem

PRICING_SYSTEM_PROMPT = """You are the Pricing Recommendation Agent for the FoodLoop AI Service.

Your sole responsibility is to evaluate a batch request belonging to a single store containing one or more products. For each product, evaluate its inventory, demand, sales velocity, expiry context, authoritative risk assessment from the Inventory Monitoring Agent, and any associated historical knowledge for that specific product to recommend a discount percentage.

RULES & BOUNDARIES:
1. STORE SCOPE & ISOLATION: The batch request belongs to a single store. Each product has its own unique product_id, sales velocity, historical sales average, inventory metrics, and expiry details. Decisions MUST remain isolated per product.
2. HISTORICAL KNOWLEDGE ISOLATION: Historical knowledge listed under a product belongs ONLY to that product. Never associate or apply knowledge from one product to another product.
3. DO NOT COMPARE UNRELATED PRODUCTS: Never compare raw sales velocity between unrelated products as a direct measure of risk. The deterministic risk assessment provided by the Inventory Monitoring Agent is authoritative.
4. EXACT PRODUCT MATCHING: You MUST return exactly one PricingDecision for every input product in the batch. The product_id in each decision MUST match the input product_id exactly.
5. STRICT SAFETY BOUNDARY: The recommended discount_percentage MUST be between 0 and 15 (inclusive) for every product. Never recommend a discount less than 0 or greater than 15.
6. NO FINAL PRICE CALCULATION: Do NOT calculate or return final monetary prices, recommended prices, or prices after discount. Financial execution is strictly handled by the .NET backend.
7. NO PRICE FLOOR MODIFICATION: Current price and price floor are context only. Do NOT modify or adjust price floors.
8. NO DONATION OR AUTOMATION DECISIONS: Do NOT make donation decisions or autonomous execution decisions.
9. NO CHAIN OF THOUGHT: Do NOT output chain-of-thought, reasoning steps, or internal analysis.
10. AUDITABLE RATIONALE: Keep the 'reason' field concise, clear, and objective for compliance auditing.
"""


from app.agents.pricing.signals import calculate_pricing_signals


def format_pricing_user_prompt(
    store_id: str,
    products: list[PricingProductContext],
    knowledge_by_product: dict[str, list[PricingKnowledgeItem]] | None = None,
) -> str:
    """Formats the batch request into a detailed prompt string for the LLM."""
    products_formatted = []
    for idx, p in enumerate(products, start=1):
        signals = calculate_pricing_signals(p)
        cov_days_str = (
            f"{signals.inventory_coverage_days:.2f} days"
            if signals.inventory_coverage_days is not None
            else "Infinite / Unconsumed (0 sales velocity)"
        )
        demand_ratio_str = (
            f"{signals.demand_ratio:.2f}"
            if signals.demand_ratio is not None
            else "N/A (Zero sales baseline)"
        )

        weather_info = "None provided"
        if p.weather_context and p.weather_context.forecast:
            forecasts = [
                f"{f.condition} ({f.temperature}°C, precip prob: {f.precipitation_probability * 100:.0f}%)"
                for f in p.weather_context.forecast
            ]
            weather_info = "; ".join(forecasts)

        events_info = "None provided"
        if p.events_context and getattr(p.events_context, "holidays", None):
            holidays = [
                f"{h.name} ({h.date}, National: {h.national_holiday})"
                for h in p.events_context.holidays
            ]
            events_info = "; ".join(holidays)

        # Retrieve knowledge scoped strictly to this product_id
        prod_knowledge = (knowledge_by_product or {}).get(p.product_id, [])
        if prod_knowledge:
            k_items = [
                f"[Relevance: {item.relevance_score:.2f}] {item.content}"
                for item in prod_knowledge
            ]
            prod_knowledge_info = "\n  ".join(k_items)
        else:
            prod_knowledge_info = "None provided"

        p_str = f"""PRODUCT {idx}:
- Product ID: {p.product_id}
- Name: {p.product_name or 'N/A'}
- Category: {p.category or 'N/A'}
- Inventory Quantity: {p.inventory.quantity}
- Original Price: {p.inventory.original_price}
- Current Price: {p.inventory.current_price}
- Price Floor: {p.inventory.price_floor}
- Expiry Date: {p.expiry.expires_at.isoformat()}
- Hours Remaining: {p.expiry.hours_remaining:.1f}
- Sales Velocity: {p.demand.sales_velocity:.2f}
- Historical Avg Daily Sales: {p.demand.historical_sales.average_daily_sales:.2f}
- Deterministic Signals:
  * Inventory Coverage: {cov_days_str} (Pressure: {signals.inventory_pressure.value})
  * Demand Ratio: {demand_ratio_str} (Pressure: {signals.demand_pressure.value})
  * Expiry Pressure: {signals.expiry_pressure.value}
- Monitoring Risk Level: {p.risk_assessment.risk_level.value}
- Monitoring Risk Reason: {p.risk_assessment.reason}
- Monitoring Risk Confidence: {p.risk_assessment.confidence:.2f}
- Weather Context: {weather_info}
- Public Holidays / Events Context: {events_info}
- Historical Knowledge:
  {prod_knowledge_info}"""
        products_formatted.append(p_str)

    all_products_text = "\n\n".join(products_formatted)

    prompt = f"""BATCH REQUEST CONTEXT:
- Store ID: {store_id}
- Total Products in Batch: {len(products)}

ITEMS IN BATCH & HISTORICAL KNOWLEDGE:
{all_products_text}

Provide exactly one PricingDecision per input product in the batch, matching each product_id.
Return only structured output conforming to PricingBatchLLMResult.
"""
    return prompt
