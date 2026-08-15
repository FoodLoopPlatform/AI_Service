from app.schemas.historical_pricing import HistoricalPricingEvent
from app.schemas.pricing_knowledge_document import PricingKnowledgeDocument


def build_pricing_knowledge_document(
    event: HistoricalPricingEvent,
) -> PricingKnowledgeDocument:
    """Builds a deterministic natural language PricingKnowledgeDocument from a HistoricalPricingEvent.
    
    This function is a pure, deterministic application-layer function with no external dependencies
    (no LLM, vector database, or HTTP network calls). It converts historical facts into a retrieval-ready
    document suitable for semantic search and vector database filtering.
    """
    recorded_at_iso = event.recorded_at.isoformat()

    content = (
        f"Historical pricing event for product {event.product_id} (Category: {event.category}) "
        f"at store {event.store_id} recorded at {recorded_at_iso}. "
        f"The product had {event.hours_remaining:.1f} hours remaining before expiry. "
        f"Inventory quantity was {event.quantity:.1f} units. "
        f"Current price was {event.current_price:.2f}, original price was {event.original_price:.2f}, "
        f"and price floor was {event.price_floor:.2f}. "
        f"Sales velocity was {event.sales_velocity:.2f} units per day with a historical average of "
        f"{event.historical_average_daily_sales:.2f} units per day. "
        f"The applied historical discount was {event.discount_percentage:.1f}%. "
        f"After the discount, {event.units_sold_after_discount:.1f} units were sold, "
        f"resulting in a sell-through rate of {event.sell_through_rate:.2f} and outcome {event.outcome.value}."
    )

    metadata = {
        "store_id": event.store_id,
        "product_id": event.product_id,
        "category": event.category,
        "outcome": event.outcome.value,
        "recorded_at": recorded_at_iso,
        "discount_percentage": event.discount_percentage,
    }

    return PricingKnowledgeDocument(
        document_id=f"doc-{event.event_id}",
        store_id=event.store_id,
        product_id=event.product_id,
        content=content,
        metadata=metadata,
    )
