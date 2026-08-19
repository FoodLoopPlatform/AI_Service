from abc import ABC, abstractmethod

from app.config.settings import settings
from app.embeddings.base import EmbeddingProvider
from app.schemas.pricing import PricingProductContext
from app.schemas.pricing_knowledge import PricingKnowledgeItem
from app.vector_store.base import VectorStore


def build_product_query_text(product: PricingProductContext) -> str:
    """Builds a deterministic factual search query string representing current product context."""
    product_name = product.product_name or product.product_id
    category = product.category or "General"
    risk_level = (
        product.risk_assessment.risk_level.value
        if hasattr(product.risk_assessment.risk_level, "value")
        else str(product.risk_assessment.risk_level)
    )

    if hasattr(product.demand, "historical_sales") and hasattr(product.demand.historical_sales, "average_daily_sales"):
        hist_avg = product.demand.historical_sales.average_daily_sales
    else:
        hist_avg = getattr(product.demand, "historical_average_daily_sales", 0.0)

    return (
        f"Product: {product_name}\n"
        f"Product ID: {product.product_id}\n"
        f"Category: {category}\n"
        f"Current inventory quantity: {product.inventory.quantity}\n"
        f"Current price: {product.inventory.current_price}\n"
        f"Original price: {product.inventory.original_price}\n"
        f"Price floor: {product.inventory.price_floor}\n"
        f"Sales velocity: {product.demand.sales_velocity} units/day\n"
        f"Historical average daily sales: {hist_avg} units/day\n"
        f"Hours remaining: {product.expiry.hours_remaining}\n"
        f"Risk level: {risk_level}\n"
        f"Risk reason: {product.risk_assessment.reason}"
    )



class PricingKnowledgeRetriever(ABC):
    """Abstract interface for retrieving store-aware pricing knowledge items."""

    @abstractmethod
    def retrieve(
        self,
        store_id: str,
        products: list[PricingProductContext],
    ) -> list[PricingKnowledgeItem]:
        """Retrieves relevant pricing knowledge items for the specified store and products."""
        pass


class VectorPricingKnowledgeRetriever(PricingKnowledgeRetriever):
    """Production pricing knowledge retriever combining an EmbeddingProvider and VectorStore."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        top_k: int | None = None,
    ):
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store

        top_k_val = top_k if top_k is not None else settings.PRICING_RETRIEVAL_TOP_K
        if top_k_val < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k_val}.")
        self.top_k = top_k_val

    def retrieve(
        self,
        store_id: str,
        products: list[PricingProductContext],
    ) -> list[PricingKnowledgeItem]:
        if not products:
            return []

        if not store_id or not isinstance(store_id, str) or not store_id.strip():
            raise ValueError("store_id is mandatory and must be a non-empty string.")

        try:
            query_texts = [build_product_query_text(p) for p in products]
            query_embeddings = self.embedding_provider.embed_queries(query_texts)

            all_items: list[PricingKnowledgeItem] = []
            for product, query_vec in zip(products, query_embeddings):
                items = self.vector_store.search(
                    query_vec,
                    store_id=store_id,
                    product_id=product.product_id,
                    category=product.category,
                    top_k=self.top_k,
                )
                all_items.extend(items)

            return all_items
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Vector pricing knowledge retrieval failed (%s), defaulting to empty knowledge context.", e)
            return []


class DefaultPricingKnowledgeRetriever(PricingKnowledgeRetriever):
    """Default implementation when vector database is not attached.
    
    Returns an empty list to avoid introducing fake knowledge.
    """

    def retrieve(
        self,
        store_id: str,
        products: list[PricingProductContext],
    ) -> list[PricingKnowledgeItem]:
        return []


class InMemoryPricingKnowledgeRetriever(PricingKnowledgeRetriever):
    """In-memory retriever for testing store and product isolation."""

    def __init__(self, records: list[PricingKnowledgeItem] | None = None):
        self.records: list[PricingKnowledgeItem] = records or []

    def retrieve(
        self,
        store_id: str,
        products: list[PricingProductContext],
    ) -> list[PricingKnowledgeItem]:
        requested_product_ids = {p.product_id for p in products}
        return [
            item
            for item in self.records
            if item.store_id == store_id and item.product_id in requested_product_ids
        ]


def group_knowledge_by_product(
    products: list[PricingProductContext],
    knowledge: list[PricingKnowledgeItem] | None,
) -> dict[str, list[PricingKnowledgeItem]]:
    """Groups retrieved knowledge items by product_id.
    
    Ensures every requested product_id exists as a key in the returned dictionary,
    defaulting to an empty list if no knowledge items exist for that product.
    """
    grouped: dict[str, list[PricingKnowledgeItem]] = {
        p.product_id: [] for p in products
    }

    if knowledge:
        for item in knowledge:
            if item.product_id in grouped:
                grouped[item.product_id].append(item)

    return grouped


def get_production_pricing_knowledge_retriever() -> PricingKnowledgeRetriever:
    """Factory function constructing VectorPricingKnowledgeRetriever with configured providers."""
    from app.embeddings.factory import get_embedding_provider
    from app.vector_store.factory import get_vector_store

    return VectorPricingKnowledgeRetriever(
        embedding_provider=get_embedding_provider(),
        vector_store=get_vector_store(),
    )
