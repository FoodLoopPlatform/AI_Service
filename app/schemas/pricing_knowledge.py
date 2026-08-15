from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class PricingKnowledgeItem(BaseModel):
    """Schema for a store-aware and product-specific domain knowledge item."""

    model_config = ConfigDict(populate_by_name=True)

    product_id: str = Field(..., min_length=1, description="Associated product ID")
    store_id: str = Field(..., min_length=1, description="Originating store ID")
    content: str = Field(..., min_length=1, description="Knowledge content or guideline text")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata dictionary for additional context",
    )
    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Relevance score constrained between 0.0 and 1.0 inclusive.",
    )
