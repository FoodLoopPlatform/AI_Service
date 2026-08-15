from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config.settings import settings
from app.policies.store_policy import ActionRequirement
from app.schemas.monitoring import (
    DemandContext,
    ExpiryContext,
    InventoryMetrics,
)
from app.schemas.risk_assessment import RiskAssessmentResult
from app.schemas.store_policy import StorePolicy
from app.tools.events import LocalEventsContext
from app.tools.weather import WeatherContext


class PricingProductContext(BaseModel):
    """Product pricing context for a single item within a batch request."""

    model_config = ConfigDict(populate_by_name=True)

    product_id: str = Field(..., min_length=1, description="Unique product ID")
    product_name: str | None = Field(default=None, description="Optional product name")
    category: str | None = Field(default=None, description="Optional product category")
    inventory: InventoryMetrics
    demand: DemandContext
    expiry: ExpiryContext
    risk_assessment: RiskAssessmentResult
    weather_context: WeatherContext | None = Field(
        default=None, alias="weather"
    )
    events_context: LocalEventsContext | None = Field(
        default=None, alias="local_events_context"
    )


class PricingBatchRequest(BaseModel):
    """Input contract for batch pricing recommendation request."""

    model_config = ConfigDict(populate_by_name=True)

    store_id: str = Field(..., min_length=1, description="Originating store ID")
    store_policy: StorePolicy | None = Field(
        default=None, description="Optional backend store policy configuration."
    )
    products: list[PricingProductContext] = Field(
        ...,
        min_length=1,
        description="List of product pricing context items (must contain at least 1).",
    )

    @model_validator(mode="after")
    def validate_batch_request(self) -> "PricingBatchRequest":
        if self.store_policy is not None:
            if self.store_id != self.store_policy.store_id:
                raise ValueError(
                    f"store_id mismatch: request store_id ('{self.store_id}') "
                    f"does not match store_policy store_id ('{self.store_policy.store_id}')."
                )

        if len(self.products) > settings.MAX_PRICING_BATCH_SIZE:
            raise ValueError(
                f"Batch size ({len(self.products)}) exceeds maximum allowed batch size of {settings.MAX_PRICING_BATCH_SIZE}."
            )

        product_ids = [p.product_id for p in self.products]
        if len(product_ids) != len(set(product_ids)):
            raise ValueError("Duplicate product_id found in batch pricing request.")

        return self


class PricingDecision(BaseModel):
    """Output decision for a single product within a batch recommendation.
    
    The Pricing Agent ONLY recommends a discount percentage.
    It MUST NOT calculate or return final monetary prices, recommended prices,
    price floors, donation decisions, or automation decisions.
    """

    model_config = ConfigDict(extra="forbid")

    product_id: str = Field(..., min_length=1, description="Preserved product ID")
    discount_percentage: float = Field(
        ...,
        ge=0.0,
        le=15.0,
        description="Recommended discount percentage between 0 and 15 inclusive.",
    )
    reason: str = Field(
        ...,
        min_length=1,
        description="Concise auditable rationale for the discount recommendation.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0 inclusive.",
    )
    action_requirement: ActionRequirement | None = Field(
        default=None,
        description="Deterministic action policy requirement derived from backend store policy.",
    )
    action_reason: str | None = Field(
        default=None,
        description="Deterministic action policy rationale explaining operating mode semantics.",
    )


class PricingBatchLLMResult(BaseModel):
    """LLM structured output contract containing decisions for all products in a batch."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[PricingDecision] = Field(
        ...,
        description="List of pricing decisions returned by the LLM.",
    )


class PricingBatchResponse(BaseModel):
    """Response contract for batch pricing recommendation."""

    model_config = ConfigDict(populate_by_name=True)

    store_id: str = Field(..., min_length=1, description="Originating store ID")
    decisions: list[PricingDecision] = Field(
        ...,
        description="List of pricing decisions matching each input product.",
    )
