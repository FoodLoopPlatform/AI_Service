from datetime import datetime
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class Outcome(str, Enum):
    SOLD_OUT = "SOLD_OUT"
    PARTIALLY_SOLD = "PARTIALLY_SOLD"
    UNSOLD = "UNSOLD"
    EXPIRED = "EXPIRED"


class HistoricalPricingEvent(BaseModel):
    """Canonical model for a recorded historical pricing snapshot event."""

    model_config = ConfigDict(populate_by_name=True)

    event_id: str = Field(..., min_length=1, description="Unique event identifier")
    store_id: str = Field(..., min_length=1, description="Store identifier")
    product_id: str = Field(..., min_length=1, description="Product identifier")
    category: str = Field(..., min_length=1, description="Product category")
    recorded_at: datetime = Field(..., description="Timestamp when event was recorded")

    # Operational snapshot
    quantity: float = Field(..., ge=0.0, description="Available inventory quantity at event time")
    current_price: float = Field(..., ge=0.0, description="Current price at event time")
    original_price: float = Field(..., ge=0.0, description="Original un-discounted price")
    price_floor: float = Field(..., ge=0.0, description="Enforced price floor limit")

    # Demand snapshot
    sales_velocity: float = Field(..., ge=0.0, description="Recent sales velocity")
    historical_average_daily_sales: float = Field(
        ..., ge=0.0, description="Historical average daily sales"
    )

    # Expiry snapshot
    hours_remaining: float = Field(..., ge=0.0, description="Hours remaining before expiry")

    # Pricing action
    discount_percentage: float = Field(
        ...,
        ge=0.0,
        le=15.0,
        description="Historical discount percentage applied (0 to 15)",
    )

    # Outcome
    units_sold_after_discount: float = Field(
        ..., ge=0.0, description="Units sold following discount application"
    )
    sell_through_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Sell-through rate between 0.0 and 1.0"
    )
    outcome: Outcome = Field(..., description="Categorical outcome of the pricing action")
