from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class ExpiryPressure(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class InventoryPressure(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


class DemandPressure(str, Enum):
    STRONG_DEMAND = "STRONG_DEMAND"
    NORMAL_DEMAND = "NORMAL_DEMAND"
    WEAK_DEMAND = "WEAK_DEMAND"
    VERY_WEAK_DEMAND = "VERY_WEAK_DEMAND"
    NO_CURRENT_SALES = "NO_CURRENT_SALES"
    NO_DEMAND_BASELINE = "NO_DEMAND_BASELINE"


class PricingSignals(BaseModel):
    """Deterministic pricing signals calculated from product operational context."""

    model_config = ConfigDict(extra="forbid")

    inventory_coverage_days: float | None = Field(
        default=None,
        description="Days of inventory coverage (quantity / sales_velocity). None if sales_velocity is 0.",
    )
    inventory_pressure: InventoryPressure = Field(
        ..., description="Categorized inventory pressure based on coverage days."
    )
    demand_ratio: float | None = Field(
        default=None,
        description="Ratio of sales velocity to historical average daily sales. None if historical average is 0.",
    )
    demand_pressure: DemandPressure = Field(
        ..., description="Categorized demand pressure."
    )
    expiry_pressure: ExpiryPressure = Field(
        ..., description="Categorized expiry pressure based on hours remaining."
    )
    sales_velocity_zero: bool = Field(
        ..., description="True if current sales_velocity == 0."
    )
    historical_average_zero: bool = Field(
        ..., description="True if historical_average_daily_sales == 0."
    )
