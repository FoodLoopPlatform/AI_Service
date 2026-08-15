from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, model_validator



class Route(str, Enum):
    PRICING = "PRICING"
    NO_ACTION = "NO_ACTION"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ProductMetadata(BaseModel):
    id: str
    name: str
    category: str


class InventoryMetrics(BaseModel):
    quantity: int = Field(..., ge=0)
    original_price: float = Field(..., ge=0.0)
    current_price: float = Field(..., ge=0.0)
    price_floor: float = Field(..., ge=0.0)


class HistoricalSales(BaseModel):
    average_daily_sales: float = Field(..., ge=0.0)
    weekday_average: float | None = Field(default=None, ge=0.0)
    weekend_average: float | None = Field(default=None, ge=0.0)


class DemandContext(BaseModel):
    sales_velocity: float = Field(..., ge=0.0)
    historical_sales: HistoricalSales


class ExpiryContext(BaseModel):
    expires_at: datetime
    hours_remaining: float = Field(..., ge=0.0)


class LocationContext(BaseModel):
    latitude: float
    longitude: float
    store_id: str


from app.schemas.store_policy import StorePolicy


class MonitoringRequest(BaseModel):
    product: ProductMetadata
    inventory: InventoryMetrics
    demand: DemandContext
    expiry: ExpiryContext
    location: LocationContext
    timestamp: datetime
    store_policy: StorePolicy | None = Field(default=None, description="Optional backend store policy configuration.")

    @model_validator(mode="after")
    def validate_store_id_consistency(self) -> "MonitoringRequest":
        if self.store_policy is not None:
            if self.location.store_id != self.store_policy.store_id:
                raise ValueError(
                    f"store_id mismatch: location store_id ('{self.location.store_id}') "
                    f"does not match store_policy store_id ('{self.store_policy.store_id}')."
                )
        return self



class MonitoringResponse(BaseModel):
    route: Route
    risk_level: RiskLevel
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
