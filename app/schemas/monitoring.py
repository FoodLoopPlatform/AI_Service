from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, model_validator

from typing import Any


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
    name: str = Field(default="Unknown Product")
    category: str = Field(default="General")

    @model_validator(mode="before")
    @classmethod
    def normalize_product_metadata(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "id" not in data and "product_id" in data:
                data["id"] = data["product_id"]
            if "name" not in data and "product_name" in data:
                data["name"] = data["product_name"]
        return data


class InventoryMetrics(BaseModel):
    quantity: int = Field(..., ge=0)
    original_price: float = Field(..., ge=0.0)
    current_price: float = Field(..., ge=0.0)
    price_floor: float = Field(..., ge=0.0)


class HistoricalSales(BaseModel):
    average_daily_sales: float = Field(default=0.0, ge=0.0)
    weekday_average: float | None = Field(default=None, ge=0.0)
    weekend_average: float | None = Field(default=None, ge=0.0)


class DemandContext(BaseModel):
    sales_velocity: float = Field(..., ge=0.0)
    historical_sales: HistoricalSales

    @model_validator(mode="before")
    @classmethod
    def normalize_demand_context(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "historical_sales" in data and isinstance(data["historical_sales"], (int, float)):
                data["historical_sales"] = {"average_daily_sales": float(data["historical_sales"])}
            elif "historical_average_daily_sales" in data and "historical_sales" not in data:
                data["historical_sales"] = {"average_daily_sales": float(data["historical_average_daily_sales"])}
        return data


class ExpiryContext(BaseModel):
    expires_at: datetime
    hours_remaining: float = Field(..., ge=0.0)


class LocationContext(BaseModel):
    latitude: float = Field(default=30.0444)
    longitude: float = Field(default=31.2357)
    store_id: str


from app.schemas.store_policy import StorePolicy


from pydantic import ConfigDict

class MonitoringRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "store_id": "store-cairo-01",
                "product_id": "prod-100",
                "product_name": "Fresh Milk 1L",
                "category": "Dairy",
                "inventory": {
                    "quantity": 25,
                    "original_price": 35.0,
                    "current_price": 35.0,
                    "price_floor": 20.0,
                },
                "demand": {
                    "sales_velocity": 2.5,
                    "historical_average_daily_sales": 5.0,
                },
                "expiry": {
                    "expires_at": "2026-08-20T12:00:00Z",
                    "hours_remaining": 18.0,
                },
                "location": {
                    "latitude": 30.0444,
                    "longitude": 31.2357,
                    "store_id": "store-cairo-01",
                },
                "timestamp": "2026-08-19T18:00:00Z",
            }
        }
    )

    product: ProductMetadata
    inventory: InventoryMetrics
    demand: DemandContext
    expiry: ExpiryContext
    location: LocationContext
    timestamp: datetime
    store_policy: StorePolicy | None = Field(default=None, description="Optional backend store policy configuration.")

    @model_validator(mode="before")
    @classmethod
    def normalize_monitoring_request(cls, data: Any) -> Any:
        if isinstance(data, dict):
            top_store_id = data.get("store_id")
            loc = data.get("location")
            if isinstance(loc, dict):
                if top_store_id and "store_id" not in loc:
                    loc["store_id"] = top_store_id
            elif not loc and top_store_id:
                data["location"] = {"store_id": top_store_id, "latitude": 30.0444, "longitude": 31.2357}
            
            if "product" not in data and ("product_id" in data or "id" in data):
                pid = data.get("product_id") or data.get("id")
                pname = data.get("product_name") or data.get("name") or pid
                pcat = data.get("category") or "General"
                data["product"] = {"id": pid, "name": pname, "category": pcat}
        return data

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
