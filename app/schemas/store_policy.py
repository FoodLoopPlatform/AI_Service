from enum import Enum
from pydantic import BaseModel, ConfigDict, Field


class OperatingMode(str, Enum):
    ASSISTED = "assisted"
    AUTONOMOUS = "autonomous"


class StorePolicy(BaseModel):
    """Store operating policy configuration owned by the backend microservice."""

    model_config = ConfigDict(extra="forbid")

    store_id: str = Field(..., min_length=1, description="Originating store ID")
    operating_mode: OperatingMode = Field(
        ...,
        description="Backend-configured operating mode for store automation policy",
    )
