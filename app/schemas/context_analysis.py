from enum import Enum
from pydantic import BaseModel, Field


class AllowedContext(str, Enum):
    WEATHER = "weather"
    LOCAL_EVENTS = "local_events"


class ContextAnalysisResult(BaseModel):
    """Schema for structured context analysis output from the Monitoring Agent."""

    is_sufficient: bool
    missing_context: list[AllowedContext]
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
