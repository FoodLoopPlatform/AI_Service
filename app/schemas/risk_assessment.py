from pydantic import BaseModel, Field

from app.schemas.monitoring import RiskLevel


class RiskAssessmentResult(BaseModel):
    """Schema for structured risk assessment output from the Monitoring Agent."""

    risk_level: RiskLevel
    reason: str
    confidence: float = Field(..., ge=0.0, le=1.0)
