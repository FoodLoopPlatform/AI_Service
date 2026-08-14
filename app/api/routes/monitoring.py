from fastapi import APIRouter

from app.agents.monitoring import run_monitoring_agent
from app.schemas.monitoring import (
    MonitoringRequest,
    MonitoringResponse,
)

router = APIRouter(prefix="/api/v1/monitoring", tags=["monitoring"])


@router.post(
    "/analyze",
    response_model=MonitoringResponse,
    summary="Analyze inventory monitoring request",
    description="Analyzes inventory state and context through the Monitoring Agent workflow to determine routing and risk level.",
)
async def analyze_inventory(request: MonitoringRequest) -> MonitoringResponse:
    return run_monitoring_agent(request)
