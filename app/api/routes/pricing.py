from fastapi import APIRouter, Request, status

from app.agents.pricing import run_pricing_agent
from app.schemas.pricing import PricingBatchRequest, PricingBatchResponse
from app.schemas.pricing_knowledge_ingestion import (
    HistoricalPricingIngestionRequest,
    HistoricalPricingIngestionResponse,
)
from app.services.pricing_knowledge_ingestion import get_ingestion_service

router = APIRouter(prefix="/api/v1/pricing", tags=["pricing"])


@router.post(
    "/recommend",
    response_model=PricingBatchResponse,
    summary="Recommend batch discount percentages",
    description="Recommends discount percentages for a batch of products within a store context.",
)
async def recommend_discount(request: PricingBatchRequest) -> PricingBatchResponse:
    return run_pricing_agent(request)


@router.post(
    "/knowledge/ingest",
    response_model=HistoricalPricingIngestionResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest authoritative historical pricing events",
    description=(
        "Internal endpoint accepting authoritative historical pricing events from the .NET backend "
        "to build, embed, and store knowledge documents in Qdrant."
    ),
)
async def ingest_historical_pricing_knowledge(
    request_body: HistoricalPricingIngestionRequest,
    raw_request: Request,
) -> HistoricalPricingIngestionResponse:
    request_id = (
        getattr(raw_request.state, "request_id", None)
        or raw_request.headers.get("X-Request-ID")
    )
    service = get_ingestion_service()
    return service.ingest(request_body, request_id=request_id)
