from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.config.settings import settings
from app.schemas.historical_pricing import HistoricalPricingEvent


class HistoricalPricingIngestionRequest(BaseModel):
    """Request payload for ingesting authoritative historical pricing events from .NET backend."""

    model_config = ConfigDict(populate_by_name=True)

    events: list[HistoricalPricingEvent] = Field(
        ...,
        description="Batch of authoritative historical pricing events to ingest.",
    )

    @field_validator("events")
    @classmethod
    def validate_events_batch(
        cls, events: list[HistoricalPricingEvent]
    ) -> list[HistoricalPricingEvent]:
        if not events:
            raise ValueError("Historical pricing ingestion request must contain at least one event.")
        max_batch = settings.HISTORICAL_INGESTION_MAX_BATCH_SIZE
        if len(events) > max_batch:
            raise ValueError(
                f"Ingestion batch size ({len(events)}) exceeds configured maximum limit of {max_batch}."
            )
        return events


class HistoricalPricingIngestionResponse(BaseModel):
    """Response payload returned following historical pricing event ingestion."""

    model_config = ConfigDict(populate_by_name=True)

    accepted_count: int = Field(..., ge=0, description="Total number of events validated and accepted.")
    upserted_count: int = Field(..., ge=0, description="Total number of documents upserted to vector store.")
    failed_count: int = Field(..., ge=0, description="Total number of events that failed ingestion.")
    document_ids: list[str] = Field(
        default_factory=list,
        description="List of deterministic document IDs created or updated in the vector store.",
    )
