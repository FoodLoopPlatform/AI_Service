from datetime import datetime, timedelta
from pydantic import BaseModel, Field


class LocalEventsToolError(Exception):
    """Raised when the local events context retrieval tool fails."""

    pass


class LocalEvent(BaseModel):
    name: str
    starts_at: datetime
    distance_km: float = Field(..., ge=0.0)
    expected_attendance: int = Field(..., ge=0)


class LocalEventsContext(BaseModel):
    events: list[LocalEvent]


def get_local_events(
    latitude: float,
    longitude: float,
    from_time: datetime,
    to_time: datetime,
) -> LocalEventsContext:
    """Retrieves deterministic local events context for a location and time window."""
    if not (-90.0 <= latitude <= 90.0):
        raise LocalEventsToolError(
            f"Invalid latitude value: {latitude}. Must be between -90 and 90."
        )
    if not (-180.0 <= longitude <= 180.0):
        raise LocalEventsToolError(
            f"Invalid longitude value: {longitude}. Must be between -180 and 180."
        )

    if from_time >= to_time:
        return LocalEventsContext(events=[])

    # Deterministic mock event generation based on location and time span
    event_start = from_time + timedelta(hours=12)
    if event_start < to_time:
        sample_event = LocalEvent(
            name="Community Farmers Market & Food Festival",
            starts_at=event_start,
            distance_km=1.2,
            expected_attendance=1500,
        )
        return LocalEventsContext(events=[sample_event])

    return LocalEventsContext(events=[])
