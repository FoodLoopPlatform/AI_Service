from datetime import datetime, timezone
import pytest
from pydantic import ValidationError

from app.tools.events import (
    LocalEvent,
    LocalEventsContext,
    LocalEventsToolError,
    get_local_events,
)
from app.tools.weather import (
    WeatherContext,
    WeatherForecast,
    WeatherToolError,
    get_weather_forecast,
)


def test_weather_tool_valid():
    from_time = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone.utc)

    context = get_weather_forecast(37.7749, -122.4194, from_time, to_time)
    assert isinstance(context, WeatherContext)
    assert len(context.forecast) > 0
    for forecast in context.forecast:
        assert 0.0 <= forecast.precipitation_probability <= 1.0


def test_weather_tool_empty_forecast_when_from_after_to():
    from_time = datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)

    context = get_weather_forecast(37.7749, -122.4194, from_time, to_time)
    assert isinstance(context, WeatherContext)
    assert context.forecast == []


def test_weather_forecast_invalid_precipitation():
    with pytest.raises(ValidationError):
        WeatherForecast(
            timestamp=datetime.now(timezone.utc),
            condition="Storm",
            precipitation_probability=1.5,  # > 1.0
            temperature=18.0,
        )


def test_weather_tool_invalid_coordinates_raises_exception():
    from_time = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(WeatherToolError, match="Invalid latitude"):
        get_weather_forecast(100.0, -122.4194, from_time, to_time)


def test_events_tool_valid():
    from_time = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone.utc)

    context = get_local_events(37.7749, -122.4194, from_time, to_time)
    assert isinstance(context, LocalEventsContext)
    assert len(context.events) > 0
    for event in context.events:
        assert event.distance_km >= 0
        assert event.expected_attendance >= 0


def test_events_tool_empty_when_window_too_short():
    from_time = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 8, 15, 11, 0, 0, tzinfo=timezone.utc)

    context = get_local_events(37.7749, -122.4194, from_time, to_time)
    assert isinstance(context, LocalEventsContext)
    assert context.events == []


def test_local_event_negative_distance_fails():
    with pytest.raises(ValidationError):
        LocalEvent(
            name="Test Event",
            starts_at=datetime.now(timezone.utc),
            distance_km=-2.0,
            expected_attendance=500,
        )


def test_local_event_negative_attendance_fails():
    with pytest.raises(ValidationError):
        LocalEvent(
            name="Test Event",
            starts_at=datetime.now(timezone.utc),
            distance_km=1.0,
            expected_attendance=-50,
        )


def test_events_tool_invalid_coordinates_raises_exception():
    from_time = datetime(2026, 8, 15, 10, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 8, 16, 10, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(LocalEventsToolError, match="Invalid longitude"):
        get_local_events(37.7749, -200.0, from_time, to_time)
