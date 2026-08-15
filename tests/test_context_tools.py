from datetime import date, datetime, timezone
import pytest
from pydantic import ValidationError

from app.tools.events import (
    Holiday,
    HolidayContext,
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
    from_time = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 1, 31, 23, 59, 59, tzinfo=timezone.utc)

    context = get_local_events(30.0444, 31.2357, from_time, to_time)
    assert isinstance(context, HolidayContext)
    assert isinstance(context, LocalEventsContext)
    assert len(context.holidays) > 0
    for holiday in context.holidays:
        assert isinstance(holiday.date, date)
        assert len(holiday.name) > 0
        assert holiday.country_code == "EG"


def test_events_tool_empty_when_window_outside_holidays():
    from_time = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    to_time = datetime(2026, 2, 5, 0, 0, 0, tzinfo=timezone.utc)

    context = get_local_events(30.0444, 31.2357, from_time, to_time)
    assert isinstance(context, HolidayContext)
    assert context.holidays == []


def test_events_tool_naive_datetime_raises_exception():
    from_time = datetime(2026, 1, 1, 0, 0, 0)
    to_time = datetime(2026, 1, 10, 0, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(LocalEventsToolError, match="timezone-aware"):
        get_local_events(30.0444, 31.2357, from_time, to_time)
