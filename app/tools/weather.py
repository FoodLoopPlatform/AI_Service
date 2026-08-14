from datetime import datetime, timedelta
from pydantic import BaseModel, Field


class WeatherToolError(Exception):
    """Raised when the weather context retrieval tool fails."""

    pass


class WeatherForecast(BaseModel):
    timestamp: datetime
    condition: str
    precipitation_probability: float = Field(..., ge=0.0, le=1.0)
    temperature: float


class WeatherContext(BaseModel):
    forecast: list[WeatherForecast]


def get_weather_forecast(
    latitude: float,
    longitude: float,
    from_time: datetime,
    to_time: datetime,
) -> WeatherContext:
    """Retrieves deterministic weather forecast context for a location and time window."""
    if not (-90.0 <= latitude <= 90.0):
        raise WeatherToolError(
            f"Invalid latitude value: {latitude}. Must be between -90 and 90."
        )
    if not (-180.0 <= longitude <= 180.0):
        raise WeatherToolError(
            f"Invalid longitude value: {longitude}. Must be between -180 and 180."
        )

    if from_time >= to_time:
        return WeatherContext(forecast=[])

    # Deterministic mock weather forecast generation based on location and time span
    forecast_items: list[WeatherForecast] = []
    current = from_time
    # Generate 6-hourly sample forecast entries up to to_time
    interval = timedelta(hours=6)

    while current < to_time:
        # Deterministic condition selection based on hour
        hour = current.hour
        if hour in (6, 12):
            condition = "Sunny"
            precip = 0.05
            temp = 22.5
        elif hour == 18:
            condition = "Partly Cloudy"
            precip = 0.15
            temp = 19.0
        else:
            condition = "Clear"
            precip = 0.0
            temp = 15.0

        forecast_items.append(
            WeatherForecast(
                timestamp=current,
                condition=condition,
                precipitation_probability=precip,
                temperature=temp,
            )
        )
        current += interval

    return WeatherContext(forecast=forecast_items)
